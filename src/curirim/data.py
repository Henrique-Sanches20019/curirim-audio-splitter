"""MUSDB18 access and the PyTorch datasets that feed the model.

The dataset root comes from the MUSDB_PATH environment variable (the same
variable the `musdb` package reads) or an explicit argument; never a
hard-coded path. The root may hold the original `.stem.mp4` files or the WAV
tree written by `scripts/decode_musdb.py`; the format is detected.

The training and validation datasets read only the WAV tree: `soundfile`
reads a 6 s crop of a WAV file in milliseconds, while every `.stem.mp4` read
starts an ffmpeg process and decodes AAC.

Tensors are channels-first: a mixture is `(2, samples)`, the stems
`(4, 2, samples)` in `STEMS` order.
"""

import os
from pathlib import Path

import musdb
import soundfile as sf
import torch
from torch.utils.data import Dataset

STEMS = ("vocals", "drums", "bass", "other")
SAMPLE_RATE = 44100


def musdb_root(root: str | os.PathLike | None = None) -> Path:
    """Resolve the dataset root from the argument or MUSDB_PATH."""
    root = root or os.environ.get("MUSDB_PATH")
    if not root:
        raise RuntimeError("Set MUSDB_PATH or pass the dataset root explicitly.")
    path = Path(root)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset root {path} does not exist.")
    return path


def is_wav_root(root: Path) -> bool:
    """True for a decoded WAV tree, False for the original .stem.mp4 files."""
    return not any(root.glob("*/*.stem.mp4"))


def open_musdb(
    subset: str,
    split: str | None = None,
    root: str | os.PathLike | None = None,
) -> musdb.DB:
    """Open one standard MUSDB18 split.

    subset="train", split="train" -> 86 tracks
    subset="train", split="valid" -> 14 tracks
    subset="test"                 -> 50 tracks (final evaluation only)
    """
    path = musdb_root(root)
    if not (path / subset).is_dir():
        raise FileNotFoundError(f"{path} has no {subset}/ folder.")
    return musdb.DB(root=str(path), subsets=subset, split=split, is_wav=is_wav_root(path))


def track_dirs(subset: str, split: str | None = None, root: str | os.PathLike | None = None) -> list[Path]:
    """Folders of one standard split in a decoded WAV tree, e.g. `track_dirs("train", "valid")`.

    Each folder holds `vocals.wav`, `drums.wav`, `bass.wav`, `other.wav` (and
    `mixture.wav`). A partially decoded tree yields only the tracks present.
    """
    path = musdb_root(root)
    if not is_wav_root(path):
        raise ValueError(f"{path} holds .stem.mp4 files; decode it first with scripts/decode_musdb.py.")
    return [Path(track.path).parent for track in open_musdb(subset, split, path).tracks]


def _stem_frames(track_dir: Path) -> int:
    """Usable length of a track in samples: the shortest of its four stem files."""
    return min(sf.info(track_dir / f"{stem}.wav").frames for stem in STEMS)


def _read_crop(path: Path, start: int, frames: int) -> torch.Tensor:
    """Read `frames` samples from `start` of a stereo WAV as a `(2, frames)` float32 tensor."""
    audio, _ = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    return torch.from_numpy(audio.T.copy())


def _check_lengths(track_dirs: list[Path], frames: int) -> list[int]:
    """Length of every track; fail early if a track is shorter than one crop."""
    if not track_dirs:
        raise ValueError("No tracks given: check MUSDB_PATH and the decoded WAV tree.")
    lengths = [_stem_frames(d) for d in track_dirs]
    short = [d.name for d, n in zip(track_dirs, lengths) if n < frames]
    if short:
        raise ValueError(f"Tracks shorter than one crop ({frames} samples): {short}")
    return lengths


class TrainDataset(Dataset):
    """Random augmented crops for training. Every item is a new random mix.

    Each of the four stems is cut from an independently chosen track at an
    independent random offset (cross-song remix), then gets a random gain and a
    random left/right swap. The mixture is the sum of the augmented stems, so
    input and targets always match exactly (MUSDB18's own mixture.wav does not
    equal the sum of its separately AAC-encoded stems).

    Why remix: with only 86 training songs the model would memorise them;
    combining stems across songs creates mixes it has never heard.

    Tracks are drawn in proportion to their length, so every second of audio is
    equally likely: a 13 s track is not sampled as often as a 10 min one.

    Randomness comes from torch's global generator, which the DataLoader seeds
    differently in each worker process, so workers never repeat each other.

    Returns `(mixture (2, samples), stems (4, 2, samples))`.
    """

    def __init__(
        self,
        track_dirs: list[Path],
        crop_seconds: float = 6.0,
        crops_per_track: int = 64,
        gain_range: tuple[float, float] = (0.25, 1.25),
        swap_probability: float = 0.5,
    ):
        """`crops_per_track` sets the epoch length: len = tracks × crops_per_track."""
        self.track_dirs = list(track_dirs)
        self.frames = round(crop_seconds * SAMPLE_RATE)
        self.lengths = _check_lengths(self.track_dirs, self.frames)
        self.weights = torch.tensor(self.lengths, dtype=torch.float64)
        self.crops_per_track = crops_per_track
        self.gain_range = gain_range
        self.swap_probability = swap_probability

    def __len__(self) -> int:
        """Number of items per epoch. The index itself is ignored: every item is random."""
        return len(self.track_dirs) * self.crops_per_track

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """One random remixed, augmented crop."""
        low, high = self.gain_range
        stems = []
        for stem in STEMS:
            track = int(torch.multinomial(self.weights, 1))
            start = int(torch.randint(0, self.lengths[track] - self.frames + 1, ()))
            audio = _read_crop(self.track_dirs[track] / f"{stem}.wav", start, self.frames)
            audio = audio * (low + (high - low) * torch.rand(()))
            if torch.rand(()) < self.swap_probability:
                audio = audio.flip(0)
            stems.append(audio)
        stems = torch.stack(stems)
        return stems.sum(0), stems


class ValidDataset(Dataset):
    """Fixed, unaugmented crops for validation, identical in every run.

    Each track gives `crops_per_track` crops at evenly spaced offsets from the
    start to the end of the song. That is deterministic without a seed and
    covers intro, middle and outro, so validation loss is comparable across
    epochs and runs. Stems come from the same track at the same offset (no
    remix), and the mixture is their sum, as in training.

    Returns `(mixture (2, samples), stems (4, 2, samples))`.
    """

    def __init__(self, track_dirs: list[Path], crop_seconds: float = 6.0, crops_per_track: int = 8):
        """Precompute every (track, offset) pair once."""
        self.track_dirs = list(track_dirs)
        self.frames = round(crop_seconds * SAMPLE_RATE)
        lengths = _check_lengths(self.track_dirs, self.frames)
        self.items = [
            (track, start)
            for track, length in enumerate(lengths)
            for start in torch.linspace(0, length - self.frames, crops_per_track).long().tolist()
        ]

    def __len__(self) -> int:
        """Total number of fixed crops: tracks × crops_per_track."""
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """The `index`-th fixed crop, always the same audio."""
        track, start = self.items[index]
        stems = torch.stack(
            [_read_crop(self.track_dirs[track] / f"{stem}.wav", start, self.frames) for stem in STEMS]
        )
        return stems.sum(0), stems
