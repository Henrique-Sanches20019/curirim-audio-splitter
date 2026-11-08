"""Checks for the training and validation datasets.

Most tests build a tiny synthetic WAV tree, so they need no dataset. Each
synthetic stem stores its sample position in the left channel and its track
number in the right channel; with augmentation off, that reveals exactly which
track and offset every crop came from.

Track numbers in the right channel start at 1, so a gain shows up directly as
right-channel value / track number.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from curirim.data import STEMS, SAMPLE_RATE, TrainDataset, ValidDataset, is_wav_root, track_dirs

CROP_SECONDS = 0.1
FRAMES = round(CROP_SECONDS * SAMPLE_RATE)  # 4410 samples
NO_AUGMENTATION = {"gain_range": (1.0, 1.0), "swap_probability": 0.0}


def make_tree(root: Path, lengths: list[int]) -> list[Path]:
    """Write one fake track per length; left = sample position, right = track number (from 1).

    Float WAV keeps the values exact (16-bit PCM would clip and round them).
    """
    dirs = []
    for number, length in enumerate(lengths):
        track = root / f"track{number}"
        track.mkdir(parents=True)
        audio = np.stack([np.arange(length), np.full(length, number + 1)], axis=1).astype(np.float32)
        for stem in STEMS:
            sf.write(track / f"{stem}.wav", audio, SAMPLE_RATE, subtype="FLOAT")
        dirs.append(track)
    return dirs


@pytest.fixture
def tracks(tmp_path):
    """Five 1 s fake tracks."""
    return make_tree(tmp_path, [SAMPLE_RATE] * 5)


def test_train_item_shapes_and_mixture_is_sum(tracks):
    mixture, stems = TrainDataset(tracks, CROP_SECONDS)[0]
    assert mixture.shape == (2, FRAMES)
    assert stems.shape == (len(STEMS), 2, FRAMES)
    assert mixture.dtype == stems.dtype == torch.float32
    assert torch.equal(mixture, stems.sum(0))


def test_train_epoch_length(tracks):
    assert len(TrainDataset(tracks, CROP_SECONDS, crops_per_track=7)) == 5 * 7


def test_train_crops_are_contiguous_and_in_bounds(tracks):
    dataset = TrainDataset(tracks, CROP_SECONDS, **NO_AUGMENTATION)
    for i in range(50):
        _, stems = dataset[i]
        positions = stems[:, 0]
        assert torch.all(positions.diff(dim=-1) == 1)  # one unbroken piece of audio
        assert positions.min() >= 0 and positions.max() < SAMPLE_RATE


def test_train_remixes_stems_across_tracks(tracks):
    # With 5 tracks, the 4 stems of an item come from one track only 1 time in 125.
    dataset = TrainDataset(tracks, CROP_SECONDS, **NO_AUGMENTATION)
    sources = [dataset[i][1][:, 1, 0] for i in range(20)]
    assert any(len(set(s.tolist())) > 1 for s in sources)


def test_train_gain_stays_in_range_and_varies(tracks):
    # One track whose right channel is 1.0, so the right channel equals the gain.
    dataset = TrainDataset(tracks[:1], CROP_SECONDS, gain_range=(0.25, 1.25), swap_probability=0.0)
    gains = torch.cat([dataset[i][1][:, 1, 0] for i in range(50)])
    assert gains.min() >= 0.25 and gains.max() <= 1.25
    assert gains.std() > 0.1  # each stem gets its own random gain


def test_train_swap_flips_left_and_right(tracks):
    _, stems = TrainDataset(tracks, CROP_SECONDS, gain_range=(1.0, 1.0), swap_probability=1.0)[0]
    # Swapped: the track number is now on the left, the position ramp on the right.
    assert torch.all(stems[:, 1].diff(dim=-1) == 1)
    assert torch.all(stems[:, 0] == stems[:, 0, :1])


def test_train_prefers_longer_tracks(tmp_path):
    # Track 1 is 9 times longer, so it should provide about 90% of the stems.
    dirs = make_tree(tmp_path, [SAMPLE_RATE, 9 * SAMPLE_RATE])
    dataset = TrainDataset(dirs, CROP_SECONDS, **NO_AUGMENTATION)
    torch.manual_seed(0)
    picks = torch.cat([dataset[i][1][:, 1, 0] for i in range(200)])
    assert 0.8 < (picks == 2).float().mean() < 1.0


def test_dataloader_workers_draw_different_crops(tracks):
    # A classic silent bug: workers sharing one random state return identical items.
    loader = DataLoader(TrainDataset(tracks, CROP_SECONDS, **NO_AUGMENTATION), batch_size=1, num_workers=2)
    first, second = [stems for _, stems in list(loader)[:2]]
    assert not torch.equal(first, second)


def test_valid_is_deterministic_and_unaugmented(tracks):
    a = ValidDataset(tracks, CROP_SECONDS, crops_per_track=4)
    b = ValidDataset(tracks, CROP_SECONDS, crops_per_track=4)
    assert len(a) == 5 * 4
    for i in range(len(a)):
        mixture, stems = a[i]
        assert torch.equal(stems, b[i][1])
        assert torch.equal(mixture, stems.sum(0))
        # No remix: all stems from one track at one offset; no swap: ramp on the left.
        assert torch.all(stems == stems[:1])
        assert torch.all(stems[:, 0].diff(dim=-1) == 1)


def test_valid_offsets_span_the_whole_track(tracks):
    dataset = ValidDataset(tracks[:1], CROP_SECONDS, crops_per_track=3)
    starts = [int(dataset[i][1][0, 0, 0]) for i in range(3)]
    assert starts == [0, (SAMPLE_RATE - FRAMES) // 2, SAMPLE_RATE - FRAMES]


def test_track_shorter_than_a_crop_is_rejected(tmp_path):
    dirs = make_tree(tmp_path, [FRAMES - 1])
    with pytest.raises(ValueError, match="shorter than one crop"):
        TrainDataset(dirs, CROP_SECONDS)


def test_track_dirs_rejects_mp4_root(tmp_path):
    (tmp_path / "train").mkdir()
    (tmp_path / "train" / "song.stem.mp4").touch()
    with pytest.raises(ValueError, match="decode it first"):
        track_dirs("train", root=tmp_path)


real_wav = os.environ.get("MUSDB_PATH")
needs_wav = pytest.mark.skipif(
    not real_wav or not Path(real_wav).is_dir() or not is_wav_root(Path(real_wav)),
    reason="MUSDB_PATH does not point to a decoded WAV tree",
)


@needs_wav
def test_real_train_item():
    mixture, stems = TrainDataset(track_dirs("train", "train"))[0]
    assert mixture.shape == (2, 6 * SAMPLE_RATE)
    assert stems.shape == (4, 2, 6 * SAMPLE_RATE)
    assert torch.equal(mixture, stems.sum(0))
