"""Step 1 data check: split summary, one track's spectrograms, and listening clips.

Prints track counts and durations for the three standard splits, then for one
track: the mixture-vs-sum-of-stems error, a PNG with log-magnitude
spectrograms (same STFT as the model: n_fft 2048, hop 512, Hann), and short
WAV clips of the mixture and each stem to listen to.

Usage:
    MUSDB_PATH=musdb18 python scripts/inspect_track.py --track 0
    MUSDB_PATH=musdb18 python scripts/inspect_track.py --split valid --track "Actions - One Minute Smile"
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch

from curirim.data import STEMS, open_musdb

SPLITS = {"train": ("train", "train"), "valid": ("train", "valid"), "test": ("test", None)}
N_FFT, HOP = 2048, 512


def summarize() -> None:
    for label, (subset, split) in SPLITS.items():
        durations = np.array([t.duration for t in open_musdb(subset, split).tracks])
        print(
            f"{label:5s} {len(durations):3d} tracks  total {durations.sum() / 3600:.2f} h  "
            f"min {durations.min():.0f} s  median {np.median(durations):.0f} s  max {durations.max():.0f} s"
        )


def log_magnitude(audio: np.ndarray) -> np.ndarray:
    """Left-channel log-magnitude spectrogram in dB, shape (1025, frames)."""
    x = torch.from_numpy(audio[:, 0]).float()
    spec = torch.stft(x, N_FFT, HOP, window=torch.hann_window(N_FFT), return_complex=True)
    return 20 * torch.log10(spec.abs() + 1e-7).numpy()


def inspect(split: str, track_ref: str, clip_seconds: float, out_root: Path) -> None:
    tracks = open_musdb(*SPLITS[split]).tracks
    track = tracks[int(track_ref)] if track_ref.isdigit() else next(t for t in tracks if t.name == track_ref)
    audio = {"mixture": track.audio} | {s: track.targets[s].audio for s in STEMS}

    print(f"\n{track.name}  ({split}, {track.rate} Hz, {track.duration:.1f} s, shape {track.audio.shape})")
    # MUSDB18 stems are separately AAC-encoded, so their sum only approximates
    # the mixture; training builds its own mixture from the stems instead.
    residual = audio["mixture"] - sum(audio[s] for s in STEMS)
    snr = 10 * np.log10(np.sum(audio["mixture"] ** 2) / np.sum(residual**2))
    print(f"mixture vs sum of stems: max abs error {np.abs(residual).max():.4f}, SNR {snr:.1f} dB")

    out = out_root / track.name
    out.mkdir(parents=True, exist_ok=True)

    # Clip from the middle of the song, where all stems are usually active.
    start = max(0, len(track.audio) // 2 - int(clip_seconds * track.rate) // 2)
    stop = start + int(clip_seconds * track.rate)
    fig, axes = plt.subplots(len(audio), 1, figsize=(12, 2.2 * len(audio)), sharex=True)
    for ax, (name, data) in zip(axes, audio.items()):
        clip = data[start:stop]
        sf.write(out / f"{name}.wav", clip, track.rate, subtype="PCM_16")
        db = log_magnitude(clip)
        ax.imshow(
            db, origin="lower", aspect="auto", cmap="magma", vmin=db.max() - 80, vmax=db.max(),
            extent=(0, clip.shape[0] / track.rate, 0, track.rate / 2000),
        )
        ax.set_ylabel(f"{name}\nkHz")
    axes[-1].set_xlabel("seconds")
    fig.suptitle(track.name)
    fig.tight_layout()
    fig.savefig(out / "spectrograms.png", dpi=100)
    print(f"wrote {out}/spectrograms.png and {len(audio)} WAV clips of {clip_seconds:g} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=SPLITS, default="train")
    parser.add_argument("--track", default="0", help="index within the split, or exact track name")
    parser.add_argument("--clip-seconds", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=Path("outputs/inspect"))
    parser.add_argument("--no-summary", action="store_true", help="skip the split duration summary")
    args = parser.parse_args()
    if not args.no_summary:
        summarize()
    inspect(args.split, args.track, args.clip_seconds, args.out)


if __name__ == "__main__":
    main()
