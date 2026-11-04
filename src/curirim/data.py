"""MUSDB18 access.

The dataset root comes from the MUSDB_PATH environment variable (the same
variable the `musdb` package reads) or an explicit argument; never a
hard-coded path. The root may hold the original `.stem.mp4` files or the WAV
tree written by `scripts/decode_musdb.py`; the format is detected.
"""

import os
from pathlib import Path

import musdb

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
