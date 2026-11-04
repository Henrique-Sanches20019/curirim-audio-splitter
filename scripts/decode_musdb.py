"""Decode MUSDB18 .stem.mp4 files into a WAV tree, once per machine.

Decoding AAC on every random training crop is slow, so training reads WAV.
Output layout matches what `musdb.DB(is_wav=True)` expects:

    <out>/<train|test>/<track name>/{mixture,vocals,drums,bass,other}.wav

Unlike the bundled `musdbconvert`, this writes only the five files training
and evaluation need (no accompaniment/linear_mixture), skips tracks already
decoded so an interrupted Colab/Kaggle session can rerun it, and can limit the
number of tracks for a small local debug copy.

Usage:
    python scripts/decode_musdb.py --src musdb18 --out musdb18-wav
    python scripts/decode_musdb.py --src musdb18 --out musdb18-wav --limit 3
"""

import argparse
from pathlib import Path

import musdb
import soundfile as sf

from curirim.data import STEMS, musdb_root

FILES = ("mixture", *STEMS)


def decode(src: Path, out: Path, subsets: list[str], limit: int | None) -> None:
    for subset in subsets:
        tracks = musdb.DB(root=str(src), subsets=subset, is_wav=False).tracks[:limit]
        for i, track in enumerate(tracks, 1):
            track_dir = out / subset / track.name
            if all((track_dir / f"{name}.wav").is_file() for name in FILES):
                print(f"[{subset} {i}/{len(tracks)}] skip {track.name}")
                continue
            track_dir.mkdir(parents=True, exist_ok=True)
            audio = {"mixture": track.audio} | {s: track.targets[s].audio for s in STEMS}
            for name, data in audio.items():
                # Write to a temp name first so a killed session never leaves a
                # truncated file that the skip check would accept.
                tmp = track_dir / f"{name}.tmp.wav"
                sf.write(tmp, data, track.rate, subtype="PCM_16")
                tmp.replace(track_dir / f"{name}.wav")
            print(f"[{subset} {i}/{len(tracks)}] {track.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", help="folder with train/ and test/ .stem.mp4 (default: MUSDB_PATH)")
    parser.add_argument("--out", required=True, type=Path, help="output WAV root")
    parser.add_argument("--subsets", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--limit", type=int, help="decode only the first N tracks per subset")
    args = parser.parse_args()
    decode(musdb_root(args.src), args.out, args.subsets, args.limit)


if __name__ == "__main__":
    main()
