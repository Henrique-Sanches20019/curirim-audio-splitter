"""Step 4 exit check: run the whole separation pipeline on full validation songs.

For each song, two separations are scored with museval median SDR:

- oracle:  ideal ratio masks from the true stems. The best this design
           (mask × mixture magnitude, mixture phase) can do. If these scores
           are low, the pipeline has a bug.
- tiny:    the untrained 1-layer TinyNet (seeded). Expected to be bad; it
           proves a network's masks flow through STFT, inverse STFT and
           museval with the right shapes.

Both use the song's real mixture, as published MUSDB18 results do.

Usage:
    MUSDB_PATH=musdb18 python scripts/placeholder_end_to_end.py
    MUSDB_PATH=musdb18 python scripts/placeholder_end_to_end.py --tracks 3
"""

import argparse
import time

import numpy as np
import torch

from curirim.audio import stft
from curirim.data import STEMS, open_musdb
from curirim.evaluate import median_sdr, oracle_masks
from curirim.model import TinyNet
from curirim.separate import apply_masks, separate


def load(track) -> tuple[torch.Tensor, torch.Tensor]:
    """A musdb track's real mixture `(2, samples)` and stems `(4, 2, samples)` as float32 tensors."""
    mixture = torch.from_numpy(track.audio.T).float()
    stems = torch.stack([torch.from_numpy(track.targets[s].audio.T).float() for s in STEMS])
    return mixture, stems


def print_row(label: str, scores: dict[str, float]) -> None:
    """One table row: a label, then SDR per stem in dB."""
    print(f"{label:38s}" + "".join(f"{scores[s]:8.2f}" for s in STEMS))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracks", type=int, default=2, help="first N validation songs")
    parser.add_argument("--seed", type=int, default=0, help="seed for TinyNet's random weights")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model = TinyNet()
    results = {"oracle": [], "tiny": []}

    print(f"{'median SDR (dB)':38s}" + "".join(f"{s:>8s}" for s in STEMS))
    for track in open_musdb("train", "valid").tracks[: args.tracks]:
        start = time.time()
        mixture, stems = load(track)
        estimates = {
            "oracle": apply_masks(oracle_masks(stems), stft(mixture), mixture.shape[-1]),
            "tiny": separate(model, mixture),
        }
        for name, estimate in estimates.items():
            # Softmax masks must give stems that add back up to the mixture.
            assert torch.allclose(estimate.sum(0), mixture, atol=1e-4), f"{name}: stems do not sum to the mixture"
            results[name].append(median_sdr(stems, estimate))
            print_row(f"{track.name[:28]} {name}", results[name][-1])
        print(f"  ({track.duration:.0f} s song, {time.time() - start:.0f} s to separate and score)")

    for name, rows in results.items():
        print_row(f"median over songs: {name}", {s: float(np.median([r[s] for r in rows])) for s in STEMS})


if __name__ == "__main__":
    main()
