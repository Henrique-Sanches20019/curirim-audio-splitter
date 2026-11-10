"""Separation quality measurement with museval (BSS Eval v4).

Scores follow the SiSEC/MUSDB18 convention so they compare with published
results: SDR is computed on 1 s windows (hop 1 s), the median over windows
gives one number per stem and song, and the median over songs gives the
reported number per stem.
"""

import numpy as np
import torch

from curirim.audio import crop_freq, stft
from curirim.data import SAMPLE_RATE, STEMS


def median_sdr(references: torch.Tensor, estimates: torch.Tensor) -> dict[str, float]:
    """Median SDR in dB per stem for one song.

    Both inputs are `(4, 2, samples)` in STEMS order. Windows where a reference
    stem is silent give NaN in museval and are ignored by the median (a song
    without vocals still gets scores for the other stems).
    """
    import museval  # imported here: slow to import and only needed for evaluation

    sdr, _, _, _ = museval.evaluate(
        references.permute(0, 2, 1).double().cpu().numpy(),
        estimates.permute(0, 2, 1).double().cpu().numpy(),
        win=SAMPLE_RATE,
        hop=SAMPLE_RATE,
    )
    return {stem: float(np.nanmedian(values)) for stem, values in zip(STEMS, sdr)}


def oracle_masks(stems: torch.Tensor) -> torch.Tensor:
    """Ideal ratio masks `(4, 2, 1024, frames)` computed from the true stems `(4, 2, samples)`.

    mask_j = |stem_j| / sum_k |stem_k| at every time-frequency point: the
    masks a perfect network would come close to. Separating with them gives
    the best SDR this mask-and-mixture-phase design can reach, which makes
    them a pipeline check: low oracle SDR means a bug, not a weak model.
    """
    magnitude = crop_freq(stft(stems)).abs()
    return magnitude / magnitude.sum(0, keepdim=True).clamp_min(1e-8)
