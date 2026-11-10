"""Checks for the mask head, mixture-phase reconstruction and SDR scoring.

All signals are synthetic, so these tests need no dataset.
"""

import math

import pytest
import torch

from curirim.audio import MODEL_BINS, stft
from curirim.data import SAMPLE_RATE, STEMS
from curirim.evaluate import median_sdr, oracle_masks
from curirim.model import TinyNet, stem_masks
from curirim.separate import apply_masks, separate


def tones(seconds: float = 2.0) -> torch.Tensor:
    """Four stereo stems `(4, 2, samples)`, each a pure tone at its own frequency.

    The tones barely overlap in frequency, so ideal masks can separate them
    almost perfectly.
    """
    t = torch.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    stems = torch.stack([0.2 * torch.sin(2 * math.pi * f * t) for f in (220, 880, 3000, 7000)])
    return stems.unsqueeze(1).repeat(1, 2, 1)


def test_stem_masks_shape_and_sum_to_one():
    masks = stem_masks(torch.randn(3, 8, 16, 10))
    assert masks.shape == (3, 4, 2, 16, 10)
    assert torch.allclose(masks.sum(1), torch.ones(3, 2, 16, 10))
    assert masks.min() >= 0


def test_tiny_net_keeps_any_input_size():
    for bins, frames in [(MODEL_BINS, 517), (64, 3), (33, 7)]:
        masks = TinyNet()(torch.rand(2, 2, bins, frames))
        assert masks.shape == (2, 4, 2, bins, frames)


def test_separate_output_shape_and_stems_sum_to_mixture():
    # Softmax masks sum to 1 in every bin, including the copied top bin,
    # so the stems must add back up to the input. White noise has energy in
    # the top bin too, so this would fail if that bin's mask were zero.
    mixture = torch.rand(2, 3 * SAMPLE_RATE) * 2 - 1
    stems = separate(TinyNet(), mixture)
    assert stems.shape == (len(STEMS), 2, mixture.shape[-1])
    assert (stems.sum(0) - mixture).abs().max() < 1e-5


def test_equal_masks_give_a_quarter_of_the_mixture():
    mixture = torch.rand(2, SAMPLE_RATE) * 2 - 1
    spec = stft(mixture)
    masks = torch.full((4, 2, MODEL_BINS, spec.shape[-1]), 0.25)
    stems = apply_masks(masks, spec, mixture.shape[-1])
    assert torch.allclose(stems, (mixture / 4).expand(4, -1, -1), atol=1e-5)


def test_oracle_masks_separate_distinct_tones():
    stems = tones()
    mixture = stems.sum(0)
    estimate = apply_masks(oracle_masks(stems), stft(mixture), mixture.shape[-1])
    scores = median_sdr(stems, estimate)
    assert all(score > 20 for score in scores.values()), scores


def test_median_sdr_ranks_good_above_bad():
    stems = tones()
    bad = stems.sum(0).expand(4, -1, -1) / 4  # every stem = mixture / 4
    good = stems + 0.01 * torch.randn_like(stems)
    good_scores, bad_scores = median_sdr(stems, good), median_sdr(stems, bad)
    assert all(good_scores[s] > bad_scores[s] + 10 for s in STEMS)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_separate_runs_on_gpu():
    stems = separate(TinyNet().cuda(), torch.rand(2, SAMPLE_RATE))
    assert stems.is_cuda
