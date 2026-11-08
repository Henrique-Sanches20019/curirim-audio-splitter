"""Checks for the STFT signal path: shapes and lossless round trips."""

import math

import pytest
import torch

from curirim.audio import HOP, MODEL_BINS, N_BINS, crop_freq, istft, pad_freq, stft
from curirim.data import SAMPLE_RATE

SIX_SECONDS = 6 * SAMPLE_RATE  # 264,600 samples, the training crop length


def noise(*shape: int) -> torch.Tensor:
    """Seeded white noise in [-1, 1): energy in every bin, the hardest round-trip case."""
    generator = torch.Generator().manual_seed(0)
    return torch.rand(*shape, generator=generator) * 2 - 1


def test_six_second_crop_shape():
    # (batch, channels, samples) -> (batch, channels, 1025 bins, 517 frames)
    spec = stft(noise(3, 2, SIX_SECONDS))
    assert spec.shape == (3, 2, N_BINS, 517)
    assert spec.is_complex()


@pytest.mark.parametrize("samples", [SIX_SECONDS, SIX_SECONDS + 1, 12345])
def test_round_trip_is_lossless(samples):
    # Lengths that are and are not multiples of the hop must all come back exactly.
    audio = noise(2, 2, samples)
    restored = istft(stft(audio), length=samples)
    assert restored.shape == audio.shape
    assert (restored - audio).abs().max() < 1e-5


def test_frame_count_formula():
    for samples in (HOP * 10, HOP * 10 + 1, HOP * 10 - 1):
        assert stft(noise(samples)).shape[-1] == 1 + samples // HOP


def test_crop_then_pad_restores_shape_with_zero_top_bin():
    spec = stft(noise(2, 2, SIX_SECONDS))
    cropped = crop_freq(spec)
    assert cropped.shape[-2] == MODEL_BINS
    padded = pad_freq(cropped)
    assert padded.shape == spec.shape
    assert torch.equal(padded[..., :MODEL_BINS, :], cropped)
    assert padded[..., MODEL_BINS:, :].abs().max() == 0


def test_crop_pad_istft_keeps_band_limited_audio():
    # Music in MUSDB18 has nothing near 22.05 kHz, so dropping the top bin
    # must barely change a signal with no energy there (a 1 kHz tone here).
    # Not exactly lossless: the Hann window leaks a trace of the tone into the
    # top bin. Measured max error 1.74e-5 (-89 dB re the tone), the same in
    # float64, so it is real content, not rounding. On a real MUSDB18 clip the
    # crop round trip measured 120 dB SNR.
    t = torch.arange(SIX_SECONDS) / SAMPLE_RATE
    tone = 0.5 * torch.sin(2 * math.pi * 1000 * t).repeat(2, 1)
    restored = istft(pad_freq(crop_freq(stft(tone))), length=SIX_SECONDS)
    assert restored.shape == tone.shape
    assert (restored - tone).abs().max() < 1e-4
