"""Checks for the U-Net: shapes for any song length, valid masks, dropout and gradients.

All inputs are random magnitudes, so these tests need no dataset.
"""

import pytest
import torch

from curirim.audio import MODEL_BINS
from curirim.data import SAMPLE_RATE, STEMS
from curirim.model import UNet
from curirim.separate import separate


@pytest.fixture(scope="module")
def unet() -> UNet:
    """One seeded U-Net shared by the tests (building it takes ~0.1 s, running it more)."""
    torch.manual_seed(0)
    return UNet()


@pytest.mark.parametrize(
    "bins, frames",
    [
        (MODEL_BINS, 517),  # a 6 s training crop: frames padded 517 -> 576 inside
        (MODEL_BINS, 64),  # already a multiple of 64: no padding
        (MODEL_BINS, 1),  # shorter than one 64-frame block
        (MODEL_BINS, 130),  # just over a multiple of 64
        (100, 37),  # bins not a multiple of 64 either
    ],
)
def test_output_shape_equals_input_shape(unet, bins, frames):
    # Padding to multiples of 64 and cropping back must give one mask per input point,
    # which also proves every skip connection lines up (torch.cat fails otherwise).
    with torch.no_grad():
        masks = unet.eval()(torch.rand(2, 2, bins, frames))
    assert masks.shape == (2, len(STEMS), 2, bins, frames)


def test_masks_sum_to_one(unet):
    with torch.no_grad():
        masks = unet.eval()(torch.rand(1, 2, MODEL_BINS, 200) * 50)
    assert masks.min() >= 0
    assert torch.allclose(masks.sum(1), torch.ones(1, 2, MODEL_BINS, 200), atol=1e-6)


def test_dropout_only_in_training_mode(unet):
    x = torch.rand(1, 2, 128, 64)
    with torch.no_grad():
        unet.eval()
        assert torch.equal(unet(x), unet(x))  # eval: deterministic
        unet.train()
        assert not torch.equal(unet(x), unet(x))  # train: dropout changes the output
    unet.eval()


def test_gradients_reach_the_first_layer(unet):
    # A broken skip or a detached tensor would leave the first encoder layer untrained.
    unet.train()
    unet.zero_grad()
    unet(torch.rand(2, 2, 128, 64))[:, 0].mean().backward()
    first = unet.encoder[0][0].weight.grad
    assert first is not None and first.abs().sum() > 0
    unet.zero_grad()
    unet.eval()


def test_parameter_count(unet):
    # Documents the architecture: 6 encoder + 6 decoder 5×5 blocks (16→512 channels) + 1×1 head.
    assert sum(p.numel() for p in unet.parameters()) == 9_835_896


def test_separate_with_unet_sums_to_mixture(unet):
    mixture = torch.rand(2, 2 * SAMPLE_RATE) * 2 - 1
    stems = separate(unet, mixture)
    assert stems.shape == (len(STEMS), 2, mixture.shape[-1])
    assert (stems.sum(0) - mixture).abs().max() < 1e-5
