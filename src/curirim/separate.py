"""Turn a mixture waveform into four stem waveforms with a separation network.

Reconstruction rule (fixed project decision): stem = mask × mixture
magnitude, with the mixture's phase, then inverse STFT. Never Griffin-Lim.

A real-valued mask times the complex mixture STFT is exactly that: it scales
the magnitude and leaves the phase untouched. So the code multiplies the
complex spectrogram directly instead of splitting it into magnitude and phase.

Long songs are processed in one pass for now; chunked overlap-add arrives in
step 8, when the U-Net makes one-pass memory too high.
"""

import torch
from torch import nn

from curirim.audio import crop_freq, istft, stft


def apply_masks(masks: torch.Tensor, mixture_spec: torch.Tensor, length: int) -> torch.Tensor:
    """Masks `(..., 4, 2, 1024, frames)` and complex mixture STFT `(..., 2, 1025, frames)` to stems `(..., 4, 2, length)`.

    The model never sees the top (22.05 kHz) bin, so its mask is copied from
    the bin below. Copying keeps the masks summing to 1 in every bin, so the
    stems still add up exactly to the mixture; padding the mask with zeros
    would silently delete that bin from every stem.
    """
    masks = torch.cat([masks, masks[..., -1:, :]], dim=-2)
    return istft(masks * mixture_spec.unsqueeze(-4), length)


@torch.no_grad()
def separate(model: nn.Module, mixture: torch.Tensor) -> torch.Tensor:
    """Separate one stereo waveform `(2, samples)` into stems `(4, 2, samples)`, in STEMS order.

    Runs the model in eval mode (dropout off, BatchNorm statistics frozen) on
    the device the model's parameters live on, and returns the stems there.
    """
    model.eval()
    device = next(model.parameters()).device
    mixture = mixture.to(device)
    spec = stft(mixture)
    masks = model(crop_freq(spec).abs().unsqueeze(0))[0]
    return apply_masks(masks, spec, mixture.shape[-1])
