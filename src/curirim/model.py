"""Separation networks.

Every network here has the same interface, so the rest of the pipeline never
depends on which one is used:

    masks = model(magnitude)
    magnitude: (batch, 2, bins, frames)      mixture STFT magnitude, raw (not log)
    masks:     (batch, 4, 2, bins, frames)   one mask per stem and channel, in STEMS order

The masks are a softmax over the stem axis, so at every time-frequency point
they are between 0 and 1 and sum to 1: the four stems always add up to the
mixture. Each network compresses the magnitude itself (log1p) before its
layers, so callers pass raw magnitudes and the masks multiply raw magnitudes.
"""

import torch
from torch import nn

from curirim.data import STEMS

N_STEMS = len(STEMS)
N_CHANNELS = 2  # stereo
N_OUTPUTS = N_STEMS * N_CHANNELS  # 8 output channels before the softmax


def stem_masks(logits: torch.Tensor) -> torch.Tensor:
    """Turn `(batch, 8, bins, frames)` network outputs into softmax masks `(batch, 4, 2, bins, frames)`.

    Output channel `2 * stem + channel` belongs to that stem and stereo channel.
    The softmax over stems makes the masks sum to 1 at every point.
    """
    batch, _, bins, frames = logits.shape
    return torch.softmax(logits.view(batch, N_STEMS, N_CHANNELS, bins, frames), dim=1)


class TinyNet(nn.Module):
    """Placeholder network for step 4: one 3×3 convolution, then the mask head.

    Far too small to separate anything well. It exists to run the whole
    pipeline (STFT, masks, mixture phase, inverse STFT, museval) before the
    U-Net is written. Padding 1 keeps any input size, so no shape constraints.
    """

    def __init__(self):
        """One Conv2d from the 2 stereo magnitude channels to the 8 mask logits."""
        super().__init__()
        self.conv = nn.Conv2d(N_CHANNELS, N_OUTPUTS, kernel_size=3, padding=1)

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Raw magnitude `(batch, 2, bins, frames)` to masks `(batch, 4, 2, bins, frames)`."""
        return stem_masks(self.conv(torch.log1p(magnitude)))
