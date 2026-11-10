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
import torch.nn.functional as F
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


ENCODER_CHANNELS = (16, 32, 64, 128, 256, 512)
DEPTH = len(ENCODER_CHANNELS)  # 6 stride-2 layers
MULTIPLE = 2**DEPTH  # 64: both spectrogram axes must divide by this inside the U-Net


class UNet(nn.Module):
    """Spectrogram U-Net (Spleeter-style), one network shared by all four stems.

    Encoder: 6 × (Conv2d 5×5 stride 2 → BatchNorm → LeakyReLU 0.2), channels
    16→32→64→128→256→512. Each layer halves bins and frames, so a 1024 × 576
    input reaches the bottleneck at 512 channels × 16 × 9.

    Decoder: 6 × (ConvTranspose2d 5×5 stride 2 → BatchNorm → ReLU), each
    doubling bins and frames. After each of the first 5 blocks, the output is
    concatenated with the encoder output of the same size (skip connection),
    which brings back the fine time-frequency detail lost on the way down.
    Dropout 0.5 on the first 3 blocks regularizes the deep, wide layers: there
    are only 100 training songs.

    Head: 1×1 convolution to 8 logits, then `stem_masks` (softmax over stems).

    Both spectrogram axes must divide by 64. `forward` pads frames (and bins,
    if needed) with zeros to the next multiple of 64 and crops the masks back,
    so the network accepts any song length.
    """

    def __init__(self):
        """Build the 6 encoder blocks, 6 decoder blocks and the 1×1 mask head."""
        super().__init__()
        self.encoder = nn.ModuleList()
        in_channels = N_CHANNELS
        for out_channels in ENCODER_CHANNELS:
            self.encoder.append(
                nn.Sequential(
                    # 5×5, stride 2, padding 2: output is exactly half the input (even sizes).
                    nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=2, padding=2),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                )
            )
            in_channels = out_channels

        # Decoder outputs: 256, 128, 64, 32, 16 (each matched by a skip), then 16 at full size.
        decoder_channels = ENCODER_CHANNELS[-2::-1] + (ENCODER_CHANNELS[0],)
        self.decoder = nn.ModuleList()
        in_channels = ENCODER_CHANNELS[-1]
        for index, out_channels in enumerate(decoder_channels):
            layers = [
                # output_padding 1 makes the output exactly twice the input.
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
            if index < 3:
                layers.append(nn.Dropout(0.5))
            self.decoder.append(nn.Sequential(*layers))
            # The next block reads this output concatenated with the same-sized encoder output.
            in_channels = 2 * out_channels if index < DEPTH - 1 else out_channels

        self.head = nn.Conv2d(in_channels, N_OUTPUTS, kernel_size=1)

    def forward(self, magnitude: torch.Tensor) -> torch.Tensor:
        """Raw magnitude `(batch, 2, bins, frames)` to masks `(batch, 4, 2, bins, frames)`.

        log1p compresses the magnitude's huge dynamic range (near 0 to
        hundreds) before the first layer; the masks still multiply the raw
        magnitude outside the network.
        """
        bins, frames = magnitude.shape[-2:]
        # Pad bottom/right with zeros up to multiples of 64 so every stride-2
        # layer halves an even size and the skip connections line up exactly.
        x = F.pad(torch.log1p(magnitude), (0, -frames % MULTIPLE, 0, -bins % MULTIPLE))

        skips = []
        for block in self.encoder:
            x = block(x)
            skips.append(x)

        # The deepest encoder output is the decoder's input, not a skip.
        skips.pop()
        for block in self.decoder:
            x = block(x)
            if skips:
                x = torch.cat([x, skips.pop()], dim=1)

        return stem_masks(self.head(x)[..., :bins, :frames])
