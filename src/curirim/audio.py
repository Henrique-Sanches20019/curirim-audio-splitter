"""Signal path between waveforms and spectrograms.

Fixed STFT settings for the whole project: n_fft 2048, hop 512, Hann window,
`center=True`. That gives 1025 frequency bins of ~21.5 Hz each and 517 frames
for a 6 s crop. The U-Net needs the frequency axis to divide by 2^6 = 64, so
the model sees only the lowest 1024 bins; the dropped top bin (22.05 kHz,
empty in MUSDB18) is restored as zeros before the inverse STFT.

Tensors keep any leading dimensions, e.g. waveforms `(batch, channels,
samples)` map to spectrograms `(batch, channels, bins, frames)`.
"""

import torch
import torch.nn.functional as F

N_FFT = 2048
HOP = 512
N_BINS = N_FFT // 2 + 1  # 1025: 0 Hz up to Nyquist (22.05 kHz at 44.1 kHz)
MODEL_BINS = 1024  # lowest bins kept for the model; a multiple of 2^6


def _window(like: torch.Tensor) -> torch.Tensor:
    """Hann window on the same device as `like`, in its real dtype.

    Creating it per call keeps the functions stateless and device-agnostic;
    the cost is negligible next to the FFT itself.
    """
    dtype = like.real.dtype if like.is_complex() else like.dtype
    return torch.hann_window(N_FFT, device=like.device, dtype=dtype)


def stft(audio: torch.Tensor) -> torch.Tensor:
    """Waveform `(..., samples)` to complex spectrogram `(..., 1025, frames)`.

    `frames = 1 + samples // HOP` because `center=True` pads the signal so the
    first window is centred on sample 0.
    """
    lead, samples = audio.shape[:-1], audio.shape[-1]
    # torch.stft accepts at most one batch dimension, so flatten the rest.
    spec = torch.stft(
        audio.reshape(-1, samples),
        n_fft=N_FFT,
        hop_length=HOP,
        window=_window(audio),
        center=True,
        return_complex=True,
    )
    return spec.reshape(*lead, *spec.shape[-2:])


def istft(spec: torch.Tensor, length: int) -> torch.Tensor:
    """Complex spectrogram `(..., 1025, frames)` back to waveform `(..., length)`.

    `length` is required: several sample counts give the same number of
    frames, so the original length cannot be recovered from the spectrogram.
    """
    lead = spec.shape[:-2]
    audio = torch.istft(
        spec.reshape(-1, *spec.shape[-2:]),
        n_fft=N_FFT,
        hop_length=HOP,
        window=_window(spec),
        center=True,
        length=length,
    )
    return audio.reshape(*lead, length)


def crop_freq(spec: torch.Tensor, bins: int = MODEL_BINS) -> torch.Tensor:
    """Keep the lowest `bins` frequency bins (axis -2) for the model."""
    return spec[..., :bins, :]


def pad_freq(spec: torch.Tensor) -> torch.Tensor:
    """Zero-pad a cropped spectrogram back to all 1025 bins, ready for `istft`."""
    missing = N_BINS - spec.shape[-2]
    # F.pad pads the last dims first: (frames_left, frames_right, bins_low, bins_high).
    return F.pad(spec, (0, 0, 0, missing))
