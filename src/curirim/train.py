"""Train the U-Net on MUSDB18 (step 6).

Start a run from a config, or resume a killed one from its run directory:

    MUSDB_PATH=/path/to/musdb18-wav python -m curirim.train --config configs/unet.yaml --run-dir runs/unet
    MUSDB_PATH=/path/to/musdb18-wav python -m curirim.train --run-dir runs/unet --resume

Everything a run produces lives in its run directory:

    config.yaml   the config the run was started with
    last.pt       full training state, written every `checkpoint_minutes` and after every epoch
    best.pt       model with the lowest validation loss so far
    events.*      TensorBoard logs: `tensorboard --logdir runs`

Jobs on the university GPU server can hit a time limit, be preempted or lose
their SSH session, so `last.pt` holds everything
needed to continue: model, optimizer, epoch, position inside the epoch, step,
best validation loss, RNG states and the config. It is written to a temporary
file and renamed, so a kill during the write never leaves a broken checkpoint.

Loss: L1 between the estimated and the true stem magnitudes, where an
estimate is the stem's mask × the mixture magnitude (on the model's 1024 bins).
"""

import argparse
import itertools
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from curirim.audio import HOP, crop_freq, stft
from curirim.data import SAMPLE_RATE, STEMS, TrainDataset, ValidDataset, track_dirs
from curirim.model import UNet
from curirim.separate import separate


@dataclass
class Config:
    """Every training hyperparameter. A YAML config overrides any subset of these defaults.

    The dataset location is not here: it comes from MUSDB_PATH, because it
    differs between the laptop and the GPU server while the config does not.
    """

    seed: int = 0
    # Crop length in STFT frames. 512 frames = 511 × 512 samples ≈ 5.93 s: a
    # multiple of 64, so the U-Net needs no padding (517 frames for 6 s would
    # be padded to 576, ~11% wasted compute).
    crop_frames: int = 512
    crops_per_track: int = 64  # epoch length = training tracks × this
    valid_crops_per_track: int = 8
    gain_range: tuple[float, float] = (0.25, 1.25)
    swap_probability: float = 0.5
    # Name of one track to overfit: train and validate on the same fixed,
    # unaugmented crops of that song (exit check 1). None for normal training.
    overfit_track: str | None = None
    batch_size: int = 16
    num_workers: int = 4
    lr: float = 1e-3
    epochs: int = 300
    checkpoint_minutes: float = 10.0
    log_every_steps: int = 50
    audio_every_epochs: int = 5

    @classmethod
    def load(cls, path: str | os.PathLike) -> "Config":
        """Read a YAML file. Unknown keys raise a TypeError, so typos never pass silently."""
        values = yaml.safe_load(Path(path).read_text()) or {}
        if "gain_range" in values:
            values["gain_range"] = tuple(values["gain_range"])
        return cls(**values)

    @property
    def crop_seconds(self) -> float:
        """Crop length in seconds for the datasets: exactly `(crop_frames - 1) × HOP` samples."""
        return (self.crop_frames - 1) * HOP / SAMPLE_RATE


def magnitude_loss(model: torch.nn.Module, mixture: torch.Tensor, stems: torch.Tensor) -> torch.Tensor:
    """Mean L1 between estimated and true stem magnitudes for a batch.

    mixture `(B, 2, samples)` and stems `(B, 4, 2, samples)` are waveforms;
    both go through the STFT here, on the device they are on (the GPU in
    training). Only the model's 1024 bins count.
    """
    mixture_magnitude = crop_freq(stft(mixture)).abs()
    target = crop_freq(stft(stems)).abs()
    estimate = model(mixture_magnitude) * mixture_magnitude.unsqueeze(1)
    return (estimate - target).abs().mean()


def make_datasets(config: Config, train_dirs: list[Path], valid_dirs: list[Path]):
    """Training and validation datasets.

    In overfit mode both are the same fixed, unaugmented crops of one song,
    so the network can memorise them: loss going towards 0 proves the model,
    loss and optimizer are wired correctly.
    """
    if config.overfit_track:
        matches = [d for d in train_dirs + valid_dirs if d.name == config.overfit_track]
        if not matches:
            raise ValueError(f"Track {config.overfit_track!r} not found in the WAV tree.")
        fixed = ValidDataset(matches[:1], config.crop_seconds, config.crops_per_track)
        return fixed, fixed
    train = TrainDataset(
        train_dirs, config.crop_seconds, config.crops_per_track, config.gain_range, config.swap_probability
    )
    valid = ValidDataset(valid_dirs, config.crop_seconds, config.valid_crops_per_track)
    return train, valid


def save_atomic(state: dict, path: Path) -> None:
    """Write a checkpoint to `path.tmp`, then rename it over `path`.

    The rename is atomic, so a session killed mid-write leaves the previous
    checkpoint intact instead of a truncated file.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Mean validation loss over every crop, in eval mode (no dropout, frozen BatchNorm)."""
    model.eval()
    total, count = 0.0, 0
    for mixture, stems in loader:
        loss = magnitude_loss(model, mixture.to(device), stems.to(device))
        total += loss.item() * len(mixture)
        count += len(mixture)
    model.train()
    return total / count


def log_audio(writer: SummaryWriter, model: torch.nn.Module, valid: ValidDataset, step: int) -> None:
    """Separate the first validation crop and log each stem (mono) to TensorBoard.

    The mixture and true stems are logged too, under `reference/`, to compare by ear.
    """
    mixture, stems = valid[0]
    estimates = separate(model, mixture).cpu()
    model.train()
    for name, audio in [("reference/mixture", mixture)] + [
        (f"{kind}/{stem}", source[i]) for kind, source in (("estimate", estimates), ("reference", stems))
        for i, stem in enumerate(STEMS)
    ]:
        writer.add_audio(name, audio.mean(0, keepdim=True).clamp(-1, 1), step, SAMPLE_RATE)


def train(
    config: Config,
    run_dir: Path,
    train_dirs: list[Path],
    valid_dirs: list[Path],
    resume: bool = False,
    device: torch.device | None = None,
) -> float:
    """Run (or continue) training until `config.epochs`; returns the best validation loss.

    A new run refuses to start in a directory that already has `last.pt`, so a
    forgotten `--resume` never overwrites a run.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(run_dir)
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"

    if resume:
        state = torch.load(last_path, map_location="cpu")
        config = Config(**state["config"])
    else:
        if last_path.exists():
            raise FileExistsError(f"{last_path} exists: pass --resume or choose another --run-dir.")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(asdict(config), sort_keys=False))
        torch.manual_seed(config.seed)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    epoch, done_in_epoch, step, best = 0, 0, 0, math.inf
    if resume:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        epoch, done_in_epoch, step, best = state["epoch"], state["batch_in_epoch"], state["step"], state["best_valid"]
        torch.set_rng_state(state["rng"])
        if device.type == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        print(f"Resumed {run_dir} at epoch {epoch + 1}, batch {done_in_epoch}, step {step}")

    train_set, valid_set = make_datasets(config, train_dirs, valid_dirs)
    workers = {"num_workers": config.num_workers, "persistent_workers": config.num_workers > 0}
    pin = device.type == "cuda"
    # Every training item is a fresh random mix, so shuffling adds nothing.
    # drop_last avoids a final batch of 1, which breaks BatchNorm statistics.
    train_loader = DataLoader(train_set, config.batch_size, drop_last=True, pin_memory=pin, **workers)
    valid_loader = DataLoader(valid_set, config.batch_size, pin_memory=pin, **workers)
    if len(train_loader) == 0:
        raise ValueError("Fewer training crops than one batch: lower batch_size or raise crops_per_track.")

    def checkpoint(batch_in_epoch: int) -> None:
        """Save the full training state as last.pt."""
        save_atomic(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "batch_in_epoch": batch_in_epoch,
                "step": step,
                "best_valid": best,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "config": asdict(config),
            },
            last_path,
        )

    writer = SummaryWriter(run_dir)
    model.train()
    last_save = time.monotonic()
    while epoch < config.epochs:
        start, running, batches = time.monotonic(), 0.0, 0
        # After a mid-epoch resume, run only the batches the killed session had not done.
        remaining = len(train_loader) - done_in_epoch
        for batch, (mixture, stems) in enumerate(itertools.islice(train_loader, remaining), start=done_in_epoch + 1):
            loss = magnitude_loss(model, mixture.to(device, non_blocking=True), stems.to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            step += 1
            running += loss.item()
            batches += 1
            if step % config.log_every_steps == 0:
                writer.add_scalar("loss/train", running / batches, step)
            if time.monotonic() - last_save >= config.checkpoint_minutes * 60:
                checkpoint(batch)
                last_save = time.monotonic()

        valid_loss = validate(model, valid_loader, device)
        train_loss = running / max(batches, 1)
        writer.add_scalar("loss/train_epoch", train_loss, step)
        writer.add_scalar("loss/valid", valid_loss, step)
        epoch += 1
        done_in_epoch = 0
        if valid_loss < best:
            best = valid_loss
            save_atomic({"model": model.state_dict(), "epoch": epoch, "valid_loss": best, "config": asdict(config)}, best_path)
        if epoch % config.audio_every_epochs == 0 or epoch == config.epochs:
            log_audio(writer, model, valid_set, step)
        checkpoint(0)
        last_save = time.monotonic()
        writer.flush()
        print(
            f"epoch {epoch}/{config.epochs}  step {step}  train {train_loss:.4f}  valid {valid_loss:.4f}"
            f"  best {best:.4f}  ({time.monotonic() - start:.0f} s, {remaining} batches)",
            flush=True,
        )

    writer.close()
    return best


def main() -> None:
    """Command line: new run from `--config`, or `--resume` an existing `--run-dir`."""
    parser = argparse.ArgumentParser(description="Train the U-Net on MUSDB18 (decoded WAV tree in MUSDB_PATH).")
    parser.add_argument("--config", type=Path, help="YAML config for a new run")
    parser.add_argument("--run-dir", type=Path, required=True, help="where checkpoints and logs go")
    parser.add_argument("--resume", action="store_true", help="continue from <run-dir>/last.pt with its saved config")
    args = parser.parse_args()
    if args.resume == (args.config is not None):
        parser.error("give --config for a new run, or --resume (the config comes from the checkpoint), not both")

    if args.resume:
        config = Config(**torch.load(args.run_dir / "last.pt", map_location="cpu")["config"])
    else:
        config = Config.load(args.config)
    if config.overfit_track:
        # Any decoded track will do; the valid split may not be decoded at all.
        train_dirs, valid_dirs = track_dirs("train"), []
    else:
        train_dirs, valid_dirs = track_dirs("train", "train"), track_dirs("train", "valid")
    print(f"{len(train_dirs)} training tracks, {len(valid_dirs)} validation tracks")
    best = train(config, args.run_dir, train_dirs, valid_dirs, resume=args.resume)
    print(f"done, best validation loss {best:.4f}")


if __name__ == "__main__":
    main()
