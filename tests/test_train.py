"""Checks for the training loop: outputs of a run, resume, overfitting, config parsing.

Runs on a tiny synthetic WAV tree with 64-frame crops, so it needs no dataset
and a few seconds of CPU. Each stem is a tone at its own frequency plus a
little noise: separable, so a working training loop learns it quickly.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from curirim.data import SAMPLE_RATE, STEMS
from curirim.model import UNet
from curirim.train import Config, make_datasets, train, validate

TINY = Config(
    crop_frames=64,  # 63 × 512 samples ≈ 0.73 s
    crops_per_track=2,
    valid_crops_per_track=2,
    batch_size=2,
    num_workers=0,
    epochs=2,
    checkpoint_minutes=60,
    log_every_steps=1,
    audio_every_epochs=1,
)


def make_tracks(root: Path, count: int) -> list[Path]:
    """`count` track folders with 1 s per stem: a stereo tone (220/880/3000/7000 Hz) plus noise."""
    rng = np.random.default_rng(0)
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    dirs = []
    for index in range(count):
        track = root / f"track{index}"
        track.mkdir(parents=True)
        for stem, frequency in zip(STEMS, (220, 880, 3000, 7000)):
            tone = 0.3 * np.sin(2 * np.pi * frequency * t + rng.uniform(0, 2 * np.pi))
            audio = tone[:, None] + rng.normal(0, 0.01, (SAMPLE_RATE, 2))
            sf.write(track / f"{stem}.wav", audio.astype(np.float32), SAMPLE_RATE, subtype="FLOAT")
        dirs.append(track)
    return dirs


@pytest.fixture
def tracks(tmp_path):
    """Two training tracks and one validation track."""
    return make_tracks(tmp_path / "train", 2), make_tracks(tmp_path / "valid", 1)


def test_run_writes_config_checkpoints_and_logs(tmp_path, tracks):
    run = tmp_path / "run"
    best = train(TINY, run, *tracks)
    assert np.isfinite(best)
    state = torch.load(run / "last.pt")
    # 2 tracks × 2 crops / batch 2 = 2 steps per epoch, 2 epochs.
    assert (state["epoch"], state["batch_in_epoch"], state["step"]) == (2, 0, 4)
    assert state["config"]["crop_frames"] == 64
    assert torch.load(run / "best.pt")["valid_loss"] == best
    assert (run / "config.yaml").exists()
    assert list(run.glob("events.out.tfevents.*"))


def test_new_run_refuses_to_overwrite(tmp_path, tracks):
    run = tmp_path / "run"
    train(replace(TINY, epochs=1), run, *tracks)
    with pytest.raises(FileExistsError):
        train(TINY, run, *tracks)


def test_resume_runs_only_the_missing_batches(tmp_path, tracks):
    # Simulate a session killed after 1 of 2 batches of epoch 2: the resumed
    # run must do just 1 more batch, then stop at the configured 2 epochs.
    run = tmp_path / "run"
    train(replace(TINY, epochs=1), run, *tracks)
    state = torch.load(run / "last.pt")
    state["config"]["epochs"] = 2
    state["batch_in_epoch"] = 1
    torch.save(state, run / "last.pt")

    train(TINY, run, *tracks, resume=True)
    resumed = torch.load(run / "last.pt")
    assert (resumed["epoch"], resumed["batch_in_epoch"], resumed["step"]) == (2, 0, 2 + 1)


def test_overfit_one_track_lowers_the_loss(tmp_path, tracks):
    config = replace(TINY, overfit_track="track0", epochs=25, crops_per_track=2, audio_every_epochs=100)
    _, fixed = make_datasets(config, *tracks)
    torch.manual_seed(config.seed)  # same initial weights as the run below
    before = validate(UNet(), DataLoader(fixed, 2), torch.device("cpu"))
    after = train(config, tmp_path / "run", *tracks)
    assert after < 0.7 * before, (before, after)


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("learning_rate: 0.01\n")
    with pytest.raises(TypeError):
        Config.load(path)


def test_shipped_configs_load():
    paths = list((Path(__file__).parents[1] / "configs").glob("*.yaml"))
    assert paths
    for path in paths:
        Config.load(path)
