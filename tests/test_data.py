"""Checks for dataset access. Tests needing audio skip when MUSDB_PATH is unset."""

import os

import pytest

from curirim.data import STEMS, musdb_root, open_musdb


def test_missing_root_is_an_error(monkeypatch):
    monkeypatch.delenv("MUSDB_PATH", raising=False)
    with pytest.raises(RuntimeError):
        musdb_root()


needs_data = pytest.mark.skipif(not os.environ.get("MUSDB_PATH"), reason="MUSDB_PATH not set")


@needs_data
def test_standard_split_sizes():
    # Standard MUSDB18 split: 86 train + 14 valid tracks from train/, 50 test.
    assert len(open_musdb("train", "train")) == 86
    assert len(open_musdb("train", "valid")) == 14
    assert len(open_musdb("test")) == 50


@needs_data
def test_track_has_four_stereo_stems():
    track = open_musdb("train", "valid").tracks[0]
    assert track.rate == 44100
    for stem in STEMS:
        audio = track.targets[stem].audio
        assert audio.ndim == 2 and audio.shape[1] == 2
