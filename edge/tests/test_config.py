"""Configuration must reject contradictions rather than guess."""

from __future__ import annotations

import pytest

from config import Config, ConfigError


def _cfg(**over):
    base = dict(
        ingest_mode="file",
        rtsp_url="",
        target_fps=8.0,
        env="development",
        ingest_key="",
    )
    base.update(over)
    return Config(**base)


def test_file_mode_is_valid_without_an_rtsp_url():
    _cfg().validate()


def test_rtsp_mode_requires_a_url():
    with pytest.raises(ConfigError, match="RTSP_URL"):
        _cfg(ingest_mode="rtsp").validate()


def test_unknown_mode_is_rejected():
    with pytest.raises(ConfigError, match="INGEST_MODE"):
        _cfg(ingest_mode="onvif").validate()


def test_non_positive_fps_is_rejected():
    with pytest.raises(ConfigError, match="TARGET_FPS"):
        _cfg(target_fps=0).validate()


def test_production_refuses_to_start_without_an_ingest_key():
    with pytest.raises(ConfigError, match="TRUEWATCH_INGEST_KEY"):
        _cfg(env="production").validate()


def test_production_accepts_a_supplied_key():
    _cfg(env="production", ingest_key="a" * 64).validate()


def test_source_label_never_calls_a_file_a_camera():
    assert _cfg(ingest_mode="file").source_label == "file"
    assert _cfg(ingest_mode="rtsp", rtsp_url="rtsp://x/y").source_label == "rtsp"
