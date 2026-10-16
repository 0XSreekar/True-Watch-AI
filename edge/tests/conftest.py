"""Shared fixtures: no test may reach the network for model weights."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_plate_locator_download(monkeypatch):
    """Default to the classical plate finder; tests that want the locator opt in explicitly."""
    from anpr import plate_locator

    monkeypatch.setenv("ANPR_FETCH_PLATE_LOCATOR", "0")
    monkeypatch.delenv("ANPR_PLATE_LOCATOR_MODEL", raising=False)
    plate_locator.reset_locator()
    yield
    plate_locator.reset_locator()
