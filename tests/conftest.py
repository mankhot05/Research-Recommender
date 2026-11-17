"""Fixtures shared by every test module."""

from __future__ import annotations

import pytest

from app import config


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    # Keep a developer's real .env out of the tests, and rebuild settings per test
    # so monkeypatched env vars are picked up.
    monkeypatch.setattr(config, "ENV_FILE", None)
    config.get_settings.cache_clear()
    yield
    config.get_settings.cache_clear()
