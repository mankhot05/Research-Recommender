"""Tests for app.config: defaults, env vars, the .env file and precedence."""

from __future__ import annotations

import pytest

from app import config
from app.config import get_settings


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("OPENAI_API_KEY", "PINECONE_API_KEY", "OPENAI_CHAT_MODEL", "EMBEDDER"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_defaults(clean_env):
    settings = get_settings()

    assert settings.embedder == "openai"
    assert settings.openai_chat_model == "gpt-4o-mini"
    assert settings.openai_api_key is None
    assert settings.pinecone_api_key is None


def test_reads_env_vars(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-env")
    clean_env.setenv("EMBEDDER", "specter")

    settings = get_settings()

    assert settings.openai_api_key.get_secret_value() == "sk-env"
    assert settings.embedder == "specter"


def test_reads_env_file(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("PINECONE_API_KEY=pc-file\nOPENAI_CHAT_MODEL=gpt-test\n")
    clean_env.setattr(config, "ENV_FILE", env_file)

    settings = get_settings()

    assert settings.pinecone_api_key.get_secret_value() == "pc-file"
    assert settings.openai_chat_model == "gpt-test"


def test_env_var_beats_env_file(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_API_KEY=sk-file\n")
    clean_env.setattr(config, "ENV_FILE", env_file)
    clean_env.setenv("OPENAI_API_KEY", "sk-env")

    assert get_settings().openai_api_key.get_secret_value() == "sk-env"


def test_blank_value_falls_back_to_default(clean_env, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OPENAI_CHAT_MODEL=\n")
    clean_env.setattr(config, "ENV_FILE", env_file)

    assert get_settings().openai_chat_model == "gpt-4o-mini"


def test_api_keys_are_hidden_in_repr(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-secret")

    assert "sk-secret" not in repr(get_settings())


def test_settings_are_cached(clean_env):
    assert get_settings() is get_settings()
