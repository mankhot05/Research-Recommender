"""App settings, read once from environment variables and the repo's .env file.

Precedence: real environment variables, then .env, then the defaults below.
The .env file is parsed into Settings without being copied into os.environ.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# The repo-root .env, so loading works from any working directory.
# Tests set this to None so a developer's real .env is never read.
ENV_FILE: Path | None = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    """Configuration; each field maps to the env var of the same name, uppercased."""

    # Blank values like OPENAI_CHAT_MODEL= fall back to the default.
    model_config = SettingsConfigDict(env_ignore_empty=True, extra="ignore")

    openai_api_key: SecretStr | None = None
    pinecone_api_key: SecretStr | None = None
    openai_chat_model: str = "gpt-4o-mini"
    embedder: str = "openai"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """The settings, built on first use and cached for the rest of the process."""
    return Settings(_env_file=ENV_FILE)
