from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv


class SettingsError(RuntimeError):
    """Raised when mandatory configuration values are missing."""


@dataclass(slots=True)
class Settings:
    azure_ai_endpoint: str
    azure_ai_key: str
    news_api_key: str
    cosmos_endpoint: str
    cosmos_key: str
    database_name: str = "NewsDatabase"
    container_name: str = "Analyses"
    news_query: str = "Generative AI"
    news_language: str = "fr"
    news_batch_size: int = 5


_REQUIRED_ENV_VARS: tuple[str, ...] = (
    "AZURE_AI_ENDPOINT",
    "AZURE_AI_KEY",
    "NEWS_API_KEY",
    "COSMOS_ENDPOINT",
    "COSMOS_KEY",
)


@lru_cache(maxsize=1)
def load_settings(dotenv_path: str | os.PathLike[str] | None = None) -> Settings:
    """Load validated settings from environment variables."""

    candidate_paths: Iterable[Path] = (
        Path(dotenv_path) if dotenv_path else Path.cwd() / ".env",
    )
    for path in candidate_paths:
        if path and path.exists():
            load_dotenv(path)
            break
    else:
        load_dotenv()

    env = os.environ
    missing = [var for var in _REQUIRED_ENV_VARS if not env.get(var)]
    if missing:
        raise SettingsError(
            "Missing required environment variables: " + ", ".join(missing)
        )

    return Settings(
        azure_ai_endpoint=env["AZURE_AI_ENDPOINT"],
        azure_ai_key=env["AZURE_AI_KEY"],
        news_api_key=env["NEWS_API_KEY"],
        cosmos_endpoint=env["COSMOS_ENDPOINT"],
        cosmos_key=env["COSMOS_KEY"],
        database_name=env.get("COSMOS_DATABASE", "NewsDatabase"),
        container_name=env.get("COSMOS_CONTAINER", "Analyses"),
        news_query=env.get("NEWS_QUERY", "Generative AI"),
        news_language=env.get("NEWS_LANGUAGE", "fr"),
        news_batch_size=int(env.get("NEWS_BATCH_SIZE", 5)),
    )
