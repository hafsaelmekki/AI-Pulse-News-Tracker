from __future__ import annotations

import pytest

from ai_pulse_tracker import config


REQUIRED_KEYS = (
    "AZURE_AI_ENDPOINT",
    "AZURE_AI_KEY",
    "NEWS_API_KEY",
    "COSMOS_ENDPOINT",
    "COSMOS_KEY",
)


def test_load_settings_success(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    content = """
AZURE_AI_ENDPOINT=https://example.cognitiveservices.azure.com/
AZURE_AI_KEY=dummy
NEWS_API_KEY=news
COSMOS_ENDPOINT=https://example.documents.azure.com:443/
COSMOS_KEY=cosmos
NEWS_QUERY=machine learning
NEWS_LANGUAGE=en
NEWS_BATCH_SIZE=10
NEWS_MAX_LOOKBACK_DAYS=14
""".strip()
    env_file.write_text(content, encoding="utf-8")

    for key in REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)

    config.load_settings.cache_clear()
    settings = config.load_settings(env_file)

    assert settings.news_query == "machine learning"
    assert settings.news_language == "en"
    assert settings.news_batch_size == 10
    assert settings.news_max_lookback_days == 14


def test_load_settings_missing_keys(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("AZURE_AI_KEY=value\n", encoding="utf-8")

    for key in REQUIRED_KEYS:
        monkeypatch.delenv(key, raising=False)

    config.load_settings.cache_clear()
    with pytest.raises(config.SettingsError):
        config.load_settings(env_file)
