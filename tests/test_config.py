import pytest

from ragbot.config import get_settings


def test_get_settings_reads_moderation_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODERATION_ENABLED", "true")
    monkeypatch.setenv("OPENAI_MODERATION_MODEL", "omni-moderation-latest")
    monkeypatch.setenv("MODERATION_FAIL_BEHAVIOR", "block")
    monkeypatch.setenv("MODERATION_BLOCK_CATEGORIES", "harassment,violence_graphic")
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.moderation_enabled is True
    assert settings.moderation_model == "omni-moderation-latest"
    assert settings.moderation_fail_behavior == "block"
    assert settings.moderation_block_categories == ("harassment", "violence_graphic")

    get_settings.cache_clear()


def test_get_settings_rejects_invalid_moderation_fail_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("MODERATION_FAIL_BEHAVIOR", "sometimes")
    get_settings.cache_clear()

    with pytest.raises(ValueError):
        get_settings()

    get_settings.cache_clear()


def test_get_settings_parses_embed_fallback_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
    monkeypatch.setenv(
        "OPENAI_EMBED_FALLBACK_MODELS",
        "text-embedding-3-small, text-embedding-3-large",
    )
    get_settings.cache_clear()

    settings = get_settings()

    assert settings.embed_model == "text-embedding-3-large"
    assert settings.embed_fallback_models == (
        "text-embedding-3-small",
        "text-embedding-3-large",
    )

    get_settings.cache_clear()


def test_get_settings_rejects_non_positive_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("TOP_K", "0")
    get_settings.cache_clear()

    with pytest.raises(ValueError):
        get_settings()

    get_settings.cache_clear()