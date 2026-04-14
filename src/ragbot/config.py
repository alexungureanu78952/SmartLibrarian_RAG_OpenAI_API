"""Application configuration loading from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import dotenv_values


DEFAULT_MODERATION_BLOCK_CATEGORIES = (
    "harassment",
    "hate",
    "self_harm",
    "self_harm_intent",
    "self_harm_instructions",
    "sexual",
    "sexual_minors",
    "violence",
    "violence_graphic",
)

def _build_env_map() -> dict[str, str]:
    """Merge .env values with process env where process env has precedence."""
    env_file_values = {k: str(v) for k, v in dotenv_values(".env").items() if v is not None}
    merged = dict(env_file_values)
    merged.update({k: v for k, v in os.environ.items() if isinstance(v, str)})
    return merged


def _parse_bool(name: str, default: bool, env_map: dict[str, str]) -> bool:
    """Parse a boolean environment variable with simple validation."""
    raw = env_map.get(name)
    if raw is None:
        return default

    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value.")


def _parse_csv(name: str, default: tuple[str, ...], env_map: dict[str, str]) -> tuple[str, ...]:
    """Parse a comma-separated environment variable into a tuple."""
    raw = env_map.get(name)
    if raw is None:
        return default

    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    return values or default


@dataclass(frozen=True)
class Settings:
    """Runtime settings shared by CLI and web entrypoints."""

    openai_api_key: str
    chat_model: str
    embed_model: str
    embed_fallback_models: tuple[str, ...]
    tts_model: str
    stt_model: str
    image_model: str
    tts_voice: str
    chroma_dir: Path
    summaries_json: Path
    top_k: int
    moderation_enabled: bool
    moderation_model: str
    moderation_fail_behavior: str
    moderation_block_categories: tuple[str, ...]


def _parse_settings(
    env_map: dict[str, str],
    top_k_raw: str,
    moderation_fail_behavior: str,
) -> Settings:
    """Build settings object from merged .env and process environment values."""
    try:
        top_k = int(top_k_raw)
    except ValueError as exc:
        raise ValueError("TOP_K must be an integer.") from exc
    if top_k <= 0:
        raise ValueError("TOP_K must be greater than 0.")

    api_key = env_map.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")

    return Settings(
        openai_api_key=api_key,
        chat_model=env_map.get("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
        embed_model=env_map.get("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
        embed_fallback_models=_parse_csv("OPENAI_EMBED_FALLBACK_MODELS", (), env_map),
        tts_model=env_map.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts"),
        stt_model=env_map.get("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe"),
        image_model=env_map.get("OPENAI_IMAGE_MODEL", "gpt-image-1"),
        tts_voice=env_map.get("OPENAI_TTS_VOICE", "alloy"),
        chroma_dir=Path(env_map.get("CHROMA_DIR", "chroma_db")),
        summaries_json=Path(env_map.get("BOOK_SUMMARIES_JSON", "data/book_summaries.json")),
        top_k=top_k,
        moderation_enabled=_parse_bool("MODERATION_ENABLED", True, env_map),
        moderation_model=env_map.get("OPENAI_MODERATION_MODEL", "omni-moderation-latest"),
        moderation_fail_behavior=moderation_fail_behavior,
        moderation_block_categories=_parse_csv(
            "MODERATION_BLOCK_CATEGORIES",
            DEFAULT_MODERATION_BLOCK_CATEGORIES,
            env_map,
        ),
    )


@lru_cache(maxsize=1)
def _get_settings_cached() -> Settings:
    """Create settings once per process; restart to apply config changes."""
    env_map = _build_env_map()
    moderation_fail_behavior = env_map.get("MODERATION_FAIL_BEHAVIOR", "allow").strip().lower()
    if moderation_fail_behavior not in {"allow", "block"}:
        raise ValueError("MODERATION_FAIL_BEHAVIOR must be 'allow' or 'block'.")

    top_k_raw = env_map.get("TOP_K", "4").strip()
    return _parse_settings(
        env_map=env_map,
        top_k_raw=top_k_raw,
        moderation_fail_behavior=moderation_fail_behavior,
    )


def get_settings() -> Settings:
    """Load, validate, and cache settings for process lifetime."""
    return _get_settings_cached()


def _clear_settings_cache() -> None:
    """Compatibility shim used by tests and local admin scripts."""
    _get_settings_cached.cache_clear()


# Backward-compatible API for existing tests that call get_settings.cache_clear().
get_settings.cache_clear = _clear_settings_cache  # type: ignore[attr-defined]
