"""Input moderation helpers shared by CLI and web entrypoints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ragbot.config import Settings
from ragbot.openai_retry import call_with_retry

# Small local list for demo purposes only.
BLOCKED_TERMS = {
    "fuck",
    "shit",
    "bitch",
    "asshole",
    "bastard",
}

OPENAI_CATEGORY_ALIASES = {
    "harassment": ("harassment",),
    "harassment_threatening": ("harassment_threatening", "harassment/threatening"),
    "hate": ("hate",),
    "hate_threatening": ("hate_threatening", "hate/threatening"),
    "self_harm": ("self_harm", "self-harm"),
    "self_harm_intent": ("self_harm_intent", "self-harm/intent"),
    "self_harm_instructions": (
        "self_harm_instructions",
        "self-harm/instructions",
    ),
    "sexual": ("sexual",),
    "sexual_minors": ("sexual_minors", "sexual/minors"),
    "violence": ("violence",),
    "violence_graphic": ("violence_graphic", "violence/graphic"),
}


@dataclass(frozen=True)
class ModerationDecision:
    """Structured moderation outcome used by CLI and API handlers."""

    blocked: bool
    source: str
    reason: str


def is_inappropriate(text: str) -> bool:
    """Return True when input contains blocked standalone terms."""
    lowered = text.lower()
    tokens = set(re.findall(r"[a-zA-Z']+", lowered))
    return any(term in tokens for term in BLOCKED_TERMS)


def _category_dict(categories: Any) -> dict[str, Any]:
    """Convert moderation categories to a dictionary when possible."""
    if categories is None:
        return {}
    if isinstance(categories, dict):
        return categories

    model_dump = getattr(categories, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}

    return vars(categories)


def _selected_openai_category(categories: Any, selected: tuple[str, ...]) -> str | None:
    """Return the first configured OpenAI category that is flagged."""
    category_data = _category_dict(categories)
    for category_name in selected:
        aliases = OPENAI_CATEGORY_ALIASES.get(category_name, (category_name,))
        for alias in aliases:
            if alias in category_data and bool(category_data[alias]):
                return category_name
            if hasattr(categories, alias) and bool(getattr(categories, alias)):
                return category_name
    return None


def moderate_text(text: str, *, settings: Settings, client: Any | None) -> ModerationDecision:
    """Apply OpenAI-first moderation with local fallback behavior."""
    if not text.strip():
        return ModerationDecision(blocked=False, source="none", reason="empty")

    if settings.moderation_enabled and client is not None:
        try:
            response = call_with_retry(
                lambda: client.moderations.create(
                    model=settings.moderation_model,
                    input=text,
                )
            )
            results = getattr(response, "results", [])
            if results:
                flagged_category = _selected_openai_category(
                    getattr(results[0], "categories", None),
                    settings.moderation_block_categories,
                )
                if flagged_category:
                    return ModerationDecision(
                        blocked=True,
                        source="openai",
                        reason=flagged_category,
                    )
        except Exception:
            if is_inappropriate(text):
                return ModerationDecision(blocked=True, source="local", reason="blocked_term")
            if settings.moderation_fail_behavior == "block":
                return ModerationDecision(
                    blocked=True,
                    source="fallback",
                    reason="moderation_unavailable",
                )
            return ModerationDecision(
                blocked=False,
                source="fallback",
                reason="moderation_unavailable",
            )

    if is_inappropriate(text):
        return ModerationDecision(blocked=True, source="local", reason="blocked_term")

    source = "openai" if settings.moderation_enabled and client is not None else "local"
    return ModerationDecision(blocked=False, source=source, reason="allowed")


def polite_block_message() -> str:
    """Return a user-friendly response when content is blocked."""
    return (
        "I want to keep this space respectful. "
        "Please rephrase your request and I will gladly recommend a book."
    )
