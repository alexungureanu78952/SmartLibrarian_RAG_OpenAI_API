"""Data loading and normalization helpers for local book summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_book_entries(json_path: Path) -> list[dict[str, Any]]:
    """Load and validate the local JSON summaries dataset."""
    with json_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError("book_summaries.json must contain a list of books.")

    required = {"title", "summary", "themes"}
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Book entry at index {i} must be an object.")
        missing = required - set(entry.keys())
        if missing:
            raise ValueError(f"Book entry '{entry}' is missing fields: {sorted(missing)}")

    return raw


def build_summary_dict(entries: list[dict[str, Any]]) -> dict[str, str]:
    """Build title->summary lookup used by get_summary_by_title."""
    return {str(e["title"]): str(e["summary"]) for e in entries}
