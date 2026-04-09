"""Local function tool definitions used in recommendation flow."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SummaryTool:
    """Local lookup for full summaries by exact title."""

    summaries_by_title: dict[str, str]

    def get_summary_by_title(self, title: str) -> str:
        """Return a full summary if title matches exactly (case-insensitive)."""
        normalized = title.strip().lower()
        for book_title, summary in self.summaries_by_title.items():
            if book_title.lower() == normalized:
                return summary
        return (
            f"I could not find an exact match for '{title}'. "
            "Please provide the exact book title from the recommendation."
        )


RESPONSES_TOOL_SPEC = [
    {
        "type": "function",
        "name": "get_summary_by_title",
        "description": "Returns the full local summary for an exact book title.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Exact book title, for example '1984' or 'The Hobbit'",
                }
            },
            "required": ["title"],
            "additionalProperties": False,
        },
    }
]
