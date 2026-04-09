from pathlib import Path

from ragbot.data_loader import build_summary_dict, load_book_entries


def test_load_entries_minimum_books() -> None:
    data = load_book_entries(Path("data/book_summaries.json"))
    assert len(data) >= 10


def test_build_summary_dict() -> None:
    sample = [{"title": "1984", "summary": "Text", "themes": ["dystopia"]}]
    summary_dict = build_summary_dict(sample)
    assert summary_dict["1984"] == "Text"
