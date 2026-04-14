from pathlib import Path

import pytest

from ragbot.data_loader import build_summary_dict, load_book_entries


def test_load_entries_minimum_books() -> None:
    data = load_book_entries(Path("data/book_summaries.json"))
    assert len(data) >= 10


def test_build_summary_dict() -> None:
    sample = [{"title": "1984", "summary": "Text", "themes": ["dystopia"]}]
    summary_dict = build_summary_dict(sample)
    assert summary_dict["1984"] == "Text"


def test_load_entries_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        load_book_entries(tmp_path / "missing.json")


def test_load_entries_rejects_empty_title(tmp_path: Path) -> None:
    path = tmp_path / "books.json"
    path.write_text('[{"title":" ","summary":"A","themes":["x"]}]', encoding="utf-8")

    with pytest.raises(ValueError, match="empty title"):
        load_book_entries(path)


def test_load_entries_rejects_non_list_themes(tmp_path: Path) -> None:
    path = tmp_path / "books.json"
    path.write_text('[{"title":"1984","summary":"A","themes":"dystopia"}]', encoding="utf-8")

    with pytest.raises(ValueError, match="themes as a list"):
        load_book_entries(path)
