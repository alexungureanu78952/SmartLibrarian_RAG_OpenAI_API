from pathlib import Path
from types import SimpleNamespace

import pytest

from ragbot.config import Settings
from ragbot.indexer import _extract_embedding_vector, build_index


def _settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        chat_model="gpt-4.1-mini",
        embed_model="text-embedding-3-small",
        embed_fallback_models=(),
        tts_model="gpt-4o-mini-tts",
        stt_model="gpt-4o-mini-transcribe",
        image_model="gpt-image-1",
        tts_voice="alloy",
        chroma_dir=Path("chroma_db"),
        summaries_json=Path("data/book_summaries.json"),
        top_k=4,
        moderation_enabled=True,
        moderation_model="omni-moderation-latest",
        moderation_fail_behavior="allow",
        moderation_block_categories=("harassment", "hate", "self_harm", "sexual", "violence"),
    )


def test_extract_embedding_vector_rejects_missing_data() -> None:
    with pytest.raises(ValueError, match="contained no data"):
        _extract_embedding_vector(SimpleNamespace(data=[]), title="1984")


def test_extract_embedding_vector_rejects_empty_vector() -> None:
    response = SimpleNamespace(data=[SimpleNamespace(embedding=[])])
    with pytest.raises(ValueError, match="empty vector"):
        _extract_embedding_vector(response, title="1984")


def test_extract_embedding_vector_rejects_non_numeric_values() -> None:
    response = SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, "bad", 0.3])])
    with pytest.raises(ValueError, match="non-numeric"):
        _extract_embedding_vector(response, title="1984")


def test_build_index_rejects_dimension_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("ragbot.indexer.get_settings", _settings)
    monkeypatch.setattr(
        "ragbot.indexer.load_book_entries",
        lambda _path: [
            {"title": "Book One", "summary": "Summary one", "themes": ["theme"]},
            {"title": "Book Two", "summary": "Summary two", "themes": ["theme"]},
        ],
    )

    responses = [
        SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2, 0.3])]),
        SimpleNamespace(data=[SimpleNamespace(embedding=[0.4, 0.5])]),
    ]

    fake_openai = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **_: responses.pop(0))
    )
    monkeypatch.setattr("ragbot.indexer.OpenAI", lambda api_key: fake_openai)

    fake_collection = SimpleNamespace(
        get=lambda include: {"ids": []},
        delete=lambda ids: None,
        add=lambda **kwargs: None,
    )
    fake_chroma = SimpleNamespace(get_or_create_collection=lambda name: fake_collection)
    monkeypatch.setattr("ragbot.indexer.chromadb.PersistentClient", lambda path: fake_chroma)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        build_index()
