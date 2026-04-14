import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ragbot.chat import BookChatbot
from ragbot.config import Settings


class _FakeResponses:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return SimpleNamespace(output_text=json.dumps({"title": "The Hobbit", "reason": "Fantasy adventure."}))

        function_call = SimpleNamespace(
            type="function_call",
            call_id="call_1",
            name="get_summary_by_title",
            arguments=json.dumps({"title": "The Hobbit"}),
        )
        return SimpleNamespace(output=[function_call], output_text="")


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.responses = _FakeResponses()


class _FakeRetriever:
    collection_name = "book_summaries__text_embedding_3_small"

    def search(self, query: str, top_k: int) -> list[dict[str, str]]:  # noqa: ARG002
        return [
            {
                "title": "The Hobbit",
                "themes": "friendship, adventure",
                "summary": "Bilbo goes on a journey.",
                "distance": "0.1234",
            }
        ]


@pytest.fixture
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


def test_chat_uses_exact_local_summary(monkeypatch: pytest.MonkeyPatch, _settings: Settings) -> None:
    monkeypatch.setattr("ragbot.chat.OpenAI", lambda api_key: _FakeOpenAIClient())
    monkeypatch.setattr(
        "ragbot.chat.load_book_entries",
        lambda _path: [{"title": "The Hobbit", "summary": "Exact local summary.", "themes": ["adventure"]}],
    )
    monkeypatch.setattr(
        "ragbot.chat.build_summary_dict",
        lambda _entries: {"The Hobbit": "Exact local summary."},
    )
    monkeypatch.setattr("ragbot.chat.Retriever.from_paths", lambda **_kwargs: _FakeRetriever())

    chatbot = BookChatbot(_settings)
    result = chatbot.ask("I want a fantasy adventure.")

    assert result.title == "The Hobbit"
    assert result.full_summary == "Exact local summary."


def test_chat_rejects_empty_user_query(monkeypatch: pytest.MonkeyPatch, _settings: Settings) -> None:
    monkeypatch.setattr("ragbot.chat.OpenAI", lambda api_key: _FakeOpenAIClient())
    monkeypatch.setattr(
        "ragbot.chat.load_book_entries",
        lambda _path: [{"title": "The Hobbit", "summary": "Exact local summary.", "themes": ["adventure"]}],
    )
    monkeypatch.setattr(
        "ragbot.chat.build_summary_dict",
        lambda _entries: {"The Hobbit": "Exact local summary."},
    )
    monkeypatch.setattr("ragbot.chat.Retriever.from_paths", lambda **_kwargs: _FakeRetriever())

    chatbot = BookChatbot(_settings)

    with pytest.raises(ValueError, match="User query cannot be empty"):
        chatbot.ask("   ")
