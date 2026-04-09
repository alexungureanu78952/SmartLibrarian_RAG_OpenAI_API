from pathlib import Path

from ragbot.config import Settings
from ragbot.safety import is_inappropriate, moderate_text, polite_block_message


class _FakeCategories:
    def __init__(self, **values: bool) -> None:
        self.__dict__.update(values)


class _FakeResult:
    def __init__(self, categories: _FakeCategories) -> None:
        self.categories = categories


class _FakeModerationResponse:
    def __init__(self, categories: _FakeCategories) -> None:
        self.results = [_FakeResult(categories)]


class _FakeModerations:
    def __init__(self, response: _FakeModerationResponse | None = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error

    def create(self, **_: str) -> _FakeModerationResponse:
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class _FakeClient:
    def __init__(self, response: _FakeModerationResponse | None = None, error: Exception | None = None) -> None:
        self.moderations = _FakeModerations(response=response, error=error)


def _settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "openai_api_key": "test-key",
        "chat_model": "gpt-4.1-mini",
        "embed_model": "text-embedding-3-small",
        "embed_fallback_models": (),
        "tts_model": "gpt-4o-mini-tts",
        "stt_model": "gpt-4o-mini-transcribe",
        "image_model": "gpt-image-1",
        "tts_voice": "alloy",
        "chroma_dir": Path("chroma_db"),
        "summaries_json": Path("data/book_summaries.json"),
        "top_k": 4,
        "moderation_enabled": True,
        "moderation_model": "omni-moderation-latest",
        "moderation_fail_behavior": "allow",
        "moderation_block_categories": (
            "harassment",
            "hate",
            "self_harm",
            "self_harm_intent",
            "self_harm_instructions",
            "sexual",
            "sexual_minors",
            "violence",
            "violence_graphic",
        ),
    }
    values.update(overrides)
    return Settings(**values)


def test_inappropriate_detection() -> None:
    assert is_inappropriate("You are an asshole") is True


def test_clean_message() -> None:
    assert is_inappropriate("I want a fantasy book") is False


def test_polite_message() -> None:
    msg = polite_block_message()
    assert "respectful" in msg.lower()


def test_moderate_text_blocks_selected_openai_category() -> None:
    client = _FakeClient(response=_FakeModerationResponse(_FakeCategories(harassment=True)))

    decision = moderate_text("Recommend a book.", settings=_settings(), client=client)

    assert decision.blocked is True
    assert decision.source == "openai"
    assert decision.reason == "harassment"


def test_moderate_text_allows_clean_message_when_openai_does_not_flag() -> None:
    client = _FakeClient(response=_FakeModerationResponse(_FakeCategories(harassment=False)))

    decision = moderate_text("I want a fantasy book.", settings=_settings(), client=client)

    assert decision.blocked is False
    assert decision.reason == "allowed"


def test_moderate_text_falls_back_to_local_block_when_openai_fails() -> None:
    client = _FakeClient(error=RuntimeError("moderation unavailable"))

    decision = moderate_text("You are an asshole", settings=_settings(), client=client)

    assert decision.blocked is True
    assert decision.source == "local"


def test_moderate_text_allows_clean_message_when_openai_fails_and_fail_behavior_allows() -> None:
    client = _FakeClient(error=RuntimeError("moderation unavailable"))

    decision = moderate_text("I want a fantasy book.", settings=_settings(), client=client)

    assert decision.blocked is False
    assert decision.source == "fallback"
