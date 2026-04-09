from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

from ragbot.config import Settings
from ragbot.safety import ModerationDecision
from ragbot.web_api import MAX_STT_BYTES, _validate_stt_upload


def _upload(filename: str) -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(b"abc"))


def test_validate_stt_upload_rejects_empty() -> None:
    file = _upload("voice.webm")
    with pytest.raises(HTTPException) as exc:
        _validate_stt_upload(file, b"")
    assert exc.value.status_code == 400


def test_validate_stt_upload_rejects_large_file() -> None:
    file = _upload("voice.webm")
    with pytest.raises(HTTPException) as exc:
        _validate_stt_upload(file, b"x" * (MAX_STT_BYTES + 1))
    assert exc.value.status_code == 413


def test_validate_stt_upload_rejects_extension() -> None:
    file = _upload("voice.txt")
    with pytest.raises(HTTPException) as exc:
        _validate_stt_upload(file, b"valid")
    assert exc.value.status_code == 400


def test_validate_stt_upload_accepts_supported_extension() -> None:
    file = _upload("voice.wav")
    _validate_stt_upload(file, b"valid")


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


class _FakeChatbot:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = object()
        self.retrievers = [SimpleNamespace(collection=SimpleNamespace(count=lambda: 1))]

    def ask(self, message: str) -> SimpleNamespace:
        return SimpleNamespace(
            title="1984",
            reason=f"Matched query: {message}",
            full_summary="A dystopian novel about surveillance.",
            retrieval_hits=[],
        )


def _make_test_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, decision: ModerationDecision) -> TestClient:
    import ragbot.web_api as web_api

    (tmp_path / "audio_out").mkdir()
    (tmp_path / "image_out").mkdir()
    (tmp_path / "src" / "ragbot" / "web").mkdir(parents=True)
    (tmp_path / "src" / "ragbot" / "web" / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_api, "get_settings", lambda: _settings())
    monkeypatch.setattr(web_api, "BookChatbot", _FakeChatbot)
    monkeypatch.setattr(web_api, "moderate_text", lambda *_args, **_kwargs: decision)
    return TestClient(web_api.create_app())


def test_chat_returns_blocked_response_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _make_test_client(
        monkeypatch,
        tmp_path,
        ModerationDecision(blocked=True, source="openai", reason="harassment"),
    ) as client:
        response = client.post("/api/chat", json={"message": "Bad message"})

    assert response.status_code == 200
    data = response.json()
    assert data["blocked"] is True
    assert "respectful" in data["detailed_summary"].lower()
    assert data["recommendation_title"] is None


def test_chat_returns_recommendation_when_not_blocked(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with _make_test_client(
        monkeypatch,
        tmp_path,
        ModerationDecision(blocked=False, source="openai", reason="allowed"),
    ) as client:
        response = client.post("/api/chat", json={"message": "Recommend a classic dystopian book."})

    assert response.status_code == 200
    data = response.json()
    assert data["blocked"] is False
    assert data["recommendation_title"] == "1984"
    assert "surveillance" in data["detailed_summary"].lower()


def test_chat_returns_generic_error_message(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ragbot.web_api as web_api

    class _FailingChatbot(_FakeChatbot):
        def ask(self, message: str) -> SimpleNamespace:  # noqa: ARG002
            raise RuntimeError("openai timeout")

    (tmp_path / "audio_out").mkdir()
    (tmp_path / "image_out").mkdir()
    (tmp_path / "src" / "ragbot" / "web").mkdir(parents=True)
    (tmp_path / "src" / "ragbot" / "web" / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_api, "get_settings", lambda: _settings())
    monkeypatch.setattr(web_api, "BookChatbot", _FailingChatbot)
    monkeypatch.setattr(
        web_api,
        "moderate_text",
        lambda *_args, **_kwargs: ModerationDecision(blocked=False, source="openai", reason="allowed"),
    )

    with TestClient(web_api.create_app()) as client:
        response = client.post("/api/chat", json={"message": "Recommend a classic dystopian book."})

    assert response.status_code == 500
    assert response.json()["detail"] == "I couldn't generate a recommendation right now. Please try again shortly."


def test_startup_fails_when_vector_store_is_empty(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import ragbot.web_api as web_api

    class _EmptyIndexChatbot(_FakeChatbot):
        def __init__(self, settings: Settings) -> None:
            super().__init__(settings)
            self.retrievers = [SimpleNamespace(collection=SimpleNamespace(count=lambda: 0))]

    (tmp_path / "audio_out").mkdir()
    (tmp_path / "image_out").mkdir()
    (tmp_path / "src" / "ragbot" / "web").mkdir(parents=True)
    (tmp_path / "src" / "ragbot" / "web" / "index.html").write_text("<html></html>", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(web_api, "get_settings", lambda: _settings())
    monkeypatch.setattr(web_api, "BookChatbot", _EmptyIndexChatbot)

    with pytest.raises(RuntimeError, match="Vector store is empty"):
        with TestClient(web_api.create_app()):
            pass
