from pathlib import Path
from types import SimpleNamespace

import pytest

from ragbot.stt import transcribe_audio_bytes
from ragbot.tts import synthesize_to_mp3


class _FakeSpeechResponse:
    def stream_to_file(self, path: Path) -> None:
        path.write_bytes(b"fake-mp3")


class _FakeSpeech:
    def create(self, **_: str) -> _FakeSpeechResponse:
        return _FakeSpeechResponse()


class _FakeAudio:
    def __init__(self) -> None:
        self.speech = _FakeSpeech()


class _FakeClient:
    def __init__(self) -> None:
        self.audio = _FakeAudio()


class _FakeTranscriptions:
    def __init__(self, text: str) -> None:
        self._text = text

    def create(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(text=self._text)


class _FakeTranscriptionClient:
    def __init__(self, text: str) -> None:
        self.audio = SimpleNamespace(transcriptions=_FakeTranscriptions(text=text))


def test_transcribe_audio_bytes_rejects_empty_audio() -> None:
    with pytest.raises(ValueError, match="audio_bytes cannot be empty"):
        transcribe_audio_bytes(client=_FakeTranscriptionClient(text="ok"), audio_bytes=b"", model="stt-model")


def test_transcribe_audio_bytes_rejects_empty_transcript() -> None:
    with pytest.raises(ValueError, match="empty result"):
        transcribe_audio_bytes(
            client=_FakeTranscriptionClient(text="  "),
            audio_bytes=b"audio-data",
            model="stt-model",
        )


def test_synthesize_to_mp3_rejects_empty_text(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="text cannot be empty"):
        synthesize_to_mp3(
            client=_FakeClient(),
            text="   ",
            model="tts-model",
            voice="alloy",
            output_dir=tmp_path,
        )


def test_synthesize_to_mp3_rejects_empty_voice(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="voice cannot be empty"):
        synthesize_to_mp3(
            client=_FakeClient(),
            text="Hello",
            model="tts-model",
            voice="   ",
            output_dir=tmp_path,
        )
