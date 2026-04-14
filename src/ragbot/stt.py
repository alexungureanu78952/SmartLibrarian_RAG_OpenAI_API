"""Speech-to-text helpers for audio transcription."""

from __future__ import annotations

from io import BytesIO

from openai import OpenAI

from ragbot.openai_retry import call_with_retry


def transcribe_audio_bytes(client: OpenAI, audio_bytes: bytes, model: str) -> str:
    """Transcribe in-memory audio bytes using the configured OpenAI STT model."""
    if not audio_bytes:
        raise ValueError("audio_bytes cannot be empty.")

    audio_file = BytesIO(audio_bytes)
    audio_file.name = "voice_input.webm"
    transcript = call_with_retry(
        lambda: client.audio.transcriptions.create(
            model=model,
            file=audio_file,
        )
    )
    text = str(getattr(transcript, "text", "")).strip()
    if not text:
        raise ValueError("Transcription returned an empty result.")
    return text
