"""Text-to-speech helpers for optional audio output."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from openai import OpenAI

from ragbot.openai_retry import call_with_retry


def synthesize_to_mp3(
    client: OpenAI,
    text: str,
    model: str,
    voice: str,
    output_dir: Path,
) -> Path:
    """Generate MP3 audio from text and return the saved file path."""
    clean_text = text.strip()
    if not clean_text:
        raise ValueError("text cannot be empty for TTS synthesis.")
    if not model.strip():
        raise ValueError("model cannot be empty for TTS synthesis.")
    if not voice.strip():
        raise ValueError("voice cannot be empty for TTS synthesis.")

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    output_path = output_dir / filename

    response = call_with_retry(
        lambda: client.audio.speech.create(
            model=model,
            voice=voice,
            input=clean_text,
        )
    )
    response.stream_to_file(output_path)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise ValueError("TTS generation did not produce a valid audio file.")
    return output_path
