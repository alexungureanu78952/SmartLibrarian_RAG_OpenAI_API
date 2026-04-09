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
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"recommendation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
    output_path = output_dir / filename

    response = call_with_retry(
        lambda: client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
        )
    )
    response.stream_to_file(output_path)
    return output_path
