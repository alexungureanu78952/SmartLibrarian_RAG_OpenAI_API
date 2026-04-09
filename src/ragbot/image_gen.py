"""Image generation helpers for optional visual recommendations."""

from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from ragbot.openai_retry import call_with_retry


def build_image_prompt(title: str, reason: str) -> str:
    """Create a concise prompt for a book-themed image."""
    return (
        "Create a cinematic book-inspired illustration with no text overlays. "
        f"Book title: {title}. "
        f"Core recommendation reason: {reason}. "
        "Style: atmospheric digital painting, detailed environment, rich lighting, modern cover-art vibe."
    )


def generate_book_image(
    client: OpenAI,
    title: str,
    reason: str,
    model: str,
    output_dir: Path,
) -> Path:
    """Generate and persist a single representative image for a recommendation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_image_prompt(title=title, reason=reason)

    result = call_with_retry(
        lambda: client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
        )
    )

    b64_data = result.data[0].b64_json
    if not b64_data:
        raise ValueError("Image generation returned no image data.")

    image_bytes = base64.b64decode(b64_data)
    filename = f"{title.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    output_path = output_dir / filename
    output_path.write_bytes(image_bytes)
    return output_path
