"""Image generation helpers for optional visual recommendations."""

from __future__ import annotations

import base64
import re
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
    clean_title = title.strip()
    clean_reason = reason.strip()
    if not clean_title:
        raise ValueError("title cannot be empty for image generation.")
    if not clean_reason:
        raise ValueError("reason cannot be empty for image generation.")

    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_image_prompt(title=clean_title, reason=clean_reason)

    result = call_with_retry(
        lambda: client.images.generate(
            model=model,
            prompt=prompt,
            size="1024x1024",
        )
    )

    if not getattr(result, "data", None):
        raise ValueError("Image generation returned no data payload.")

    b64_data = result.data[0].b64_json
    if not b64_data:
        raise ValueError("Image generation returned no image data.")

    image_bytes = base64.b64decode(b64_data, validate=True)
    safe_title = re.sub(r"[^a-z0-9]+", "_", clean_title.lower()).strip("_") or "book"
    filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    output_path = output_dir / filename
    output_path.write_bytes(image_bytes)
    return output_path
