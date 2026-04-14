from pathlib import Path
from types import SimpleNamespace

import pytest

from ragbot.image_gen import build_image_prompt
from ragbot.image_gen import generate_book_image


def test_build_image_prompt_contains_title_and_reason() -> None:
    prompt = build_image_prompt(
        title="1984",
        reason="It strongly matches freedom and social control themes.",
    )
    assert "1984" in prompt
    assert "freedom and social control" in prompt
    assert "no text overlays" in prompt


def test_generate_book_image_rejects_empty_title(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="title cannot be empty"):
        generate_book_image(
            client=SimpleNamespace(),
            title=" ",
            reason="Strong match",
            model="gpt-image-1",
            output_dir=tmp_path,
        )


def test_generate_book_image_rejects_empty_reason(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reason cannot be empty"):
        generate_book_image(
            client=SimpleNamespace(),
            title="1984",
            reason=" ",
            model="gpt-image-1",
            output_dir=tmp_path,
        )


def test_generate_book_image_rejects_empty_data_payload(tmp_path: Path) -> None:
    fake_client = SimpleNamespace(images=SimpleNamespace(generate=lambda **_: SimpleNamespace(data=[])))

    with pytest.raises(ValueError, match="no data payload"):
        generate_book_image(
            client=fake_client,
            title="1984",
            reason="Strong dystopian match",
            model="gpt-image-1",
            output_dir=tmp_path,
        )
