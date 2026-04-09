from ragbot.image_gen import build_image_prompt


def test_build_image_prompt_contains_title_and_reason() -> None:
    prompt = build_image_prompt(
        title="1984",
        reason="It strongly matches freedom and social control themes.",
    )
    assert "1984" in prompt
    assert "freedom and social control" in prompt
    assert "no text overlays" in prompt
