"""Command-line interface for Smart Librarian chatbot."""

from __future__ import annotations

from pathlib import Path

from ragbot.chat import BookChatbot
from ragbot.config import get_settings
from ragbot.image_gen import generate_book_image
from ragbot.safety import moderate_text, polite_block_message
from ragbot.stt import transcribe_audio_bytes
from ragbot.tts import synthesize_to_mp3


HELP_TEXT = """
Commands:
  /help        Show commands
  /quit        Exit
  /tts on      Enable text-to-speech output
  /tts off     Disable text-to-speech output
    /image on    Enable image generation output
    /image off   Disable image generation output
    /stt <path>  Transcribe local audio file and send as query

Example prompts:
  I want a book about freedom and social control.
  What do you recommend for someone who loves war stories?
  I want a book about friendship and magic.
""".strip()


def main() -> None:
    """Run interactive CLI chat loop with optional media features."""
    settings = get_settings()
    chatbot = BookChatbot(settings)
    tts_enabled = False
    image_enabled = False

    print("Book RAG Chatbot (English-only)")
    print("Type /help for commands. Run indexing first with: bookbot-index")

    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue

        if user_input.lower() in {"/quit", "quit", "exit"}:
            print("Goodbye.")
            break

        if user_input.lower() == "/help":
            print(HELP_TEXT)
            continue

        if user_input.lower() == "/tts on":
            tts_enabled = True
            print("TTS enabled.")
            continue

        if user_input.lower() == "/tts off":
            tts_enabled = False
            print("TTS disabled.")
            continue

        if user_input.lower() == "/image on":
            image_enabled = True
            print("Image generation enabled.")
            continue

        if user_input.lower() == "/image off":
            image_enabled = False
            print("Image generation disabled.")
            continue

        if user_input.lower().startswith("/stt "):
            audio_path = Path(user_input[5:].strip())
            if not audio_path.exists() or not audio_path.is_file():
                print("\nAssistant: Audio file not found.")
                continue
            try:
                transcription = transcribe_audio_bytes(
                    client=chatbot.client,
                    audio_bytes=audio_path.read_bytes(),
                    model=settings.stt_model,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"\nAssistant: STT failed: {exc}")
                continue
            print(f"\nTranscribed: {transcription}")
            user_input = transcription

        moderation = moderate_text(
            user_input,
            settings=settings,
            client=chatbot.client,
        )
        if moderation.blocked:
            print(f"\nAssistant: {polite_block_message()}")
            continue

        try:
            result = chatbot.ask(user_input)
        except Exception as exc:  # noqa: BLE001
            print(f"\nAssistant: Sorry, something went wrong: {exc}")
            continue

        final_response = (
            f"Recommended book: {result.title}\n"
            f"Why: {result.reason}\n\n"
            f"Detailed summary:\n{result.full_summary}"
        )
        print(f"\nAssistant:\n{final_response}")

        if tts_enabled:
            try:
                audio_path = synthesize_to_mp3(
                    client=chatbot.client,
                    text=final_response,
                    model=settings.tts_model,
                    voice=settings.tts_voice,
                    output_dir=Path("audio_out"),
                )
                print(f"\nAudio saved to: {audio_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"\nTTS failed: {exc}")

        if image_enabled:
            try:
                image_path = generate_book_image(
                    client=chatbot.client,
                    title=result.title,
                    reason=result.reason,
                    model=settings.image_model,
                    output_dir=Path("image_out"),
                )
                print(f"\nImage saved to: {image_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"\nImage generation failed: {exc}")


if __name__ == "__main__":
    main()
