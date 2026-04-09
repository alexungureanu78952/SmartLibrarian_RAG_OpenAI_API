"""FastAPI backend for Smart Librarian web UI and optional media features."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ragbot.chat import BookChatbot
from ragbot.config import get_settings
from ragbot.image_gen import generate_book_image
from ragbot.safety import moderate_text, polite_block_message
from ragbot.stt import transcribe_audio_bytes
from ragbot.tts import synthesize_to_mp3


MAX_STT_BYTES = 25 * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
LOGGER = logging.getLogger(__name__)


def _validate_stt_upload(file: UploadFile, raw: bytes) -> None:
    """Validate STT upload size and extension before API calls."""
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")
    if len(raw) > MAX_STT_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Audio file exceeds 25 MB limit. Please compress or split the file.",
        )

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio format. Allowed: mp3, mp4, mpeg, mpga, "
                "m4a, wav, webm."
            ),
        )


def _cors_origins_from_env() -> list[str]:
    """Read demo-friendly CORS origins without requiring full app settings."""
    if os.getenv("ALLOW_ALL_ORIGINS", "false").strip().lower() == "true":
        return ["*"]
    raw = os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5173,http://localhost:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _ensure_index_ready(chatbot: BookChatbot) -> None:
    """Fail startup when no indexed vectors are available for retrieval."""
    retrievers = getattr(chatbot, "retrievers", [])
    if not retrievers:
        raise RuntimeError("No retrievers were configured. Check embedding model settings.")

    total_vectors = 0
    for retriever in retrievers:
        try:
            total_vectors += int(retriever.collection.count())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed to read vector store state during startup.") from exc

    if total_vectors <= 0:
        raise RuntimeError("Vector store is empty. Run 'bookbot-index' before starting the web API.")


class ChatRequest(BaseModel):
    """Request payload for book recommendation chat."""

    message: str
    enable_tts: bool = False
    enable_image: bool = False


class ChatResponse(BaseModel):
    """Response payload returned by chat endpoint."""

    blocked: bool
    user_message: str
    recommendation_title: str | None = None
    reason: str | None = None
    detailed_summary: str | None = None
    audio_url: str | None = None
    image_url: str | None = None


def create_app() -> FastAPI:
    """Create and configure FastAPI application with runtime state."""
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings = get_settings()
        chatbot = BookChatbot(settings)
        _ensure_index_ready(chatbot)
        app.state.settings = settings
        app.state.chatbot = chatbot
        yield

    app = FastAPI(title="Smart Librarian API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins_from_env(),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    project_root = Path.cwd()
    web_root = project_root / "src" / "ragbot" / "web"
    audio_root = project_root / "audio_out"
    image_root = project_root / "image_out"

    app.mount("/audio", StaticFiles(directory=audio_root), name="audio")
    app.mount("/images", StaticFiles(directory=image_root), name="images")

    app.state.web_root = web_root
    app.state.audio_root = audio_root
    app.state.image_root = image_root

    @app.get("/")
    def home() -> FileResponse:
        return FileResponse(app.state.web_root / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/stt")
    async def transcribe(file: UploadFile = File(...)) -> dict[str, str]:
        raw = await file.read()
        _validate_stt_upload(file=file, raw=raw)

        try:
            text = transcribe_audio_bytes(
                client=app.state.chatbot.client,
                audio_bytes=raw,
                model=app.state.settings.stt_model,
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("STT transcription failed.")
            raise HTTPException(
                status_code=502,
                detail="I couldn't process the voice input right now. Please try again.",
            ) from exc
        return {"text": text}

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, request: Request) -> ChatResponse:
        message = req.message.strip()
        if not message:
            raise HTTPException(status_code=400, detail="Message cannot be empty.")

        moderation = moderate_text(
            message,
            settings=request.app.state.settings,
            client=request.app.state.chatbot.client,
        )
        if moderation.blocked:
            return ChatResponse(
                blocked=True,
                user_message=message,
                detailed_summary=polite_block_message(),
            )

        try:
            result = request.app.state.chatbot.ask(message)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("Chat request failed.")
            raise HTTPException(
                status_code=500,
                detail="I couldn't generate a recommendation right now. Please try again shortly.",
            ) from exc

        final_text = (
            f"Recommended book: {result.title}\n"
            f"Why: {result.reason}\n\n"
            f"Detailed summary:\n{result.full_summary}"
        )

        audio_url = None
        if req.enable_tts:
            try:
                audio_path = synthesize_to_mp3(
                    client=request.app.state.chatbot.client,
                    text=final_text,
                    model=request.app.state.settings.tts_model,
                    voice=request.app.state.settings.tts_voice,
                    output_dir=request.app.state.audio_root,
                )
                audio_url = f"/audio/{audio_path.name}"
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("TTS generation failed.")
                raise HTTPException(
                    status_code=502,
                    detail="I couldn't generate audio right now. Please try again.",
                ) from exc

        image_url = None
        if req.enable_image:
            try:
                image_path = generate_book_image(
                    client=request.app.state.chatbot.client,
                    title=result.title,
                    reason=result.reason,
                    model=request.app.state.settings.image_model,
                    output_dir=request.app.state.image_root,
                )
                image_url = f"/images/{image_path.name}"
            except Exception as exc:  # noqa: BLE001
                LOGGER.exception("Image generation failed.")
                raise HTTPException(
                    status_code=502,
                    detail="I couldn't generate an image right now. Please try again.",
                ) from exc

        return ChatResponse(
            blocked=False,
            user_message=message,
            recommendation_title=result.title,
            reason=result.reason,
            detailed_summary=result.full_summary,
            audio_url=audio_url,
            image_url=image_url,
        )

    return app


app = create_app()


def run() -> None:
    """CLI entrypoint for web API server."""
    uvicorn.run("ragbot.web_api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
