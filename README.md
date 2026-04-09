# Book Recommendation Chatbot (OpenAI + RAG + ChromaDB)

This project implements an English-only AI chatbot that recommends books based on user interests.
It uses:

- OpenAI GPT for conversational reasoning
- ChromaDB as a local vector store (not OpenAI vector store)
- OpenAI embeddings (`text-embedding-3-small`) for semantic search
- OpenAI tool calling to fetch a full local summary by exact title

After the model recommends a book, it automatically calls the tool `get_summary_by_title(title)` and returns a detailed summary.

## Features

- Local corpus with 10+ books in `data/book_summaries.md` and `data/book_summaries.json`
- Semantic retrieval over book themes and context
- CLI chatbot
- Web chatbot UI (Vue frontend + FastAPI backend)
- Tool calling workflow (`get_summary_by_title`)
- Optional language filter for offensive words
- Hybrid moderation flow with OpenAI moderation plus local profanity fallback
- Optional TTS mode that saves recommendation audio as `.mp3`
- Optional STT voice mode (microphone in web UI or file-based command in CLI)
- Optional image generation for the recommended book

## Project Structure

- `data/book_summaries.md`: Human-readable dataset (10+ books)
- `data/book_summaries.json`: Structured source for indexing and tool lookup
- `src/ragbot/indexer.py`: Build Chroma index with OpenAI embeddings
- `src/ragbot/retriever.py`: Semantic retrieval from ChromaDB
- `src/ragbot/tools.py`: Tool function + tool schema
- `src/ragbot/chat.py`: LLM orchestration with retrieval + function calling
- `src/ragbot/ui_cli.py`: CLI interface
- `src/ragbot/web_api.py`: FastAPI backend for browser UI
- `src/ragbot/web/index.html`: Vue-based browser frontend
- `src/ragbot/safety.py`: Optional basic inappropriate language filter
- `src/ragbot/tts.py`: Optional TTS using OpenAI audio API
- `src/ragbot/stt.py`: Optional speech-to-text transcription
- `src/ragbot/image_gen.py`: Optional image generation

## Setup

### 1) Install dependencies

Using `uv`:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .[dev]
```

### 2) Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY`.

Optional model settings:

- `OPENAI_STT_MODEL=gpt-4o-mini-transcribe`
- `OPENAI_IMAGE_MODEL=gpt-image-1`
- `OPENAI_EMBED_FALLBACK_MODELS=text-embedding-3-small`
- `OPENAI_MODERATION_MODEL=omni-moderation-latest`
- `MODERATION_BLOCK_CATEGORIES=harassment,hate,self_harm,self_harm_intent,self_harm_instructions,sexual,sexual_minors,violence,violence_graphic`
- `MODERATION_FAIL_BEHAVIOR=allow`

Runtime behavior:

- The web API reads `.env` changes at runtime and applies settings without process restart.
- Process environment variables still take precedence over `.env` values.
- For embedding model changes, index data is stored per-model collection. Use fallbacks during rollout to avoid retrieval downtime.

## Build the Vector Store

```bash
bookbot-index
```

Expected output includes something like:

```text
Indexed 12 books into 'book_summaries__text_embedding_3_small' (embedding model: text-embedding-3-small) ...
```

### No-downtime embedding model rollout

1. Keep current model as fallback:
	- `OPENAI_EMBED_MODEL=text-embedding-3-large`
	- `OPENAI_EMBED_FALLBACK_MODELS=text-embedding-3-small`
2. Build index with the new primary model:
	- `bookbot-index`
3. Keep serving traffic while the app tries primary first and fallback second.
4. After validating new retrieval quality, clear fallbacks.

## Run the Chatbot

```bash
bookbot-chat
```

## Run the Web App (Backend + Frontend)

```bash
bookbot-web
```

Then open `http://127.0.0.1:8000` in your browser.

## CLI Commands

- `/help` - show commands and sample prompts
- `/tts on` - enable text-to-speech output
- `/tts off` - disable text-to-speech output
- `/image on` - enable image generation output
- `/image off` - disable image generation output
- `/stt <path>` - transcribe a local audio file and use it as the query
- `/quit` - exit

## Test Prompts

- `I want a book about freedom and social control.`
- `What do you recommend for someone who loves fantasy stories?`
- `What do you recommend for someone who loves war stories?`
- `What is 1984 about?`

## Tool Calling Flow

1. User asks for a recommendation.
2. Retriever gets top-k semantic matches from ChromaDB.
3. LLM recommends one book title.
4. LLM is forced to call `get_summary_by_title(title)`.
5. Local tool returns the detailed summary.
6. Final answer is shown with recommendation + detailed summary.

## Optional Features Flow

1. **Hybrid moderation** checks input before retrieval and LLM call using OpenAI moderation first, then local profanity fallback on moderation errors.
2. **TTS** can convert final answer to audio and save it under `audio_out/`.
3. **STT** can transcribe voice input to text:
	- in web UI through microphone recording
	- in CLI through `/stt <audio_file_path>`
4. **Image generation** can create a themed visual and save it under `image_out/`.

## Tests

Run tests:

```bash
pytest -q
```

## Notes

- If ChromaDB setup fails in your environment, you can swap to another local vector store while preserving the same retriever interface.
- Moderation defaults to OpenAI-first with local fallback; if the moderation API is unavailable, clean requests are allowed by default and obvious blocked terms still stop locally.
- TTS depends on OpenAI audio API availability and key permissions.
