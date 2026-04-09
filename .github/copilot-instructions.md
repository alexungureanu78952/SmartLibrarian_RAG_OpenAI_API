# Copilot Instructions for This Repository

These instructions define implementation best practices for this Python + ChromaDB + OpenAI project.
Follow them for all code changes unless the user explicitly asks otherwise.

## Project Goals

- Build and maintain a small, understandable RAG chatbot for book recommendations.
- Prioritize correctness, readability, low cost, and reproducibility.
- Keep the implementation simple and explicit over clever abstractions.

## Python Best Practices

Derived from official Python docs and style guidance:
- https://docs.python.org/3/tutorial/venv.html
- https://peps.python.org/pep-0008/

Rules:
1. Use a project-local virtual environment (`.venv`) and never rely on global packages.
2. Use `python -m pip ...` style invocation when using pip directly.
3. Follow PEP 8 naming and formatting conventions:
- 4-space indentation
- snake_case for functions/variables/modules
- CapWords for classes
- clear, short module names
4. Keep functions focused and small; prefer explicit control flow.
5. Use type hints on public functions and dataclasses where helpful.
6. Handle exceptions narrowly (`except Exception` only when justified).
7. Avoid wildcard imports and side effects at import time.
8. Write docstrings for public modules/classes/functions.
9. Keep comments useful and maintenance-friendly; remove stale comments.

## ChromaDB Best Practices

Derived from official Chroma docs:
- https://docs.trychroma.com/docs/overview/introduction
- https://docs.trychroma.com/docs/collections/add-data

Rules:
1. Use a persistent local path for Chroma data (`CHROMA_DIR`).
2. Every vector record must have a unique, deterministic string `id`.
3. Store both `documents` and explicit `embeddings` when embedding externally with OpenAI.
4. Include useful metadata (`title`, `themes`, source tags) for filtering/debugging.
5. Keep metadata types valid and simple (string/number/bool/arrays of a single type).
6. Keep embedding dimensions consistent inside each collection.
7. Do not silently rely on duplicate `id` behavior; when re-indexing, clear or update deterministically.
8. Keep collection schema stable over time; document any migration needed.

## OpenAI API Best Practices

Derived from official OpenAI docs:
- https://developers.openai.com/api/docs/guides/text
- https://developers.openai.com/api/docs/guides/embeddings
- https://developers.openai.com/api/docs/guides/function-calling

Rules:
1. Prefer the Responses API for new features; if Chat Completions is used, keep usage consistent and documented.
2. Pin model names in configuration and avoid hidden model changes.
3. Keep prompts structured with clear system/developer/user intent.
4. Use tool/function schemas with explicit JSON schema:
- include descriptions
- set `additionalProperties: false`
- require all expected fields
- enable strict behavior where supported
5. Implement full tool-calling loop correctly:
- send tools
- parse one or more tool calls
- execute app-side code
- return tool outputs
- ask model for final response
6. Validate tool arguments before execution; never trust raw model arguments blindly.
7. Treat tool output as data, not executable code.
8. Add retry/backoff for transient API failures and clear user-facing errors.
9. Never log or commit secrets (`OPENAI_API_KEY` must come from env).
10. Control token cost:
- keep retrieved context concise
- keep tool list small
- prefer smaller models for default paths
- avoid unnecessary repeated embedding calls

## OpenAI Audio Best Practices (TTS/STT)

Derived from official OpenAI docs:
- https://developers.openai.com/api/docs/guides/text-to-speech
- https://developers.openai.com/api/docs/guides/speech-to-text

Rules:
1. Use pinned audio model names in config (`OPENAI_TTS_MODEL`, `OPENAI_STT_MODEL`) and avoid hardcoded model strings in feature code.
2. For TTS, default to `gpt-4o-mini-tts` for reliability and configurable voices; keep voice configurable from env.
3. For low-latency playback pathways, prefer `wav` or `pcm`; for simple file output, `mp3` is acceptable.
4. Always disclose to end users that generated voice is AI-generated.
5. For STT, validate upload type and size before API call; supported audio formats include `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, and `webm`.
6. Enforce the 25 MB STT file limit with actionable fallback guidance (chunk/compress audio).
7. Use `prompt` for STT only when domain-specific words/acronyms need accuracy improvements.
8. Return graceful errors for failed transcription/synthesis and never crash the chat loop.

## OpenAI Image Generation Best Practices

Derived from official OpenAI docs:
- https://developers.openai.com/api/docs/guides/image-generation

Rules:
1. Use Image API for single-shot generation and Responses image tool for conversational multi-turn image workflows.
2. Keep image model configurable from env (`OPENAI_IMAGE_MODEL`) and document quality/cost tradeoffs per model.
3. For demo speed, default to square `1024x1024` and medium quality unless user explicitly asks otherwise.
4. Decode and persist returned base64 image safely; validate that image payload exists before writing files.
5. Handle potentially high latency in UI/API (image calls can take significantly longer than text).
6. Avoid prompt instructions that depend on exact text rendering or pixel-perfect layout.
7. Respect content policy filtering and provide user-safe fallback messages when generation is blocked.

## FastAPI Backend Best Practices

Derived from official FastAPI docs:
- https://fastapi.tiangolo.com/tutorial/

Rules:
1. Use Pydantic request/response models for endpoint contracts; avoid untyped dict payloads in public handlers.
2. Use `HTTPException` with meaningful status codes/messages for user and integration errors.
3. For file uploads, use `UploadFile` and validate empty files/invalid formats early.
4. Keep endpoint logic thin; delegate business logic to service modules (chat/retrieval/audio/image helpers).
5. Add a simple health endpoint for runtime checks (`/api/health`).
6. Keep CORS policy explicit and restricted in production scenarios.
7. Serve static assets and generated media through explicit mounted paths and never expose secrets via static routes.

## Frontend (Vue) Best Practices

Derived from official Vue docs:
- https://vuejs.org/guide/introduction.html

Rules:
1. Keep UI state reactive and centralized (request state, toggles, results, and error messages).
2. Use declarative rendering for all response states (loading, success, blocked, error) instead of manual DOM manipulation.
3. Isolate async API calls in dedicated methods and always handle `try/catch/finally` for fetch workflows.
4. Show user-friendly network diagnostics when backend is unreachable.
5. Keep voice-mode and media controls explicit and reversible (start/stop toggles, clear status text).

## RAG-Specific Rules

1. Keep dataset language consistent (English-only in this project).
2. Use deterministic indexing inputs (`Title + Themes + Summary`) so retrieval is reproducible.
3. Retrieve top-k small enough for cost and quality (default 3-4 unless tests show otherwise).
4. Return at least one recommendation reason grounded in retrieved context.
5. After recommending a title, call `get_summary_by_title(title)` for full local summary.
6. If title not found, return a graceful fallback and ask for exact title.

## Safety and Reliability

1. Run local inappropriate-language filtering before LLM calls when enabled.
2. Fail gracefully when:
- vector store is empty
- env vars are missing
- tool arguments are invalid
3. Do not crash the CLI loop on request-level errors.
4. Provide actionable error messages for setup problems.

## Testing and Validation Expectations

1. For each change, keep or add tests when practical:
- summary tool lookup
- safety filter behavior
- data loader validation
2. Prefer deterministic tests that do not require live API calls.
3. For API flows, isolate logic so behavior can be mocked.
4. Run syntax checks and tests before finalizing significant edits.

## Documentation Expectations

1. Keep README runnable by a new developer in under 10 minutes.
2. Document required env vars, index step, run step, and sample prompts.
3. Keep examples aligned with actual CLI commands and code paths.
4. Update docs whenever command names, models, or config keys change.

## Cost-Aware Development Mode

When asked to minimize token usage:
1. Disable optional token-heavy features (TTS/STT/image generation).
2. Keep retrieval context and completions concise.
3. Favor local/mock pathways for development tests.
4. Reserve live API calls for final verification only.
