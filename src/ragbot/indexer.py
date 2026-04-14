"""Indexing pipeline that stores local summaries into ChromaDB."""

from __future__ import annotations

from pathlib import Path

import chromadb
from openai import OpenAI

from ragbot.config import get_settings
from ragbot.data_loader import load_book_entries
from ragbot.openai_retry import call_with_retry
from ragbot.retriever import collection_name_for_embedding


def _extract_embedding_vector(response: object, *, title: str) -> list[float]:
    """Extract and validate the embedding vector from OpenAI response payload."""
    data = getattr(response, "data", None)
    if not data:
        raise ValueError(f"Embedding response for '{title}' contained no data.")

    first_item = data[0]
    vector = getattr(first_item, "embedding", None)
    if not isinstance(vector, list) or not vector:
        raise ValueError(f"Embedding response for '{title}' contained an empty vector.")

    if any(not isinstance(value, (int, float)) for value in vector):
        raise ValueError(f"Embedding response for '{title}' contained non-numeric values.")

    return [float(value) for value in vector]


def build_index() -> None:
    """Generate embeddings for all books and persist them to ChromaDB."""
    settings = get_settings()
    collection_name = collection_name_for_embedding(settings.embed_model)
    entries = load_book_entries(settings.summaries_json)

    openai_client = OpenAI(api_key=settings.openai_api_key)
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    collection = chroma_client.get_or_create_collection(name=collection_name)

    ids: list[str] = []
    docs: list[str] = []
    metas: list[dict[str, str]] = []
    embeddings: list[list[float]] = []
    expected_dimension: int | None = None

    for i, entry in enumerate(entries):
        title = str(entry["title"])
        summary = str(entry["summary"])
        themes = ", ".join(str(t) for t in entry.get("themes", []))
        combined_text = f"Title: {title}\nThemes: {themes}\nSummary: {summary}"

        emb = call_with_retry(
            lambda: openai_client.embeddings.create(
                model=settings.embed_model,
                input=combined_text,
            )
        )

        vector = _extract_embedding_vector(emb, title=title)
        if expected_dimension is None:
            expected_dimension = len(vector)
        elif len(vector) != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch for '{title}': "
                f"expected {expected_dimension}, got {len(vector)}."
            )

        ids.append(f"book-{i}")
        docs.append(combined_text)
        metas.append({"title": title, "themes": themes})
        embeddings.append(vector)

    # Reset collection so repeated indexing stays deterministic for demo.
    existing = collection.get(include=[])
    existing_ids = existing.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)

    collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    print(
        f"Indexed {len(ids)} books into '{collection_name}' "
        f"(embedding model: {settings.embed_model}) at {Path(settings.chroma_dir).resolve()}."
    )


def main() -> None:
    """CLI entrypoint for indexing."""
    build_index()


if __name__ == "__main__":
    main()
