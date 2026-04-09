"""Indexing pipeline that stores local summaries into ChromaDB."""

from __future__ import annotations

from pathlib import Path

import chromadb
from openai import OpenAI

from ragbot.config import get_settings
from ragbot.data_loader import load_book_entries
from ragbot.openai_retry import call_with_retry
from ragbot.retriever import collection_name_for_embedding


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

        ids.append(f"book-{i}")
        docs.append(combined_text)
        metas.append({"title": title, "themes": themes})
        embeddings.append(emb.data[0].embedding)

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
