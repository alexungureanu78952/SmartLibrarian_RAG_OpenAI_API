"""Semantic retrieval helpers backed by ChromaDB and OpenAI embeddings."""

from __future__ import annotations

import re
from dataclasses import dataclass

import chromadb
from chromadb.api.models.Collection import Collection
from openai import OpenAI

from ragbot.openai_retry import call_with_retry


COLLECTION_NAME = "book_summaries"


def collection_name_for_embedding(model: str) -> str:
    """Build a deterministic Chroma collection name from embedding model."""
    normalized = re.sub(r"[^a-z0-9]+", "_", model.strip().lower()).strip("_")
    suffix = normalized or "default"
    return f"{COLLECTION_NAME}__{suffix}"


@dataclass
class Retriever:
    """Vector retriever with OpenAI embedding-based query encoding."""

    client: OpenAI
    collection: Collection
    embedding_model: str
    collection_name: str

    @classmethod
    def from_paths(
        cls,
        openai_client: OpenAI,
        chroma_dir: str,
        embedding_model: str,
        collection_name: str | None = None,
    ) -> "Retriever":
        """Build a retriever from persistent Chroma path, model, and collection."""
        chroma_client = chromadb.PersistentClient(path=chroma_dir)
        target_collection_name = collection_name or COLLECTION_NAME
        collection = chroma_client.get_or_create_collection(name=target_collection_name)
        return cls(
            client=openai_client,
            collection=collection,
            embedding_model=embedding_model,
            collection_name=target_collection_name,
        )

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Embedding input text cannot be empty.")

        emb = call_with_retry(
            lambda: self.client.embeddings.create(model=self.embedding_model, input=text)
        )
        if not getattr(emb, "data", None):
            raise ValueError("Embedding response contained no data.")

        embedding = emb.data[0].embedding
        if not embedding:
            raise ValueError("Embedding response contained an empty vector.")

        return embedding

    def search(self, query: str, top_k: int) -> list[dict[str, str]]:
        """Return top-k semantic matches from the vector store."""
        if not query.strip():
            raise ValueError("Search query cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0.")

        query_embedding = self.embed_text(query)
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]

        hits: list[dict[str, str]] = []
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append(
                {
                    "title": str(meta.get("title", "Unknown")),
                    "themes": str(meta.get("themes", "")),
                    "summary": str(doc),
                    "distance": f"{dist:.4f}",
                }
            )
        return hits
