from types import SimpleNamespace

import pytest

from ragbot.retriever import collection_name_for_embedding
from ragbot.retriever import Retriever


def test_collection_name_for_embedding_is_deterministic() -> None:
    name = collection_name_for_embedding("text-embedding-3-small")
    assert name == "book_summaries__text_embedding_3_small"


def test_collection_name_for_embedding_normalizes_symbols() -> None:
    name = collection_name_for_embedding("My Model/v2")
    assert name == "book_summaries__my_model_v2"


def test_search_rejects_empty_query() -> None:
    retriever = Retriever(
        client=SimpleNamespace(),
        collection=SimpleNamespace(),
        embedding_model="text-embedding-3-small",
        collection_name="book_summaries__text_embedding_3_small",
    )

    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.search("   ", top_k=4)


def test_search_rejects_non_positive_top_k() -> None:
    retriever = Retriever(
        client=SimpleNamespace(),
        collection=SimpleNamespace(),
        embedding_model="text-embedding-3-small",
        collection_name="book_summaries__text_embedding_3_small",
    )

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        retriever.search("war novels", top_k=0)
