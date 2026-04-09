from ragbot.retriever import collection_name_for_embedding


def test_collection_name_for_embedding_is_deterministic() -> None:
    name = collection_name_for_embedding("text-embedding-3-small")
    assert name == "book_summaries__text_embedding_3_small"


def test_collection_name_for_embedding_normalizes_symbols() -> None:
    name = collection_name_for_embedding("My Model/v2")
    assert name == "book_summaries__my_model_v2"
