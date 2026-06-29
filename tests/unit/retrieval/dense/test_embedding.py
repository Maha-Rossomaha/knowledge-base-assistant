import pytest

from knowledge_base_assistant.retrieval.dense.embedding import (
    EmbeddingModelConfig,
)


def test_embedding_model_config_stores_values() -> None:
    config = EmbeddingModelConfig(
        provider="sentence-transformers",
        model_name="test-model",
        dimension=768,
        query_prefix="query: ",
        document_prefix="passage: ",
    )

    assert config.provider == "sentence-transformers"
    assert config.model_name == "test-model"
    assert config.dimension == 768
    assert config.query_prefix == "query: "
    assert config.document_prefix == "passage: "


@pytest.mark.parametrize(
    ("provider", "model_name", "dimension", "message"),
    [
        (
            "",
            "test-model",
            768,
            "Embedding model provider must not be empty",
        ),
        (
            "   ",
            "test-model",
            768,
            "Embedding model provider must not be empty",
        ),
        (
            "sentence-transformers",
            "",
            768,
            "Embedding model name must not be empty",
        ),
        (
            "sentence-transformers",
            "   ",
            768,
            "Embedding model name must not be empty",
        ),
        (
            "sentence-transformers",
            "test-model",
            0,
            "Embedding model dimension must be at least 1",
        ),
        (
            "sentence-transformers",
            "test-model",
            -1,
            "Embedding model dimension must be at least 1",
        ),
    ],
)
def test_embedding_model_config_rejects_invalid_values(
    provider: str,
    model_name: str,
    dimension: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EmbeddingModelConfig(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
            query_prefix="",
            document_prefix="",
        )


def test_embedding_model_config_allows_empty_prefixes() -> None:
    config = EmbeddingModelConfig(
        provider="sentence-transformers",
        model_name="test-model",
        dimension=384,
        query_prefix="",
        document_prefix="",
    )

    assert config.query_prefix == ""
    assert config.document_prefix == ""