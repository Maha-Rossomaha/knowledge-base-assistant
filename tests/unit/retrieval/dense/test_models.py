from dataclasses import FrozenInstanceError

import pytest

from knowledge_base_assistant.retrieval.dense.embedding import (
    EmbeddingModelConfig,
)
from knowledge_base_assistant.retrieval.dense.models import (
    DenseIndexMetadata,
)


def test_dense_index_metadata_can_be_created() -> None:
    embedding_model = EmbeddingModelConfig(
        provider="sentence-transformers",
        model_name="intfloat/multilingual-e5-base",
        dimension=768,
        query_prefix="query: ",
        document_prefix="passage: ",
    )

    metadata = DenseIndexMetadata(
        schema_version=2,
        embedding_model=embedding_model,
        normalized=True,
        chunks_sha256="chunks-hash-1",
        chunk_ids=(
            "chunk-1",
            "chunk-2",
        ),
    )

    assert metadata.schema_version == 2
    assert metadata.embedding_model == embedding_model
    assert metadata.embedding_model.provider == (
        "sentence-transformers"
    )
    assert metadata.embedding_model.model_name == (
        "intfloat/multilingual-e5-base"
    )
    assert metadata.embedding_model.dimension == 768
    assert metadata.embedding_model.query_prefix == "query: "
    assert metadata.embedding_model.document_prefix == "passage: "
    assert metadata.normalized is True
    assert metadata.chunks_sha256 == "chunks-hash-1"
    assert metadata.chunk_ids == (
        "chunk-1",
        "chunk-2",
    )


def test_dense_index_metadata_is_immutable() -> None:
    metadata = _make_metadata()

    with pytest.raises(FrozenInstanceError):
        metadata.normalized = False  # type: ignore[misc]


def test_nested_embedding_model_config_is_immutable() -> None:
    metadata = _make_metadata()

    with pytest.raises(FrozenInstanceError):
        metadata.embedding_model.dimension = 384  # type: ignore[misc]


def _make_metadata() -> DenseIndexMetadata:
    return DenseIndexMetadata(
        schema_version=2,
        embedding_model=EmbeddingModelConfig(
            provider="sentence-transformers",
            model_name="intfloat/multilingual-e5-base",
            dimension=768,
            query_prefix="query: ",
            document_prefix="passage: ",
        ),
        normalized=True,
        chunks_sha256="chunks-hash-1",
        chunk_ids=(
            "chunk-1",
            "chunk-2",
        ),
    )