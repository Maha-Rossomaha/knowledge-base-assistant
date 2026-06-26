from dataclasses import FrozenInstanceError

import pytest

from knowledge_base_assistant.retrieval.dense.models import DenseIndexMetadata


def test_dense_index_metadata_can_be_created() -> None:
    metadata = DenseIndexMetadata(
        schema_version=1,
        model_name="intfloat/multilingual-e5-base",
        dimension=768,
        normalized=True,
        chunks_hash="chunks-hash-1",
        chunk_ids=("chunk-1", "chunk-2"),
    )

    assert metadata.schema_version == 1
    assert metadata.model_name == "intfloat/multilingual-e5-base"
    assert metadata.dimension == 768
    assert metadata.normalized is True
    assert metadata.chunks_hash == "chunks-hash-1"
    assert metadata.chunk_ids == ("chunk-1", "chunk-2")


def test_dense_index_metadata_is_immutable() -> None:
    metadata = DenseIndexMetadata(
        schema_version=1,
        model_name="intfloat/multilingual-e5-base",
        dimension=768,
        normalized=True,
        chunks_hash="chunks-hash-1",
        chunk_ids=("chunk-1", "chunk-2"),
    )

    with pytest.raises(FrozenInstanceError):
        metadata.dimension = 384  # type: ignore[misc]