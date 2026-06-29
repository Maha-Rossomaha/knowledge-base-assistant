import hashlib
from pathlib import Path

import numpy as np
import pytest

from knowledge_base_assistant.application.dense_search import (
    load_dense_index,
    search_dense_index,
)
from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.retrieval.dense.embedding import (
    EmbeddingModelConfig,
)
from knowledge_base_assistant.retrieval.dense.index_storage import (
    write_dense_embeddings,
)
from knowledge_base_assistant.retrieval.dense.models import (
    DenseIndexMetadata,
)
from knowledge_base_assistant.retrieval.dense.serialization import (
    write_dense_index_metadata,
)
from knowledge_base_assistant.serialization.jsonl import (
    write_chunks_jsonl,
)


class FakeEmbeddingModel:
    def __init__(
        self,
        query_embedding: np.ndarray | None = None,
        *,
        provider: str = "fake",
        model_name: str = "fake-model",
        dimension: int = 3,
        query_prefix: str = "",
        document_prefix: str = "",
    ) -> None:
        self._query_embedding = (
            query_embedding
            if query_embedding is not None
            else np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            )
        )
        self._config = EmbeddingModelConfig(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        )

    @property
    def config(self) -> EmbeddingModelConfig:
        return self._config

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def dimension(self) -> int:
        return self._config.dimension

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        raise NotImplementedError

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        return self._query_embedding


def test_load_dense_index_loads_valid_index(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"

    chunks = [
        _make_chunk("chunk-1", 0),
        _make_chunk("chunk-2", 1),
    ]
    write_chunks_jsonl(
        chunks,
        chunks_path,
    )

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    write_dense_embeddings(
        embeddings=embeddings,
        path=embeddings_path,
    )

    metadata = _make_metadata(
        chunks_sha256=_calculate_sha256(
            chunks_path,
        ),
        chunk_ids=(
            "chunk-1",
            "chunk-2",
        ),
    )
    write_dense_index_metadata(
        metadata=metadata,
        path=metadata_path,
    )

    index = load_dense_index(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedding_model=FakeEmbeddingModel(),
    )

    results = index.search(
        query="test query",
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk-1"
    assert results[0].rank == 1
    assert results[0].score == pytest.approx(1.0)


def test_load_dense_index_rejects_changed_chunks_file(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"

    chunks = [
        _make_chunk("chunk-1", 0),
    ]
    write_chunks_jsonl(
        chunks,
        chunks_path,
    )

    original_sha256 = _calculate_sha256(
        chunks_path,
    )

    write_dense_embeddings(
        embeddings=np.array(
            [[1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        path=embeddings_path,
    )

    write_dense_index_metadata(
        metadata=_make_metadata(
            chunks_sha256=original_sha256,
            chunk_ids=("chunk-1",),
        ),
        path=metadata_path,
    )

    write_chunks_jsonl(
        [
            _make_chunk("chunk-1", 0),
            _make_chunk("chunk-2", 1),
        ],
        chunks_path,
    )

    with pytest.raises(
        ValueError,
        match=(
            "Dense index metadata does not match "
            "the current chunks file"
        ),
    ):
        load_dense_index(
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            embedding_model=FakeEmbeddingModel(),
        )


@pytest.mark.parametrize(
    (
        "metadata_config",
        "model_config",
    ),
    [
        (
            {
                "provider": "sentence-transformers",
            },
            {
                "provider": "fake",
            },
        ),
        (
            {
                "model_name": "index-model",
            },
            {
                "model_name": "query-model",
            },
        ),
        (
            {
                "query_prefix": "query: ",
            },
            {
                "query_prefix": "",
            },
        ),
        (
            {
                "document_prefix": "passage: ",
            },
            {
                "document_prefix": "",
            },
        ),
    ],
)
def test_load_dense_index_rejects_mismatched_embedding_configuration(
    tmp_path: Path,
    metadata_config: dict[str, object],
    model_config: dict[str, object],
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"

    chunks = [
        _make_chunk("chunk-1", 0),
    ]
    write_chunks_jsonl(
        chunks,
        chunks_path,
    )

    write_dense_embeddings(
        embeddings=np.array(
            [[1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        path=embeddings_path,
    )

    metadata = _make_metadata(
        chunks_sha256=_calculate_sha256(
            chunks_path,
        ),
        chunk_ids=("chunk-1",),
        provider=str(
            metadata_config.get(
                "provider",
                "fake",
            )
        ),
        model_name=str(
            metadata_config.get(
                "model_name",
                "fake-model",
            )
        ),
        query_prefix=str(
            metadata_config.get(
                "query_prefix",
                "",
            )
        ),
        document_prefix=str(
            metadata_config.get(
                "document_prefix",
                "",
            )
        ),
    )
    write_dense_index_metadata(
        metadata=metadata,
        path=metadata_path,
    )

    embedding_model = FakeEmbeddingModel(
        provider=str(
            model_config.get(
                "provider",
                "fake",
            )
        ),
        model_name=str(
            model_config.get(
                "model_name",
                "fake-model",
            )
        ),
        query_prefix=str(
            model_config.get(
                "query_prefix",
                "",
            )
        ),
        document_prefix=str(
            model_config.get(
                "document_prefix",
                "",
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Embedding model configuration must match "
            "metadata embedding model configuration"
        ),
    ):
        load_dense_index(
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            embedding_model=embedding_model,
        )


def test_search_dense_index_returns_ranked_results(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"

    chunks = [
        _make_chunk("chunk-1", 0),
        _make_chunk("chunk-2", 1),
        _make_chunk("chunk-3", 2),
    ]
    write_chunks_jsonl(
        chunks,
        chunks_path,
    )

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    write_dense_embeddings(
        embeddings=embeddings,
        path=embeddings_path,
    )

    metadata = _make_metadata(
        chunks_sha256=_calculate_sha256(
            chunks_path,
        ),
        chunk_ids=(
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ),
    )
    write_dense_index_metadata(
        metadata=metadata,
        path=metadata_path,
    )

    results = search_dense_index(
        query="test query",
        top_k=2,
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedding_model=FakeEmbeddingModel(
            query_embedding=np.array(
                [2.0, 0.0, 0.0],
                dtype=np.float32,
            ),
        ),
    )

    assert [
        result.chunk.chunk_id
        for result in results
    ] == [
        "chunk-1",
        "chunk-2",
    ]
    assert [
        result.rank
        for result in results
    ] == [1, 2]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.8)


def _make_metadata(
    *,
    chunks_sha256: str,
    chunk_ids: tuple[str, ...],
    provider: str = "fake",
    model_name: str = "fake-model",
    dimension: int = 3,
    query_prefix: str = "",
    document_prefix: str = "",
) -> DenseIndexMetadata:
    return DenseIndexMetadata(
        schema_version=2,
        embedding_model=EmbeddingModelConfig(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        ),
        normalized=True,
        chunks_sha256=chunks_sha256,
        chunk_ids=chunk_ids,
    )


def _calculate_sha256(
    path: Path,
) -> str:
    return hashlib.sha256(
        path.read_bytes(),
    ).hexdigest()


def _make_chunk(
    chunk_id: str,
    chunk_index: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="document-1",
        source_name="test-source",
        relative_path="notes/test.md",
        title="Test",
        section_path=("Test",),
        content=f"Content {chunk_index}",
        searchable_text=f"Searchable text {chunk_index}",
        chunk_index=chunk_index,
        section_chunk_index=chunk_index,
        start_line=1,
        end_line=2,
        content_hash=f"content-hash-{chunk_index}",
    )