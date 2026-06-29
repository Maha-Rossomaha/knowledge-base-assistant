import numpy as np
import pytest

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.retrieval.dense.embedding import (
    EmbeddingModelConfig,
)
from knowledge_base_assistant.retrieval.dense.index import DenseIndex
from knowledge_base_assistant.retrieval.dense.models import (
    DenseIndexMetadata,
)

_CONFIG_MISMATCH_MESSAGE = (
    "Embedding model configuration must match "
    "metadata embedding model configuration"
)


class FakeEmbeddingModel:
    def __init__(
        self,
        *,
        query_embedding: np.ndarray | None = None,
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
        self.received_queries: list[str] = []

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
        self.received_queries.append(query)
        return self._query_embedding


def test_dense_index_search_returns_results_in_score_order() -> None:
    chunks = [
        _make_chunk("chunk-1", 0),
        _make_chunk("chunk-2", 1),
        _make_chunk("chunk-3", 2),
    ]

    embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    model = FakeEmbeddingModel(
        query_embedding=np.array(
            [2.0, 0.0, 0.0],
            dtype=np.float32,
        ),
    )

    index = DenseIndex(
        chunks=chunks,
        embeddings=embeddings,
        metadata=_make_metadata(),
        embedding_model=model,
    )

    results = index.search(
        query="test query",
        top_k=2,
    )

    assert model.received_queries == ["test query"]

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


def test_dense_index_search_returns_all_chunks_when_top_k_is_larger() -> None:
    index = _make_index()

    results = index.search(
        query="test query",
        top_k=100,
    )

    assert len(results) == 3
    assert [
        result.rank
        for result in results
    ] == [1, 2, 3]


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
    ],
)
def test_dense_index_search_rejects_empty_query(
    query: str,
) -> None:
    index = _make_index()

    with pytest.raises(
        ValueError,
        match="Query must not be empty",
    ):
        index.search(
            query=query,
            top_k=1,
        )


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_dense_index_search_rejects_invalid_top_k(
    top_k: int,
) -> None:
    index = _make_index()

    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        index.search(
            query="test query",
            top_k=top_k,
        )


def test_dense_index_search_rejects_non_vector_query_embedding() -> None:
    model = FakeEmbeddingModel(
        query_embedding=np.array(
            [[1.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
    )

    index = _make_index(
        embedding_model=model,
    )

    with pytest.raises(
        ValueError,
        match="Query embedding must be a one-dimensional array",
    ):
        index.search(
            query="test query",
            top_k=1,
        )


def test_dense_index_search_rejects_wrong_query_dimension() -> None:
    model = FakeEmbeddingModel(
        query_embedding=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )

    index = _make_index(
        embedding_model=model,
    )

    with pytest.raises(
        ValueError,
        match="Query embedding dimension must match index dimension",
    ):
        index.search(
            query="test query",
            top_k=1,
        )


def test_dense_index_search_rejects_zero_query_vector() -> None:
    model = FakeEmbeddingModel(
        query_embedding=np.zeros(
            3,
            dtype=np.float32,
        ),
    )

    index = _make_index(
        embedding_model=model,
    )

    with pytest.raises(
        ValueError,
        match="Query embedding must not be a zero vector",
    ):
        index.search(
            query="test query",
            top_k=1,
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_dense_index_search_rejects_non_finite_query_embedding(
    invalid_value: float,
) -> None:
    model = FakeEmbeddingModel(
        query_embedding=np.array(
            [invalid_value, 0.0, 0.0],
            dtype=np.float32,
        ),
    )

    index = _make_index(
        embedding_model=model,
    )

    with pytest.raises(
        ValueError,
        match="Query embedding must contain only finite values",
    ):
        index.search(
            query="test query",
            top_k=1,
        )


def test_dense_index_rejects_unsupported_schema_version() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported dense index schema version: 1",
    ):
        DenseIndex(
            chunks=[
                _make_chunk("chunk-1", 0),
            ],
            embeddings=np.array(
                [[1.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            metadata=_make_metadata(
                schema_version=1,
                chunk_ids=("chunk-1",),
            ),
            embedding_model=FakeEmbeddingModel(),
        )


def test_dense_index_rejects_non_matrix_embeddings() -> None:
    with pytest.raises(
        ValueError,
        match="Dense embeddings must be a two-dimensional array",
    ):
        DenseIndex(
            chunks=[
                _make_chunk("chunk-1", 0),
            ],
            embeddings=np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            metadata=_make_metadata(
                chunk_ids=("chunk-1",),
            ),
            embedding_model=FakeEmbeddingModel(),
        )


def test_dense_index_rejects_wrong_embedding_row_count() -> None:
    with pytest.raises(
        ValueError,
        match="Number of embedding rows must equal number of chunks",
    ):
        DenseIndex(
            chunks=[
                _make_chunk("chunk-1", 0),
                _make_chunk("chunk-2", 1),
            ],
            embeddings=np.array(
                [[1.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            metadata=_make_metadata(
                chunk_ids=(
                    "chunk-1",
                    "chunk-2",
                ),
            ),
            embedding_model=FakeEmbeddingModel(),
        )


def test_dense_index_rejects_wrong_embedding_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="Embedding dimension must match metadata dimension",
    ):
        DenseIndex(
            chunks=[
                _make_chunk("chunk-1", 0),
            ],
            embeddings=np.array(
                [[1.0, 0.0]],
                dtype=np.float32,
            ),
            metadata=_make_metadata(
                dimension=3,
                chunk_ids=("chunk-1",),
            ),
            embedding_model=FakeEmbeddingModel(),
        )


def test_dense_index_rejects_not_normalized_metadata() -> None:
    with pytest.raises(
        ValueError,
        match="Dense index embeddings must be normalized",
    ):
        DenseIndex(
            chunks=[
                _make_chunk("chunk-1", 0),
            ],
            embeddings=np.array(
                [[1.0, 0.0, 0.0]],
                dtype=np.float32,
            ),
            metadata=_make_metadata(
                normalized=False,
                chunk_ids=("chunk-1",),
            ),
            embedding_model=FakeEmbeddingModel(),
        )


def test_dense_index_rejects_wrong_chunk_id_order() -> None:
    with pytest.raises(
        ValueError,
        match=(
            "Chunk IDs must match metadata chunk IDs "
            "in the same order"
        ),
    ):
        DenseIndex(
            chunks=[
                _make_chunk("chunk-1", 0),
                _make_chunk("chunk-2", 1),
            ],
            embeddings=np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            metadata=_make_metadata(
                chunk_ids=(
                    "chunk-2",
                    "chunk-1",
                ),
            ),
            embedding_model=FakeEmbeddingModel(),
        )


@pytest.mark.parametrize(
    (
        "metadata_overrides",
        "model_overrides",
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
                "dimension": 3,
            },
            {
                "dimension": 2,
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
def test_dense_index_rejects_mismatched_embedding_model_configuration(
    metadata_overrides: dict[str, object],
    model_overrides: dict[str, object],
) -> None:
    metadata = _make_metadata(
        provider=str(
            metadata_overrides.get(
                "provider",
                "fake",
            )
        ),
        model_name=str(
            metadata_overrides.get(
                "model_name",
                "fake-model",
            )
        ),
        dimension=int(
            metadata_overrides.get(
                "dimension",
                3,
            )
        ),
        query_prefix=str(
            metadata_overrides.get(
                "query_prefix",
                "",
            )
        ),
        document_prefix=str(
            metadata_overrides.get(
                "document_prefix",
                "",
            )
        ),
    )

    model = FakeEmbeddingModel(
        provider=str(
            model_overrides.get(
                "provider",
                "fake",
            )
        ),
        model_name=str(
            model_overrides.get(
                "model_name",
                "fake-model",
            )
        ),
        dimension=int(
            model_overrides.get(
                "dimension",
                3,
            )
        ),
        query_prefix=str(
            model_overrides.get(
                "query_prefix",
                "",
            )
        ),
        document_prefix=str(
            model_overrides.get(
                "document_prefix",
                "",
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match=_CONFIG_MISMATCH_MESSAGE,
    ):
        DenseIndex(
            chunks=_make_chunks(),
            embeddings=_make_embeddings(),
            metadata=metadata,
            embedding_model=model,
        )


def test_dense_index_accepts_matching_embedding_model_configuration() -> None:
    metadata = _make_metadata(
        provider="sentence-transformers",
        model_name="intfloat/multilingual-e5-base",
        dimension=3,
        query_prefix="query: ",
        document_prefix="passage: ",
    )

    model = FakeEmbeddingModel(
        provider="sentence-transformers",
        model_name="intfloat/multilingual-e5-base",
        dimension=3,
        query_prefix="query: ",
        document_prefix="passage: ",
    )

    index = DenseIndex(
        chunks=_make_chunks(),
        embeddings=_make_embeddings(),
        metadata=metadata,
        embedding_model=model,
    )

    assert index.chunks == tuple(_make_chunks())


def test_dense_index_chunks_returns_immutable_sequence() -> None:
    index = _make_index()

    first_result = index.chunks
    second_result = index.chunks

    assert first_result == second_result
    assert isinstance(first_result, tuple)


def _make_index(
    *,
    embedding_model: FakeEmbeddingModel | None = None,
) -> DenseIndex:
    return DenseIndex(
        chunks=_make_chunks(),
        embeddings=_make_embeddings(),
        metadata=_make_metadata(),
        embedding_model=(
            embedding_model
            if embedding_model is not None
            else FakeEmbeddingModel()
        ),
    )


def _make_metadata(
    *,
    schema_version: int = 2,
    provider: str = "fake",
    model_name: str = "fake-model",
    dimension: int = 3,
    query_prefix: str = "",
    document_prefix: str = "",
    normalized: bool = True,
    chunk_ids: tuple[str, ...] = (
        "chunk-1",
        "chunk-2",
        "chunk-3",
    ),
) -> DenseIndexMetadata:
    return DenseIndexMetadata(
        schema_version=schema_version,
        embedding_model=EmbeddingModelConfig(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
            query_prefix=query_prefix,
            document_prefix=document_prefix,
        ),
        normalized=normalized,
        chunks_sha256="test-sha256",
        chunk_ids=chunk_ids,
    )


def _make_chunks() -> list[Chunk]:
    return [
        _make_chunk("chunk-1", 0),
        _make_chunk("chunk-2", 1),
        _make_chunk("chunk-3", 2),
    ]


def _make_embeddings() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


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