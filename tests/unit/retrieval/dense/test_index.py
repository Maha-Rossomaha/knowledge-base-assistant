import numpy as np
import pytest

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.retrieval.dense.index import DenseIndex
from knowledge_base_assistant.retrieval.dense.models import DenseIndexMetadata


class FakeEmbeddingModel:
    def __init__(
        self,
        *,
        query_embedding: np.ndarray | None = None,
        model_name: str = "fake-model",
        dimension: int = 3,
    ) -> None:
        self._query_embedding = (
            query_embedding
            if query_embedding is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        self._model_name = model_name
        self._dimension = dimension
        self.received_queries: list[str] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

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

    assert [result.chunk.chunk_id for result in results] == [
        "chunk-1",
        "chunk-2",
    ]
    assert [result.rank for result in results] == [1, 2]

    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.8)


def test_dense_index_search_returns_all_chunks_when_top_k_is_larger() -> None:
    index = _make_index()

    results = index.search(
        query="test query",
        top_k=100,
    )

    assert len(results) == 3
    assert [result.rank for result in results] == [1, 2, 3]


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


def test_dense_index_rejects_non_matrix_embeddings() -> None:
    with pytest.raises(
        ValueError,
        match="Dense embeddings must be a two-dimensional array",
    ):
        DenseIndex(
            chunks=[_make_chunk("chunk-1", 0)],
            embeddings=np.array(
                [1.0, 0.0, 0.0],
                dtype=np.float32,
            ),
            metadata=DenseIndexMetadata(
                schema_version=1,
                model_name="fake-model",
                dimension=3,
                normalized=True,
                chunks_sha256="test-sha256",
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
            metadata=DenseIndexMetadata(
                schema_version=1,
                model_name="fake-model",
                dimension=3,
                normalized=True,
                chunks_sha256="test-sha256",
                chunk_ids=("chunk-1", "chunk-2"),
            ),
            embedding_model=FakeEmbeddingModel(),
        )


def test_dense_index_rejects_wrong_embedding_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="Embedding dimension must match metadata dimension",
    ):
        DenseIndex(
            chunks=[_make_chunk("chunk-1", 0)],
            embeddings=np.array(
                [[1.0, 0.0]],
                dtype=np.float32,
            ),
            metadata=DenseIndexMetadata(
                schema_version=1,
                model_name="fake-model",
                dimension=3,
                normalized=True,
                chunks_sha256="test-sha256",
                chunk_ids=("chunk-1",),
            ),
            embedding_model=FakeEmbeddingModel(),
        )


def test_dense_index_rejects_wrong_chunk_id_order() -> None:
    with pytest.raises(
        ValueError,
        match="Chunk IDs must match metadata chunk IDs in the same order",
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
            metadata=DenseIndexMetadata(
                schema_version=1,
                model_name="fake-model",
                dimension=3,
                normalized=True,
                chunks_sha256="test-sha256",
                chunk_ids=("chunk-2", "chunk-1"),
            ),
            embedding_model=FakeEmbeddingModel(),
        )


def test_dense_index_rejects_wrong_model_name() -> None:
    with pytest.raises(
        ValueError,
        match="Embedding model name must match metadata model name",
    ):
        _make_index(
            embedding_model=FakeEmbeddingModel(
                model_name="another-model",
            ),
        )


def test_dense_index_rejects_wrong_model_dimension() -> None:
    with pytest.raises(
        ValueError,
        match="Embedding model dimension must match metadata dimension",
    ):
        _make_index(
            embedding_model=FakeEmbeddingModel(
                dimension=2,
            ),
        )


def _make_index(
    *,
    embedding_model: FakeEmbeddingModel | None = None,
) -> DenseIndex:
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

    return DenseIndex(
        chunks=chunks,
        embeddings=embeddings,
        metadata=_make_metadata(),
        embedding_model=embedding_model or FakeEmbeddingModel(),
    )


def _make_metadata() -> DenseIndexMetadata:
    return DenseIndexMetadata(
        schema_version=1,
        model_name="fake-model",
        dimension=3,
        normalized=True,
        chunks_sha256="test-sha256",
        chunk_ids=(
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ),
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
    
    
def test_dense_index_rejects_not_normalized_metadata() -> None:
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

    metadata = DenseIndexMetadata(
        schema_version=1,
        model_name="fake-model",
        dimension=3,
        normalized=False,
        chunks_sha256="test-sha256",
        chunk_ids=(
            "chunk-1",
            "chunk-2",
            "chunk-3",
        ),
    )

    with pytest.raises(
        ValueError,
        match="Dense index embeddings must be normalized",
    ):
        DenseIndex(
            chunks=chunks,
            embeddings=embeddings,
            metadata=metadata,
            embedding_model=FakeEmbeddingModel(),
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
        
        
def test_dense_index_chunks_returns_immutable_sequence() -> None:
    index = _make_index()

    first_result = index.chunks
    second_result = index.chunks

    assert first_result == second_result
    assert isinstance(first_result, tuple)