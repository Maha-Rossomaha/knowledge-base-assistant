import numpy as np
import pytest

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.retrieval.dense.index_builder import (
    build_dense_index_data,
    normalize_embeddings,
)


class FakeEmbeddingModel:
    def __init__(
        self,
        embeddings: object,
        *,
        model_name: str = "fake-model",
        dimension: int = 3,
    ) -> None:
        self._embeddings = embeddings
        self._model_name = model_name
        self._dimension = dimension
        self.received_texts: list[str] | None = None

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
        self.received_texts = texts
        return self._embeddings  # type: ignore[return-value]

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        raise NotImplementedError


def test_build_dense_index_data_uses_searchable_texts_in_order() -> None:
    chunks = [
        _make_chunk(
            chunk_id="chunk-1",
            searchable_text="First searchable text",
            chunk_index=0,
        ),
        _make_chunk(
            chunk_id="chunk-2",
            searchable_text="Second searchable text",
            chunk_index=1,
        ),
    ]
    model = FakeEmbeddingModel(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
    )

    build_dense_index_data(
        chunks,
        model,
        chunks_hash="chunks-hash",
    )

    assert model.received_texts == [
        "First searchable text",
        "Second searchable text",
    ]


def test_build_dense_index_data_returns_normalized_float32_embeddings() -> None:
    chunks = [
        _make_chunk(
            chunk_id="chunk-1",
            searchable_text="First",
            chunk_index=0,
        ),
        _make_chunk(
            chunk_id="chunk-2",
            searchable_text="Second",
            chunk_index=1,
        ),
    ]
    model = FakeEmbeddingModel(
        np.array(
            [
                [3.0, 4.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float64,
        )
    )

    embeddings, _ = build_dense_index_data(
        chunks,
        model,
        chunks_hash="chunks-hash",
    )

    assert embeddings.dtype == np.float32

    np.testing.assert_allclose(
        np.linalg.norm(embeddings, axis=1),
        np.ones(2),
        rtol=1e-6,
        atol=1e-6,
    )

    np.testing.assert_allclose(
        embeddings,
        np.array(
            [
                [0.6, 0.8, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_build_dense_index_data_creates_expected_metadata() -> None:
    chunks = [
        _make_chunk(
            chunk_id="chunk-1",
            searchable_text="First",
            chunk_index=0,
        ),
        _make_chunk(
            chunk_id="chunk-2",
            searchable_text="Second",
            chunk_index=1,
        ),
    ]
    model = FakeEmbeddingModel(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        model_name="test-embedding-model",
        dimension=3,
    )

    _, metadata = build_dense_index_data(
        chunks,
        model,
        chunks_hash="chunks-hash",
    )

    assert metadata.schema_version == 1
    assert metadata.model_name == "test-embedding-model"
    assert metadata.dimension == 3
    assert metadata.normalized is True
    assert metadata.chunks_hash == "chunks-hash"
    assert metadata.chunk_ids == ("chunk-1", "chunk-2")


def test_build_dense_index_data_rejects_empty_model_name() -> None:
    model = FakeEmbeddingModel(
        np.empty((0, 3), dtype=np.float32),
        model_name="",
    )

    with pytest.raises(
        ValueError,
        match="Embedding model name must not be empty",
    ):
        build_dense_index_data(
            [],
            model,
            chunks_hash="chunks-hash",
        )


@pytest.mark.parametrize("dimension", [0, -1])
def test_build_dense_index_data_rejects_invalid_model_dimension(
    dimension: int,
) -> None:
    model = FakeEmbeddingModel(
        np.empty((0, 0), dtype=np.float32),
        dimension=dimension,
    )

    with pytest.raises(
        ValueError,
        match="Embedding dimension must be at least 1",
    ):
        build_dense_index_data(
            [],
            model,
            chunks_hash="chunks-hash",
        )


def test_build_dense_index_data_rejects_empty_chunks_hash() -> None:
    model = FakeEmbeddingModel(
        np.empty((0, 3), dtype=np.float32)
    )

    with pytest.raises(
        ValueError,
        match="Chunks hash must not be empty",
    ):
        build_dense_index_data(
            [],
            model,
            chunks_hash="",
        )


def test_build_dense_index_data_rejects_duplicate_chunk_ids() -> None:
    chunks = [
        _make_chunk(
            chunk_id="duplicate",
            searchable_text="First",
            chunk_index=0,
        ),
        _make_chunk(
            chunk_id="duplicate",
            searchable_text="Second",
            chunk_index=1,
        ),
    ]
    model = FakeEmbeddingModel(
        np.ones((2, 3), dtype=np.float32)
    )

    with pytest.raises(
        ValueError,
        match="Chunk IDs must be unique",
    ):
        build_dense_index_data(
            chunks,
            model,
            chunks_hash="chunks-hash",
        )


def test_build_dense_index_data_rejects_non_numpy_result() -> None:
    chunks = [
        _make_chunk(
            chunk_id="chunk-1",
            searchable_text="Text",
            chunk_index=0,
        )
    ]
    model = FakeEmbeddingModel(
        [[1.0, 2.0, 3.0]]
    )

    with pytest.raises(
        TypeError,
        match="Embedding model must return a NumPy array",
    ):
        build_dense_index_data(
            chunks,
            model,
            chunks_hash="chunks-hash",
        )


@pytest.mark.parametrize(
    "embeddings",
    [
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array(1.0, dtype=np.float32),
        np.ones((1, 1, 3), dtype=np.float32),
    ],
)
def test_build_dense_index_data_rejects_non_two_dimensional_result(
    embeddings: np.ndarray,
) -> None:
    chunks = [
        _make_chunk(
            chunk_id="chunk-1",
            searchable_text="Text",
            chunk_index=0,
        )
    ]
    model = FakeEmbeddingModel(embeddings)

    with pytest.raises(
        ValueError,
        match="Dense embeddings must be a two-dimensional matrix",
    ):
        build_dense_index_data(
            chunks,
            model,
            chunks_hash="chunks-hash",
        )


def test_build_dense_index_data_rejects_wrong_row_count() -> None:
    chunks = [
        _make_chunk(
            chunk_id="chunk-1",
            searchable_text="First",
            chunk_index=0,
        ),
        _make_chunk(
            chunk_id="chunk-2",
            searchable_text="Second",
            chunk_index=1,
        ),
    ]
    model = FakeEmbeddingModel(
        np.ones((1, 3), dtype=np.float32)
    )

    with pytest.raises(
        ValueError,
        match="Dense embeddings row count does not match chunks count",
    ):
        build_dense_index_data(
            chunks,
            model,
            chunks_hash="chunks-hash",
        )


def test_build_dense_index_data_rejects_wrong_dimension() -> None:
    chunks = [
        _make_chunk(
            chunk_id="chunk-1",
            searchable_text="Text",
            chunk_index=0,
        )
    ]
    model = FakeEmbeddingModel(
        np.ones((1, 2), dtype=np.float32),
        dimension=3,
    )

    with pytest.raises(
        ValueError,
        match="Dense embeddings dimension does not match embedding model",
    ):
        build_dense_index_data(
            chunks,
            model,
            chunks_hash="chunks-hash",
        )


def test_build_dense_index_data_rejects_non_numeric_embeddings() -> None:
    chunks = [
        _make_chunk(
            chunk_id="chunk-1",
            searchable_text="Text",
            chunk_index=0,
        )
    ]
    model = FakeEmbeddingModel(
        np.array(
            [["a", "b", "c"]],
            dtype=np.str_,
        )
    )

    with pytest.raises(
        ValueError,
        match="Dense embeddings must contain numeric values",
    ):
        build_dense_index_data(
            chunks,
            model,
            chunks_hash="chunks-hash",
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_build_dense_index_data_rejects_non_finite_values(
    invalid_value: float,
) -> None:
    chunks = [
        _make_chunk(
            chunk_id="chunk-1",
            searchable_text="Text",
            chunk_index=0,
        )
    ]
    model = FakeEmbeddingModel(
        np.array(
            [[invalid_value, 1.0, 2.0]],
            dtype=np.float32,
        )
    )

    with pytest.raises(
        ValueError,
        match="Dense embeddings must contain only finite values",
    ):
        build_dense_index_data(
            chunks,
            model,
            chunks_hash="chunks-hash",
        )


def test_normalize_embeddings_normalizes_each_row_independently() -> None:
    embeddings = np.array(
        [
            [3.0, 4.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )

    normalized = normalize_embeddings(embeddings)

    np.testing.assert_allclose(
        normalized,
        np.array(
            [
                [0.6, 0.8],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_normalize_embeddings_rejects_zero_vector() -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(
        ValueError,
        match="Dense embeddings must not contain zero vectors",
    ):
        normalize_embeddings(embeddings)


def _make_chunk(
    *,
    chunk_id: str,
    searchable_text: str,
    chunk_index: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="document-1",
        source_name="test-source",
        relative_path="notes/test.md",
        title="Test",
        section_path=("Test",),
        content="Original chunk content",
        searchable_text=searchable_text,
        chunk_index=chunk_index,
        section_chunk_index=chunk_index,
        start_line=1,
        end_line=2,
        content_hash=f"content-hash-{chunk_index}",
    )