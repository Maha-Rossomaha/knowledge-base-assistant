import numpy as np
import pytest

from knowledge_base_assistant.retrieval.dense.index_storage import (
    read_dense_embeddings,
    write_dense_embeddings,
)
from knowledge_base_assistant.retrieval.dense.models import DenseIndexMetadata


def test_write_dense_embeddings_creates_parent_directories(
    tmp_path,
) -> None:
    path = tmp_path / "nested" / "dense" / "embeddings.npy"
    embeddings = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )

    write_dense_embeddings(embeddings, path)

    assert path.is_file()


def test_dense_embeddings_round_trip(
    tmp_path,
) -> None:
    path = tmp_path / "embeddings.npy"
    embeddings = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )
    metadata = _make_metadata(
        dimension=3,
        chunk_ids=("chunk-1", "chunk-2"),
    )

    write_dense_embeddings(embeddings, path)
    loaded_embeddings = read_dense_embeddings(path, metadata)

    np.testing.assert_array_equal(
        loaded_embeddings,
        embeddings,
    )


def test_write_dense_embeddings_converts_float64_to_float32(
    tmp_path,
) -> None:
    path = tmp_path / "embeddings.npy"
    embeddings = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    write_dense_embeddings(embeddings, path)

    loaded_embeddings = np.load(
        path,
        allow_pickle=False,
    )

    assert loaded_embeddings.dtype == np.float32


def test_write_dense_embeddings_preserves_float32(
    tmp_path,
) -> None:
    path = tmp_path / "embeddings.npy"
    embeddings = np.array(
        [
            [1.0, 2.0],
        ],
        dtype=np.float32,
    )

    write_dense_embeddings(embeddings, path)

    loaded_embeddings = np.load(
        path,
        allow_pickle=False,
    )

    assert loaded_embeddings.dtype == np.float32
    np.testing.assert_array_equal(
        loaded_embeddings,
        embeddings,
    )


@pytest.mark.parametrize(
    "embeddings",
    [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array(1.0, dtype=np.float32),
        np.zeros((1, 2, 3), dtype=np.float32),
    ],
)
def test_write_dense_embeddings_rejects_non_two_dimensional_array(
    tmp_path,
    embeddings,
) -> None:
    path = tmp_path / "embeddings.npy"

    with pytest.raises(
        ValueError,
        match="Dense embeddings must be a two-dimensional matrix",
    ):
        write_dense_embeddings(embeddings, path)


@pytest.mark.parametrize(
    "embeddings",
    [
        np.array([1.0, 2.0], dtype=np.float32),
        np.array(1.0, dtype=np.float32),
        np.zeros((1, 2, 3), dtype=np.float32),
    ],
)
def test_read_dense_embeddings_rejects_non_two_dimensional_array(
    tmp_path,
    embeddings,
) -> None:
    path = tmp_path / "embeddings.npy"
    np.save(path, embeddings)

    metadata = _make_metadata(
        dimension=2,
        chunk_ids=("chunk-1",),
    )

    with pytest.raises(
        ValueError,
        match="Dense embeddings must be a two-dimensional matrix",
    ):
        read_dense_embeddings(path, metadata)


def test_read_dense_embeddings_rejects_wrong_row_count(
    tmp_path,
) -> None:
    path = tmp_path / "embeddings.npy"
    embeddings = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    np.save(path, embeddings)

    metadata = _make_metadata(
        dimension=2,
        chunk_ids=("chunk-1",),
    )

    with pytest.raises(
        ValueError,
        match="Dense embeddings row count does not match chunk IDs count",
    ):
        read_dense_embeddings(path, metadata)


def test_read_dense_embeddings_rejects_wrong_dimension(
    tmp_path,
) -> None:
    path = tmp_path / "embeddings.npy"
    embeddings = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dtype=np.float32,
    )
    np.save(path, embeddings)

    metadata = _make_metadata(
        dimension=3,
        chunk_ids=("chunk-1", "chunk-2"),
    )

    with pytest.raises(
        ValueError,
        match="Dense embeddings dimension does not match index metadata",
    ):
        read_dense_embeddings(path, metadata)


def test_read_dense_embeddings_returns_numpy_array(
    tmp_path,
) -> None:
    path = tmp_path / "embeddings.npy"
    embeddings = np.array(
        [
            [1.0, 2.0],
        ],
        dtype=np.float32,
    )
    np.save(path, embeddings)

    metadata = _make_metadata(
        dimension=2,
        chunk_ids=("chunk-1",),
    )

    loaded_embeddings = read_dense_embeddings(
        path,
        metadata,
    )

    assert isinstance(loaded_embeddings, np.ndarray)


def test_read_dense_embeddings_accepts_empty_matrix(
    tmp_path,
) -> None:
    path = tmp_path / "embeddings.npy"
    embeddings = np.empty(
        (0, 3),
        dtype=np.float32,
    )
    np.save(path, embeddings)

    metadata = _make_metadata(
        dimension=3,
        chunk_ids=(),
    )

    loaded_embeddings = read_dense_embeddings(
        path,
        metadata,
    )

    assert loaded_embeddings.shape == (0, 3)


def _make_metadata(
    *,
    dimension: int,
    chunk_ids: tuple[str, ...],
) -> DenseIndexMetadata:
    return DenseIndexMetadata(
        schema_version=1,
        model_name="test-model",
        dimension=dimension,
        normalized=True,
        chunks_hash="chunks-hash",
        chunk_ids=chunk_ids,
    )