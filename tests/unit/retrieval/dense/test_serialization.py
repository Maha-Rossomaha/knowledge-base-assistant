import json

import pytest

from knowledge_base_assistant.retrieval.dense.models import DenseIndexMetadata
from knowledge_base_assistant.retrieval.dense.serialization import (
    read_dense_index_metadata,
    write_dense_index_metadata,
)


def test_dense_index_metadata_round_trip(tmp_path) -> None:
    metadata = DenseIndexMetadata(
        schema_version=1,
        model_name="intfloat/multilingual-e5-base",
        dimension=768,
        normalized=True,
        chunks_sha256="chunks-hash-1",
        chunk_ids=("chunk-1", "chunk-2"),
    )
    path = tmp_path / "dense" / "index_metadata.json"

    write_dense_index_metadata(metadata, path)
    loaded_metadata = read_dense_index_metadata(path)

    assert loaded_metadata == metadata


def test_write_dense_index_metadata_creates_parent_directories(
    tmp_path,
) -> None:
    metadata = DenseIndexMetadata(
        schema_version=1,
        model_name="model",
        dimension=3,
        normalized=True,
        chunks_sha256="hash",
        chunk_ids=("chunk-1",),
    )
    path = tmp_path / "nested" / "dense" / "index_metadata.json"

    write_dense_index_metadata(metadata, path)

    assert path.is_file()


def test_write_dense_index_metadata_writes_expected_json(
    tmp_path,
) -> None:
    metadata = DenseIndexMetadata(
        schema_version=1,
        model_name="многоязычная-модель",
        dimension=3,
        normalized=False,
        chunks_sha256="hash",
        chunk_ids=("chunk-1", "chunk-2"),
    )
    path = tmp_path / "index_metadata.json"

    write_dense_index_metadata(metadata, path)

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data == {
        "schema_version": 1,
        "model_name": "многоязычная-модель",
        "dimension": 3,
        "normalized": False,
        "chunks_sha256": "hash",
        "chunk_ids": ["chunk-1", "chunk-2"],
    }


def test_read_dense_index_metadata_rejects_invalid_json(
    tmp_path,
) -> None:
    path = tmp_path / "index_metadata.json"
    path.write_text("{invalid", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Invalid dense index metadata JSON",
    ):
        read_dense_index_metadata(path)


@pytest.mark.parametrize(
    "json_value",
    [
        [],
        "metadata",
        42,
        None,
    ],
)
def test_read_dense_index_metadata_rejects_non_object_root(
    tmp_path,
    json_value,
) -> None:
    path = tmp_path / "index_metadata.json"
    path.write_text(
        json.dumps(json_value),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="JSON value must be an object",
    ):
        read_dense_index_metadata(path)


def test_read_dense_index_metadata_rejects_missing_field(
    tmp_path,
) -> None:
    path = tmp_path / "index_metadata.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_name": "model",
                "dimension": 3,
                "normalized": True,
                "chunks_sha256": "hash",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid dense index metadata",
    ):
        read_dense_index_metadata(path)


@pytest.mark.parametrize(
    "chunk_ids",
    [
        "chunk-1",
        {"chunk-1": True},
        42,
        None,
    ],
)
def test_read_dense_index_metadata_rejects_non_list_chunk_ids(
    tmp_path,
    chunk_ids,
) -> None:
    path = tmp_path / "index_metadata.json"
    _write_metadata_json(
        path,
        chunk_ids=chunk_ids,
    )

    with pytest.raises(
        ValueError,
        match="chunk_ids must be a list",
    ):
        read_dense_index_metadata(path)


@pytest.mark.parametrize(
    "chunk_ids",
    [
        [1],
        ["chunk-1", 2],
        [None],
        [True],
    ],
)
def test_read_dense_index_metadata_rejects_non_string_chunk_ids(
    tmp_path,
    chunk_ids,
) -> None:
    path = tmp_path / "index_metadata.json"
    _write_metadata_json(
        path,
        chunk_ids=chunk_ids,
    )

    with pytest.raises(
        ValueError,
        match="chunk_ids must contain only strings",
    ):
        read_dense_index_metadata(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "schema_version",
            "1",
            "schema_version must be an integer",
        ),
        (
            "schema_version",
            True,
            "schema_version must be an integer",
        ),
        (
            "model_name",
            123,
            "model_name must be a string",
        ),
        (
            "dimension",
            "768",
            "dimension must be an integer",
        ),
        (
            "dimension",
            False,
            "dimension must be an integer",
        ),
        (
            "normalized",
            1,
            "normalized must be a boolean",
        ),
        (
            "chunks_sha256",
            123,
            "chunks_sha256 must be a string",
        ),
    ],
)
def test_read_dense_index_metadata_rejects_invalid_field_types(
    tmp_path,
    field,
    value,
    message,
) -> None:
    path = tmp_path / "index_metadata.json"
    _write_metadata_json(
        path,
        **{field: value},
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        read_dense_index_metadata(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "schema_version",
            0,
            "schema_version must be at least 1",
        ),
        (
            "schema_version",
            -1,
            "schema_version must be at least 1",
        ),
        (
            "dimension",
            0,
            "dimension must be at least 1",
        ),
        (
            "dimension",
            -1,
            "dimension must be at least 1",
        ),
        (
            "model_name",
            "",
            "model_name must not be empty",
        ),
        (
            "chunks_sha256",
            "",
            "chunks_sha256 must not be empty",
        ),
    ],
)
def test_read_dense_index_metadata_rejects_invalid_field_values(
    tmp_path,
    field,
    value,
    message,
) -> None:
    path = tmp_path / "index_metadata.json"
    _write_metadata_json(
        path,
        **{field: value},
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        read_dense_index_metadata(path)


def test_read_dense_index_metadata_accepts_empty_chunk_ids(
    tmp_path,
) -> None:
    path = tmp_path / "index_metadata.json"
    _write_metadata_json(
        path,
        chunk_ids=[],
    )

    metadata = read_dense_index_metadata(path)

    assert metadata.chunk_ids == ()


def _write_metadata_json(
    path,
    **overrides,
) -> None:
    data = {
        "schema_version": 1,
        "model_name": "model",
        "dimension": 3,
        "normalized": True,
        "chunks_sha256": "hash",
        "chunk_ids": ["chunk-1", "chunk-2"],
    }
    data.update(overrides)

    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )