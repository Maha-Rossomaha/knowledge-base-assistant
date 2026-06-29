import json
from pathlib import Path
from typing import Any

import pytest

from knowledge_base_assistant.retrieval.dense.embedding import (
    EmbeddingModelConfig,
)
from knowledge_base_assistant.retrieval.dense.models import (
    DenseIndexMetadata,
)
from knowledge_base_assistant.retrieval.dense.serialization import (
    read_dense_index_metadata,
    write_dense_index_metadata,
)


def test_dense_index_metadata_round_trip(
    tmp_path: Path,
) -> None:
    metadata = DenseIndexMetadata(
        schema_version=2,
        embedding_model=_make_embedding_model_config(
            model_name="intfloat/multilingual-e5-base",
            dimension=768,
        ),
        normalized=True,
        chunks_sha256="chunks-hash-1",
        chunk_ids=("chunk-1", "chunk-2"),
    )
    path = tmp_path / "dense" / "index_metadata.json"

    write_dense_index_metadata(metadata, path)
    loaded_metadata = read_dense_index_metadata(path)

    assert loaded_metadata == metadata


def test_write_dense_index_metadata_creates_parent_directories(
    tmp_path: Path,
) -> None:
    metadata = DenseIndexMetadata(
        schema_version=2,
        embedding_model=_make_embedding_model_config(),
        normalized=True,
        chunks_sha256="hash",
        chunk_ids=("chunk-1",),
    )
    path = (
        tmp_path
        / "nested"
        / "dense"
        / "index_metadata.json"
    )

    write_dense_index_metadata(metadata, path)

    assert path.is_file()


def test_write_dense_index_metadata_writes_expected_json(
    tmp_path: Path,
) -> None:
    metadata = DenseIndexMetadata(
        schema_version=2,
        embedding_model=_make_embedding_model_config(
            model_name="многоязычная-модель",
        ),
        normalized=False,
        chunks_sha256="hash",
        chunk_ids=("chunk-1", "chunk-2"),
    )
    path = tmp_path / "index_metadata.json"

    write_dense_index_metadata(metadata, path)

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    assert data == {
        "schema_version": 2,
        "embedding_model": {
            "provider": "sentence-transformers",
            "model_name": "многоязычная-модель",
            "dimension": 3,
            "query_prefix": "query: ",
            "document_prefix": "passage: ",
        },
        "normalized": False,
        "chunks_sha256": "hash",
        "chunk_ids": ["chunk-1", "chunk-2"],
    }


def test_read_dense_index_metadata_rejects_invalid_json(
    tmp_path: Path,
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
    tmp_path: Path,
    json_value: object,
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
    tmp_path: Path,
) -> None:
    path = tmp_path / "index_metadata.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "embedding_model": {
                    "provider": "sentence-transformers",
                    "model_name": "model",
                    "dimension": 3,
                    "query_prefix": "",
                    "document_prefix": "",
                },
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
    "embedding_model",
    [
        [],
        "model",
        42,
        None,
    ],
)
def test_read_dense_index_metadata_rejects_non_object_embedding_model(
    tmp_path: Path,
    embedding_model: object,
) -> None:
    path = tmp_path / "index_metadata.json"

    _write_metadata_json(
        path,
        embedding_model=embedding_model,
    )

    with pytest.raises(
        ValueError,
        match="embedding_model must be an object",
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
    tmp_path: Path,
    chunk_ids: object,
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
    tmp_path: Path,
    chunk_ids: list[object],
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
            "2",
            "schema_version must be an integer",
        ),
        (
            "schema_version",
            True,
            "schema_version must be an integer",
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
def test_read_dense_index_metadata_rejects_invalid_top_level_field_types(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
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
            "provider",
            123,
            "embedding_model.provider must be a string",
        ),
        (
            "model_name",
            123,
            "embedding_model.model_name must be a string",
        ),
        (
            "dimension",
            "768",
            "embedding_model.dimension must be an integer",
        ),
        (
            "dimension",
            False,
            "embedding_model.dimension must be an integer",
        ),
        (
            "query_prefix",
            123,
            "embedding_model.query_prefix must be a string",
        ),
        (
            "document_prefix",
            123,
            "embedding_model.document_prefix must be a string",
        ),
    ],
)
def test_read_dense_index_metadata_rejects_invalid_embedding_model_field_types(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = tmp_path / "index_metadata.json"

    _write_metadata_json(
        path,
        embedding_model_overrides={
            field: value,
        },
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        read_dense_index_metadata(path)


@pytest.mark.parametrize(
    "schema_version",
    [
        0,
        1,
        3,
        -1,
    ],
)
def test_read_dense_index_metadata_rejects_unsupported_schema_version(
    tmp_path: Path,
    schema_version: int,
) -> None:
    path = tmp_path / "index_metadata.json"

    _write_metadata_json(
        path,
        schema_version=schema_version,
    )

    with pytest.raises(
        ValueError,
        match=f"Unsupported schema_version: {schema_version}",
    ):
        read_dense_index_metadata(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "provider",
            "",
            "Embedding model provider must not be empty",
        ),
        (
            "provider",
            "   ",
            "Embedding model provider must not be empty",
        ),
        (
            "model_name",
            "",
            "Embedding model name must not be empty",
        ),
        (
            "model_name",
            "   ",
            "Embedding model name must not be empty",
        ),
        (
            "dimension",
            0,
            "Embedding model dimension must be at least 1",
        ),
        (
            "dimension",
            -1,
            "Embedding model dimension must be at least 1",
        ),
    ],
)
def test_read_dense_index_metadata_rejects_invalid_embedding_model_values(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    path = tmp_path / "index_metadata.json"

    _write_metadata_json(
        path,
        embedding_model_overrides={
            field: value,
        },
    )

    with pytest.raises(
        ValueError,
        match=message,
    ):
        read_dense_index_metadata(path)


def test_read_dense_index_metadata_rejects_empty_chunks_sha256(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index_metadata.json"

    _write_metadata_json(
        path,
        chunks_sha256="",
    )

    with pytest.raises(
        ValueError,
        match="chunks_sha256 must not be empty",
    ):
        read_dense_index_metadata(path)


def test_read_dense_index_metadata_allows_empty_prefixes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index_metadata.json"

    _write_metadata_json(
        path,
        embedding_model_overrides={
            "query_prefix": "",
            "document_prefix": "",
        },
    )

    metadata = read_dense_index_metadata(path)

    assert metadata.embedding_model.query_prefix == ""
    assert metadata.embedding_model.document_prefix == ""


def test_read_dense_index_metadata_accepts_empty_chunk_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "index_metadata.json"

    _write_metadata_json(
        path,
        chunk_ids=[],
    )

    metadata = read_dense_index_metadata(path)

    assert metadata.chunk_ids == ()


def _make_embedding_model_config(
    *,
    provider: str = "sentence-transformers",
    model_name: str = "model",
    dimension: int = 3,
    query_prefix: str = "query: ",
    document_prefix: str = "passage: ",
) -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        provider=provider,
        model_name=model_name,
        dimension=dimension,
        query_prefix=query_prefix,
        document_prefix=document_prefix,
    )


def _write_metadata_json(
    path: Path,
    *,
    embedding_model_overrides: dict[str, object] | None = None,
    **overrides: object,
) -> None:
    embedding_model: dict[str, object] = {
        "provider": "sentence-transformers",
        "model_name": "model",
        "dimension": 3,
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
    }

    if embedding_model_overrides is not None:
        embedding_model.update(
            embedding_model_overrides
        )

    data: dict[str, Any] = {
        "schema_version": 2,
        "embedding_model": embedding_model,
        "normalized": True,
        "chunks_sha256": "hash",
        "chunk_ids": ["chunk-1", "chunk-2"],
    }
    data.update(overrides)

    path.write_text(
        json.dumps(data),
        encoding="utf-8",
    )