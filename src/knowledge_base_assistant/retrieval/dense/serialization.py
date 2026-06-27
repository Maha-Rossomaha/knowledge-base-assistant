import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from knowledge_base_assistant.retrieval.dense.models import DenseIndexMetadata


def write_dense_index_metadata(
    metadata: DenseIndexMetadata,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            asdict(metadata),
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")


def read_dense_index_metadata(
    path: Path,
) -> DenseIndexMetadata:
    with path.open("r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Invalid dense index metadata JSON"
            ) from error

    try:
        return _dense_index_metadata_from_dict(data)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid dense index metadata: {error}"
        ) from error


def _dense_index_metadata_from_dict(
    data: Any,
) -> DenseIndexMetadata:
    if not isinstance(data, dict):
        raise TypeError("JSON value must be an object")

    chunk_ids = data["chunk_ids"]

    if not isinstance(chunk_ids, list):
        raise TypeError("chunk_ids must be a list")

    if not all(isinstance(chunk_id, str) for chunk_id in chunk_ids):
        raise TypeError("chunk_ids must contain only strings")

    schema_version = data["schema_version"]
    model_name = data["model_name"]
    dimension = data["dimension"]
    normalized = data["normalized"]
    chunks_sha256 = data["chunks_sha256"]

    if type(schema_version) is not int:
        raise TypeError("schema_version must be an integer")

    if not isinstance(model_name, str):
        raise TypeError("model_name must be a string")

    if type(dimension) is not int:
        raise TypeError("dimension must be an integer")

    if not isinstance(normalized, bool):
        raise TypeError("normalized must be a boolean")

    if not isinstance(chunks_sha256, str):
        raise TypeError("chunks_sha256 must be a string")

    if schema_version < 1:
        raise ValueError("schema_version must be at least 1")

    if dimension < 1:
        raise ValueError("dimension must be at least 1")

    if not model_name:
        raise ValueError("model_name must not be empty")

    if not chunks_sha256:
        raise ValueError("chunks_sha256 must not be empty")

    return DenseIndexMetadata(
        schema_version=schema_version,
        model_name=model_name,
        dimension=dimension,
        normalized=normalized,
        chunks_sha256=chunks_sha256,
        chunk_ids=tuple(chunk_ids),
    )