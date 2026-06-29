import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from knowledge_base_assistant.retrieval.dense.embedding import (
    EmbeddingModelConfig,
)
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
    embedding_model_data = data["embedding_model"]
    normalized = data["normalized"]
    chunks_sha256 = data["chunks_sha256"]

    if type(schema_version) is not int:
        raise TypeError("schema_version must be an integer")
    
    if not isinstance(embedding_model_data, dict):
        raise TypeError("embedding_model must be an object")

    if not isinstance(normalized, bool):
        raise TypeError("normalized must be a boolean")

    if not isinstance(chunks_sha256, str):
        raise TypeError("chunks_sha256 must be a string")

    if schema_version != 2:
        raise ValueError(
            f"Unsupported schema_version: {schema_version}"
        )

    if not chunks_sha256:
        raise ValueError("chunks_sha256 must not be empty")
    
    provider = embedding_model_data["provider"]
    model_name = embedding_model_data["model_name"]
    dimension = embedding_model_data["dimension"]
    query_prefix = embedding_model_data["query_prefix"]
    document_prefix = embedding_model_data["document_prefix"]

    if not isinstance(provider, str):
        raise TypeError("embedding_model.provider must be a string")

    if not isinstance(model_name, str):
        raise TypeError("embedding_model.model_name must be a string")

    if type(dimension) is not int:
        raise TypeError("embedding_model.dimension must be an integer")

    if not isinstance(query_prefix, str):
        raise TypeError("embedding_model.query_prefix must be a string")

    if not isinstance(document_prefix, str):
        raise TypeError(
            "embedding_model.document_prefix must be a string"
        )

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
        chunks_sha256=chunks_sha256,
        chunk_ids=tuple(chunk_ids),
    )