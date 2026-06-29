from dataclasses import dataclass

from knowledge_base_assistant.retrieval.dense.embedding import (
    EmbeddingModelConfig,
)


@dataclass(frozen=True, slots=True)
class DenseIndexMetadata:
    schema_version: int
    embedding_model: EmbeddingModelConfig
    normalized: bool
    chunks_sha256: str
    chunk_ids: tuple[str, ...]