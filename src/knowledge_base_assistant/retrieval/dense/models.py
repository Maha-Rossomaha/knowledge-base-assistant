from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DenseIndexMetadata:
    schema_version: int
    model_name: str
    dimension: int
    normalized: bool
    chunks_hash: str
    chunk_ids: tuple[str, ...]