import hashlib
from dataclasses import dataclass
from pathlib import Path

from knowledge_base_assistant.retrieval.dense.embedding import EmbeddingModel
from knowledge_base_assistant.retrieval.dense.index_builder import build_dense_index_data
from knowledge_base_assistant.retrieval.dense.index_storage import write_dense_embeddings
from knowledge_base_assistant.retrieval.dense.serialization import write_dense_index_metadata
from knowledge_base_assistant.serialization.jsonl import read_chunks_jsonl

_BLOCK_SIZE = 64 * 1024


@dataclass(frozen=True, slots=True)
class DenseIndexingResult:
    chunk_count: int
    dimension: int
    embeddings_path: Path
    metadata_path: Path
    

def calculate_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(_BLOCK_SIZE):
            digest.update(chunk)

    return digest.hexdigest()


def build_dense_index(
    *,
    chunks_path: Path,
    embeddings_path: Path,
    metadata_path: Path,
    embedding_model: EmbeddingModel,
) -> DenseIndexingResult:
    chunks = read_chunks_jsonl(chunks_path)
    chunks_sha256 = calculate_file_sha256(chunks_path)
    embeddings, metadata = build_dense_index_data(
        chunks=chunks,
        chunks_sha256=chunks_sha256,
        embedding_model=embedding_model,
    )
    write_dense_embeddings(embeddings=embeddings, path=embeddings_path)
    write_dense_index_metadata(metadata=metadata, path=metadata_path)
    
    return DenseIndexingResult(
        chunk_count=len(chunks),
        dimension=metadata.embedding_model.dimension,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
    )