from pathlib import Path

from knowledge_base_assistant.application.dense_indexing import calculate_file_sha256
from knowledge_base_assistant.retrieval.dense.embedding import EmbeddingModel
from knowledge_base_assistant.retrieval.dense.index import DenseIndex
from knowledge_base_assistant.retrieval.dense.index_storage import read_dense_embeddings
from knowledge_base_assistant.retrieval.dense.serialization import read_dense_index_metadata
from knowledge_base_assistant.retrieval.models import SearchResult
from knowledge_base_assistant.serialization.jsonl import read_chunks_jsonl


def load_dense_index(
    *,
    chunks_path: Path,
    embeddings_path: Path,
    metadata_path: Path,
    embedding_model: EmbeddingModel,
) -> DenseIndex:
    metadata = read_dense_index_metadata(metadata_path)
        
    if metadata.chunks_sha256 != calculate_file_sha256(chunks_path):
        raise ValueError("Dense index metadata does not match the current chunks file")
    
    chunks = read_chunks_jsonl(chunks_path)
    
    embeddings = read_dense_embeddings(path=embeddings_path, metadata=metadata)
    
    return DenseIndex(
        chunks=chunks,
        embeddings=embeddings,
        metadata=metadata,
        embedding_model=embedding_model,
    )
    
    
def search_dense_index(
    *,
    query: str,
    top_k: int,
    chunks_path: Path,
    embeddings_path: Path,
    metadata_path: Path,
    embedding_model: EmbeddingModel,
) -> list[SearchResult]:
    index = load_dense_index(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model
    )
    
    return index.search(query=query, top_k=top_k)
    