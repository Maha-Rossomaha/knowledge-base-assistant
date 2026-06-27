import numpy as np

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.retrieval.dense.embedding import EmbeddingModel
from knowledge_base_assistant.retrieval.dense.models import DenseIndexMetadata


def build_dense_index_data(
    chunks: list[Chunk],
    embedding_model: EmbeddingModel,
    chunks_sha256: str,
) -> tuple[np.ndarray, DenseIndexMetadata]:
    if not embedding_model.model_name:
        raise ValueError("Embedding model name must not be empty")

    if embedding_model.dimension < 1:
        raise ValueError("Embedding dimension must be at least 1")

    if not chunks_sha256:
        raise ValueError("Chunks hash must not be empty")
    
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)

    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError("Chunk IDs must be unique")

    texts = [
        chunk.searchable_text
        for chunk in chunks
    ]

    embeddings = embedding_model.embed_documents(texts)

    if not isinstance(embeddings, np.ndarray):
        raise TypeError("Embedding model must return a NumPy array")

    if embeddings.ndim != 2:
        raise ValueError(
            "Dense embeddings must be a two-dimensional matrix"
        )

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Dense embeddings row count does not match chunks count"
        )

    if embeddings.shape[1] != embedding_model.dimension:
        raise ValueError(
            "Dense embeddings dimension does not match embedding model"
        )

    if not np.issubdtype(embeddings.dtype, np.number):
        raise ValueError("Dense embeddings must contain numeric values")

    embeddings = embeddings.astype(
        np.float32,
        copy=False,
    )
    
    if not np.all(np.isfinite(embeddings)):
        raise ValueError(
            "Dense embeddings must contain only finite values"
        )
    
    embeddings = normalize_embeddings(embeddings)

    metadata = DenseIndexMetadata(
        schema_version=1,
        model_name=embedding_model.model_name,
        dimension=embedding_model.dimension,
        normalized=True,
        chunks_sha256=chunks_sha256,
        chunk_ids=chunk_ids,
    )

    return embeddings, metadata


def normalize_embeddings(
    embeddings: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True,
    )
    
    if np.any(norms == 0):
        raise ValueError("Dense embeddings must not contain zero vectors")
    
    normalized = embeddings / norms
    
    return normalized.astype( # type: ignore[no-any-return]
        np.float32,
        copy=False,
    )