from pathlib import Path

import numpy as np

from knowledge_base_assistant.retrieval.dense.models import DenseIndexMetadata


def write_dense_embeddings(
    embeddings: np.ndarray,
    path: Path,
) -> None:
    if embeddings.ndim != 2:
        raise ValueError("Dense embeddings must be a two-dimensional matrix")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    
    embeddings = embeddings.astype(np.float32, copy=False)
    
    np.save(path, embeddings)
    
    
def read_dense_embeddings(
    path: Path,
    metadata: DenseIndexMetadata,
) -> np.ndarray:
    embeddings = np.load(path, allow_pickle=False)
    
    if embeddings.ndim != 2:
        raise ValueError("Dense embeddings must be a two-dimensional matrix")
    
    if embeddings.shape[0] != len(metadata.chunk_ids):
        raise ValueError("Dense embeddings row count does not match chunk IDs count")
    
    if embeddings.shape[1] != metadata.embedding_model.dimension:
        raise ValueError("Dense embeddings dimension does not match index metadata")
    
    return embeddings  # type: ignore[no-any-return]