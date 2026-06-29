import numpy as np

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.retrieval.dense.embedding import EmbeddingModel
from knowledge_base_assistant.retrieval.dense.models import DenseIndexMetadata
from knowledge_base_assistant.retrieval.models import SearchResult


def __init__(
    self,
    *,
    chunks: list[Chunk],
    embeddings: np.ndarray,
    metadata: DenseIndexMetadata,
    embedding_model: EmbeddingModel,
) -> None:
    if metadata.schema_version != 2:
        raise ValueError(
            f"Unsupported dense index schema version: "
            f"{metadata.schema_version}"
        )

    if embeddings.ndim != 2:
        raise ValueError(
            "Dense embeddings must be a two-dimensional array"
        )

    if embeddings.shape[0] != len(chunks):
        raise ValueError(
            "Number of embedding rows must equal number of chunks"
        )

    if (
        embeddings.shape[1]
        != metadata.embedding_model.dimension
    ):
        raise ValueError(
            "Embedding dimension must match metadata dimension"
        )

    if not metadata.normalized:
        raise ValueError(
            "Dense index embeddings must be normalized"
        )

    chunk_ids = tuple(
        chunk.chunk_id
        for chunk in chunks
    )

    if chunk_ids != metadata.chunk_ids:
        raise ValueError(
            "Chunk IDs must match metadata chunk IDs "
            "in the same order"
        )

    if embedding_model.config != metadata.embedding_model:
        raise ValueError(
            "Embedding model configuration must match "
            "metadata embedding model configuration"
        )

    self._chunks = chunks
    self._embeddings = embeddings
    self._metadata = metadata
    self._embedding_model = embedding_model
        
        
    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[SearchResult]:
        if not query.strip():
            raise ValueError("Query must not be empty")
        
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        
        query_embedding = self._embedding_model.embed_query(query=query)
        
        if query_embedding.ndim != 1:
            raise ValueError("Query embedding must be a one-dimensional array")
        
        if query_embedding.shape[0] != self._metadata.embedding_model.dimension:
            raise ValueError("Query embedding dimension must match index dimension")
        
        query_embedding = query_embedding.astype(
            np.float32,
            copy=False,
        )
        
        if not np.all(np.isfinite(query_embedding)):
            raise ValueError("Query embedding must contain only finite values")

        norm = np.linalg.norm(query_embedding)

        if norm == 0:
            raise ValueError(
                "Query embedding must not be a zero vector"
            )

        normalized_query = query_embedding / norm
        scores = self._embeddings @ normalized_query

        result_count = min(top_k, len(self._chunks))
        best_indices = np.argsort(scores)[::-1][:result_count]

        return [
            SearchResult(
                chunk=self._chunks[index],
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(best_indices, start=1)
        ]
            
            
    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)