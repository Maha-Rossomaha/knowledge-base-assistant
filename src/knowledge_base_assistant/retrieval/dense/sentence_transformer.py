import numpy as np
from sentence_transformers import SentenceTransformer

from knowledge_base_assistant.retrieval.dense.embedding import EmbeddingModel


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(
        self, 
        model_name: str, 
        batch_size: int,
        device: str = "cpu",
    ):
        if not model_name.strip():
            raise ValueError("Sentence Transformer model name must not be empty")
        
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        
        model = SentenceTransformer(model_name, device=device)
        dimension = model.get_embedding_dimension()
        
        if dimension is None or dimension < 1:
            raise ValueError("Embedding model dimension must be at least 1")
                
        self._model_name = model_name
        self._batch_size = batch_size 
        self._model = model
        self._dimension = dimension
        

    @property
    def model_name(self) -> str:
        return self._model_name


    @property
    def dimension(self) -> int:
        return self._dimension
    
    
    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        embeddings = self._model.encode_document(
            texts,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return embeddings # type: ignore[no-any-return]
        

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        embeddings = self._model.encode_query(
            query,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return embeddings # type: ignore[no-any-return]