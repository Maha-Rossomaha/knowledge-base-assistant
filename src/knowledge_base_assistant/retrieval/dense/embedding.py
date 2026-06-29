from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True, slots=True)
class EmbeddingModelConfig:
    provider: str
    model_name: str
    dimension: int
    query_prefix: str
    document_prefix: str

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError(
                "Embedding model provider must not be empty"
            )

        if not self.model_name.strip():
            raise ValueError(
                "Embedding model name must not be empty"
            )

        if self.dimension < 1:
            raise ValueError(
                "Embedding model dimension must be at least 1"
            )


@runtime_checkable
class EmbeddingModel(Protocol):
    @property
    def config(self) -> EmbeddingModelConfig:
        ...

    @property
    def model_name(self) -> str:
        ...

    @property
    def dimension(self) -> int:
        ...

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        ...

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        ...