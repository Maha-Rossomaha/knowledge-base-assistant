from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingModel(Protocol):
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