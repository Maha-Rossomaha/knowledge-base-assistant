from typing import Protocol

from knowledge_base_assistant.retrieval.models import SearchResult


class Retriever(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[SearchResult]:
        ...