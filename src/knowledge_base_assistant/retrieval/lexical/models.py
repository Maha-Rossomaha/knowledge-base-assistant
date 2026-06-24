from dataclasses import dataclass

from knowledge_base_assistant.domain.models import Chunk


@dataclass(frozen=True, slots=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int