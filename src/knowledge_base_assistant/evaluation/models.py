from dataclasses import dataclass
    

@dataclass(frozen=True, slots=True)
class RelevantChunk:
    chunk_id: str
    relative_path: str
    section_path: tuple[str, ...]
    relevance: int


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    query_id: str
    query: str
    relevant_chunks: tuple[RelevantChunk, ...]
    notes: str
    

@dataclass(frozen=True, slots=True) 
class GoldenValidationResult:
    query_count: int
    answerable_query_count: int
    no_answer_query_count: int
    relevance_judgment_count: int