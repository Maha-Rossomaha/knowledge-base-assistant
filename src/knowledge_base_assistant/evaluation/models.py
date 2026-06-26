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
    
    
@dataclass(frozen=True, slots=True)
class QueryMetrics:
    hit_rate_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    

@dataclass(frozen=True, slots=True)
class RetrievalEvaluationResult:
    top_k: int
    query_count: int
    evaluated_query_count: int
    no_answer_query_count: int
    hit_rate_at_k: float
    recall_at_k: float
    mean_reciprocal_rank: float
    ndcg_at_k: float
    

@dataclass(frozen=True, slots=True)
class RetrievedChunkEvaluation:
    chunk_id: str
    rank: int
    score: float
    relevance: int
    relative_path: str
    section_path: tuple[str, ...]
    content: str    


@dataclass(frozen=True, slots=True)
class QueryEvaluationResult:
    query_id: str
    query: str
    relevant_chunk_ids: tuple[str, ...]
    retrieved_chunks: tuple[RetrievedChunkEvaluation, ...]
    first_relevant_rank: int | None
    hit_rate_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float