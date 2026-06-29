from pathlib import Path

from knowledge_base_assistant.application.retrieval_evaluation import (
    evaluate_queries,
    evaluate_retrieval,
)
from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
    RetrievalEvaluationResult,
)
from knowledge_base_assistant.retrieval.lexical.bm25 import BM25Index
from knowledge_base_assistant.serialization.jsonl import (
    read_chunks_jsonl,
    read_golden_queries_jsonl,
)


def evaluate_bm25_retrieval(
    *,
    golden_path: Path,
    chunks_path: Path,
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> RetrievalEvaluationResult:
    queries = read_golden_queries_jsonl(golden_path)
    chunks = read_chunks_jsonl(chunks_path)

    index = BM25Index.build(
        chunks,
        k1=k1,
        b=b,
    )

    return evaluate_retrieval(
        queries=queries,
        chunks=chunks,
        retriever=index,
        top_k=top_k,
    )
    

def evaluate_bm25_queries(
    *,
    golden_path: Path,
    chunks_path: Path,
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> tuple[QueryEvaluationResult, ...]:
    queries = read_golden_queries_jsonl(golden_path)
    chunks = read_chunks_jsonl(chunks_path)

    index = BM25Index.build(
        chunks,
        k1=k1,
        b=b,
    )

    return evaluate_queries(
        queries=queries,
        chunks=chunks,
        retriever=index,
        top_k=top_k,
    )