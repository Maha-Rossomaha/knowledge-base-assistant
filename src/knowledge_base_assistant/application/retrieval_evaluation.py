from pathlib import Path

from knowledge_base_assistant.evaluation.metrics import (
    calculate_query_metrics,
)
from knowledge_base_assistant.evaluation.models import (
    RetrievalEvaluationResult,
)
from knowledge_base_assistant.evaluation.validation import (
    validate_golden_queries,
)
from knowledge_base_assistant.retrieval.bm25 import BM25Index
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
    if top_k < 1:
        raise ValueError(
            f"top_k must be at least 1, got {top_k}"
        )

    queries = read_golden_queries_jsonl(golden_path)
    chunks = read_chunks_jsonl(chunks_path)

    validation_result = validate_golden_queries(
        queries,
        chunks,
    )

    index = BM25Index.build(
        chunks,
        k1=k1,
        b=b,
    )

    evaluated_query_count = 0

    total_hit_rate = 0.0
    total_recall = 0.0
    total_reciprocal_rank = 0.0
    total_ndcg = 0.0

    for query in queries:
        if not query.relevant_chunks:
            continue

        relevance_by_chunk_id = {
            relevant_chunk.chunk_id: relevant_chunk.relevance
            for relevant_chunk in query.relevant_chunks
        }

        search_results = index.search(
            query.query,
            top_k=top_k,
        )

        retrieved_chunk_ids = [
            result.chunk.chunk_id
            for result in search_results
        ]

        query_metrics = calculate_query_metrics(
            relevance_by_chunk_id,
            retrieved_chunk_ids,
            k=top_k,
        )

        total_hit_rate += query_metrics.hit_rate_at_k
        total_recall += query_metrics.recall_at_k
        total_reciprocal_rank += (
            query_metrics.reciprocal_rank
        )
        total_ndcg += query_metrics.ndcg_at_k

        evaluated_query_count += 1

    if evaluated_query_count == 0:
        return RetrievalEvaluationResult(
            top_k=top_k,
            query_count=validation_result.query_count,
            evaluated_query_count=0,
            no_answer_query_count=(
                validation_result.no_answer_query_count
            ),
            hit_rate_at_k=0.0,
            recall_at_k=0.0,
            mean_reciprocal_rank=0.0,
            ndcg_at_k=0.0,
        )

    return RetrievalEvaluationResult(
        top_k=top_k,
        query_count=validation_result.query_count,
        evaluated_query_count=evaluated_query_count,
        no_answer_query_count=(
            validation_result.no_answer_query_count
        ),
        hit_rate_at_k=(
            total_hit_rate / evaluated_query_count
        ),
        recall_at_k=(
            total_recall / evaluated_query_count
        ),
        mean_reciprocal_rank=(
            total_reciprocal_rank
            / evaluated_query_count
        ),
        ndcg_at_k=(
            total_ndcg / evaluated_query_count
        ),
    )
