from pathlib import Path

from knowledge_base_assistant.application.dense_search import load_dense_index
from knowledge_base_assistant.evaluation.metrics import (
    calculate_query_metrics,
)
from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
    RetrievalEvaluationResult,
    RetrievedChunkEvaluation,
)
from knowledge_base_assistant.evaluation.validation import (
    validate_golden_queries,
)
from knowledge_base_assistant.retrieval.dense.embedding import EmbeddingModel
from knowledge_base_assistant.serialization.jsonl import read_golden_queries_jsonl


def evaluate_dense_retrieval(
    *,
    golden_path: Path,
    chunks_path: Path,
    embeddings_path: Path,
    metadata_path: Path,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> RetrievalEvaluationResult:
    if top_k < 1:
        raise ValueError(
            f"top_k must be at least 1, got {top_k}"
        )

    queries = read_golden_queries_jsonl(golden_path)

    index = load_dense_index(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
    )

    validation_result = validate_golden_queries(
        queries,
        index.chunks,
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
            query=query.query,
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
    
    

def evaluate_dense_queries(
    *,
    golden_path: Path,
    chunks_path: Path,
    embeddings_path: Path,
    metadata_path: Path,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> tuple[QueryEvaluationResult, ...]:
    if top_k < 1:
        raise ValueError(
            f"top_k must be at least 1, got {top_k}"
        )

    queries = read_golden_queries_jsonl(golden_path)

    index = load_dense_index(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
    )

    validate_golden_queries(
        queries,
        index.chunks,
    )

    evaluation_results: list[QueryEvaluationResult] = []

    for query in queries:
        if not query.relevant_chunks:
            continue

        relevance_by_chunk_id = {
            relevant_chunk.chunk_id: relevant_chunk.relevance
            for relevant_chunk in query.relevant_chunks
        }

        search_results = index.search(
            query=query.query,
            top_k=top_k,
        )

        retrieved_chunk_ids = [
            result.chunk.chunk_id
            for result in search_results
        ]

        metrics = calculate_query_metrics(
            relevance_by_chunk_id,
            retrieved_chunk_ids,
            k=top_k,
        )

        retrieved_chunks = tuple(
            RetrievedChunkEvaluation(
                chunk_id=result.chunk.chunk_id,
                rank=result.rank,
                score=result.score,
                relevance=relevance_by_chunk_id.get(
                    result.chunk.chunk_id,
                    0,
                ),
                relative_path=result.chunk.relative_path,
                section_path=result.chunk.section_path,
                content=result.chunk.content,
            )
            for result in search_results
        )

        first_relevant_rank = next(
            (
                result.rank
                for result in search_results
                if result.chunk.chunk_id
                in relevance_by_chunk_id
            ),
            None,
        )

        evaluation_results.append(
            QueryEvaluationResult(
                query_id=query.query_id,
                query=query.query,
                relevant_chunk_ids=tuple(
                    relevance_by_chunk_id
                ),
                retrieved_chunks=retrieved_chunks,
                first_relevant_rank=first_relevant_rank,
                hit_rate_at_k=metrics.hit_rate_at_k,
                recall_at_k=metrics.recall_at_k,
                reciprocal_rank=metrics.reciprocal_rank,
                ndcg_at_k=metrics.ndcg_at_k,
            )
        )

    return tuple(evaluation_results)