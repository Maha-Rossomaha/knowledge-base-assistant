import math
from collections.abc import Mapping, Sequence

from knowledge_base_assistant.evaluation.models import (
    QueryMetrics,
)


def hit_rate_at_k(
    relevance_by_chunk_id: Mapping[str, int],
    retrieved_chunk_ids: Sequence[str],
    *,
    k: int,
) -> float:
    _validate_k(k)

    relevant_chunk_ids = set(relevance_by_chunk_id)

    return float(
        any(
            chunk_id in relevant_chunk_ids
            for chunk_id in retrieved_chunk_ids[:k]
        )
    )


def recall_at_k(
    relevance_by_chunk_id: Mapping[str, int],
    retrieved_chunk_ids: Sequence[str],
    *,
    k: int,
) -> float:
    _validate_k(k)

    if not relevance_by_chunk_id:
        return 0.0

    relevant_chunk_ids = set(relevance_by_chunk_id)
    retrieved_relevant_chunk_ids = {
        chunk_id
        for chunk_id in retrieved_chunk_ids[:k]
        if chunk_id in relevant_chunk_ids
    }

    return (
        len(retrieved_relevant_chunk_ids)
        / len(relevant_chunk_ids)
    )


def reciprocal_rank(
    relevance_by_chunk_id: Mapping[str, int],
    retrieved_chunk_ids: Sequence[str],
) -> float:
    relevant_chunk_ids = set(relevance_by_chunk_id)

    for rank, chunk_id in enumerate(
        retrieved_chunk_ids,
        start=1,
    ):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    relevance_by_chunk_id: Mapping[str, int],
    retrieved_chunk_ids: Sequence[str],
    *,
    k: int,
) -> float:
    _validate_k(k)

    if not relevance_by_chunk_id:
        return 0.0

    actual_relevances = [
        relevance_by_chunk_id.get(chunk_id, 0)
        for chunk_id in retrieved_chunk_ids[:k]
    ]

    actual_dcg = _discounted_cumulative_gain(
        actual_relevances
    )

    ideal_relevances = sorted(
        relevance_by_chunk_id.values(),
        reverse=True,
    )[:k]

    ideal_dcg = _discounted_cumulative_gain(
        ideal_relevances
    )

    if ideal_dcg == 0.0:
        return 0.0

    return actual_dcg / ideal_dcg


def calculate_query_metrics(
    relevance_by_chunk_id: Mapping[str, int],
    retrieved_chunk_ids: Sequence[str],
    *,
    k: int,
) -> QueryMetrics:
    return QueryMetrics(
        hit_rate_at_k=hit_rate_at_k(
            relevance_by_chunk_id,
            retrieved_chunk_ids,
            k=k,
        ),
        recall_at_k=recall_at_k(
            relevance_by_chunk_id,
            retrieved_chunk_ids,
            k=k,
        ),
        reciprocal_rank=reciprocal_rank(
            relevance_by_chunk_id,
            retrieved_chunk_ids,
        ),
        ndcg_at_k=ndcg_at_k(
            relevance_by_chunk_id,
            retrieved_chunk_ids,
            k=k,
        ),
    )


def _discounted_cumulative_gain(
    relevances: Sequence[int],
) -> float:
    return sum(
        (2**relevance - 1)
        / math.log2(rank + 1)
        for rank, relevance in enumerate(
            relevances,
            start=1,
        )
    )


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError(
            f"k must be at least 1, got {k}"
        )