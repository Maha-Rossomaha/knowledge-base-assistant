import pytest
from knowledge_base_assistant.evaluation.metrics import (
    calculate_query_metrics,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_hit_rate_is_one_when_relevant_chunk_is_found() -> None:
    result = hit_rate_at_k(
        {
            "chunk-a": 2,
            "chunk-b": 1,
        },
        [
            "chunk-x",
            "chunk-a",
            "chunk-y",
        ],
        k=2,
    )

    assert result == 1.0


def test_hit_rate_is_zero_when_no_relevant_chunk_is_found() -> None:
    result = hit_rate_at_k(
        {
            "chunk-a": 2,
        },
        [
            "chunk-x",
            "chunk-y",
        ],
        k=2,
    )

    assert result == 0.0


def test_hit_rate_respects_k() -> None:
    result = hit_rate_at_k(
        {
            "chunk-a": 2,
        },
        [
            "chunk-x",
            "chunk-a",
        ],
        k=1,
    )

    assert result == 0.0


def test_recall_at_k_returns_fraction_of_relevant_chunks() -> None:
    result = recall_at_k(
        {
            "chunk-a": 2,
            "chunk-b": 1,
            "chunk-c": 1,
        },
        [
            "chunk-a",
            "chunk-x",
            "chunk-b",
        ],
        k=3,
    )

    assert result == pytest.approx(2 / 3)


def test_recall_does_not_count_duplicate_retrieved_chunks() -> None:
    result = recall_at_k(
        {
            "chunk-a": 2,
            "chunk-b": 1,
        },
        [
            "chunk-a",
            "chunk-a",
        ],
        k=2,
    )

    assert result == 0.5


def test_recall_is_zero_for_empty_relevance_mapping() -> None:
    assert recall_at_k(
        {},
        ["chunk-a"],
        k=1,
    ) == 0.0


def test_reciprocal_rank_uses_first_relevant_result() -> None:
    result = reciprocal_rank(
        {
            "chunk-a": 2,
            "chunk-b": 1,
        },
        [
            "chunk-x",
            "chunk-y",
            "chunk-b",
            "chunk-a",
        ],
    )

    assert result == pytest.approx(1 / 3)


def test_reciprocal_rank_is_one_for_first_result() -> None:
    result = reciprocal_rank(
        {
            "chunk-a": 2,
        },
        [
            "chunk-a",
            "chunk-x",
        ],
    )

    assert result == 1.0


def test_reciprocal_rank_is_zero_without_relevant_results() -> None:
    result = reciprocal_rank(
        {
            "chunk-a": 2,
        },
        [
            "chunk-x",
            "chunk-y",
        ],
    )

    assert result == 0.0


def test_ndcg_is_one_for_ideal_ranking() -> None:
    result = ndcg_at_k(
        {
            "chunk-a": 2,
            "chunk-b": 1,
        },
        [
            "chunk-a",
            "chunk-b",
        ],
        k=2,
    )

    assert result == pytest.approx(1.0)


def test_ndcg_penalizes_wrong_order() -> None:
    ideal_result = ndcg_at_k(
        {
            "chunk-a": 2,
            "chunk-b": 1,
        },
        [
            "chunk-a",
            "chunk-b",
        ],
        k=2,
    )

    reversed_result = ndcg_at_k(
        {
            "chunk-a": 2,
            "chunk-b": 1,
        },
        [
            "chunk-b",
            "chunk-a",
        ],
        k=2,
    )

    assert reversed_result < ideal_result


def test_ndcg_penalizes_irrelevant_results_before_relevant() -> None:
    result = ndcg_at_k(
        {
            "chunk-a": 2,
        },
        [
            "chunk-x",
            "chunk-a",
        ],
        k=2,
    )

    assert 0.0 < result < 1.0


def test_ndcg_is_zero_without_relevant_results() -> None:
    assert ndcg_at_k(
        {
            "chunk-a": 2,
        },
        [
            "chunk-x",
            "chunk-y",
        ],
        k=2,
    ) == 0.0


def test_ndcg_is_zero_for_empty_relevance_mapping() -> None:
    assert ndcg_at_k(
        {},
        ["chunk-a"],
        k=1,
    ) == 0.0


def test_calculate_query_metrics_returns_all_metrics() -> None:
    result = calculate_query_metrics(
        {
            "chunk-a": 2,
            "chunk-b": 1,
        },
        [
            "chunk-x",
            "chunk-a",
            "chunk-b",
        ],
        k=3,
    )

    assert result.hit_rate_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 0.5
    assert 0.0 < result.ndcg_at_k < 1.0


@pytest.mark.parametrize(
    "metric_name",
    [
        "hit_rate",
        "recall",
        "ndcg",
    ],
)
def test_metrics_reject_invalid_k(
    metric_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="k must be at least 1",
    ):
        if metric_name == "hit_rate":
            hit_rate_at_k(
                {"chunk-a": 2},
                ["chunk-a"],
                k=0,
            )
        elif metric_name == "recall":
            recall_at_k(
                {"chunk-a": 2},
                ["chunk-a"],
                k=0,
            )
        else:
            ndcg_at_k(
                {"chunk-a": 2},
                ["chunk-a"],
                k=0,
            )
            

def test_ndcg_is_zero_when_ideal_dcg_is_zero() -> None:
    result = ndcg_at_k(
        {
            "chunk-a": 0,
            "chunk-b": 0,
        },
        [
            "chunk-a",
            "chunk-b",
        ],
        k=2,
    )

    assert result == 0.0