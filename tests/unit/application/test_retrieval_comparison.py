import json
from pathlib import Path

import pytest

from knowledge_base_assistant.application.retrieval_comparison import (
    ComparisonCategory,
    compare_retrieval_results,
    run_retrieval_comparison,
)
from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
)
from knowledge_base_assistant.evaluation.serialization import (
    write_query_evaluation_results_jsonl,
)


def test_compare_retrieval_results_splits_all_categories() -> None:
    lexical_results = (
        _make_result("dense-only", None),
        _make_result("lexical-only", 1),
        _make_result("both-hit", 2),
        _make_result("both-miss", None),
    )
    dense_results = (
        _make_result("dense-only", 1),
        _make_result("lexical-only", None),
        _make_result("both-hit", 1),
        _make_result("both-miss", None),
    )

    result = compare_retrieval_results(
        lexical_results=lexical_results,
        dense_results=dense_results,
    )

    assert [
        item.query_id
        for item in result.dense_only_hits
    ] == ["dense-only"]

    assert [
        item.query_id
        for item in result.lexical_only_hits
    ] == ["lexical-only"]

    assert [
        item.query_id
        for item in result.both_hit
    ] == ["both-hit"]

    assert [
        item.query_id
        for item in result.both_miss
    ] == ["both-miss"]

    compared = result.dense_only_hits[0]

    assert (
        compared.category
        is ComparisonCategory.DENSE_ONLY_HIT
    )
    assert compared.lexical_result.first_relevant_rank is None
    assert compared.dense_result.first_relevant_rank == 1


def test_compare_retrieval_results_rejects_different_query_ids() -> None:
    with pytest.raises(
        ValueError,
        match="same query IDs",
    ):
        compare_retrieval_results(
            lexical_results=(
                _make_result("query-1", 1),
            ),
            dense_results=(
                _make_result("query-2", 1),
            ),
        )


def test_run_retrieval_comparison_writes_both_results(
    tmp_path: Path,
) -> None:
    lexical_path = tmp_path / "lexical.jsonl"
    dense_path = tmp_path / "dense.jsonl"
    output_dir = tmp_path / "comparison"

    write_query_evaluation_results_jsonl(
        (
            _make_result("query-1", None),
        ),
        lexical_path,
    )
    write_query_evaluation_results_jsonl(
        (
            _make_result("query-1", 1),
        ),
        dense_path,
    )

    artifacts = run_retrieval_comparison(
        lexical_results_path=lexical_path,
        dense_results_path=dense_path,
        output_dir=output_dir,
    )

    lines = artifacts.dense_only_hits_path.read_text(
        encoding="utf-8",
    ).splitlines()

    assert len(lines) == 1

    data = json.loads(lines[0])

    assert data["query_id"] == "query-1"
    assert data["category"] == "dense_only_hit"
    assert (
        data["lexical_result"]["first_relevant_rank"]
        is None
    )
    assert (
        data["dense_result"]["first_relevant_rank"]
        == 1
    )


def _make_result(
    query_id: str,
    first_relevant_rank: int | None,
    *,
    query: str | None = None,
) -> QueryEvaluationResult:
    return QueryEvaluationResult(
        query_id=query_id,
        query=query or f"Query {query_id}",
        relevant_chunk_ids=("chunk-1",),
        retrieved_chunks=(),
        first_relevant_rank=first_relevant_rank,
        hit_rate_at_k=(
            1.0
            if first_relevant_rank is not None
            else 0.0
        ),
        recall_at_k=(
            1.0
            if first_relevant_rank is not None
            else 0.0
        ),
        reciprocal_rank=(
            1.0
            if first_relevant_rank is not None
            else 0.0
        ),
        ndcg_at_k=(
            1.0
            if first_relevant_rank is not None
            else 0.0
        ),
    )
    
    
def test_compare_retrieval_results_rejects_different_query_text() -> None:
    lexical_result = _make_result(
        "query-1",
        1,
        query="Lexical query text",
    )
    dense_result = _make_result(
        "query-1",
        1,
        query="Dense query text",
    )

    with pytest.raises(
        ValueError,
        match=(
            "same query text for query ID query-1"
        ),
    ):
        compare_retrieval_results(
            lexical_results=(lexical_result,),
            dense_results=(dense_result,),
        )