import json
from pathlib import Path

from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
    RetrievedChunkEvaluation,
)
from knowledge_base_assistant.evaluation.serialization import (
    read_query_evaluation_results_jsonl,
    write_query_evaluation_misses_jsonl,
    write_query_evaluation_results_jsonl,
)


def test_write_query_evaluation_results_jsonl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested" / "results.jsonl"
    result = _make_result(
        query_id="query-1",
        first_relevant_rank=1,
    )

    write_query_evaluation_results_jsonl(
        (result,),
        path,
    )

    lines = path.read_text(
        encoding="utf-8",
    ).splitlines()

    assert len(lines) == 1

    data = json.loads(lines[0])

    assert data["query_id"] == "query-1"
    assert data["query"] == "test query"
    assert data["relevant_chunk_ids"] == [
        "chunk-1",
    ]
    assert data["first_relevant_rank"] == 1

    assert data["retrieved_chunks"] == [
        {
            "chunk_id": "chunk-1",
            "rank": 1,
            "score": 0.9,
            "relevance": 2,
            "relative_path": "notes/test.md",
            "section_path": [
                "Test",
            ],
            "content": "Relevant content",
        }
    ]


def test_write_query_evaluation_misses_jsonl_writes_only_misses(
    tmp_path: Path,
) -> None:
    path = tmp_path / "misses.jsonl"

    hit = _make_result(
        query_id="query-hit",
        first_relevant_rank=1,
    )
    miss = _make_result(
        query_id="query-miss",
        first_relevant_rank=None,
    )

    miss_count = write_query_evaluation_misses_jsonl(
        (hit, miss),
        path,
    )

    lines = path.read_text(
        encoding="utf-8",
    ).splitlines()

    assert miss_count == 1
    assert len(lines) == 1
    assert json.loads(lines[0])["query_id"] == "query-miss"


def test_write_query_evaluation_misses_jsonl_creates_empty_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "misses.jsonl"

    hit = _make_result(
        query_id="query-hit",
        first_relevant_rank=1,
    )

    miss_count = write_query_evaluation_misses_jsonl(
        (hit,),
        path,
    )

    assert miss_count == 0
    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""


def _make_result(
    *,
    query_id: str,
    first_relevant_rank: int | None,
) -> QueryEvaluationResult:
    return QueryEvaluationResult(
        query_id=query_id,
        query="test query",
        relevant_chunk_ids=("chunk-1",),
        retrieved_chunks=(
            RetrievedChunkEvaluation(
                chunk_id="chunk-1",
                rank=1,
                score=0.9,
                relevance=2,
                relative_path="notes/test.md",
                section_path=("Test",),
                content="Relevant content",
            ),
        ),
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
    
    
def test_read_query_evaluation_results_jsonl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.jsonl"
    expected = _make_result(
        query_id="query-1",
        first_relevant_rank=1,
    )

    write_query_evaluation_results_jsonl(
        (expected,),
        path,
    )

    actual = read_query_evaluation_results_jsonl(
        path,
    )

    assert actual == (expected,)