import json
from pathlib import Path
from typing import Any

import pytest

from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
    RetrievedChunkEvaluation,
)
from knowledge_base_assistant.evaluation.serialization import (
    read_query_evaluation_results_jsonl,
    write_query_evaluation_misses_jsonl,
    write_query_evaluation_results_jsonl,
)


def _valid_serialized_result() -> dict[str, Any]:
    return {
        "query_id": "query-1",
        "query": "Test query",
        "relevant_chunk_ids": ["chunk-1"],
        "retrieved_chunks": [
            {
                "chunk_id": "chunk-1",
                "rank": 1,
                "score": 0.9,
                "relevance": 1,
                "relative_path": "document.md",
                "section_path": ["Section"],
                "content": "Content",
            }
        ],
        "first_relevant_rank": 1,
        "hit_rate_at_k": 1.0,
        "recall_at_k": 1.0,
        "reciprocal_rank": 1.0,
        "ndcg_at_k": 1.0,
    }
    

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
    
    
def test_read_query_evaluation_results_skips_blank_lines(
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

    original = path.read_text(encoding="utf-8")
    path.write_text(
        f"\n{original}\n\n",
        encoding="utf-8",
    )

    assert read_query_evaluation_results_jsonl(path) == (
        expected,
    )
    
    
def test_read_query_evaluation_results_rejects_invalid_json(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(
        '{"query_id":',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSONL at line 1",
    ):
        read_query_evaluation_results_jsonl(path)
        
        
def test_read_query_evaluation_results_reports_invalid_result(
    tmp_path: Path,
) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid query evaluation result at line 1",
    ):
        read_query_evaluation_results_jsonl(path)
        
        
@pytest.mark.parametrize(
    ("update", "expected_message"),
    [
        (
            {"root": []},
            "JSON value must be an object",
        ),
        (
            {"relevant_chunk_ids": "chunk-1"},
            "relevant_chunk_ids must be a list",
        ),
        (
            {"relevant_chunk_ids": [123]},
            "relevant_chunk_ids must contain only strings",
        ),
        (
            {"retrieved_chunks": {}},
            "retrieved_chunks must be a list",
        ),
        (
            {"first_relevant_rank": "1"},
            "first_relevant_rank must be an integer or null",
        ),
        (
            {"retrieved_chunks": ["invalid"]},
            "retrieved_chunks item must be an object",
        ),
    ],
)
def test_read_query_evaluation_results_rejects_invalid_root_fields(
    tmp_path: Path,
    update: dict[str, Any],
    expected_message: str,
) -> None:
    path = tmp_path / "results.jsonl"
    data: Any = _valid_serialized_result()

    if "root" in update:
        data = update["root"]
    else:
        data.update(update)

    path.write_text(
        json.dumps(data) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        read_query_evaluation_results_jsonl(path)
        
        
@pytest.mark.parametrize(
    ("section_path", "expected_message"),
    [
        (
            "Section",
            "section_path must be a list",
        ),
        (
            [123],
            "section_path must contain only strings",
        ),
    ],
)
def test_read_query_evaluation_results_rejects_invalid_section_path(
    tmp_path: Path,
    section_path: Any,
    expected_message: str,
) -> None:
    path = tmp_path / "results.jsonl"
    data = _valid_serialized_result()
    data["retrieved_chunks"][0]["section_path"] = (
        section_path
    )

    path.write_text(
        json.dumps(data) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        read_query_evaluation_results_jsonl(path)
        
        
@pytest.mark.parametrize(
    ("target", "field"),
    [
        ("root", "query_id"),
        ("root", "query"),
        ("chunk", "chunk_id"),
        ("chunk", "relative_path"),
        ("chunk", "content"),
    ],
)
def test_read_query_evaluation_results_rejects_non_string_fields(
    tmp_path: Path,
    target: str,
    field: str,
) -> None:
    path = tmp_path / "results.jsonl"
    data = _valid_serialized_result()

    if target == "root":
        data[field] = 123
    else:
        data["retrieved_chunks"][0][field] = 123

    path.write_text(
        json.dumps(data) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=rf"{field} must be a string",
    ):
        read_query_evaluation_results_jsonl(path)