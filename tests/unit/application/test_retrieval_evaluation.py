import json
from pathlib import Path

import pytest

from knowledge_base_assistant.application.lexical_retrieval_evaluation import (
    evaluate_bm25_queries,
    evaluate_bm25_retrieval,
)
from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.serialization.jsonl import (
    write_chunks_jsonl,
)


def make_chunk(
    *,
    chunk_id: str,
    searchable_text: str,
    relative_path: str | None = None,
    section_path: tuple[str, ...] | None = None,
) -> Chunk:
    resolved_relative_path = (
        relative_path
        if relative_path is not None
        else f"notes/{chunk_id}.md"
    )
    resolved_section_path = (
        section_path
        if section_path is not None
        else (chunk_id,)
    )

    return Chunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        source_name="test-source",
        relative_path=resolved_relative_path,
        title=(
            resolved_section_path[-1]
            if resolved_section_path
            else None
        ),
        section_path=resolved_section_path,
        content=searchable_text,
        searchable_text=searchable_text,
        chunk_index=0,
        section_chunk_index=0,
        start_line=1,
        end_line=1,
        content_hash=f"hash-{chunk_id}",
    )


def write_golden_queries(
    path: Path,
    records: list[dict[str, object]],
) -> None:
    path.write_text(
        "".join(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def test_evaluate_bm25_retrieval_returns_average_metrics(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    golden_path = tmp_path / "golden.jsonl"

    write_chunks_jsonl(
        [
            make_chunk(
                chunk_id="bm25",
                searchable_text=(
                    "BM25 lexical retrieval ranking"
                ),
            ),
            make_chunk(
                chunk_id="dense",
                searchable_text=(
                    "Dense semantic vector retrieval"
                ),
            ),
            make_chunk(
                chunk_id="docker",
                searchable_text=(
                    "Docker container image"
                ),
            ),
        ],
        chunks_path,
    )

    write_golden_queries(
        golden_path,
        [
            {
                "query_id": "q001",
                "query": "BM25 lexical retrieval",
                "relevant_chunks": [
                    {
                        "chunk_id": "bm25",
                        "relative_path": "notes/bm25.md",
                        "section_path": ["bm25"],
                        "relevance": 2,
                    }
                ],
                "notes": "BM25 query",
            },
            {
                "query_id": "q002",
                "query": "Dense semantic retrieval",
                "relevant_chunks": [
                    {
                        "chunk_id": "dense",
                        "relative_path": "notes/dense.md",
                        "section_path": ["dense"],
                        "relevance": 2,
                    }
                ],
                "notes": "Dense retrieval query",
            },
            {
                "query_id": "q003",
                "query": "Question without answer",
                "relevant_chunks": [],
                "notes": "No answer",
            },
        ],
    )

    result = evaluate_bm25_retrieval(
        golden_path=golden_path,
        chunks_path=chunks_path,
        top_k=2,
    )

    assert result.top_k == 2
    assert result.query_count == 3
    assert result.evaluated_query_count == 2
    assert result.no_answer_query_count == 1

    assert result.hit_rate_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.mean_reciprocal_rank == 1.0
    assert result.ndcg_at_k == 1.0


def test_evaluation_averages_metrics_across_queries(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    golden_path = tmp_path / "golden.jsonl"

    write_chunks_jsonl(
        [
            make_chunk(
                chunk_id="chunk-a",
                searchable_text="bm25 lexical retrieval",
            ),
            make_chunk(
                chunk_id="chunk-b",
                searchable_text="dense semantic retrieval",
            ),
            make_chunk(
                chunk_id="chunk-x",
                searchable_text=(
                    "bm25 semantic unrelated retrieval"
                ),
            ),
        ],
        chunks_path,
    )

    write_golden_queries(
        golden_path,
        [
            {
                "query_id": "q001",
                "query": "bm25 lexical",
                "relevant_chunks": [
                    {
                        "chunk_id": "chunk-a",
                        "relative_path": "notes/chunk-a.md",
                        "section_path": ["chunk-a"],
                        "relevance": 2,
                    }
                ],
                "notes": "",
            },
            {
                "query_id": "q002",
                "query": "semantic retrieval",
                "relevant_chunks": [
                    {
                        "chunk_id": "chunk-b",
                        "relative_path": "notes/chunk-b.md",
                        "section_path": ["chunk-b"],
                        "relevance": 2,
                    }
                ],
                "notes": "",
            },
        ],
    )

    result = evaluate_bm25_retrieval(
        golden_path=golden_path,
        chunks_path=chunks_path,
        top_k=2,
    )

    assert result.query_count == 2
    assert result.evaluated_query_count == 2

    assert 0.0 <= result.hit_rate_at_k <= 1.0
    assert 0.0 <= result.recall_at_k <= 1.0
    assert 0.0 <= result.mean_reciprocal_rank <= 1.0
    assert 0.0 <= result.ndcg_at_k <= 1.0


def test_evaluation_skips_no_answer_queries(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    golden_path = tmp_path / "golden.jsonl"

    write_chunks_jsonl(
        [
            make_chunk(
                chunk_id="bm25",
                searchable_text="BM25 lexical retrieval",
            )
        ],
        chunks_path,
    )

    write_golden_queries(
        golden_path,
        [
            {
                "query_id": "q001",
                "query": "BM25",
                "relevant_chunks": [
                    {
                        "chunk_id": "bm25",
                        "relative_path": "notes/bm25.md",
                        "section_path": ["bm25"],
                        "relevance": 2,
                    }
                ],
                "notes": "",
            },
            {
                "query_id": "q002",
                "query": "Unknown information",
                "relevant_chunks": [],
                "notes": "No answer",
            },
        ],
    )

    result = evaluate_bm25_retrieval(
        golden_path=golden_path,
        chunks_path=chunks_path,
        top_k=1,
    )

    assert result.query_count == 2
    assert result.evaluated_query_count == 1
    assert result.no_answer_query_count == 1

    assert result.hit_rate_at_k == 1.0
    assert result.recall_at_k == 1.0


def test_evaluation_returns_zero_metrics_when_all_queries_are_no_answer(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    golden_path = tmp_path / "golden.jsonl"

    write_chunks_jsonl(
        [
            make_chunk(
                chunk_id="bm25",
                searchable_text="BM25 lexical retrieval",
            )
        ],
        chunks_path,
    )

    write_golden_queries(
        golden_path,
        [
            {
                "query_id": "q001",
                "query": "No answer one",
                "relevant_chunks": [],
                "notes": "",
            },
            {
                "query_id": "q002",
                "query": "No answer two",
                "relevant_chunks": [],
                "notes": "",
            },
        ],
    )

    result = evaluate_bm25_retrieval(
        golden_path=golden_path,
        chunks_path=chunks_path,
        top_k=5,
    )

    assert result.query_count == 2
    assert result.evaluated_query_count == 0
    assert result.no_answer_query_count == 2

    assert result.hit_rate_at_k == 0.0
    assert result.recall_at_k == 0.0
    assert result.mean_reciprocal_rank == 0.0
    assert result.ndcg_at_k == 0.0


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_evaluation_rejects_invalid_top_k(
    tmp_path: Path,
    top_k: int,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    golden_path = tmp_path / "golden.jsonl"

    chunks_path.write_text("", encoding="utf-8")
    golden_path.write_text("", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        evaluate_bm25_retrieval(
            golden_path=golden_path,
            chunks_path=chunks_path,
            top_k=top_k,
        )


def test_evaluation_validates_golden_dataset_against_chunks(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    golden_path = tmp_path / "golden.jsonl"

    write_chunks_jsonl(
        [
            make_chunk(
                chunk_id="existing",
                searchable_text="BM25 retrieval",
            )
        ],
        chunks_path,
    )

    write_golden_queries(
        golden_path,
        [
            {
                "query_id": "q001",
                "query": "BM25",
                "relevant_chunks": [
                    {
                        "chunk_id": "missing",
                        "relative_path": "notes/missing.md",
                        "section_path": ["missing"],
                        "relevance": 2,
                    }
                ],
                "notes": "",
            }
        ],
    )

    with pytest.raises(
        ValueError,
        match="unknown chunk_id",
    ):
        evaluate_bm25_retrieval(
            golden_path=golden_path,
            chunks_path=chunks_path,
            top_k=5,
        )
        
        
def test_evaluate_bm25_queries_returns_query_details(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    golden_path = tmp_path / "golden.jsonl"

    write_chunks_jsonl(
        [
            make_chunk(
                chunk_id="bm25",
                searchable_text="BM25 lexical retrieval ranking",
            ),
            make_chunk(
                chunk_id="dense",
                searchable_text="Dense semantic vector search",
            ),
        ],
        chunks_path,
    )

    write_golden_queries(
        golden_path,
        [
            {
                "query_id": "q001",
                "query": "BM25 lexical retrieval",
                "relevant_chunks": [
                    {
                        "chunk_id": "bm25",
                        "relative_path": "notes/bm25.md",
                        "section_path": ["bm25"],
                        "relevance": 2,
                    }
                ],
                "notes": "",
            }
        ],
    )

    results = evaluate_bm25_queries(
        golden_path=golden_path,
        chunks_path=chunks_path,
        top_k=2,
    )

    assert len(results) == 1

    result = results[0]

    assert result.query_id == "q001"
    assert result.query == "BM25 lexical retrieval"
    assert result.relevant_chunk_ids == ("bm25",)
    assert result.first_relevant_rank == 1

    assert result.hit_rate_at_k == 1.0
    assert result.recall_at_k == 1.0
    assert result.reciprocal_rank == 1.0
    assert result.ndcg_at_k == 1.0

    assert len(result.retrieved_chunks) == 1

    retrieved = result.retrieved_chunks[0]

    assert retrieved.chunk_id == "bm25"
    assert retrieved.rank == 1
    assert retrieved.score > 0.0
    assert retrieved.relevance == 2
    
    
def test_evaluate_bm25_queries_marks_retrieved_relevance(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    golden_path = tmp_path / "golden.jsonl"

    write_chunks_jsonl(
        [
            make_chunk(
                chunk_id="relevant",
                searchable_text="semantic retrieval",
            ),
            make_chunk(
                chunk_id="irrelevant",
                searchable_text=(
                    "semantic retrieval semantic retrieval"
                ),
            ),
        ],
        chunks_path,
    )

    write_golden_queries(
        golden_path,
        [
            {
                "query_id": "q001",
                "query": "semantic retrieval",
                "relevant_chunks": [
                    {
                        "chunk_id": "relevant",
                        "relative_path": "notes/relevant.md",
                        "section_path": ["relevant"],
                        "relevance": 2,
                    }
                ],
                "notes": "",
            }
        ],
    )

    results = evaluate_bm25_queries(
        golden_path=golden_path,
        chunks_path=chunks_path,
        top_k=2,
    )

    result = results[0]

    assert len(result.retrieved_chunks) == 2
    assert result.retrieved_chunks[0].chunk_id == "irrelevant"
    assert result.retrieved_chunks[0].relevance == 0
    assert result.retrieved_chunks[1].chunk_id == "relevant"
    assert result.retrieved_chunks[1].relevance == 2

    assert result.first_relevant_rank == 2
    assert result.reciprocal_rank == 0.5
    
    
def test_evaluate_bm25_queries_sets_no_first_relevant_rank(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    golden_path = tmp_path / "golden.jsonl"

    write_chunks_jsonl(
        [
            make_chunk(
                chunk_id="relevant",
                searchable_text="docker container images",
            ),
            make_chunk(
                chunk_id="retrieved",
                searchable_text="BM25 lexical retrieval",
            ),
        ],
        chunks_path,
    )

    write_golden_queries(
        golden_path,
        [
            {
                "query_id": "q001",
                "query": "BM25 retrieval",
                "relevant_chunks": [
                    {
                        "chunk_id": "relevant",
                        "relative_path": "notes/relevant.md",
                        "section_path": ["relevant"],
                        "relevance": 2,
                    }
                ],
                "notes": "",
            }
        ],
    )

    results = evaluate_bm25_queries(
        golden_path=golden_path,
        chunks_path=chunks_path,
        top_k=1,
    )

    result = results[0]

    assert result.first_relevant_rank is None
    assert result.hit_rate_at_k == 0.0
    assert result.recall_at_k == 0.0
    assert result.reciprocal_rank == 0.0
    assert result.ndcg_at_k == 0.0

    assert result.retrieved_chunks[0].chunk_id == "retrieved"
    assert result.retrieved_chunks[0].relevance == 0
    
    
def test_evaluate_bm25_queries_skips_no_answer_queries(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    golden_path = tmp_path / "golden.jsonl"

    write_chunks_jsonl(
        [
            make_chunk(
                chunk_id="bm25",
                searchable_text="BM25 lexical retrieval",
            )
        ],
        chunks_path,
    )

    write_golden_queries(
        golden_path,
        [
            {
                "query_id": "answerable",
                "query": "BM25",
                "relevant_chunks": [
                    {
                        "chunk_id": "bm25",
                        "relative_path": "notes/bm25.md",
                        "section_path": ["bm25"],
                        "relevance": 2,
                    }
                ],
                "notes": "",
            },
            {
                "query_id": "no-answer",
                "query": "Missing topic",
                "relevant_chunks": [],
                "notes": "",
            },
        ],
    )

    results = evaluate_bm25_queries(
        golden_path=golden_path,
        chunks_path=chunks_path,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].query_id == "answerable"
    
    
@pytest.mark.parametrize("top_k", [0, -1])
def test_evaluate_bm25_queries_rejects_invalid_top_k(
    tmp_path: Path,
    top_k: int,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    golden_path = tmp_path / "golden.jsonl"

    chunks_path.write_text("", encoding="utf-8")
    golden_path.write_text("", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        evaluate_bm25_queries(
            golden_path=golden_path,
            chunks_path=chunks_path,
            top_k=top_k,
        )