import json
from pathlib import Path
from typing import Any

import pytest

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.evaluation.models import GoldenQuery, RelevantChunk
from knowledge_base_assistant.serialization.jsonl import (
    read_golden_queries_jsonl,
    validate_golden_queries,
    write_chunks_jsonl,
)


def make_chunk(
    *,
    chunk_id: str = "chunk-1",
    relative_path: str = "notes/rag.md",
    section_path: tuple[str, ...] = ("RAG", "Chunking"),
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_name="test-source",
        relative_path=relative_path,
        title=section_path[-1],
        section_path=section_path,
        content="Chunk content",
        searchable_text="RAG > Chunking\n\nChunk content",
        chunk_index=0,
        section_chunk_index=0,
        start_line=10,
        end_line=10,
        content_hash=f"hash-{chunk_id}",
    )


def golden_query_line(
    *,
    query_id: str = "q001",
    query: str = "Что такое чанкинг?",
    chunk_id: str = "chunk-1",
    relative_path: str = "notes/rag.md",
    section_path: list[str] | None = None,
    relevance: int = 2,
) -> str:
    sections = ["RAG", "Chunking"] if section_path is None else section_path
    return json.dumps(
        {
            "query_id": query_id,
            "query": query,
            "relevant_chunks": [
                {
                    "chunk_id": chunk_id,
                    "relative_path": relative_path,
                    "section_path": sections,
                    "relevance": relevance,
                }
            ],
            "notes": "Должно найти определение чанкинга.",
        },
        ensure_ascii=False,
    )


def write_golden_value(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def test_read_golden_queries_jsonl(tmp_path: Path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    queries_path = tmp_path / "golden_queries.jsonl"
    chunk = make_chunk()

    write_chunks_jsonl([chunk], chunks_path)
    queries_path.write_text(golden_query_line(), encoding="utf-8")

    queries = read_golden_queries_jsonl(queries_path, chunks_path)

    assert queries == [
        GoldenQuery(
            query_id="q001",
            query="Что такое чанкинг?",
            relevant_chunks=(
                RelevantChunk(
                    chunk_id="chunk-1",
                    relative_path="notes/rag.md",
                    section_path=("RAG", "Chunking"),
                    relevance=2,
                ),
            ),
            notes="Должно найти определение чанкинга.",
        )
    ]


def test_golden_reader_reports_invalid_json_line(tmp_path: Path) -> None:
    path = tmp_path / "golden_queries.jsonl"
    path.write_text("\nnot json\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSONL at line 2"):
        read_golden_queries_jsonl(path)


def test_golden_reader_reports_schema_line(tmp_path: Path) -> None:
    path = tmp_path / "golden_queries.jsonl"
    path.write_text('{"query_id":"q001"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid golden query at line 1"):
        read_golden_queries_jsonl(path)



@pytest.mark.parametrize(
    ("value", "message"),
    [
        (
            ["not", "an", "object"],
            "JSON value must be an object",
        ),
        (
            {
                "query_id": "q001",
                "query": "Что такое чанкинг?",
                "relevant_chunks": "chunk-1",
                "notes": "Должно найти определение чанкинга.",
            },
            "relevant_chunks must be a list",
        ),
        (
            {
                "query_id": "q001",
                "query": "Что такое чанкинг?",
                "relevant_chunks": ["chunk-1"],
                "notes": "Должно найти определение чанкинга.",
            },
            "relevant_chunks item must be an object",
        ),
        (
            {
                "query_id": "q001",
                "query": "Что такое чанкинг?",
                "relevant_chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "relative_path": "notes/rag.md",
                        "section_path": ["RAG", 1],
                        "relevance": 2,
                    }
                ],
                "notes": "Должно найти определение чанкинга.",
            },
            "section_path must contain only strings",
        ),
        (
            {
                "query_id": "q001",
                "query": "Что такое чанкинг?",
                "relevant_chunks": [
                    {
                        "chunk_id": "chunk-1",
                        "relative_path": "notes/rag.md",
                        "section_path": ["RAG", "Chunking"],
                        "relevance": "2",
                    }
                ],
                "notes": "Должно найти определение чанкинга.",
            },
            "relevance must be an integer",
        ),
        (
            {
                "query_id": 1,
                "query": "Что такое чанкинг?",
                "relevant_chunks": [],
                "notes": "Должно найти определение чанкинга.",
            },
            "query_id must be a string",
        ),
    ],
)
def test_golden_reader_rejects_invalid_schema_types(
    tmp_path: Path,
    value: Any,
    message: str,
) -> None:
    path = tmp_path / "golden_queries.jsonl"
    write_golden_value(path, value)

    with pytest.raises(
        ValueError,
        match=f"Invalid golden query at line 1: {message}",
    ):
        read_golden_queries_jsonl(path)


def test_validate_rejects_duplicate_query_id() -> None:
    query = GoldenQuery("q001", "query", (), "notes")

    with pytest.raises(ValueError, match="Duplicate query_id: q001"):
        validate_golden_queries([query, query])


def test_validate_rejects_empty_query() -> None:
    query = GoldenQuery("q001", " ", (), "notes")

    with pytest.raises(ValueError, match="query must not be empty for q001"):
        validate_golden_queries([query])


def test_validate_rejects_invalid_relevance() -> None:
    query = GoldenQuery(
        "q001",
        "query",
        (RelevantChunk("chunk-1", "notes/rag.md", ("RAG",), 3),),
        "notes",
    )

    with pytest.raises(ValueError, match="Invalid relevance for q001: 3"):
        validate_golden_queries([query])
        

def test_validate_rejects_empty_id() -> None:
    query = GoldenQuery(
        "",
        "query",
        (RelevantChunk("chunk-1", "notes/rag.md", ("RAG",), 2),),
        "notes",
    )

    with pytest.raises(ValueError, match="query_id must not be empty"):
        validate_golden_queries([query])


def test_validate_rejects_duplicate_chunk_id_inside_query() -> None:
    relevant_chunk = RelevantChunk("chunk-1", "notes/rag.md", ("RAG",), 2)
    query = GoldenQuery("q001", "query", (relevant_chunk, relevant_chunk), "notes")

    with pytest.raises(ValueError, match="Duplicate chunk_id for q001: chunk-1"):
        validate_golden_queries([query])


def test_validate_rejects_unknown_chunk_id() -> None:
    query = GoldenQuery(
        "q001",
        "query",
        (RelevantChunk("missing", "notes/rag.md", ("RAG",), 2),),
        "notes",
    )

    with pytest.raises(ValueError, match="Unknown chunk_id for q001: missing"):
        validate_golden_queries([query], [make_chunk()])


def test_validate_rejects_relative_path_mismatch() -> None:
    query = GoldenQuery(
        "q001",
        "query",
        (RelevantChunk("chunk-1", "wrong.md", ("RAG", "Chunking"), 2),),
        "notes",
    )

    with pytest.raises(ValueError, match="relative_path mismatch for q001"):
        validate_golden_queries([query], [make_chunk()])


def test_validate_rejects_section_path_mismatch() -> None:
    query = GoldenQuery(
        "q001",
        "query",
        (RelevantChunk("chunk-1", "notes/rag.md", ("Other",), 2),),
        "notes",
    )

    with pytest.raises(ValueError, match="section_path mismatch for q001"):
        validate_golden_queries([query], [make_chunk()])


def test_project_golden_dataset_is_valid() -> None:
    queries = read_golden_queries_jsonl(
        Path("data/evaluation/golden_queries.jsonl"),
        Path("artifacts/chunks.jsonl"),
    )

    assert len(queries) == 66
    assert sum(not query.relevant_chunks for query in queries) == 12
    assert sum(len(query.relevant_chunks) == 1 for query in queries) == 0
