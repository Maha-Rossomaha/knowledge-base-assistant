import pytest

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.evaluation.models import (
    GoldenQuery,
    RelevantChunk,
)
from knowledge_base_assistant.evaluation.validation import (
    validate_golden_queries,
)


def make_chunk(
    *,
    chunk_id: str = "chunk-1",
    relative_path: str = "notes/bm25.md",
    section_path: tuple[str, ...] = ("BM25",),
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        source_name="test-source",
        relative_path=relative_path,
        title=section_path[-1] if section_path else None,
        section_path=section_path,
        content="BM25 content",
        searchable_text="BM25\n\nBM25 content",
        chunk_index=0,
        section_chunk_index=0,
        start_line=1,
        end_line=1,
        content_hash=f"hash-{chunk_id}",
    )


def make_query(
    *,
    query_id: str = "q001",
    query: str = "What is BM25?",
    chunk_id: str = "chunk-1",
    relative_path: str = "notes/bm25.md",
    section_path: tuple[str, ...] = ("BM25",),
    relevance: int = 2,
) -> GoldenQuery:
    return GoldenQuery(
        query_id=query_id,
        query=query,
        relevant_chunks=(
            RelevantChunk(
                chunk_id=chunk_id,
                relative_path=relative_path,
                section_path=section_path,
                relevance=relevance,
            ),
        ),
        notes="Definition",
    )
    

def test_valid_golden_queries_return_statistics() -> None:
    result = validate_golden_queries(
        [
            make_query(),
            GoldenQuery(
                query_id="q002",
                query="No-answer query",
                relevant_chunks=(),
                notes="No answer",
            ),
        ],
        [make_chunk()],
    )

    assert result.query_count == 2
    assert result.answerable_query_count == 1
    assert result.no_answer_query_count == 1
    assert result.relevance_judgment_count == 1


def test_rejects_duplicate_query_id() -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate query_id",
    ):
        validate_golden_queries(
            [make_query(), make_query()],
            [make_chunk()],
        )


def test_rejects_empty_query_id() -> None:
    with pytest.raises(
        ValueError,
        match="query_id must not be empty",
    ):
        validate_golden_queries(
            [make_query(query_id="   ")],
            [make_chunk()],
        )


def test_rejects_empty_query() -> None:
    with pytest.raises(
        ValueError,
        match="query must not be empty",
    ):
        validate_golden_queries(
            [make_query(query="   ")],
            [make_chunk()],
        )


@pytest.mark.parametrize(
    "relevance",
    [0, 3, -1],
)
def test_rejects_invalid_relevance(
    relevance: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="relevance must be 1 or 2",
    ):
        validate_golden_queries(
            [make_query(relevance=relevance)],
            [make_chunk()],
        )


def test_rejects_unknown_chunk_id() -> None:
    with pytest.raises(
        ValueError,
        match="unknown chunk_id",
    ):
        validate_golden_queries(
            [make_query(chunk_id="missing")],
            [make_chunk()],
        )


def test_rejects_relative_path_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="relative_path mismatch",
    ):
        validate_golden_queries(
            [make_query(relative_path="wrong.md")],
            [make_chunk()],
        )


def test_rejects_section_path_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="section_path mismatch",
    ):
        validate_golden_queries(
            [make_query(section_path=("Wrong",))],
            [make_chunk()],
        )