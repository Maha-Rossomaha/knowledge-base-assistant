from pathlib import Path

import pytest

from knowledge_base_assistant.application.statistics import (
    calculate_chunk_statistics,
    calculate_chunk_statistics_from_jsonl,
)
from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.serialization.jsonl import (
    write_chunks_jsonl,
)


def make_chunk(
    *,
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    relative_path: str = "notes/test.md",
    section_path: tuple[str, ...] = ("Heading",),
    content: str = "content",
    chunk_index: int = 0,
    section_chunk_index: int = 0,
    start_line: int = 1,
    end_line: int = 1,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_name="test-source",
        relative_path=relative_path,
        title=section_path[-1] if section_path else None,
        section_path=section_path,
        content=content,
        searchable_text=content,
        chunk_index=chunk_index,
        section_chunk_index=section_chunk_index,
        start_line=start_line,
        end_line=end_line,
        content_hash=f"hash-{chunk_id}",
    )


def test_empty_chunks_return_zero_statistics() -> None:
    statistics = calculate_chunk_statistics([])

    assert statistics.chunk_count == 0
    assert statistics.document_count == 0

    assert statistics.min_char_count == 0
    assert statistics.average_char_count == 0.0
    assert statistics.max_char_count == 0

    assert statistics.min_line_count == 0
    assert statistics.average_line_count == 0.0
    assert statistics.max_line_count == 0

    assert statistics.chunks_with_code_fence == 0
    assert statistics.chunks_over_max_chars == 0
    assert statistics.chunks_over_max_lines == 0
    assert statistics.largest_chunks == ()
    assert statistics.smallest_chunks == ()
    

def test_calculates_character_statistics() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            content="abc",
        ),
        make_chunk(
            chunk_id="chunk-2",
            content="abcdef",
            chunk_index=1,
            section_chunk_index=1,
        ),
        make_chunk(
            chunk_id="chunk-3",
            content="abcdefghi",
            chunk_index=2,
            section_chunk_index=2,
        ),
    ]

    statistics = calculate_chunk_statistics(chunks)

    assert statistics.min_char_count == 3
    assert statistics.average_char_count == 6.0
    assert statistics.max_char_count == 9


def test_calculates_line_statistics() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            content="one line",
        ),
        make_chunk(
            chunk_id="chunk-2",
            content="line 1\nline 2",
            chunk_index=1,
            section_chunk_index=1,
        ),
        make_chunk(
            chunk_id="chunk-3",
            content="line 1\nline 2\nline 3",
            chunk_index=2,
            section_chunk_index=2,
        ),
    ]

    statistics = calculate_chunk_statistics(chunks)

    assert statistics.min_line_count == 1
    assert statistics.average_line_count == 2.0
    assert statistics.max_line_count == 3


def test_counts_unique_documents() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
        ),
        make_chunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            chunk_index=1,
            section_chunk_index=1,
        ),
        make_chunk(
            chunk_id="chunk-3",
            document_id="doc-2",
            chunk_index=2,
        ),
    ]

    statistics = calculate_chunk_statistics(chunks)

    assert statistics.chunk_count == 3
    assert statistics.document_count == 2


def test_counts_chunks_with_backtick_code_fence() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            content=(
                "Example:\n"
                "```python\n"
                "print('hello')\n"
                "```"
            ),
        ),
        make_chunk(
            chunk_id="chunk-2",
            content="Plain text",
            chunk_index=1,
            section_chunk_index=1,
        ),
    ]

    statistics = calculate_chunk_statistics(chunks)

    assert statistics.chunks_with_code_fence == 1


def test_counts_chunks_with_tilde_code_fence() -> None:
    chunks = [
        make_chunk(
            content=(
                "Example:\n"
                "~~~text\n"
                "definition\n"
                "~~~"
            ),
        ),
    ]

    statistics = calculate_chunk_statistics(chunks)

    assert statistics.chunks_with_code_fence == 1


def test_code_fence_may_have_leading_spaces() -> None:
    chunks = [
        make_chunk(
            content=(
                "Example:\n"
                "   ```python\n"
                "print('hello')\n"
                "   ```"
            ),
        ),
    ]

    statistics = calculate_chunk_statistics(chunks)

    assert statistics.chunks_with_code_fence == 1


def test_counts_chunks_over_limits() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            content="12345",
        ),
        make_chunk(
            chunk_id="chunk-2",
            content="line 1\nline 2\nline 3",
            chunk_index=1,
            section_chunk_index=1,
        ),
    ]

    statistics = calculate_chunk_statistics(
        chunks,
        max_chars=4,
        max_lines=2,
    )

    assert statistics.chunks_over_max_chars == 2
    assert statistics.chunks_over_max_lines == 1


def test_chunk_equal_to_limit_is_not_counted_as_over_limit() -> None:
    chunk = make_chunk(
        content="line1\nline2",
    )

    statistics = calculate_chunk_statistics(
        [chunk],
        max_chars=len(chunk.content),
        max_lines=2,
    )

    assert statistics.chunks_over_max_chars == 0
    assert statistics.chunks_over_max_lines == 0


def test_largest_chunks_are_sorted_by_character_count() -> None:
    chunks = [
        make_chunk(
            chunk_id="small",
            content="abc",
            relative_path="small.md",
        ),
        make_chunk(
            chunk_id="large",
            content="abcdefghij",
            relative_path="large.md",
            chunk_index=1,
        ),
        make_chunk(
            chunk_id="medium",
            content="abcdef",
            relative_path="medium.md",
            chunk_index=2,
        ),
    ]

    statistics = calculate_chunk_statistics(
        chunks,
        largest_limit=2,
    )

    assert [chunk.chunk_id for chunk in statistics.largest_chunks] == [
        "large",
        "medium",
    ]

    assert statistics.largest_chunks[0].char_count == 10
    assert statistics.largest_chunks[0].relative_path == "large.md"


def test_largest_chunk_contains_location_information() -> None:
    chunk = make_chunk(
        chunk_id="chunk-1",
        relative_path="notes/rag.md",
        section_path=("RAG", "Chunking"),
        content="line 1\nline 2",
        start_line=10,
        end_line=11,
    )

    statistics = calculate_chunk_statistics(
        [chunk],
        largest_limit=1,
    )

    largest = statistics.largest_chunks[0]

    assert largest.chunk_id == "chunk-1"
    assert largest.relative_path == "notes/rag.md"
    assert largest.section_path == ("RAG", "Chunking")
    assert largest.char_count == len(chunk.content)
    assert largest.line_count == 2
    assert largest.start_line == 10
    assert largest.end_line == 11


def test_largest_limit_zero_returns_no_largest_chunks() -> None:
    statistics = calculate_chunk_statistics(
        [make_chunk()],
        largest_limit=0,
    )

    assert statistics.largest_chunks == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"max_chars": 0},
            "max_chars must be positive",
        ),
        (
            {"max_chars": -1},
            "max_chars must be positive",
        ),
        (
            {"max_lines": 0},
            "max_lines must be positive",
        ),
        (
            {"max_lines": -1},
            "max_lines must be positive",
        ),
        (
            {"largest_limit": -1},
            "largest_limit must be non-negative",
        ),
        (
            {"smallest_limit": -1},
            "smallest_limit must be non-negative",
        ),
    ],
)
def test_invalid_statistics_parameters_raise_error(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        calculate_chunk_statistics(
            [make_chunk()],
            **kwargs,
        )


def test_calculates_statistics_from_jsonl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunks.jsonl"

    chunks = [
        make_chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="short",
        ),
        make_chunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            content="line 1\nline 2",
            chunk_index=1,
        ),
    ]

    write_chunks_jsonl(chunks, path)

    statistics = calculate_chunk_statistics_from_jsonl(
        path,
        largest_limit=1,
    )

    assert statistics.chunk_count == 2
    assert statistics.document_count == 2
    assert statistics.max_line_count == 2
    assert len(statistics.largest_chunks) == 1


def test_statistics_from_jsonl_rejects_missing_file(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.jsonl"

    with pytest.raises(
        FileNotFoundError,
        match="Chunks JSONL does not exist",
    ):
        calculate_chunk_statistics_from_jsonl(missing_path)


def test_statistics_from_jsonl_rejects_directory(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "chunks"
    directory.mkdir()

    with pytest.raises(
        IsADirectoryError,
        match="Chunks JSONL path is not a file",
    ):
        calculate_chunk_statistics_from_jsonl(directory)
        

def test_empty_chunk_content_has_zero_lines() -> None:
    chunk = make_chunk(content="")

    statistics = calculate_chunk_statistics([chunk])

    assert statistics.min_line_count == 0
    assert statistics.average_line_count == 0.0
    assert statistics.max_line_count == 0
    
    
def test_smallest_chunks_are_sorted_by_character_count() -> None:
    chunks = [
        make_chunk(
            chunk_id="medium",
            content="abcdef",
        ),
        make_chunk(
            chunk_id="small",
            content="abc",
            chunk_index=1,
        ),
        make_chunk(
            chunk_id="large",
            content="abcdefghij",
            chunk_index=2,
        ),
    ]

    statistics = calculate_chunk_statistics(
        chunks,
        smallest_limit=2,
    )

    assert [chunk.chunk_id for chunk in statistics.smallest_chunks] == [
        "small",
        "medium",
    ]
    

def test_smallest_limit_zero_returns_no_smallest_chunks() -> None:
    statistics = calculate_chunk_statistics(
        [make_chunk()],
        smallest_limit=0,
    )

    assert statistics.smallest_chunks == ()