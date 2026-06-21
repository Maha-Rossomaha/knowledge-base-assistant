import pytest

from knowledge_base_assistant.domain.models import (
    Document,
    MarkdownSection,
)
from knowledge_base_assistant.ingestion.chunker import (
    ChunkerConfig,
    _is_separator_only,
    chunk_sections,
)


def make_document(
    *,
    document_id: str = "doc-1",
    content: str = "",
) -> Document:
    return Document(
        document_id=document_id,
        source_name="test-source",
        relative_path="notes/test.md",
        content=content,
        content_hash="document-hash",
    )


def make_section(
    content: str,
    *,
    document_id: str = "doc-1",
    section_path: tuple[str, ...] = ("Heading",),
    start_line: int = 1,
    content_start_line: int | None = 2,
    end_line: int | None = None,
) -> MarkdownSection:
    if end_line is None:
        if content and content_start_line is not None:
            end_line = content_start_line + len(content.splitlines()) - 1
        else:
            end_line = start_line

    return MarkdownSection(
        document_id=document_id,
        section_path=section_path,
        content=content,
        start_line=start_line,
        content_start_line=content_start_line,
        end_line=end_line,
    )


def test_empty_section_produces_no_chunks() -> None:
    document = make_document()
    section = make_section(
        "",
        content_start_line=None,
    )

    chunks = chunk_sections(document, [section])

    assert chunks == []


def test_short_section_produces_single_chunk() -> None:
    document = make_document()
    section = make_section(
        "First line\nSecond line",
        content_start_line=3,
    )

    chunks = chunk_sections(document, [section])

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.document_id == document.document_id
    assert chunk.source_name == document.source_name
    assert chunk.relative_path == document.relative_path
    assert chunk.section_path == ("Heading",)
    assert chunk.title == "Heading"
    assert chunk.content == "First line\nSecond line"
    assert chunk.chunk_index == 0
    assert chunk.start_line == 3
    assert chunk.end_line == 4
    assert chunk.chunk_id
    assert chunk.content_hash


def test_section_without_heading_has_no_title() -> None:
    document = make_document()
    section = make_section(
        "Content",
        section_path=(),
        start_line=1,
        content_start_line=1,
    )

    chunks = chunk_sections(document, [section])

    assert chunks[0].title is None
    assert chunks[0].searchable_text == "Content"


def test_searchable_text_contains_heading_context() -> None:
    document = make_document()
    section = make_section(
        "Chunk content",
        section_path=("RAG", "Chunking"),
    )

    chunks = chunk_sections(document, [section])

    assert chunks[0].searchable_text == (
        "RAG > Chunking\n\nChunk content"
    )


def test_chunks_are_split_by_max_lines() -> None:
    document = make_document()
    section = make_section(
        "line 1\nline 2\nline 3\nline 4\nline 5",
        content_start_line=10,
    )
    config = ChunkerConfig(
        max_lines=2,
        max_chars=1_000,
        overlap_lines=0,
    )

    chunks = chunk_sections(document, [section], config)

    assert [chunk.content for chunk in chunks] == [
        "line 1\nline 2",
        "line 3\nline 4",
        "line 5",
    ]

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (10, 11),
        (12, 13),
        (14, 14),
    ]


def test_chunks_are_split_by_max_chars() -> None:
    document = make_document()
    section = make_section(
        "abcd\nefgh\nij",
        content_start_line=5,
    )
    config = ChunkerConfig(
        max_lines=100,
        max_chars=6,
        overlap_lines=0,
    )

    chunks = chunk_sections(document, [section], config)

    assert [chunk.content for chunk in chunks] == [
        "abcd",
        "efgh",
        "ij",
    ]

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (5, 5),
        (6, 6),
        (7, 7),
    ]


def test_very_long_single_line_is_still_included() -> None:
    document = make_document()
    section = make_section(
        "a" * 100,
        content_start_line=7,
    )
    config = ChunkerConfig(
        max_lines=10,
        max_chars=20,
        overlap_lines=0,
    )

    chunks = chunk_sections(document, [section], config)

    assert len(chunks) == 1
    assert chunks[0].content == "a" * 100
    assert chunks[0].start_line == 7
    assert chunks[0].end_line == 7


def test_chunks_have_line_overlap() -> None:
    document = make_document()
    section = make_section(
        "line 1\nline 2\nline 3\nline 4\nline 5",
        content_start_line=1,
    )
    config = ChunkerConfig(
        max_lines=3,
        max_chars=1_000,
        overlap_lines=1,
    )

    chunks = chunk_sections(document, [section], config)

    assert [chunk.content for chunk in chunks] == [
        "line 1\nline 2\nline 3",
        "line 3\nline 4\nline 5",
    ]

    assert chunks[0].end_line == 3
    assert chunks[1].start_line == 3


def test_chunk_indexes_are_continuous_across_sections() -> None:
    document = make_document()

    first_section = make_section(
        "first",
        section_path=("First",),
        content_start_line=2,
    )
    second_section = make_section(
        "second",
        section_path=("Second",),
        start_line=4,
        content_start_line=5,
    )

    chunks = chunk_sections(
        document,
        [first_section, second_section],
    )

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert [chunk.title for chunk in chunks] == ["First", "Second"]


def test_empty_sections_do_not_affect_chunk_indexes() -> None:
    document = make_document()

    empty_section = make_section(
        "",
        section_path=("Empty",),
        content_start_line=None,
    )
    non_empty_section = make_section(
        "content",
        section_path=("Non-empty",),
    )

    chunks = chunk_sections(
        document,
        [empty_section, non_empty_section],
    )

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0


def test_code_fence_is_not_split_when_limit_is_reached_inside_it() -> None:
    document = make_document()
    section = make_section(
        "Before\n"
        "```python\n"
        "first = 1\n"
        "second = 2\n"
        "```\n"
        "After",
        content_start_line=1,
    )
    config = ChunkerConfig(
        max_lines=3,
        max_chars=1_000,
        overlap_lines=0,
    )

    chunks = chunk_sections(document, [section], config)

    assert chunks[0].content == (
        "Before\n"
        "```python\n"
        "first = 1\n"
        "second = 2\n"
        "```"
    )
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 5

    assert chunks[1].content == "After"
    assert chunks[1].start_line == 6


def test_tilde_code_fence_is_not_split() -> None:
    document = make_document()
    section = make_section(
        "Before\n"
        "~~~text\n"
        "definition\n"
        "another line\n"
        "~~~\n"
        "After",
        content_start_line=1,
    )
    config = ChunkerConfig(
        max_lines=3,
        max_chars=1_000,
        overlap_lines=0,
    )

    chunks = chunk_sections(document, [section], config)

    assert "~~~text\ndefinition\nanother line\n~~~" in chunks[0].content
    assert chunks[1].content == "After"


def test_unclosed_code_fence_is_included_until_section_end() -> None:
    document = make_document()
    section = make_section(
        "Before\n"
        "```python\n"
        "first = 1\n"
        "second = 2\n"
        "third = 3",
        content_start_line=1,
    )
    config = ChunkerConfig(
        max_lines=2,
        max_chars=1_000,
        overlap_lines=0,
    )

    chunks = chunk_sections(document, [section], config)

    assert len(chunks) == 1
    assert chunks[0].content.endswith("third = 3")
    assert chunks[0].end_line == 5


def test_short_fence_is_repeated_when_overlap_enters_it() -> None:
    document = make_document()
    section = make_section(
        "Before\n"
        "```text\n"
        "Definition\n"
        "```\n"
        "After one\n"
        "After two",
        content_start_line=1,
    )
    config = ChunkerConfig(
        max_lines=3,
        max_chars=1_000,
        overlap_lines=2,
        max_fence_overlap_lines=3,
    )

    chunks = chunk_sections(document, [section], config)

    assert chunks[0].content == (
        "Before\n"
        "```text\n"
        "Definition\n"
        "```"
    )

    assert chunks[1].content.startswith(
        "```text\nDefinition\n```"
    )


def test_large_fence_is_not_repeated_when_overlap_enters_it() -> None:
    document = make_document()
    section = make_section(
        "Before\n"
        "```python\n"
        "line 1\n"
        "line 2\n"
        "line 3\n"
        "line 4\n"
        "```\n"
        "After one\n"
        "After two",
        content_start_line=1,
    )
    config = ChunkerConfig(
        max_lines=3,
        max_chars=1_000,
        overlap_lines=2,
        max_fence_overlap_lines=3,
    )

    chunks = chunk_sections(document, [section], config)

    assert chunks[0].content == (
        "Before\n"
        "```python\n"
        "line 1\n"
        "line 2\n"
        "line 3\n"
        "line 4\n"
        "```"
    )

    assert chunks[1].content == "After one\nAfter two"
    assert "```python" not in chunks[1].content


def test_section_document_id_must_match_document() -> None:
    document = make_document(document_id="doc-1")
    section = make_section(
        "content",
        document_id="doc-2",
    )

    with pytest.raises(
        ValueError,
        match="Section document_id does not match",
    ):
        chunk_sections(document, [section])


def test_non_empty_section_requires_content_start_line() -> None:
    document = make_document()
    section = make_section(
        "content",
        content_start_line=None,
    )

    with pytest.raises(
        ValueError,
        match="Non-empty section must have content_start_line",
    ):
        chunk_sections(document, [section])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_lines": 0}, "max_lines must be positive"),
        ({"max_lines": -1}, "max_lines must be positive"),
        ({"max_chars": 0}, "max_chars must be positive"),
        ({"max_chars": -1}, "max_chars must be positive"),
        (
            {"overlap_lines": -1},
            "overlap_lines must be non-negative",
        ),
        (
            {"max_lines": 5, "overlap_lines": 5},
            "overlap_lines must be smaller than max_lines",
        ),
        (
            {"max_lines": 5, "overlap_lines": 6},
            "overlap_lines must be smaller than max_lines",
        ),
        (
            {"max_fence_overlap_lines": -1},
            "max_fence_overlap_lines must be non-negative",
        ),
    ],
)
def test_invalid_chunker_config_raises_error(
    kwargs: dict[str, int],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ChunkerConfig(**kwargs)


def test_different_fence_marker_inside_code_block_does_not_close_it() -> None:
    document = make_document()
    section = make_section(
        "```python\n"
        "value = 1\n"
        "~~~\n"
        "value = 2\n"
        "```\n"
        "After",
        content_start_line=1,
    )
    config = ChunkerConfig(
        max_lines=2,
        max_chars=1_000,
        overlap_lines=0,
    )

    chunks = chunk_sections(document, [section], config)

    assert chunks[0].content == (
        "```python\n"
        "value = 1\n"
        "~~~\n"
        "value = 2\n"
        "```"
    )
    assert chunks[1].content == "After"
    

def test_section_chunk_indexes_restart_for_each_section() -> None:
    document = make_document()

    first_section = make_section(
        "a\nb",
        section_path=("First",),
    )
    second_section = make_section(
        "c\nd",
        section_path=("Second",),
    )

    config = ChunkerConfig(
        max_lines=1,
        max_chars=1_000,
        overlap_lines=0,
    )

    chunks = chunk_sections(
        document,
        [first_section, second_section],
        config,
    )

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2, 3]
    assert [chunk.section_chunk_index for chunk in chunks] == [0, 1, 0, 1]
    

def test_chunker_removes_trailing_empty_lines_from_chunk() -> None:
    document = make_document()
    section = make_section(
        "line 1\n"
        "line 2\n"
        "\n"
        "\n"
        "line 5",
        content_start_line=10,
    )
    config = ChunkerConfig(
        max_lines=4,
        max_chars=1_000,
        overlap_lines=0,
    )

    chunks = chunk_sections(document, [section], config)

    assert chunks[0].content == "line 1\nline 2"
    assert chunks[0].start_line == 10
    assert chunks[0].end_line == 11

    assert chunks[1].content == "line 5"
    assert chunks[1].start_line == 14
    assert chunks[1].end_line == 14
    
    
def test_separator_only_section_does_not_create_chunk() -> None:
    document = make_document(content="---")

    section = MarkdownSection(
        document_id=document.document_id,
        section_path=("Section",),
        content="---",
        start_line=1,
        content_start_line=1,
        end_line=1,
    )

    chunks = chunk_sections(
        document,
        [section],
        ChunkerConfig(),
    )

    assert chunks == []
    
    
def test_short_meaningful_content_is_preserved() -> None:
    document = make_document(content="GQA")

    section = MarkdownSection(
        document_id=document.document_id,
        section_path=("Attention",),
        content="GQA",
        start_line=1,
        content_start_line=1,
        end_line=1,
    )

    chunks = chunk_sections(
        document,
        [section],
        ChunkerConfig(),
    )

    assert len(chunks) == 1
    assert chunks[0].content == "GQA"
    

def test_separator_does_not_consume_section_chunk_index() -> None:
    document = make_document(
        content="---\n\nuseful content",
    )

    section = MarkdownSection(
        document_id=document.document_id,
        section_path=("Section",),
        content="---\n\nuseful content",
        start_line=1,
        content_start_line=1,
        end_line=3,
    )

    chunks = chunk_sections(
        document,
        [section],
        ChunkerConfig(
            max_lines=1,
            max_chars=100,
            overlap_lines=0,
        ),
    )

    assert len(chunks) == 1
    assert chunks[0].content == "useful content"
    assert chunks[0].chunk_index == 0
    assert chunks[0].section_chunk_index == 0
    
    
@pytest.mark.parametrize(
    "content",
    [
        "",
        "   ",
        "\n\n",
        " \n\t\n ",
    ],
)
def test_blank_content_is_separator_only(content: str) -> None:
    assert _is_separator_only(content) is True