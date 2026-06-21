from knowledge_base_assistant.domain.models import Document
from knowledge_base_assistant.ingestion.markdown_parser import parse_markdown_sections


def make_document(content: str) -> Document:
    return Document(
        document_id="doc-1",
        source_name="test-source",
        relative_path="test.md",
        content=content,
        content_hash="hash",
    )


def test_parser_parses_single_heading() -> None:
    document = make_document(
        "# RAG\n"
        "\n"
        "Введение."
    )

    sections = parse_markdown_sections(document)

    assert len(sections) == 1
    assert sections[0].document_id == "doc-1"
    assert sections[0].section_path == ("RAG",)
    assert sections[0].content == "Введение."
    assert sections[0].start_line == 1
    assert sections[0].content_start_line == 3
    assert sections[0].end_line == 3


def test_parser_parses_nested_headings() -> None:
    document = make_document(
        "# RAG\n"
        "Введение.\n"
        "## Chunking\n"
        "Описание.\n"
        "### Semantic chunking\n"
        "Подробности."
    )

    sections = parse_markdown_sections(document)

    assert [section.section_path for section in sections] == [
        ("RAG",),
        ("RAG", "Chunking"),
        ("RAG", "Chunking", "Semantic chunking"),
    ]

    assert [section.content for section in sections] == [
        "Введение.",
        "Описание.",
        "Подробности.",
    ]


def test_parser_returns_to_previous_heading_level() -> None:
    document = make_document(
        "# RAG\n"
        "## Chunking\n"
        "### Semantic chunking\n"
        "Текст.\n"
        "## Retrieval\n"
        "Другой текст."
    )

    sections = parse_markdown_sections(document)

    assert [section.section_path for section in sections] == [
        ("RAG",),
        ("RAG", "Chunking"),
        ("RAG", "Chunking", "Semantic chunking"),
        ("RAG", "Retrieval"),
    ]


def test_parser_handles_document_without_headings() -> None:
    document = make_document(
        "Первая строка.\n"
        "Вторая строка."
    )

    sections = parse_markdown_sections(document)

    assert len(sections) == 1
    assert sections[0].section_path == ()
    assert sections[0].content == "Первая строка.\nВторая строка."
    assert sections[0].start_line == 1
    assert sections[0].content_start_line == 1
    assert sections[0].end_line == 2


def test_parser_keeps_empty_sections() -> None:
    document = make_document(
        "# RAG\n"
        "## Chunking\n"
        "### Semantic chunking\n"
        "\n"
        "Подробности."
    )

    sections = parse_markdown_sections(document)

    assert sections[0].section_path == ("RAG",)
    assert sections[0].content_start_line is None
    assert sections[0].content == ""

    assert sections[1].section_path == ("RAG", "Chunking")
    assert sections[1].content_start_line is None
    assert sections[1].content == ""

    assert sections[2].section_path == (
        "RAG",
        "Chunking",
        "Semantic chunking",
    )
    assert sections[2].content_start_line == 5
    assert sections[2].content == "Подробности."


def test_parser_does_not_treat_heading_inside_code_fence_as_heading() -> None:
    document = make_document(
        "# Example\n"
        "\n"
        "```python\n"
        "# not a heading\n"
        "print('hello')\n"
        "```\n"
        "\n"
        "После кода."
    )

    sections = parse_markdown_sections(document)

    assert len(sections) == 1
    assert sections[0].section_path == ("Example",)
    assert "# not a heading" in sections[0].content


def test_parser_sets_correct_line_ranges() -> None:
    document = make_document(
        "# First\n"
        "\n"
        "First content.\n"
        "## Second\n"
        "\n"
        "Second content."
    )

    sections = parse_markdown_sections(document)

    assert sections[0].start_line == 1
    assert sections[0].content_start_line == 3
    assert sections[0].end_line == 3

    assert sections[1].start_line == 4
    assert sections[1].content_start_line == 6
    assert sections[1].end_line == 6
    

def test_parser_skips_leading_empty_lines_when_setting_content_start_line() -> None:
    document = make_document(
        "# Heading\n"
        "\n"
        "\n"
        "Content."
    )

    sections = parse_markdown_sections(document)

    assert sections[0].start_line == 1
    assert sections[0].content_start_line == 4
    assert sections[0].end_line == 4


def test_parser_returns_empty_list_for_empty_document() -> None:
    document = make_document("")

    sections = parse_markdown_sections(document)

    assert sections == []


def test_parser_removes_trailing_empty_lines_from_section() -> None:
    document = make_document(
        "# Heading\n"
        "\n"
        "Content.\n"
        "\n"
        "\n"
    )

    sections = parse_markdown_sections(document)

    assert len(sections) == 1
    assert sections[0].content == "Content."
    assert sections[0].start_line == 1
    assert sections[0].content_start_line == 3
    assert sections[0].end_line == 3