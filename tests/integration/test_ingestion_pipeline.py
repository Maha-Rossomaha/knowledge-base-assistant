from pathlib import Path

from knowledge_base_assistant.ingestion.chunker import (
    ChunkerConfig,
    chunk_sections,
)
from knowledge_base_assistant.ingestion.markdown_parser import (
    parse_markdown_sections,
)
from knowledge_base_assistant.ingestion.scanner import scan_documents
from knowledge_base_assistant.serialization.jsonl import (
    read_chunks_jsonl,
    write_chunks_jsonl,
)


def test_markdown_file_passes_through_full_ingestion_pipeline(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "knowledge-base"
    notes_directory = repository_root / "notes"
    notes_directory.mkdir(parents=True)

    markdown_file = notes_directory / "retrieval.md"
    markdown_file.write_text(
        "# Retrieval\n"
        "\n"
        "Retrieval finds relevant documents.\n"
        "\n"
        "## BM25\n"
        "\n"
        "BM25 is a lexical ranking algorithm.\n"
        "It uses term frequencies.\n"
        "It also considers document length.\n",
        encoding="utf-8",
    )

    documents = scan_documents(
        repository_root,
        "test-source",
    )

    assert len(documents) == 1

    document = documents[0]

    assert document.source_name == "test-source"
    assert document.relative_path == "notes/retrieval.md"
    assert document.content.startswith("# Retrieval")
    assert document.document_id
    assert document.content_hash

    sections = parse_markdown_sections(document)

    assert len(sections) == 2

    assert sections[0].section_path == ("Retrieval",)
    assert sections[0].content == "Retrieval finds relevant documents."
    assert sections[0].start_line == 1
    assert sections[0].content_start_line == 3
    assert sections[0].end_line == 3

    assert sections[1].section_path == ("Retrieval", "BM25")
    assert sections[1].start_line == 5
    assert sections[1].content_start_line == 7
    assert sections[1].end_line == 9

    chunks = chunk_sections(
        document,
        sections,
        ChunkerConfig(
            max_lines=2,
            max_chars=1_000,
            overlap_lines=0,
        ),
    )

    assert len(chunks) == 3

    assert chunks[0].content == "Retrieval finds relevant documents."
    assert chunks[0].section_path == ("Retrieval",)
    assert chunks[0].title == "Retrieval"
    assert chunks[0].start_line == 3
    assert chunks[0].end_line == 3

    assert chunks[1].content == (
        "BM25 is a lexical ranking algorithm.\n"
        "It uses term frequencies."
    )
    assert chunks[1].section_path == ("Retrieval", "BM25")
    assert chunks[1].title == "BM25"
    assert chunks[1].start_line == 7
    assert chunks[1].end_line == 8
    assert chunks[1].searchable_text == (
        "Retrieval > BM25\n\n"
        "BM25 is a lexical ranking algorithm.\n"
        "It uses term frequencies."
    )

    assert chunks[2].content == (
        "It also considers document length."
    )
    assert chunks[2].start_line == 9
    assert chunks[2].end_line == 9

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert len({chunk.chunk_id for chunk in chunks}) == 3
    
    jsonl_path = tmp_path / "artifacts" / "chunks.jsonl"

    write_chunks_jsonl(chunks, jsonl_path)
    restored_chunks = read_chunks_jsonl(jsonl_path)

    assert restored_chunks == chunks
    assert [chunk.section_chunk_index for chunk in chunks] == [0, 0, 1]


def test_code_fence_remains_whole_in_full_ingestion_pipeline(
    tmp_path: Path,
) -> None:
    repository_root = tmp_path / "knowledge-base"
    repository_root.mkdir()

    markdown_file = repository_root / "python.md"
    markdown_file.write_text(
        "# Python\n"
        "\n"
        "Example:\n"
        "\n"
        "```python\n"
        "def add(left: int, right: int) -> int:\n"
        "    return left + right\n"
        "```\n"
        "\n"
        "The function returns a sum.\n",
        encoding="utf-8",
    )

    documents = scan_documents(
        repository_root,
        "test-source",
    )

    document = documents[0]
    sections = parse_markdown_sections(document)

    chunks = chunk_sections(
        document,
        sections,
        ChunkerConfig(
            max_lines=3,
            max_chars=1_000,
            overlap_lines=0,
        ),
    )

    assert len(chunks) == 2

    assert chunks[0].content == (
        "Example:\n"
        "\n"
        "```python\n"
        "def add(left: int, right: int) -> int:\n"
        "    return left + right\n"
        "```"
    )
    assert chunks[0].start_line == 3
    assert chunks[0].end_line == 8

    assert chunks[1].content == "The function returns a sum."
    assert chunks[1].start_line == 10
    assert chunks[1].end_line == 10

    for chunk in chunks:
        assert chunk.document_id == document.document_id
        assert chunk.relative_path == "python.md"
        assert chunk.source_name == "test-source"
        