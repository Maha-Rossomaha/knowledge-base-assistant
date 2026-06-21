from pathlib import Path

import pytest

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.serialization.jsonl import (
    read_chunks_jsonl,
    write_chunks_jsonl,
)


def make_chunk(
    *,
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    title: str | None = "Chunking",
    section_path: tuple[str, ...] = ("RAG", "Chunking"),
    content: str = "Chunk content",
    searchable_text: str = "RAG > Chunking\n\nChunk content",
    chunk_index: int = 0,
    start_line: int = 10,
    end_line: int = 10,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        source_name="test-source",
        relative_path="notes/rag.md",
        title=title,
        section_path=section_path,
        content=content,
        searchable_text=searchable_text,
        chunk_index=chunk_index,
        start_line=start_line,
        end_line=end_line,
        content_hash=f"hash-{chunk_id}",
    )


def test_write_and_read_chunks_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"

    original_chunks = [
        make_chunk(),
        make_chunk(
            chunk_id="chunk-2",
            title="BM25",
            section_path=("Search", "BM25"),
            content="BM25 content",
            searchable_text="Search > BM25\n\nBM25 content",
            chunk_index=1,
            start_line=20,
            end_line=21,
        ),
    ]

    write_chunks_jsonl(original_chunks, path)
    restored_chunks = read_chunks_jsonl(path)

    assert restored_chunks == original_chunks


def test_write_preserves_unicode_and_newlines(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"

    chunk = make_chunk(
        content="Определение чанкинга\nВторая строка",
        searchable_text=(
            "RAG > Чанкинг\n\n"
            "Определение чанкинга\n"
            "Вторая строка"
        ),
    )

    write_chunks_jsonl([chunk], path)

    raw_content = path.read_text(encoding="utf-8")
    restored_chunks = read_chunks_jsonl(path)

    assert "Определение чанкинга" in raw_content
    assert "\\u041e" not in raw_content
    assert len(raw_content.splitlines()) == 1
    assert restored_chunks == [chunk]


def test_write_creates_parent_directories(tmp_path: Path) -> None:
    path = tmp_path / "artifacts" / "ingestion" / "chunks.jsonl"

    write_chunks_jsonl([make_chunk()], path)

    assert path.exists()
    assert read_chunks_jsonl(path) == [make_chunk()]


def test_empty_chunks_create_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"

    write_chunks_jsonl([], path)

    assert path.exists()
    assert path.read_text(encoding="utf-8") == ""
    assert read_chunks_jsonl(path) == []


def test_reader_skips_empty_lines(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"

    chunk = make_chunk()
    write_chunks_jsonl([chunk], path)

    original_content = path.read_text(encoding="utf-8")
    path.write_text(
        f"\n{original_content}\n",
        encoding="utf-8",
    )

    assert read_chunks_jsonl(path) == [chunk]


def test_invalid_json_raises_error_with_line_number(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "\n"
        "not valid json\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSONL at line 2",
    ):
        read_chunks_jsonl(path)


def test_json_value_must_be_an_object(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        '["not", "an", "object"]\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=r"Invalid chunk at line 1: JSON value must be an object",
    ):
        read_chunks_jsonl(path)


def test_missing_required_field_raises_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        '{"chunk_id":"chunk-1"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid chunk at line 1",
    ):
        read_chunks_jsonl(path)


def test_section_path_must_be_a_list(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"

    chunk = make_chunk()
    write_chunks_jsonl([chunk], path)

    raw_content = path.read_text(encoding="utf-8")
    raw_content = raw_content.replace(
        '"section_path":["RAG","Chunking"]',
        '"section_path":"RAG > Chunking"',
    )
    path.write_text(raw_content, encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=(
            r"Invalid chunk at line 1: "
            r"section_path must be a list"
        ),
    ):
        read_chunks_jsonl(path)


def test_section_path_is_restored_as_tuple(tmp_path: Path) -> None:
    path = tmp_path / "chunks.jsonl"

    write_chunks_jsonl([make_chunk()], path)
    restored_chunk = read_chunks_jsonl(path)[0]

    assert restored_chunk.section_path == ("RAG", "Chunking")
    assert isinstance(restored_chunk.section_path, tuple)


def test_reader_reports_correct_invalid_line(
    tmp_path: Path,
) -> None:
    path = tmp_path / "chunks.jsonl"

    valid_chunk = make_chunk()
    write_chunks_jsonl([valid_chunk], path)

    valid_line = path.read_text(encoding="utf-8")
    path.write_text(
        f"{valid_line}\nnot json\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSONL at line 3",
    ):
        read_chunks_jsonl(path)