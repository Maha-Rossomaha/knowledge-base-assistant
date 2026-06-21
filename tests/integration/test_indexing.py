from pathlib import Path

import pytest

from knowledge_base_assistant.application.indexing import (
    build_chunks,
)
from knowledge_base_assistant.ingestion.chunker import (
    ChunkerConfig,
)
from knowledge_base_assistant.serialization.jsonl import (
    read_chunks_jsonl,
)


def test_build_chunks_creates_jsonl_and_returns_statistics(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "knowledge-base"
    repository.mkdir()

    (repository / "search.md").write_text(
        "# Search\n"
        "\n"
        "Search finds relevant information.\n"
        "\n"
        "## BM25\n"
        "\n"
        "BM25 is a lexical ranking algorithm.\n",
        encoding="utf-8",
    )

    output = tmp_path / "artifacts" / "chunks.jsonl"

    result = build_chunks(
        repository_path=repository,
        source_name="test-source",
        output_path=output,
        chunker_config=ChunkerConfig(),
    )

    assert result.document_count == 1
    assert result.section_count == 2
    assert result.chunk_count == 2
    assert result.output_path == output

    assert output.exists()

    chunks = read_chunks_jsonl(output)

    assert len(chunks) == 2
    assert chunks[0].source_name == "test-source"
    assert chunks[0].section_path == ("Search",)
    assert chunks[1].section_path == ("Search", "BM25")


def test_build_chunks_rejects_missing_repository(
    tmp_path: Path,
) -> None:
    missing_repository = tmp_path / "missing"

    with pytest.raises(
        FileNotFoundError,
        match="Repository path does not exist",
    ):
        build_chunks(
            repository_path=missing_repository,
            source_name="test-source",
            output_path=tmp_path / "chunks.jsonl",
            chunker_config=ChunkerConfig(),
        )


def test_build_chunks_rejects_file_instead_of_directory(
    tmp_path: Path,
) -> None:
    repository_file = tmp_path / "repository.md"
    repository_file.write_text(
        "# Not a repository\n",
        encoding="utf-8",
    )

    with pytest.raises(
        NotADirectoryError,
        match="Repository path is not a directory",
    ):
        build_chunks(
            repository_path=repository_file,
            source_name="test-source",
            output_path=tmp_path / "chunks.jsonl",
            chunker_config=ChunkerConfig(),
        )


def test_build_chunks_handles_empty_repository(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "empty-repository"
    repository.mkdir()

    output = tmp_path / "chunks.jsonl"

    result = build_chunks(
        repository_path=repository,
        source_name="test-source",
        output_path=output,
        chunker_config=ChunkerConfig(),
    )

    assert result.document_count == 0
    assert result.section_count == 0
    assert result.chunk_count == 0
    assert result.output_path == output

    assert output.exists()
    assert read_chunks_jsonl(output) == []