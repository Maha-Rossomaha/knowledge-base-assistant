from pathlib import Path

from pytest import MonkeyPatch
from typer.testing import CliRunner

from knowledge_base_assistant.cli import app
from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.serialization.jsonl import (
    read_chunks_jsonl,
    write_chunks_jsonl,
)

runner = CliRunner()


def make_statistics_chunk(
    *,
    chunk_id: str = "chunk-1",
    document_id: str = "doc-1",
    relative_path: str = "notes/test.md",
    section_path: tuple[str, ...] = ("Heading",),
    content: str = "content",
    chunk_index: int = 0,
    section_chunk_index: int = 0,
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
        start_line=1,
        end_line=len(content.splitlines()),
        content_hash=f"hash-{chunk_id}",
    )


def test_index_command_builds_chunks_jsonl(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "knowledge-base"
    repository.mkdir()

    markdown_file = repository / "search.md"
    markdown_file.write_text(
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

    result = runner.invoke(
        app,
        [
            "index",
            str(repository),
            "--source-name",
            "test-source",
            "--output",
            str(output),
            "--max-lines",
            "40",
            "--max-chars",
            "3000",
            "--overlap-lines",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()

    chunks = read_chunks_jsonl(output)

    assert len(chunks) == 2
    assert chunks[0].source_name == "test-source"
    assert chunks[0].section_path == ("Search",)
    assert chunks[1].section_path == ("Search", "BM25")

    assert "Documents: 1" in result.output
    assert "Sections: 2" in result.output
    assert "Chunks: 2" in result.output
    assert f"Output: {output}" in result.output


def test_index_command_uses_default_options(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    repository = tmp_path / "knowledge-base"
    repository.mkdir()

    (repository / "note.md").write_text(
        "# Note\n\nContent.\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)

    result = runner.invoke(
        app,
        [
            "index",
            str(repository),
        ],
    )

    assert result.exit_code == 0, result.output

    output = tmp_path / "artifacts" / "chunks.jsonl"

    assert output.exists()

    chunks = read_chunks_jsonl(output)

    assert len(chunks) == 1
    assert chunks[0].source_name == "llm_projects"


def test_index_command_rejects_missing_repository(
    tmp_path: Path,
) -> None:
    missing_repository = tmp_path / "missing"

    result = runner.invoke(
        app,
        [
            "index",
            str(missing_repository),
        ],
    )

    assert result.exit_code == 2
    assert "repository" in result.output.lower()


def test_index_command_reports_invalid_chunker_config(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "knowledge-base"
    repository.mkdir()

    (repository / "note.md").write_text(
        "# Note\n\nContent.\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "index",
            str(repository),
            "--max-lines",
            "5",
            "--overlap-lines",
            "5",
        ],
    )

    assert result.exit_code == 1
    assert "Indexing failed" in result.output
    assert "overlap_lines must be smaller than max_lines" in result.output
    
    
def test_stats_command_displays_chunk_statistics(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"

    chunks = [
        make_statistics_chunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            relative_path="notes/search.md",
            section_path=("Search",),
            content="short",
        ),
        make_statistics_chunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            relative_path="notes/code.md",
            section_path=("Python", "Example"),
            content=(
                "```python\n"
                "print('hello')\n"
                "```"
            ),
            chunk_index=1,
        ),
    ]

    write_chunks_jsonl(chunks, chunks_path)

    result = runner.invoke(
        app,
        [
            "stats",
            str(chunks_path),
            "--max-chars",
            "10",
            "--max-lines",
            "2",
            "--largest-limit",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output

    assert "Documents: 2" in result.output
    assert "Chunks: 2" in result.output

    assert "Characters:" in result.output
    assert "Lines:" in result.output

    assert "Chunks with code fence: 1" in result.output
    assert "Chunks over 10 chars: 1" in result.output
    assert "Chunks over 2 lines: 1" in result.output

    assert "Largest chunks:" in result.output
    assert "notes/code.md" in result.output
    assert "Python > Example" in result.output


def test_stats_command_with_zero_largest_limit_hides_largest_chunks(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"

    write_chunks_jsonl(
        [make_statistics_chunk()],
        chunks_path,
    )

    result = runner.invoke(
        app,
        [
            "stats",
            str(chunks_path),
            "--largest-limit",
            "0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Chunks: 1" in result.output
    assert "Largest chunks:" not in result.output


def test_stats_command_handles_empty_jsonl(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("", encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "stats",
            str(chunks_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Documents: 0" in result.output
    assert "Chunks: 0" in result.output
    assert "Average: 0.00" in result.output


def test_stats_command_rejects_missing_jsonl(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / "missing.jsonl"

    result = runner.invoke(
        app,
        [
            "stats",
            str(missing_path),
        ],
    )

    assert result.exit_code == 2
    assert "does not exist" in result.output.lower()


def test_stats_command_rejects_invalid_cli_limit(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"

    write_chunks_jsonl(
        [make_statistics_chunk()],
        chunks_path,
    )

    result = runner.invoke(
        app,
        [
            "stats",
            str(chunks_path),
            "--max-lines",
            "0",
        ],
    )

    assert result.exit_code == 2


def test_stats_command_reports_invalid_jsonl(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "not valid json\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "stats",
            str(chunks_path),
        ],
    )

    assert result.exit_code == 1
    assert "Statistics failed" in result.output
    assert "Invalid JSONL at line 1" in result.output