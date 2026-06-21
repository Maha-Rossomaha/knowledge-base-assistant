from pathlib import Path

from typer.testing import CliRunner

from knowledge_base_assistant.cli import app
from knowledge_base_assistant.serialization.jsonl import (
    read_chunks_jsonl,
)

runner = CliRunner()


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
    monkeypatch,
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