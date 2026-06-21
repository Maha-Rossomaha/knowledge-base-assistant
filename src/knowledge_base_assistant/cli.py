from pathlib import Path
from typing import Annotated

import typer

from knowledge_base_assistant.application.indexing import build_chunks
from knowledge_base_assistant.ingestion.chunker import ChunkerConfig

app = typer.Typer(
    name="copilot",
    help="Search assistant for a Markdown knowledge base.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Knowledge-base assistant CLI."""


@app.command()
def index(
    repository: Annotated[
        Path,
        typer.Argument(
            help="Path to the Markdown knowledge-base repository.",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ],
    source_name: Annotated[
        str,
        typer.Option(
            "--source-name",
            help="Stable logical name of the knowledge source.",
        ),
    ] = "llm_projects",
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Path to the resulting chunks JSONL file.",
        ),
    ] = Path("artifacts/chunks.jsonl"),
    max_lines: Annotated[
        int,
        typer.Option(
            "--max-lines",
            help="Maximum number of lines in a chunk.",
            min=1,
        ),
    ] = 40,
    max_chars: Annotated[
        int,
        typer.Option(
            "--max-chars",
            help="Maximum number of characters in a chunk.",
            min=1,
        ),
    ] = 3_000,
    overlap_lines: Annotated[
        int,
        typer.Option(
            "--overlap-lines",
            help="Number of overlapping lines between chunks.",
            min=0,
        ),
    ] = 5,
    max_fence_overlap_lines: Annotated[
        int,
        typer.Option(
            "--max-fence-overlap-lines",
            help="Maximum fenced-block size that may be repeated in overlap.",
            min=0,
        ),
    ] = 10,
) -> None:
    """Build chunks.jsonl from a Markdown repository."""

    try:
        config = ChunkerConfig(
            max_lines=max_lines,
            max_chars=max_chars,
            overlap_lines=overlap_lines,
            max_fence_overlap_lines=max_fence_overlap_lines,
        )

        result = build_chunks(
            repository_path=repository,
            source_name=source_name,
            output_path=output,
            chunker_config=config,
        )
    except (ValueError, OSError) as error:
        typer.secho(
            f"Indexing failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.echo(f"Documents: {result.document_count}")
    typer.echo(f"Sections: {result.section_count}")
    typer.echo(f"Chunks: {result.chunk_count}")
    typer.echo(f"Output: {result.output_path}")