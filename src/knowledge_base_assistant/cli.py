from pathlib import Path
from typing import Annotated

import typer

from knowledge_base_assistant.application.golden_validation import validate_golden_files
from knowledge_base_assistant.application.indexing import build_chunks
from knowledge_base_assistant.application.lexical_retrieval_evaluation import (
    evaluate_bm25_retrieval,
)
from knowledge_base_assistant.application.lexical_search import search_chunks_file
from knowledge_base_assistant.application.statistics import calculate_chunk_statistics_from_jsonl
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
    

@app.command()
def stats(
    chunks_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the chunks JSONL file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    max_chars: Annotated[
        int,
        typer.Option(
            "--max-chars",
            help="Expected maximum number of characters in a chunk.",
            min=1,
        ),
    ] = 3_000,
    max_lines: Annotated[
        int,
        typer.Option(
            "--max-lines",
            help="Expected maximum number of lines in a chunk.",
            min=1,
        ),
    ] = 40,
    largest_limit: Annotated[
        int,
        typer.Option(
            "--largest-limit",
            help="Number of largest chunks to display.",
            min=0,
        ),
    ] = 10,
    smallest_limit: Annotated[
        int,
        typer.Option(
            "--smallest-limit",
            help="Number of smallest chunks to display.",
            min=0,
        ),
    ] = 20,
) -> None:
    """Show statistics for a chunks JSONL file."""

    try:
        statistics = calculate_chunk_statistics_from_jsonl(
            chunks_path,
            max_chars=max_chars,
            max_lines=max_lines,
            largest_limit=largest_limit,
            smallest_limit=smallest_limit,
        )
    except (ValueError, OSError) as error:
        typer.secho(
            f"Statistics failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.echo(f"Documents: {statistics.document_count}")
    typer.echo(f"Chunks: {statistics.chunk_count}")

    typer.echo("")
    typer.echo("Characters:")
    typer.echo(f"  Min: {statistics.min_char_count}")
    typer.echo(f"  Average: {statistics.average_char_count:.2f}")
    typer.echo(f"  Max: {statistics.max_char_count}")

    typer.echo("")
    typer.echo("Lines:")
    typer.echo(f"  Min: {statistics.min_line_count}")
    typer.echo(f"  Average: {statistics.average_line_count:.2f}")
    typer.echo(f"  Max: {statistics.max_line_count}")

    typer.echo("")
    typer.echo(
        f"Chunks with code fence: "
        f"{statistics.chunks_with_code_fence}"
    )
    typer.echo(
        f"Chunks over {max_chars} chars: "
        f"{statistics.chunks_over_max_chars}"
    )
    typer.echo(
        f"Chunks over {max_lines} lines: "
        f"{statistics.chunks_over_max_lines}"
    )

    if statistics.largest_chunks:
        typer.echo("")
        typer.echo("Largest chunks:")

        for position, chunk in enumerate(
            statistics.largest_chunks,
            start=1,
        ):
            section = (
                " > ".join(chunk.section_path)
                if chunk.section_path
                else "<no heading>"
            )

            typer.echo(
                f"{position}. {chunk.relative_path}:"
                f"{chunk.start_line}-{chunk.end_line}"
            )
            typer.echo(f"   Section: {section}")
            typer.echo(
                f"   Size: {chunk.char_count} chars, "
                f"{chunk.line_count} lines"
            )
    
    if statistics.smallest_chunks:
        typer.echo("")
        typer.echo("Smallest chunks:")

        for position, chunk in enumerate(
            statistics.smallest_chunks,
            start=1,
        ):
            section = (
                " > ".join(chunk.section_path)
                if chunk.section_path
                else "<no heading>"
            )

            typer.echo(
                f"{position}. {chunk.relative_path}:"
                f"{chunk.start_line}-{chunk.end_line}"
            )
            typer.echo(f"   Section: {section}")
            typer.echo(
                f"   Size: {chunk.char_count} chars, "
                f"{chunk.line_count} lines"
            )
            

@app.command("validate-golden")
def validate_golden(
    golden_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the golden queries JSONL file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    chunks_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the chunks JSONL file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    """Validate a golden retrieval dataset against chunks."""

    try:
        result = validate_golden_files(
            golden_path=golden_path,
            chunks_path=chunks_path,
        )
    except (ValueError, OSError) as error:
        typer.secho(
            f"Golden validation failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.secho(
        "Golden dataset is valid.",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Queries: {result.query_count}")
    typer.echo(
        f"Answerable queries: "
        f"{result.answerable_query_count}"
    )
    typer.echo(
        f"No-answer queries: "
        f"{result.no_answer_query_count}"
    )
    typer.echo(
        f"Relevance judgments: "
        f"{result.relevance_judgment_count}"
    )
    
    
@app.command()
def search(
    query: Annotated[
        str,
        typer.Argument(
            help="Search query.",
        ),
    ],
    chunks_path: Annotated[
        Path,
        typer.Option(
            "--chunks",
            help="Path to the chunks JSONL file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = Path("artifacts/chunks.jsonl"),
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            help="Maximum number of results.",
            min=1,
        ),
    ] = 5,
    k1: Annotated[
        float,
        typer.Option(
            "--k1",
            help="BM25 term-frequency saturation parameter.",
            min=0.000001,
        ),
    ] = 1.5,
    b: Annotated[
        float,
        typer.Option(
            "--b",
            help="BM25 document-length normalization parameter.",
            min=0.0,
            max=1.0,
        ),
    ] = 0.75,
    content_limit: Annotated[
        int,
        typer.Option(
            "--content-limit",
            help="Maximum number of content characters to display.",
            min=0,
        ),
    ] = 500,
) -> None:
    """Search chunks using BM25."""

    try:
        results = search_chunks_file(
            chunks_path=chunks_path,
            query=query,
            top_k=top_k,
            k1=k1,
            b=b,
        )
    except (ValueError, OSError) as error:
        typer.secho(
            f"Search failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error

    if not results:
        typer.echo("No matching chunks found.")
        return

    for result in results:
        section = (
            " > ".join(result.chunk.section_path)
            if result.chunk.section_path
            else "<no heading>"
        )

        content = result.chunk.content

        if content_limit == 0:
            content = ""
        elif len(content) > content_limit:
            content = (
                content[:content_limit].rstrip()
                + "..."
            )

        typer.echo(
            f"{result.rank}. Score: {result.score:.4f}"
        )
        typer.echo(
            f"   Path: {result.chunk.relative_path}"
        )
        typer.echo(f"   Section: {section}")
        typer.echo(
            f"   Chunk ID: {result.chunk.chunk_id}"
        )
        typer.echo("   Content:")
        typer.echo(
            "\n".join(
                f"     {line}"
                for line in content.splitlines()
            )
        )

        if result.rank != len(results):
            typer.echo("")
            
            
@app.command()
def evaluate(
    golden_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the golden queries JSONL file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    chunks_path: Annotated[
        Path,
        typer.Argument(
            help="Path to the chunks JSONL file.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            help="Number of retrieved chunks evaluated per query.",
            min=1,
        ),
    ] = 5,
    k1: Annotated[
        float,
        typer.Option(
            "--k1",
            help="BM25 term-frequency saturation parameter.",
            min=0.000001,
        ),
    ] = 1.5,
    b: Annotated[
        float,
        typer.Option(
            "--b",
            help="BM25 document-length normalization parameter.",
            min=0.0,
            max=1.0,
        ),
    ] = 0.75,
) -> None:
    """Evaluate BM25 retrieval against a golden dataset."""

    try:
        result = evaluate_bm25_retrieval(
            golden_path=golden_path,
            chunks_path=chunks_path,
            top_k=top_k,
            k1=k1,
            b=b,
        )
    except (ValueError, OSError) as error:
        typer.secho(
            f"Retrieval evaluation failed: {error}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from error

    typer.secho(
        "Retrieval evaluation completed.",
        fg=typer.colors.GREEN,
    )
    typer.echo(f"Top-K: {result.top_k}")
    typer.echo(f"Queries: {result.query_count}")
    typer.echo(
        f"Evaluated queries: "
        f"{result.evaluated_query_count}"
    )
    typer.echo(
        f"No-answer queries: "
        f"{result.no_answer_query_count}"
    )
    typer.echo("")
    typer.echo(
        f"HitRate@{result.top_k}: "
        f"{result.hit_rate_at_k:.4f}"
    )
    typer.echo(
        f"Recall@{result.top_k}: "
        f"{result.recall_at_k:.4f}"
    )
    typer.echo(
        f"MRR@{result.top_k}: "
        f"{result.mean_reciprocal_rank:.4f}"
    )
    typer.echo(
        f"nDCG@{result.top_k}: "
        f"{result.ndcg_at_k:.4f}"
    )
