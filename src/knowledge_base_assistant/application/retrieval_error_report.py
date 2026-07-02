import re
from dataclasses import dataclass
from pathlib import Path

from knowledge_base_assistant.application.retrieval_comparison import (
    ComparedQueryResult,
    compare_retrieval_results,
)
from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
)
from knowledge_base_assistant.evaluation.serialization import (
    read_query_evaluation_results_jsonl,
)
from knowledge_base_assistant.retrieval.dense.serialization import (
    read_dense_index_metadata,
)
from knowledge_base_assistant.serialization.jsonl import (
    read_golden_queries_jsonl,
)


@dataclass(frozen=True, slots=True)
class RetrievalErrorReportResult:
    output_path: Path
    total_query_count: int
    answerable_query_count: int
    dense_only_hits_count: int
    lexical_only_hits_count: int
    both_hit_count: int
    both_miss_count: int


def generate_retrieval_error_report(
    *,
    golden_path: Path,
    lexical_results_path: Path,
    dense_results_path: Path,
    metadata_path: Path,
    output_root: Path,
    top_k: int,
    content_limit: int = 1_500,
) -> RetrievalErrorReportResult:
    if top_k < 1:
        raise ValueError(
            f"top_k must be at least 1, got {top_k}"
        )

    if content_limit < 0:
        raise ValueError(
            "content_limit must be non-negative, "
            f"got {content_limit}"
        )

    golden_queries = read_golden_queries_jsonl(
        golden_path
    )
    lexical_results = (
        read_query_evaluation_results_jsonl(
            lexical_results_path
        )
    )
    dense_results = (
        read_query_evaluation_results_jsonl(
            dense_results_path
        )
    )
    metadata = read_dense_index_metadata(
        metadata_path
    )

    comparison = compare_retrieval_results(
        lexical_results=lexical_results,
        dense_results=dense_results,
    )

    total_query_count = len(golden_queries)
    answerable_query_count = sum(
        bool(query.relevant_chunks)
        for query in golden_queries
    )

    compared_query_count = (
        len(comparison.dense_only_hits)
        + len(comparison.lexical_only_hits)
        + len(comparison.both_hit)
        + len(comparison.both_miss)
    )

    if compared_query_count != answerable_query_count:
        raise ValueError(
            "Comparison query count does not match "
            "answerable golden query count: "
            f"{compared_query_count} != "
            f"{answerable_query_count}"
        )

    output_path = (
        output_root
        / f"top_{top_k}"
        / "error_analysis.md"
    )

    lines = _build_report_lines(
        total_query_count=total_query_count,
        answerable_query_count=answerable_query_count,
        top_k=top_k,
        dense_model_name=(
            metadata.embedding_model.model_name
        ),
        dense_only_hits=comparison.dense_only_hits,
        lexical_only_hits=(
            comparison.lexical_only_hits
        ),
        both_hit_count=len(comparison.both_hit),
        both_miss=comparison.both_miss,
        content_limit=content_limit,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )

    return RetrievalErrorReportResult(
        output_path=output_path,
        total_query_count=total_query_count,
        answerable_query_count=answerable_query_count,
        dense_only_hits_count=len(
            comparison.dense_only_hits
        ),
        lexical_only_hits_count=len(
            comparison.lexical_only_hits
        ),
        both_hit_count=len(comparison.both_hit),
        both_miss_count=len(comparison.both_miss),
    )


def _build_report_lines(
    *,
    total_query_count: int,
    answerable_query_count: int,
    top_k: int,
    dense_model_name: str,
    dense_only_hits: tuple[
        ComparedQueryResult,
        ...,
    ],
    lexical_only_hits: tuple[
        ComparedQueryResult,
        ...,
    ],
    both_hit_count: int,
    both_miss: tuple[
        ComparedQueryResult,
        ...,
    ],
    content_limit: int,
) -> list[str]:
    lines = [
        "# Retrieval Error Analysis",
        "",
        "## Контекст",
        "",
        f"- Golden queries: {total_query_count}",
        (
            "- Answerable queries: "
            f"{answerable_query_count}"
        ),
        f"- Top-K: {top_k}",
        "- Lexical retriever: BM25",
        (
            "- Dense retriever: "
            f"`{dense_model_name}`"
        ),
        "",
        "## Сводка",
        "",
        "| Группа | Количество |",
        "|---|---:|",
        (
            "| Dense only hit | "
            f"{len(dense_only_hits)} |"
        ),
        (
            "| Lexical only hit | "
            f"{len(lexical_only_hits)} |"
        ),
        f"| Both hit | {both_hit_count} |",
        f"| Both miss | {len(both_miss)} |",
        "",
        "## Категории ошибок",
        "",
        "- `vocabulary_mismatch`",
        "- `exact_term_needed`",
        "- `ranking_error`",
        "- `chunk_boundary`",
        "- `context_missing`",
        "- `duplicate_or_similar_chunks`",
        "- `golden_incomplete`",
        "- `golden_incorrect`",
        "- `query_ambiguous`",
        "- `content_missing`",
        "",
    ]

    _append_group(
        lines=lines,
        title="Both miss",
        results=both_miss,
        top_k=top_k,
        content_limit=content_limit,
        hypothesis=(
            "Требуется ручная проверка. Возможны "
            "`ranking_error`, `golden_incomplete`, "
            "`chunk_boundary` или `content_missing`."
        ),
    )
    _append_group(
        lines=lines,
        title="Lexical only hit",
        results=lexical_only_hits,
        top_k=top_k,
        content_limit=content_limit,
        hypothesis=(
            "Предварительная гипотеза: "
            "`exact_term_needed`."
        ),
    )
    _append_group(
        lines=lines,
        title="Dense only hit",
        results=dense_only_hits,
        top_k=top_k,
        content_limit=content_limit,
        hypothesis=(
            "Предварительная гипотеза: "
            "`vocabulary_mismatch`."
        ),
    )

    lines.extend(
        [
            "## Итоговые наблюдения",
            "",
            "### Где Dense сильнее",
            "",
            "TODO",
            "",
            "### Где BM25 сильнее",
            "",
            "TODO",
            "",
            "### Ошибки golden dataset",
            "",
            "TODO",
            "",
            "### Ошибки chunking",
            "",
            "TODO",
            "",
            "### Ограничения обоих retriever-ов",
            "",
            "TODO",
            "",
            "### Вывод для Hybrid Retrieval",
            "",
            "TODO",
            "",
        ]
    )

    return lines


def _append_group(
    *,
    lines: list[str],
    title: str,
    results: tuple[ComparedQueryResult, ...],
    top_k: int,
    content_limit: int,
    hypothesis: str,
) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
        ]
    )

    if not results:
        lines.extend(
            [
                "В этой группе нет запросов.",
                "",
            ]
        )
        return

    for result in results:
        lines.extend(
            [
                f"### query-id: `{result.query_id}`",
                "",
                f"**Запрос:** {result.query}",
                "",
                "**Golden chunks:**",
                "",
            ]
        )

        for chunk_id in (
            result.lexical_result.relevant_chunk_ids
        ):
            lines.append(f"- `{chunk_id}`")

        lines.extend(
            [
                "",
                f"**Автоматическая гипотеза:** "
                f"{hypothesis}",
                "",
            ]
        )

        _append_retriever_result(
            lines=lines,
            title="BM25",
            result=result.lexical_result,
            top_k=top_k,
            content_limit=content_limit,
        )
        _append_retriever_result(
            lines=lines,
            title="Dense",
            result=result.dense_result,
            top_k=top_k,
            content_limit=content_limit,
        )

        lines.extend(
            [
                "**Итоговая категория:** `TODO`",
                "",
                "**Причина:**",
                "",
                "TODO",
                "",
                "**Действие:**",
                "",
                "TODO",
                "",
                "**Нужно изменить golden dataset:** "
                "`TODO`",
                "",
                "**Нужно изменить chunking:** `TODO`",
                "",
                "---",
                "",
            ]
        )


def _append_retriever_result(
    *,
    lines: list[str],
    title: str,
    result: QueryEvaluationResult,
    top_k: int,
    content_limit: int,
) -> None:
    first_rank = (
        str(result.first_relevant_rank)
        if result.first_relevant_rank is not None
        else "не найден"
    )

    lines.extend(
        [
            f"#### {title} top-{top_k}",
            "",
            (
                "**Первый релевантный rank:** "
                f"{first_rank}"
            ),
            "",
        ]
    )

    if not result.retrieved_chunks:
        lines.extend(
            [
                "Нет результатов.",
                "",
            ]
        )
        return

    for retrieved in result.retrieved_chunks:
        section = (
            " > ".join(retrieved.section_path)
            if retrieved.section_path
            else "<no heading>"
        )
        relevance_label = (
            "релевантный по golden"
            if retrieved.relevance > 0
            else "не размечен как релевантный"
        )

        lines.extend(
            [
                (
                    f"{retrieved.rank}. "
                    f"`{retrieved.chunk_id}` — "
                    f"score `{retrieved.score:.6f}`"
                ),
                (
                    f"   - **Relevance:** "
                    f"{retrieved.relevance} "
                    f"({relevance_label})"
                ),
                (
                    f"   - **Path:** "
                    f"`{retrieved.relative_path}`"
                ),
                f"   - **Section:** {section}",
                "",
            ]
        )

        content = _truncate_content(
            retrieved.content,
            content_limit,
        )

        lines.extend(
            [
                "<details>",
                "<summary>Показать content</summary>",
                "",
                *_markdown_code_block(content),
                "",
                "</details>",
                "",
            ]
        )


def _truncate_content(
    content: str,
    limit: int,
) -> str:
    if limit == 0:
        return "<content hidden>"

    if len(content) <= limit:
        return content

    return (
        content[:limit].rstrip()
        + "\n\n<content truncated>"
    )


def _markdown_code_block(
    content: str,
) -> list[str]:
    backtick_runs = re.findall(
        r"`+",
        content,
    )
    longest_run = max(
        (
            len(run)
            for run in backtick_runs
        ),
        default=0,
    )
    fence = "`" * max(
        3,
        longest_run + 1,
    )

    return [
        f"{fence}text",
        content,
        fence,
    ]