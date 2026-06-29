import json
from dataclasses import asdict
from pathlib import Path

from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
)


def write_query_evaluation_results_jsonl(
    results: tuple[QueryEvaluationResult, ...],
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(
                json.dumps(
                    asdict(result),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def write_query_evaluation_misses_jsonl(
    results: tuple[QueryEvaluationResult, ...],
    path: Path,
) -> int:
    misses = tuple(
        result
        for result in results
        if result.first_relevant_rank is None
    )

    write_query_evaluation_results_jsonl(
        misses,
        path,
    )

    return len(misses)