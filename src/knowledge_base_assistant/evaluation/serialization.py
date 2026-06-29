import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
    RetrievedChunkEvaluation,
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


def read_query_evaluation_results_jsonl(
    path: Path,
) -> tuple[QueryEvaluationResult, ...]:
    results: list[QueryEvaluationResult] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at line {line_number}"
                ) from error

            try:
                result = _query_evaluation_result_from_dict(
                    data
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    "Invalid query evaluation result "
                    f"at line {line_number}: {error}"
                ) from error

            results.append(result)

    return tuple(results)


def _query_evaluation_result_from_dict(
    data: Any,
) -> QueryEvaluationResult:
    if not isinstance(data, dict):
        raise TypeError("JSON value must be an object")

    relevant_chunk_ids = data["relevant_chunk_ids"]
    retrieved_chunks = data["retrieved_chunks"]

    if not isinstance(relevant_chunk_ids, list):
        raise TypeError(
            "relevant_chunk_ids must be a list"
        )

    if not all(
        isinstance(chunk_id, str)
        for chunk_id in relevant_chunk_ids
    ):
        raise TypeError(
            "relevant_chunk_ids must contain only strings"
        )

    if not isinstance(retrieved_chunks, list):
        raise TypeError(
            "retrieved_chunks must be a list"
        )

    first_relevant_rank = data["first_relevant_rank"]

    if (
        first_relevant_rank is not None
        and not isinstance(first_relevant_rank, int)
    ):
        raise TypeError(
            "first_relevant_rank must be an integer or null"
        )

    return QueryEvaluationResult(
        query_id=_required_string(data, "query_id"),
        query=_required_string(data, "query"),
        relevant_chunk_ids=tuple(relevant_chunk_ids),
        retrieved_chunks=tuple(
            _retrieved_chunk_evaluation_from_dict(item)
            for item in retrieved_chunks
        ),
        first_relevant_rank=first_relevant_rank,
        hit_rate_at_k=float(data["hit_rate_at_k"]),
        recall_at_k=float(data["recall_at_k"]),
        reciprocal_rank=float(data["reciprocal_rank"]),
        ndcg_at_k=float(data["ndcg_at_k"]),
    )


def _retrieved_chunk_evaluation_from_dict(
    data: Any,
) -> RetrievedChunkEvaluation:
    if not isinstance(data, dict):
        raise TypeError(
            "retrieved_chunks item must be an object"
        )

    section_path = data["section_path"]

    if not isinstance(section_path, list):
        raise TypeError("section_path must be a list")

    if not all(
        isinstance(section, str)
        for section in section_path
    ):
        raise TypeError(
            "section_path must contain only strings"
        )

    return RetrievedChunkEvaluation(
        chunk_id=_required_string(data, "chunk_id"),
        rank=int(data["rank"]),
        score=float(data["score"]),
        relevance=int(data["relevance"]),
        relative_path=_required_string(
            data,
            "relative_path",
        ),
        section_path=tuple(section_path),
        content=_required_string(data, "content"),
    )


def _required_string(
    data: dict[str, Any],
    field: str,
) -> str:
    value = data[field]

    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")

    return value