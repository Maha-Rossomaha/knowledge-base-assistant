import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
)
from knowledge_base_assistant.evaluation.serialization import (
    read_query_evaluation_results_jsonl,
)


class ComparisonCategory(StrEnum):
    DENSE_ONLY_HIT = "dense_only_hit"
    LEXICAL_ONLY_HIT = "lexical_only_hit"
    BOTH_HIT = "both_hit"
    BOTH_MISS = "both_miss"


@dataclass(frozen=True, slots=True)
class ComparedQueryResult:
    query_id: str
    query: str
    category: ComparisonCategory
    lexical_result: QueryEvaluationResult
    dense_result: QueryEvaluationResult


@dataclass(frozen=True, slots=True)
class RetrievalComparisonResult:
    dense_only_hits: tuple[ComparedQueryResult, ...]
    lexical_only_hits: tuple[ComparedQueryResult, ...]
    both_hit: tuple[ComparedQueryResult, ...]
    both_miss: tuple[ComparedQueryResult, ...]


@dataclass(frozen=True, slots=True)
class RetrievalComparisonArtifacts:
    dense_only_hits_path: Path
    lexical_only_hits_path: Path
    both_hit_path: Path
    both_miss_path: Path
    dense_only_hits_count: int
    lexical_only_hits_count: int
    both_hit_count: int
    both_miss_count: int


def compare_retrieval_results(
    *,
    lexical_results: tuple[QueryEvaluationResult, ...],
    dense_results: tuple[QueryEvaluationResult, ...],
) -> RetrievalComparisonResult:
    lexical_by_query_id = {
        result.query_id: result
        for result in lexical_results
    }
    dense_by_query_id = {
        result.query_id: result
        for result in dense_results
    }

    if lexical_by_query_id.keys() != dense_by_query_id.keys():
        raise ValueError(
            "Lexical and dense results must contain "
            "the same query IDs"
        )

    dense_only_hits: list[ComparedQueryResult] = []
    lexical_only_hits: list[ComparedQueryResult] = []
    both_hit: list[ComparedQueryResult] = []
    both_miss: list[ComparedQueryResult] = []

    for query_id in lexical_by_query_id:
        lexical_result = lexical_by_query_id[query_id]
        dense_result = dense_by_query_id[query_id]

        if lexical_result.query != dense_result.query:
            raise ValueError(
                "Lexical and dense results must contain "
                f"the same query text for query ID {query_id}"
            )

        lexical_hit = (
            lexical_result.first_relevant_rank is not None
        )
        dense_hit = (
            dense_result.first_relevant_rank is not None
        )

        if dense_hit and not lexical_hit:
            category = ComparisonCategory.DENSE_ONLY_HIT
        elif lexical_hit and not dense_hit:
            category = ComparisonCategory.LEXICAL_ONLY_HIT
        elif lexical_hit and dense_hit:
            category = ComparisonCategory.BOTH_HIT
        else:
            category = ComparisonCategory.BOTH_MISS

        compared_result = ComparedQueryResult(
            query_id=query_id,
            query=lexical_result.query,
            category=category,
            lexical_result=lexical_result,
            dense_result=dense_result,
        )

        if category is ComparisonCategory.DENSE_ONLY_HIT:
            dense_only_hits.append(compared_result)
        elif category is ComparisonCategory.LEXICAL_ONLY_HIT:
            lexical_only_hits.append(compared_result)
        elif category is ComparisonCategory.BOTH_HIT:
            both_hit.append(compared_result)
        else:
            both_miss.append(compared_result)

    return RetrievalComparisonResult(
        dense_only_hits=tuple(dense_only_hits),
        lexical_only_hits=tuple(lexical_only_hits),
        both_hit=tuple(both_hit),
        both_miss=tuple(both_miss),
    )


def write_compared_query_results_jsonl(
    results: tuple[ComparedQueryResult, ...],
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


def run_retrieval_comparison(
    *,
    lexical_results_path: Path,
    dense_results_path: Path,
    output_dir: Path,
) -> RetrievalComparisonArtifacts:
    lexical_results = read_query_evaluation_results_jsonl(
        lexical_results_path
    )
    dense_results = read_query_evaluation_results_jsonl(
        dense_results_path
    )

    comparison = compare_retrieval_results(
        lexical_results=lexical_results,
        dense_results=dense_results,
    )

    dense_only_hits_path = (
        output_dir / "dense_only_hits.jsonl"
    )
    lexical_only_hits_path = (
        output_dir / "lexical_only_hits.jsonl"
    )
    both_hit_path = output_dir / "both_hit.jsonl"
    both_miss_path = output_dir / "both_miss.jsonl"

    write_compared_query_results_jsonl(
        comparison.dense_only_hits,
        dense_only_hits_path,
    )
    write_compared_query_results_jsonl(
        comparison.lexical_only_hits,
        lexical_only_hits_path,
    )
    write_compared_query_results_jsonl(
        comparison.both_hit,
        both_hit_path,
    )
    write_compared_query_results_jsonl(
        comparison.both_miss,
        both_miss_path,
    )

    return RetrievalComparisonArtifacts(
        dense_only_hits_path=dense_only_hits_path,
        lexical_only_hits_path=lexical_only_hits_path,
        both_hit_path=both_hit_path,
        both_miss_path=both_miss_path,
        dense_only_hits_count=len(
            comparison.dense_only_hits
        ),
        lexical_only_hits_count=len(
            comparison.lexical_only_hits
        ),
        both_hit_count=len(comparison.both_hit),
        both_miss_count=len(comparison.both_miss),
    )