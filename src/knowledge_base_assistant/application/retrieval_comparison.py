from pathlib import Path
from dataclasses import dataclass

from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
)
from knowledge_base_assistant.evaluation.serialization import (
    read_query_evaluation_results_jsonl,
    write_query_evaluation_results_jsonl,
)


@dataclass(frozen=True, slots=True)
class RetrievalComparisonResult:
    dense_only_hits: tuple[QueryEvaluationResult, ...]
    lexical_only_hits: tuple[QueryEvaluationResult, ...]
    both_hit: tuple[QueryEvaluationResult, ...]
    both_miss: tuple[QueryEvaluationResult, ...]
    

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

    dense_only_hits: list[QueryEvaluationResult] = []
    lexical_only_hits: list[QueryEvaluationResult] = []
    both_hit: list[QueryEvaluationResult] = []
    both_miss: list[QueryEvaluationResult] = []

    for query_id in lexical_by_query_id:
        lexical_result = lexical_by_query_id[query_id]
        dense_result = dense_by_query_id[query_id]

        lexical_hit = (
            lexical_result.first_relevant_rank is not None
        )
        dense_hit = (
            dense_result.first_relevant_rank is not None
        )

        if dense_hit and not lexical_hit:
            dense_only_hits.append(dense_result)
        elif lexical_hit and not dense_hit:
            lexical_only_hits.append(lexical_result)
        elif lexical_hit and dense_hit:
            both_hit.append(dense_result)
        else:
            both_miss.append(dense_result)

    return RetrievalComparisonResult(
        dense_only_hits=tuple(dense_only_hits),
        lexical_only_hits=tuple(lexical_only_hits),
        both_hit=tuple(both_hit),
        both_miss=tuple(both_miss),
    )
    
    
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

    write_query_evaluation_results_jsonl(
        comparison.dense_only_hits,
        dense_only_hits_path,
    )
    write_query_evaluation_results_jsonl(
        comparison.lexical_only_hits,
        lexical_only_hits_path,
    )
    write_query_evaluation_results_jsonl(
        comparison.both_hit,
        both_hit_path,
    )
    write_query_evaluation_results_jsonl(
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