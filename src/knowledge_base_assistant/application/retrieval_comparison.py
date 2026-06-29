from dataclasses import dataclass

from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
)


@dataclass(frozen=True, slots=True)
class RetrievalComparisonResult:
    dense_only_hits: tuple[QueryEvaluationResult, ...]
    lexical_only_hits: tuple[QueryEvaluationResult, ...]
    both_hit: tuple[QueryEvaluationResult, ...]
    both_miss: tuple[QueryEvaluationResult, ...]


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