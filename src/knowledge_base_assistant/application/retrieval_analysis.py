from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from knowledge_base_assistant.application.dense_retrieval_evaluation import (
    evaluate_dense_queries,
)
from knowledge_base_assistant.application.lexical_retrieval_evaluation import (
    evaluate_bm25_queries,
)
from knowledge_base_assistant.evaluation.serialization import (
    write_query_evaluation_misses_jsonl,
    write_query_evaluation_results_jsonl,
)
from knowledge_base_assistant.retrieval.dense.embedding import (
    EmbeddingModel,
)


class RetrieverType(StrEnum):
    LEXICAL = "lexical"
    DENSE = "dense"


@dataclass(frozen=True, slots=True)
class RetrievalAnalysisResult:
    retriever_type: RetrieverType
    top_k: int
    query_count: int
    miss_count: int
    results_path: Path
    misses_path: Path


def run_retrieval_analysis(
    *,
    retriever_type: RetrieverType,
    golden_path: Path,
    chunks_path: Path,
    output_root: Path,
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
    embeddings_path: Path | None = None,
    metadata_path: Path | None = None,
    embedding_model: EmbeddingModel | None = None,
) -> RetrievalAnalysisResult:
    if top_k < 1:
        raise ValueError(
            f"top_k must be at least 1, got {top_k}"
        )

    if retriever_type is RetrieverType.LEXICAL:
        results = evaluate_bm25_queries(
            golden_path=golden_path,
            chunks_path=chunks_path,
            top_k=top_k,
            k1=k1,
            b=b,
        )
    elif retriever_type is RetrieverType.DENSE:
        if embeddings_path is None:
            raise ValueError(
                "embeddings_path is required for dense analysis"
            )

        if metadata_path is None:
            raise ValueError(
                "metadata_path is required for dense analysis"
            )

        if embedding_model is None:
            raise ValueError(
                "embedding_model is required for dense analysis"
            )

        results = evaluate_dense_queries(
            golden_path=golden_path,
            chunks_path=chunks_path,
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            embedding_model=embedding_model,
            top_k=top_k,
        )
    else:
        raise ValueError(
            f"Unsupported retriever type: {retriever_type}"
        )

    output_dir = (
        output_root
        / retriever_type.value
        / f"top_{top_k}"
    )

    results_path = output_dir / "results.jsonl"
    misses_path = output_dir / "misses.jsonl"

    write_query_evaluation_results_jsonl(
        results,
        results_path,
    )

    miss_count = (
        write_query_evaluation_misses_jsonl(
            results,
            misses_path,
        )
    )

    return RetrievalAnalysisResult(
        retriever_type=retriever_type,
        top_k=top_k,
        query_count=len(results),
        miss_count=miss_count,
        results_path=results_path,
        misses_path=misses_path,
    )