from pathlib import Path

from knowledge_base_assistant.application.dense_search import load_dense_index
from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
    RetrievalEvaluationResult,
)
from knowledge_base_assistant.retrieval.dense.embedding import EmbeddingModel
from knowledge_base_assistant.serialization.jsonl import read_golden_queries_jsonl
from knowledge_base_assistant.application.retrieval_evaluation import (
    evaluate_retrieval,
    evaluate_queries,
)


def evaluate_dense_retrieval(
    *,
    golden_path: Path,
    chunks_path: Path,
    embeddings_path: Path,
    metadata_path: Path,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> RetrievalEvaluationResult:
    queries = read_golden_queries_jsonl(golden_path)

    index = load_dense_index(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
    )
    
    return evaluate_retrieval(
        queries=queries,
        chunks=index.chunks,
        retriever=index,
        top_k=top_k,
    )


def evaluate_dense_queries(
    *,
    golden_path: Path,
    chunks_path: Path,
    embeddings_path: Path,
    metadata_path: Path,
    embedding_model: EmbeddingModel,
    top_k: int,
) -> tuple[QueryEvaluationResult, ...]:
    queries = read_golden_queries_jsonl(golden_path)

    index = load_dense_index(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
    )

    return evaluate_queries(
        queries=queries,
        chunks=index.chunks,
        retriever=index,
        top_k=top_k,
    )