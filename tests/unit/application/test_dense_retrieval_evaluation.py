from pathlib import Path

import pytest

import knowledge_base_assistant.application.dense_retrieval_evaluation as module
from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.evaluation.models import (
    GoldenQuery,
    QueryEvaluationResult,
    RetrievalEvaluationResult,
)
from knowledge_base_assistant.retrieval.dense.embedding import EmbeddingModelConfig


class FakeEmbeddingModel:
    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimension(self) -> int:
        return 3
    
    @property
    def config(self) -> EmbeddingModelConfig:
        return EmbeddingModelConfig(
            provider="fake",
            model_name=self.model_name,
            dimension=self.dimension,
            query_prefix="",
            document_prefix="",
        )

    def embed_documents(self, texts: list[str]):
        raise NotImplementedError

    def embed_query(self, query: str):
        raise NotImplementedError


class FakeDenseIndex:
    def __init__(
        self,
        chunks: tuple[Chunk, ...],
    ) -> None:
        self.chunks = chunks


def test_evaluate_dense_retrieval_loads_index_and_calls_common_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden_path = Path("golden.jsonl")
    chunks_path = Path("chunks.jsonl")
    embeddings_path = Path("embeddings.npy")
    metadata_path = Path("metadata.json")

    queries = [
        GoldenQuery(
            query_id="query-1",
            query="test query",
            relevant_chunks=(),
            notes="",
        ),
    ]
    chunks = (_make_chunk(),)
    fake_index = FakeDenseIndex(chunks)
    embedding_model = FakeEmbeddingModel()

    monkeypatch.setattr(
        module,
        "read_golden_queries_jsonl",
        lambda path: queries,
    )

    load_calls: list[dict[str, object]] = []

    def fake_load_dense_index(
        *,
        chunks_path,
        embeddings_path,
        metadata_path,
        embedding_model,
    ) -> FakeDenseIndex:
        load_calls.append(
            {
                "chunks_path": chunks_path,
                "embeddings_path": embeddings_path,
                "metadata_path": metadata_path,
                "embedding_model": embedding_model,
            }
        )
        return fake_index

    monkeypatch.setattr(
        module,
        "load_dense_index",
        fake_load_dense_index,
    )

    expected_result = RetrievalEvaluationResult(
        top_k=5,
        query_count=1,
        evaluated_query_count=0,
        no_answer_query_count=1,
        hit_rate_at_k=0.0,
        recall_at_k=0.0,
        mean_reciprocal_rank=0.0,
        ndcg_at_k=0.0,
    )

    evaluator_calls: list[dict[str, object]] = []

    def fake_evaluate_retrieval(
        *,
        queries,
        chunks,
        retriever,
        top_k,
    ) -> RetrievalEvaluationResult:
        evaluator_calls.append(
            {
                "queries": queries,
                "chunks": chunks,
                "retriever": retriever,
                "top_k": top_k,
            }
        )
        return expected_result

    monkeypatch.setattr(
        module,
        "evaluate_retrieval",
        fake_evaluate_retrieval,
    )

    result = module.evaluate_dense_retrieval(
        golden_path=golden_path,
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
        top_k=5,
    )

    assert result == expected_result
    assert load_calls == [
        {
            "chunks_path": chunks_path,
            "embeddings_path": embeddings_path,
            "metadata_path": metadata_path,
            "embedding_model": embedding_model,
        },
    ]
    assert evaluator_calls == [
        {
            "queries": queries,
            "chunks": chunks,
            "retriever": fake_index,
            "top_k": 5,
        },
    ]


def _make_chunk() -> Chunk:
    return Chunk(
        chunk_id="chunk-1",
        document_id="document-1",
        source_name="test-source",
        relative_path="notes/test.md",
        title="Test",
        section_path=("Test",),
        content="Content",
        searchable_text="Searchable content",
        chunk_index=0,
        section_chunk_index=0,
        start_line=1,
        end_line=2,
        content_hash="content-hash",
    )
    
    
def test_evaluate_dense_queries_loads_index_and_calls_common_evaluator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    golden_path = Path("golden.jsonl")
    chunks_path = Path("chunks.jsonl")
    embeddings_path = Path("embeddings.npy")
    metadata_path = Path("metadata.json")

    queries = [
        GoldenQuery(
            query_id="query-1",
            query="test query",
            relevant_chunks=(),
            notes="",
        ),
    ]
    chunks = (_make_chunk(),)
    fake_index = FakeDenseIndex(chunks)
    embedding_model = FakeEmbeddingModel()
    expected_result: tuple[QueryEvaluationResult, ...] = ()

    monkeypatch.setattr(
        module,
        "read_golden_queries_jsonl",
        lambda path: queries,
    )
    monkeypatch.setattr(
        module,
        "load_dense_index",
        lambda **kwargs: fake_index,
    )

    evaluator_calls: list[dict[str, object]] = []

    def fake_evaluate_queries(
        *,
        queries,
        chunks,
        retriever,
        top_k,
    ) -> tuple[QueryEvaluationResult, ...]:
        evaluator_calls.append(
            {
                "queries": queries,
                "chunks": chunks,
                "retriever": retriever,
                "top_k": top_k,
            }
        )
        return expected_result

    monkeypatch.setattr(
        module,
        "evaluate_queries",
        fake_evaluate_queries,
    )

    result = module.evaluate_dense_queries(
        golden_path=golden_path,
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedding_model=embedding_model,
        top_k=5,
    )

    assert result == expected_result
    assert evaluator_calls == [
        {
            "queries": queries,
            "chunks": chunks,
            "retriever": fake_index,
            "top_k": 5,
        },
    ]