from pathlib import Path

import pytest

import knowledge_base_assistant.application.retrieval_analysis as module
from knowledge_base_assistant.evaluation.models import (
    QueryEvaluationResult,
)


class FakeEmbeddingModel:
    pass


def test_run_retrieval_analysis_for_lexical(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_item = _make_result(
        query_id="query-1",
        first_relevant_rank=1,
    )

    monkeypatch.setattr(
        module,
        "evaluate_bm25_queries",
        lambda **kwargs: (result_item,),
    )

    written_results: list[
        tuple[tuple[QueryEvaluationResult, ...], Path]
    ] = []

    monkeypatch.setattr(
        module,
        "write_query_evaluation_results_jsonl",
        lambda results, path: written_results.append(
            (results, path)
        ),
    )

    monkeypatch.setattr(
        module,
        "write_query_evaluation_misses_jsonl",
        lambda results, path: 0,
    )

    result = module.run_retrieval_analysis(
        retriever_type=module.RetrieverType.LEXICAL,
        golden_path=Path("golden.jsonl"),
        chunks_path=Path("chunks.jsonl"),
        output_root=tmp_path,
        top_k=5,
    )

    assert result.retriever_type is module.RetrieverType.LEXICAL
    assert result.query_count == 1
    assert result.miss_count == 0
    assert result.results_path == (
        tmp_path / "lexical" / "top_5" / "results.jsonl"
    )
    assert result.misses_path == (
        tmp_path / "lexical" / "top_5" / "misses.jsonl"
    )

    assert written_results == [
        (
            (result_item,),
            tmp_path
            / "lexical"
            / "top_5"
            / "results.jsonl",
        )
    ]


def test_run_retrieval_analysis_for_dense(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_item = _make_result(
        query_id="query-1",
        first_relevant_rank=None,
    )
    embedding_model = FakeEmbeddingModel()

    dense_calls: list[dict[str, object]] = []

    def fake_evaluate_dense_queries(
        **kwargs,
    ) -> tuple[QueryEvaluationResult, ...]:
        dense_calls.append(kwargs)
        return (result_item,)

    monkeypatch.setattr(
        module,
        "evaluate_dense_queries",
        fake_evaluate_dense_queries,
    )
    monkeypatch.setattr(
        module,
        "write_query_evaluation_results_jsonl",
        lambda results, path: None,
    )
    monkeypatch.setattr(
        module,
        "write_query_evaluation_misses_jsonl",
        lambda results, path: 1,
    )

    result = module.run_retrieval_analysis(
        retriever_type=module.RetrieverType.DENSE,
        golden_path=Path("golden.jsonl"),
        chunks_path=Path("chunks.jsonl"),
        output_root=tmp_path,
        top_k=5,
        embeddings_path=Path("embeddings.npy"),
        metadata_path=Path("metadata.json"),
        embedding_model=embedding_model,
    )

    assert result.retriever_type is module.RetrieverType.DENSE
    assert result.query_count == 1
    assert result.miss_count == 1

    assert dense_calls == [
        {
            "golden_path": Path("golden.jsonl"),
            "chunks_path": Path("chunks.jsonl"),
            "embeddings_path": Path("embeddings.npy"),
            "metadata_path": Path("metadata.json"),
            "embedding_model": embedding_model,
            "top_k": 5,
        }
    ]


@pytest.mark.parametrize(
    ("field_name", "expected_message"),
    [
        (
            "embeddings_path",
            "embeddings_path is required for dense analysis",
        ),
        (
            "metadata_path",
            "metadata_path is required for dense analysis",
        ),
        (
            "embedding_model",
            "embedding_model is required for dense analysis",
        ),
    ],
)
def test_run_retrieval_analysis_requires_dense_arguments(
    field_name: str,
    expected_message: str,
    tmp_path: Path,
) -> None:
    arguments: dict[str, object] = {
        "retriever_type": module.RetrieverType.DENSE,
        "golden_path": Path("golden.jsonl"),
        "chunks_path": Path("chunks.jsonl"),
        "output_root": tmp_path,
        "top_k": 5,
        "embeddings_path": Path("embeddings.npy"),
        "metadata_path": Path("metadata.json"),
        "embedding_model": FakeEmbeddingModel(),
    }

    arguments[field_name] = None

    with pytest.raises(
        ValueError,
        match=expected_message,
    ):
        module.run_retrieval_analysis(
            **arguments,
        )


def _make_result(
    *,
    query_id: str,
    first_relevant_rank: int | None,
) -> QueryEvaluationResult:
    return QueryEvaluationResult(
        query_id=query_id,
        query="test query",
        relevant_chunk_ids=("chunk-1",),
        retrieved_chunks=(),
        first_relevant_rank=first_relevant_rank,
        hit_rate_at_k=(
            1.0
            if first_relevant_rank is not None
            else 0.0
        ),
        recall_at_k=(
            1.0
            if first_relevant_rank is not None
            else 0.0
        ),
        reciprocal_rank=(
            1.0
            if first_relevant_rank is not None
            else 0.0
        ),
        ndcg_at_k=(
            1.0
            if first_relevant_rank is not None
            else 0.0
        ),
    )