from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import typer

import knowledge_base_assistant.cli as cli
from knowledge_base_assistant.application.retrieval_analysis import (
    RetrieverType,
)


class FakeEmbeddingModel:
    def __init__(
        self,
        model_name: str = "test-model",
    ) -> None:
        self.model_name = model_name


def test_dense_index_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"
    embedding_model = FakeEmbeddingModel()

    constructor_calls: list[dict[str, Any]] = []
    build_calls: list[dict[str, Any]] = []

    def fake_embedding_model_constructor(
        **kwargs: Any,
    ) -> FakeEmbeddingModel:
        constructor_calls.append(kwargs)
        return embedding_model

    def fake_build_dense_index(
        **kwargs: Any,
    ) -> SimpleNamespace:
        build_calls.append(kwargs)
        return SimpleNamespace(
            chunk_count=10,
            dimension=768,
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
        )

    monkeypatch.setattr(
        cli,
        "SentenceTransformerEmbeddingModel",
        fake_embedding_model_constructor,
    )
    monkeypatch.setattr(
        cli,
        "build_dense_index",
        fake_build_dense_index,
    )

    cli.dense_index(
        model_name="test-model",
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        batch_size=16,
        device="cuda",
        query_prefix="query: ",
        document_prefix="passage: ",
    )

    output = capsys.readouterr().out

    assert constructor_calls == [
        {
            "model_name": "test-model",
            "batch_size": 16,
            "device": "cuda",
            "query_prefix": "query: ",
            "document_prefix": "passage: ",
        }
    ]
    assert build_calls == [
        {
            "chunks_path": chunks_path,
            "embeddings_path": embeddings_path,
            "metadata_path": metadata_path,
            "embedding_model": embedding_model,
        }
    ]
    assert "Dense index built." in output
    assert "Model: test-model" in output
    assert "Chunks: 10" in output
    assert "Dimension: 768" in output
    assert f"Embeddings: {embeddings_path}" in output
    assert f"Metadata: {metadata_path}" in output


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid configuration"),
        OSError("cannot write index"),
    ],
)
def test_dense_index_reports_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error: Exception,
) -> None:
    def raise_error(**kwargs: Any) -> FakeEmbeddingModel:
        raise error

    monkeypatch.setattr(
        cli,
        "SentenceTransformerEmbeddingModel",
        raise_error,
    )

    with pytest.raises(typer.Exit) as exc_info:
        cli.dense_index(
            model_name="test-model",
            chunks_path=tmp_path / "chunks.jsonl",
            embeddings_path=tmp_path / "embeddings.npy",
            metadata_path=tmp_path / "metadata.json",
            batch_size=16,
            device="cpu",
            query_prefix="query: ",
            document_prefix="passage: ",
        )

    output = capsys.readouterr().err

    assert exc_info.value.exit_code == 1
    assert f"Dense indexing failed: {error}" in output


def test_dense_search_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"
    embedding_config = object()
    embedding_model = FakeEmbeddingModel()

    metadata = SimpleNamespace(
        embedding_model=embedding_config,
    )
    first_chunk = SimpleNamespace(
        section_path=("Dense Retrieval", "Search"),
        content="A" * 20,
        relative_path="dense.md",
        start_line=10,
        end_line=20,
        chunk_id="chunk-1",
    )
    second_chunk = SimpleNamespace(
        section_path=(),
        content="Short content",
        relative_path="other.md",
        start_line=1,
        end_line=5,
        chunk_id="chunk-2",
    )
    results = (
        SimpleNamespace(
            chunk=first_chunk,
            rank=1,
            score=0.98765,
        ),
        SimpleNamespace(
            chunk=second_chunk,
            rank=2,
            score=0.5,
        ),
    )

    factory_calls: list[dict[str, Any]] = []
    search_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        cli,
        "read_dense_index_metadata",
        lambda path: metadata,
    )

    def fake_factory(
        config: object,
        *,
        batch_size: int,
        device: str,
    ) -> FakeEmbeddingModel:
        factory_calls.append(
            {
                "config": config,
                "batch_size": batch_size,
                "device": device,
            }
        )
        return embedding_model

    def fake_search_dense_index(
        **kwargs: Any,
    ) -> tuple[SimpleNamespace, ...]:
        search_calls.append(kwargs)
        return results

    monkeypatch.setattr(
        cli,
        "create_sentence_transformer_embedding_model",
        fake_factory,
    )
    monkeypatch.setattr(
        cli,
        "search_dense_index",
        fake_search_dense_index,
    )

    cli.dense_search(
        query="test query",
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        top_k=2,
        batch_size=8,
        device="cuda",
        content_limit=10,
    )

    output = capsys.readouterr().out

    assert factory_calls == [
        {
            "config": embedding_config,
            "batch_size": 8,
            "device": "cuda",
        }
    ]
    assert search_calls == [
        {
            "query": "test query",
            "top_k": 2,
            "chunks_path": chunks_path,
            "embeddings_path": embeddings_path,
            "metadata_path": metadata_path,
            "embedding_model": embedding_model,
        }
    ]
    assert "1. Score: 0.9877" in output
    assert "Section: Dense Retrieval > Search" in output
    assert "Lines: 10-20" in output
    assert "Chunk ID: chunk-1" in output
    assert "AAAAAAAAAA..." in output
    assert "2. Score: 0.5000" in output
    assert "Section: <no heading>" in output
    assert "Chunk ID: chunk-2" in output


def test_dense_search_with_zero_content_limit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    metadata = SimpleNamespace(
        embedding_model=object(),
    )
    result = SimpleNamespace(
        rank=1,
        score=0.5,
        chunk=SimpleNamespace(
            section_path=(),
            content="Hidden content",
            relative_path="document.md",
            start_line=1,
            end_line=2,
            chunk_id="chunk-1",
        ),
    )

    monkeypatch.setattr(
        cli,
        "read_dense_index_metadata",
        lambda path: metadata,
    )
    monkeypatch.setattr(
        cli,
        "create_sentence_transformer_embedding_model",
        lambda config, batch_size, device: FakeEmbeddingModel(),
    )
    monkeypatch.setattr(
        cli,
        "search_dense_index",
        lambda **kwargs: (result,),
    )

    cli.dense_search(
        query="query",
        chunks_path=tmp_path / "chunks.jsonl",
        embeddings_path=tmp_path / "embeddings.npy",
        metadata_path=tmp_path / "metadata.json",
        top_k=1,
        batch_size=8,
        device="cpu",
        content_limit=0,
    )

    output = capsys.readouterr().out

    assert "Content:" in output
    assert "Hidden content" not in output


def test_dense_search_reports_no_results(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    metadata = SimpleNamespace(
        embedding_model=object(),
    )

    monkeypatch.setattr(
        cli,
        "read_dense_index_metadata",
        lambda path: metadata,
    )
    monkeypatch.setattr(
        cli,
        "create_sentence_transformer_embedding_model",
        lambda config, batch_size, device: FakeEmbeddingModel(),
    )
    monkeypatch.setattr(
        cli,
        "search_dense_index",
        lambda **kwargs: (),
    )

    cli.dense_search(
        query="query",
        chunks_path=tmp_path / "chunks.jsonl",
        embeddings_path=tmp_path / "embeddings.npy",
        metadata_path=tmp_path / "metadata.json",
        top_k=5,
        batch_size=32,
        device="cpu",
        content_limit=500,
    )

    assert (
        capsys.readouterr().out
        == "No matching chunks found.\n"
    )


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid index"),
        OSError("cannot read metadata"),
    ],
)
def test_dense_search_reports_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error: Exception,
) -> None:
    def raise_error(path: Path) -> None:
        raise error

    monkeypatch.setattr(
        cli,
        "read_dense_index_metadata",
        raise_error,
    )

    with pytest.raises(typer.Exit) as exc_info:
        cli.dense_search(
            query="query",
            chunks_path=tmp_path / "chunks.jsonl",
            embeddings_path=tmp_path / "embeddings.npy",
            metadata_path=tmp_path / "metadata.json",
            top_k=5,
            batch_size=32,
            device="cpu",
            content_limit=500,
        )

    assert exc_info.value.exit_code == 1
    assert (
        f"Dense search failed: {error}"
        in capsys.readouterr().err
    )


def test_dense_evaluate_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    embedding_config = object()
    embedding_model = FakeEmbeddingModel()
    metadata = SimpleNamespace(
        embedding_model=embedding_config,
    )
    evaluation_result = SimpleNamespace(
        top_k=5,
        query_count=66,
        evaluated_query_count=54,
        no_answer_query_count=12,
        hit_rate_at_k=0.8333,
        recall_at_k=0.4907,
        mean_reciprocal_rank=0.6043,
        ndcg_at_k=0.5136,
    )
    evaluation_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        cli,
        "read_dense_index_metadata",
        lambda path: metadata,
    )
    monkeypatch.setattr(
        cli,
        "create_sentence_transformer_embedding_model",
        lambda config, batch_size, device: embedding_model,
    )

    def fake_evaluate_dense_retrieval(
        **kwargs: Any,
    ) -> SimpleNamespace:
        evaluation_calls.append(kwargs)
        return evaluation_result

    monkeypatch.setattr(
        cli,
        "evaluate_dense_retrieval",
        fake_evaluate_dense_retrieval,
    )

    golden_path = tmp_path / "golden.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"

    cli.dense_evaluate(
        golden_path=golden_path,
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        top_k=5,
        batch_size=16,
        device="cuda",
    )

    output = capsys.readouterr().out

    assert evaluation_calls == [
        {
            "golden_path": golden_path,
            "chunks_path": chunks_path,
            "embeddings_path": embeddings_path,
            "metadata_path": metadata_path,
            "embedding_model": embedding_model,
            "top_k": 5,
        }
    ]
    assert "Dense retrieval evaluation completed." in output
    assert "Top-K: 5" in output
    assert "Queries: 66" in output
    assert "Evaluated queries: 54" in output
    assert "No-answer queries: 12" in output
    assert "HitRate@5: 0.8333" in output
    assert "Recall@5: 0.4907" in output
    assert "MRR@5: 0.6043" in output
    assert "nDCG@5: 0.5136" in output


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid evaluation"),
        OSError("cannot read index"),
    ],
)
def test_dense_evaluate_reports_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error: Exception,
) -> None:
    def raise_error(path: Path) -> None:
        raise error

    monkeypatch.setattr(
        cli,
        "read_dense_index_metadata",
        raise_error,
    )

    with pytest.raises(typer.Exit) as exc_info:
        cli.dense_evaluate(
            golden_path=tmp_path / "golden.jsonl",
            chunks_path=tmp_path / "chunks.jsonl",
            embeddings_path=tmp_path / "embeddings.npy",
            metadata_path=tmp_path / "metadata.json",
            top_k=5,
            batch_size=32,
            device="cpu",
        )

    assert exc_info.value.exit_code == 1
    assert (
        f"Dense retrieval evaluation failed: {error}"
        in capsys.readouterr().err
    )


def test_retrieval_analyze_lexical_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    analysis_result = SimpleNamespace(
        retriever_type=RetrieverType.LEXICAL,
        top_k=5,
        query_count=54,
        miss_count=13,
        results_path=tmp_path / "lexical/results.jsonl",
        misses_path=tmp_path / "lexical/misses.jsonl",
    )
    analysis_calls: list[dict[str, Any]] = []

    def fake_run_retrieval_analysis(
        **kwargs: Any,
    ) -> SimpleNamespace:
        analysis_calls.append(kwargs)
        return analysis_result

    monkeypatch.setattr(
        cli,
        "run_retrieval_analysis",
        fake_run_retrieval_analysis,
    )

    golden_path = tmp_path / "golden.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    output_root = tmp_path / "evaluation"
    embeddings_path = tmp_path / "missing-embeddings.npy"
    metadata_path = tmp_path / "missing-metadata.json"

    cli.retrieval_analyze(
        golden_path=golden_path,
        chunks_path=chunks_path,
        retriever_type=RetrieverType.LEXICAL,
        output_root=output_root,
        top_k=5,
        k1=1.5,
        b=0.75,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        batch_size=32,
        device="cpu",
    )

    output = capsys.readouterr().out

    assert analysis_calls == [
        {
            "retriever_type": RetrieverType.LEXICAL,
            "golden_path": golden_path,
            "chunks_path": chunks_path,
            "output_root": output_root,
            "top_k": 5,
            "k1": 1.5,
            "b": 0.75,
            "embeddings_path": None,
            "metadata_path": None,
            "embedding_model": None,
        }
    ]
    assert "Retrieval analysis completed." in output
    assert "Retriever: lexical" in output
    assert "Queries: 54" in output
    assert "Misses: 13" in output


def test_retrieval_analyze_dense_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    golden_path = tmp_path / "golden.jsonl"
    chunks_path = tmp_path / "chunks.jsonl"
    output_root = tmp_path / "evaluation"
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"

    embeddings_path.touch()
    metadata_path.touch()

    embedding_config = object()
    embedding_model = FakeEmbeddingModel()
    metadata = SimpleNamespace(
        embedding_model=embedding_config,
    )
    analysis_result = SimpleNamespace(
        retriever_type=RetrieverType.DENSE,
        top_k=5,
        query_count=54,
        miss_count=9,
        results_path=output_root / "dense/results.jsonl",
        misses_path=output_root / "dense/misses.jsonl",
    )
    analysis_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        cli,
        "read_dense_index_metadata",
        lambda path: metadata,
    )
    monkeypatch.setattr(
        cli,
        "create_sentence_transformer_embedding_model",
        lambda config, batch_size, device: embedding_model,
    )

    def fake_run_retrieval_analysis(
        **kwargs: Any,
    ) -> SimpleNamespace:
        analysis_calls.append(kwargs)
        return analysis_result

    monkeypatch.setattr(
        cli,
        "run_retrieval_analysis",
        fake_run_retrieval_analysis,
    )

    cli.retrieval_analyze(
        golden_path=golden_path,
        chunks_path=chunks_path,
        retriever_type=RetrieverType.DENSE,
        output_root=output_root,
        top_k=5,
        k1=1.5,
        b=0.75,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        batch_size=16,
        device="cuda",
    )

    output = capsys.readouterr().out

    assert analysis_calls == [
        {
            "retriever_type": RetrieverType.DENSE,
            "golden_path": golden_path,
            "chunks_path": chunks_path,
            "output_root": output_root,
            "top_k": 5,
            "k1": 1.5,
            "b": 0.75,
            "embeddings_path": embeddings_path,
            "metadata_path": metadata_path,
            "embedding_model": embedding_model,
        }
    ]
    assert "Retriever: dense" in output
    assert "Misses: 9" in output


@pytest.mark.parametrize(
    ("missing_file", "expected_message"),
    [
        (
            "embeddings",
            "Dense embeddings file does not exist",
        ),
        (
            "metadata",
            "Dense metadata file does not exist",
        ),
    ],
)
def test_retrieval_analyze_dense_requires_files(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    missing_file: str,
    expected_message: str,
) -> None:
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"

    if missing_file == "embeddings":
        metadata_path.touch()
    else:
        embeddings_path.touch()

    run_called = False

    def fake_run_retrieval_analysis(
        **kwargs: Any,
    ) -> None:
        nonlocal run_called
        run_called = True

    monkeypatch.setattr(
        cli,
        "run_retrieval_analysis",
        fake_run_retrieval_analysis,
    )

    with pytest.raises(typer.Exit) as exc_info:
        cli.retrieval_analyze(
            golden_path=tmp_path / "golden.jsonl",
            chunks_path=tmp_path / "chunks.jsonl",
            retriever_type=RetrieverType.DENSE,
            output_root=tmp_path / "evaluation",
            top_k=5,
            k1=1.5,
            b=0.75,
            embeddings_path=embeddings_path,
            metadata_path=metadata_path,
            batch_size=32,
            device="cpu",
        )

    assert exc_info.value.exit_code == 1
    assert not run_called
    assert expected_message in capsys.readouterr().err


@pytest.mark.parametrize(
    "error",
    [
        ValueError("analysis failed"),
        OSError("cannot write results"),
    ],
)
def test_retrieval_analyze_reports_application_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error: Exception,
) -> None:
    def raise_error(**kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(
        cli,
        "run_retrieval_analysis",
        raise_error,
    )

    with pytest.raises(typer.Exit) as exc_info:
        cli.retrieval_analyze(
            golden_path=tmp_path / "golden.jsonl",
            chunks_path=tmp_path / "chunks.jsonl",
            retriever_type=RetrieverType.LEXICAL,
            output_root=tmp_path / "evaluation",
            top_k=5,
            k1=1.5,
            b=0.75,
            embeddings_path=tmp_path / "embeddings.npy",
            metadata_path=tmp_path / "metadata.json",
            batch_size=32,
            device="cpu",
        )

    assert exc_info.value.exit_code == 1
    assert (
        f"Retrieval analysis failed: {error}"
        in capsys.readouterr().err
    )


def test_retrieval_compare_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    lexical_path = tmp_path / "lexical.jsonl"
    dense_path = tmp_path / "dense.jsonl"
    output_dir = tmp_path / "comparison"

    comparison_result = SimpleNamespace(
        dense_only_hits_count=6,
        lexical_only_hits_count=2,
        both_hit_count=39,
        both_miss_count=7,
    )
    comparison_calls: list[dict[str, Any]] = []

    def fake_run_retrieval_comparison(
        **kwargs: Any,
    ) -> SimpleNamespace:
        comparison_calls.append(kwargs)
        return comparison_result

    monkeypatch.setattr(
        cli,
        "run_retrieval_comparison",
        fake_run_retrieval_comparison,
    )

    cli.retrieval_compare(
        lexical_results_path=lexical_path,
        dense_results_path=dense_path,
        output_dir=output_dir,
    )

    output = capsys.readouterr().out

    assert comparison_calls == [
        {
            "lexical_results_path": lexical_path,
            "dense_results_path": dense_path,
            "output_dir": output_dir,
        }
    ]
    assert "Retrieval comparison completed." in output
    assert "Dense-only hits: 6" in output
    assert "Lexical-only hits: 2" in output
    assert "Both hit: 39" in output
    assert "Both miss: 7" in output
    assert f"Output: {output_dir}" in output


@pytest.mark.parametrize(
    "error",
    [
        ValueError("different query IDs"),
        OSError("cannot write comparison"),
    ],
)
def test_retrieval_compare_reports_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    error: Exception,
) -> None:
    def raise_error(**kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(
        cli,
        "run_retrieval_comparison",
        raise_error,
    )

    with pytest.raises(typer.Exit) as exc_info:
        cli.retrieval_compare(
            lexical_results_path=tmp_path / "lexical.jsonl",
            dense_results_path=tmp_path / "dense.jsonl",
            output_dir=tmp_path / "comparison",
        )

    assert exc_info.value.exit_code == 1
    assert (
        f"Retrieval comparison failed: {error}"
        in capsys.readouterr().err
    )