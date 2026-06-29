import pytest

from knowledge_base_assistant.application.retrieval_evaluation import (
    evaluate_queries,
    evaluate_retrieval,
)
from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.evaluation.models import (
    GoldenQuery,
    RelevantChunk,
)
from knowledge_base_assistant.retrieval.models import SearchResult


class FakeRetriever:
    def __init__(
        self,
        results_by_query: dict[str, list[SearchResult]],
    ) -> None:
        self._results_by_query = results_by_query
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> list[SearchResult]:
        self.calls.append((query, top_k))
        return self._results_by_query.get(query, [])[:top_k]


def test_evaluate_retrieval_calculates_aggregated_metrics() -> None:
    chunks = _make_chunks()

    queries = [
        _make_query(
            query_id="query-1",
            query="first query",
            relevant_chunk_ids=("chunk-1",),
            chunks=chunks,
        ),
        _make_query(
            query_id="query-2",
            query="second query",
            relevant_chunk_ids=("chunk-3",),
            chunks=chunks,
        ),
    ]

    retriever = FakeRetriever(
        {
            "first query": [
                _make_search_result(chunks[0], score=0.9, rank=1),
                _make_search_result(chunks[1], score=0.8, rank=2),
            ],
            "second query": [
                _make_search_result(chunks[1], score=0.9, rank=1),
                _make_search_result(chunks[2], score=0.8, rank=2),
            ],
        }
    )

    result = evaluate_retrieval(
        queries=queries,
        chunks=chunks,
        retriever=retriever,
        top_k=2,
    )

    assert result.top_k == 2
    assert result.query_count == 2
    assert result.evaluated_query_count == 2
    assert result.no_answer_query_count == 0

    assert result.hit_rate_at_k == pytest.approx(1.0)
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.mean_reciprocal_rank == pytest.approx(0.75)
    assert result.ndcg_at_k > 0.0

    assert retriever.calls == [
        ("first query", 2),
        ("second query", 2),
    ]


def test_evaluate_retrieval_skips_no_answer_queries() -> None:
    chunks = _make_chunks()

    queries = [
        _make_query(
            query_id="query-1",
            query="answerable query",
            relevant_chunk_ids=("chunk-1",),
            chunks=chunks,
        ),
        GoldenQuery(
            query_id="query-2",
            query="no answer query",
            relevant_chunks=(),
            notes="",
        ),
    ]

    retriever = FakeRetriever(
        {
            "answerable query": [
                _make_search_result(chunks[0], score=1.0, rank=1),
            ],
        }
    )

    result = evaluate_retrieval(
        queries=queries,
        chunks=chunks,
        retriever=retriever,
        top_k=1,
    )

    assert result.query_count == 2
    assert result.evaluated_query_count == 1
    assert result.no_answer_query_count == 1

    assert retriever.calls == [
        ("answerable query", 1),
    ]


def test_evaluate_retrieval_returns_zero_metrics_when_no_queries_are_answerable() -> None:
    chunks = _make_chunks()

    queries = [
        GoldenQuery(
            query_id="query-1",
            query="no answer query",
            relevant_chunks=(),
            notes="",
        ),
    ]

    retriever = FakeRetriever({})

    result = evaluate_retrieval(
        queries=queries,
        chunks=chunks,
        retriever=retriever,
        top_k=3,
    )

    assert result.query_count == 1
    assert result.evaluated_query_count == 0
    assert result.no_answer_query_count == 1
    assert result.hit_rate_at_k == 0.0
    assert result.recall_at_k == 0.0
    assert result.mean_reciprocal_rank == 0.0
    assert result.ndcg_at_k == 0.0

    assert retriever.calls == []


@pytest.mark.parametrize(
    "top_k",
    [0, -1],
)
def test_evaluate_retrieval_rejects_invalid_top_k(
    top_k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"top_k must be at least 1, got {top_k}",
    ):
        evaluate_retrieval(
            queries=[],
            chunks=[],
            retriever=FakeRetriever({}),
            top_k=top_k,
        )


def test_evaluate_queries_returns_detailed_results() -> None:
    chunks = _make_chunks()

    queries = [
        _make_query(
            query_id="query-1",
            query="test query",
            relevant_chunk_ids=("chunk-2",),
            chunks=chunks,
        ),
    ]

    retriever = FakeRetriever(
        {
            "test query": [
                _make_search_result(chunks[0], score=0.9, rank=1),
                _make_search_result(chunks[1], score=0.8, rank=2),
            ],
        }
    )

    results = evaluate_queries(
        queries=queries,
        chunks=chunks,
        retriever=retriever,
        top_k=2,
    )

    assert len(results) == 1

    result = results[0]

    assert result.query_id == "query-1"
    assert result.query == "test query"
    assert result.relevant_chunk_ids == ("chunk-2",)
    assert result.first_relevant_rank == 2

    assert result.hit_rate_at_k == pytest.approx(1.0)
    assert result.recall_at_k == pytest.approx(1.0)
    assert result.reciprocal_rank == pytest.approx(0.5)

    assert [chunk.chunk_id for chunk in result.retrieved_chunks] == [
        "chunk-1",
        "chunk-2",
    ]
    assert [chunk.relevance for chunk in result.retrieved_chunks] == [
        0,
        1,
    ]


def test_evaluate_queries_skips_no_answer_queries() -> None:
    chunks = _make_chunks()

    queries = [
        GoldenQuery(
            query_id="query-1",
            query="no answer query",
            relevant_chunks=(),
            notes="",
        ),
    ]

    retriever = FakeRetriever({})

    results = evaluate_queries(
        queries=queries,
        chunks=chunks,
        retriever=retriever,
        top_k=1,
    )

    assert results == ()
    assert retriever.calls == []


@pytest.mark.parametrize(
    "top_k",
    [0, -1],
)
def test_evaluate_queries_rejects_invalid_top_k(
    top_k: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=f"top_k must be at least 1, got {top_k}",
    ):
        evaluate_queries(
            queries=[],
            chunks=[],
            retriever=FakeRetriever({}),
            top_k=top_k,
        )


def _make_search_result(
    chunk: Chunk,
    *,
    score: float,
    rank: int,
) -> SearchResult:
    return SearchResult(
        chunk=chunk,
        score=score,
        rank=rank,
    )


def _make_query(
    *,
    query_id: str,
    query: str,
    relevant_chunk_ids: tuple[str, ...],
    chunks: list[Chunk],
) -> GoldenQuery:
    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    return GoldenQuery(
        query_id=query_id,
        query=query,
        relevant_chunks=tuple(
            RelevantChunk(
                chunk_id=chunk_id,
                relative_path=chunks_by_id[chunk_id].relative_path,
                section_path=chunks_by_id[chunk_id].section_path,
                relevance=1,
            )
            for chunk_id in relevant_chunk_ids
        ),
        notes="",
    )


def _make_chunks() -> list[Chunk]:
    return [
        _make_chunk("chunk-1", 0),
        _make_chunk("chunk-2", 1),
        _make_chunk("chunk-3", 2),
    ]


def _make_chunk(
    chunk_id: str,
    chunk_index: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="document-1",
        source_name="test-source",
        relative_path="notes/test.md",
        title="Test",
        section_path=("Test",),
        content=f"Content {chunk_index}",
        searchable_text=f"Searchable text {chunk_index}",
        chunk_index=chunk_index,
        section_chunk_index=chunk_index,
        start_line=1,
        end_line=2,
        content_hash=f"content-hash-{chunk_index}",
    )