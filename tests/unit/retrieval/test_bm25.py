import pytest

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.retrieval.bm25 import BM25Index


def make_chunk(
    *,
    chunk_id: str,
    searchable_text: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        source_name="test-source",
        relative_path=f"notes/{chunk_id}.md",
        title=chunk_id,
        section_path=(chunk_id,),
        content=searchable_text,
        searchable_text=searchable_text,
        chunk_index=0,
        section_chunk_index=0,
        start_line=1,
        end_line=1,
        content_hash=f"hash-{chunk_id}",
    )


def test_search_returns_most_relevant_chunk_first() -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="bm25",
                searchable_text=(
                    "BM25 BM25 lexical retrieval ranking"
                ),
            ),
            make_chunk(
                chunk_id="dense",
                searchable_text=(
                    "Dense semantic vector retrieval"
                ),
            ),
        ]
    )

    results = index.search(
        "BM25 lexical retrieval",
        top_k=2,
    )

    assert results[0].chunk.chunk_id == "bm25"
    assert results[0].rank == 1
    assert results[0].score > results[1].score


def test_search_returns_only_matching_chunks() -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="bm25",
                searchable_text="BM25 lexical search",
            ),
            make_chunk(
                chunk_id="docker",
                searchable_text="Docker container image",
            ),
        ]
    )

    results = index.search("BM25")

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "bm25"


def test_search_returns_empty_list_for_unknown_term() -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="bm25",
                searchable_text="BM25 lexical search",
            )
        ]
    )

    assert index.search("unseen-term") == []


@pytest.mark.parametrize(
    "query",
    [
        "",
        " ",
        "\n\t",
        "...",
    ],
)
def test_search_returns_empty_list_for_empty_query(
    query: str,
) -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="bm25",
                searchable_text="BM25 search",
            )
        ]
    )

    assert index.search(query) == []


def test_empty_index_returns_no_results() -> None:
    index = BM25Index.build([])

    assert index.search("BM25") == []


def test_top_k_limits_number_of_results() -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="chunk-1",
                searchable_text="retrieval search",
            ),
            make_chunk(
                chunk_id="chunk-2",
                searchable_text="retrieval ranking",
            ),
            make_chunk(
                chunk_id="chunk-3",
                searchable_text="retrieval system",
            ),
        ]
    )

    results = index.search(
        "retrieval",
        top_k=2,
    )

    assert len(results) == 2
    assert [result.rank for result in results] == [1, 2]


def test_results_are_sorted_by_descending_score() -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="low",
                searchable_text="bm25 search",
            ),
            make_chunk(
                chunk_id="high",
                searchable_text=(
                    "bm25 bm25 bm25 search ranking"
                ),
            ),
        ]
    )

    results = index.search("bm25")

    assert len(results) == 2
    assert results[0].score >= results[1].score


def test_repeated_document_term_increases_score() -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="once",
                searchable_text="bm25 retrieval",
            ),
            make_chunk(
                chunk_id="repeated",
                searchable_text=(
                    "bm25 bm25 bm25 retrieval"
                ),
            ),
        ],
        b=0.0,
    )

    results = index.search("bm25")

    assert results[0].chunk.chunk_id == "repeated"
    assert results[0].score > results[1].score


def test_document_length_normalization_affects_score() -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="short",
                searchable_text="bm25 retrieval",
            ),
            make_chunk(
                chunk_id="long",
                searchable_text=(
                    "bm25 retrieval additional unrelated "
                    "words make this document much longer"
                ),
            ),
        ],
        b=0.75,
    )

    results = index.search("bm25 retrieval")

    assert results[0].chunk.chunk_id == "short"
    assert results[0].score > results[1].score


def test_search_uses_searchable_text() -> None:
    chunk = make_chunk(
        chunk_id="heading-match",
        searchable_text=(
            "Sparse retrieval > Reciprocal Rank Fusion"
        ),
    )

    index = BM25Index.build([chunk])

    results = index.search("reciprocal rank fusion")

    assert len(results) == 1
    assert results[0].chunk == chunk


def test_hyphenated_terms_are_searchable() -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="tf-idf",
                searchable_text=(
                    "TF-IDF is a lexical weighting method"
                ),
            )
        ]
    )

    results = index.search("TF-IDF")

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "tf-idf"


def test_repeated_query_terms_do_not_change_score() -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="bm25",
                searchable_text="BM25 lexical retrieval",
            )
        ]
    )

    single_result = index.search("bm25")[0]
    repeated_result = index.search(
        "bm25 bm25 bm25"
    )[0]

    assert repeated_result.score == pytest.approx(
        single_result.score
    )


def test_equal_scores_are_sorted_by_chunk_id() -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="chunk-b",
                searchable_text="retrieval",
            ),
            make_chunk(
                chunk_id="chunk-a",
                searchable_text="retrieval",
            ),
        ]
    )

    results = index.search("retrieval")

    assert [
        result.chunk.chunk_id
        for result in results
    ] == [
        "chunk-a",
        "chunk-b",
    ]


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_search_rejects_invalid_top_k(
    top_k: int,
) -> None:
    index = BM25Index.build([])

    with pytest.raises(
        ValueError,
        match="top_k must be at least 1",
    ):
        index.search(
            "query",
            top_k=top_k,
        )


@pytest.mark.parametrize(
    "k1",
    [
        0.0,
        -0.1,
    ],
)
def test_build_rejects_invalid_k1(
    k1: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="k1 must be greater than 0",
    ):
        BM25Index.build(
            [],
            k1=k1,
        )


@pytest.mark.parametrize(
    "b",
    [
        -0.1,
        1.1,
    ],
)
def test_build_rejects_invalid_b(
    b: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="b must be between 0 and 1",
    ):
        BM25Index.build(
            [],
            b=b,
        )


@pytest.mark.parametrize(
    "b",
    [
        0.0,
        1.0,
    ],
)
def test_build_accepts_boundary_b_values(
    b: float,
) -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="chunk",
                searchable_text="retrieval",
            )
        ],
        b=b,
    )

    assert index.search("retrieval")


def test_zero_average_document_length_returns_zero_ratio() -> None:
    index = BM25Index.build(
        [
            make_chunk(
                chunk_id="empty",
                searchable_text="---",
            )
        ]
    )

    assert index._length_ratio(0) == 0.0