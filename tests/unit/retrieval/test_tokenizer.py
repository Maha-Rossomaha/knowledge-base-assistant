import pytest

from knowledge_base_assistant.retrieval.lexical.tokenizer import tokenize


def test_tokenize_lowercases_latin_text() -> None:
    assert tokenize(
        "BM25 Search RETRIEVAL"
    ) == (
        "bm25",
        "search",
        "retrieval",
    )


def test_tokenize_lowercases_cyrillic_text() -> None:
    assert tokenize(
        "Поиск РЕЛЕВАНТНЫХ Документов"
    ) == (
        "поиск",
        "релевантных",
        "документов",
    )


def test_tokenize_supports_mixed_cyrillic_and_latin() -> None:
    assert tokenize(
        "BM25 ищет релевантные Documents"
    ) == (
        "bm25",
        "ищет",
        "релевантные",
        "documents",
    )


def test_tokenize_removes_punctuation() -> None:
    assert tokenize(
        "BM25: ranking, retrieval! Search?"
    ) == (
        "bm25",
        "ranking",
        "retrieval",
        "search",
    )


def test_tokenize_treats_whitespace_as_separator() -> None:
    assert tokenize(
        "Sparse retrieval\nDense\tretrieval"
    ) == (
        "sparse",
        "retrieval",
        "dense",
        "retrieval",
    )


def test_tokenize_preserves_numbers() -> None:
    assert tokenize(
        "Qwen2 has 32 layers and 128 heads"
    ) == (
        "qwen2",
        "has",
        "32",
        "layers",
        "and",
        "128",
        "heads",
    )


def test_tokenize_preserves_hyphenated_terms() -> None:
    assert tokenize(
        "TF-IDF top-k cross-encoder"
    ) == (
        "tf-idf",
        "top-k",
        "cross-encoder",
    )


def test_tokenize_preserves_underscore_terms() -> None:
    assert tokenize(
        "chunk_id max_tokens document_count"
    ) == (
        "chunk_id",
        "max_tokens",
        "document_count",
    )


def test_tokenize_preserves_repeated_tokens() -> None:
    assert tokenize(
        "retrieval retrieval search"
    ) == (
        "retrieval",
        "retrieval",
        "search",
    )


def test_tokenize_splits_slash_separated_terms() -> None:
    assert tokenize(
        "sparse/dense retrieval"
    ) == (
        "sparse",
        "dense",
        "retrieval",
    )


def test_tokenize_splits_dot_separated_terms() -> None:
    assert tokenize(
        "python3.12 sentence.transformers"
    ) == (
        "python3",
        "12",
        "sentence",
        "transformers",
    )


def test_tokenize_does_not_keep_standalone_hyphens() -> None:
    assert tokenize(
        "sparse - dense -- hybrid"
    ) == (
        "sparse",
        "dense",
        "hybrid",
    )


def test_tokenize_does_not_keep_standalone_underscores() -> None:
    assert tokenize(
        "chunk _ document __ query"
    ) == (
        "chunk",
        "document",
        "query",
    )


@pytest.mark.parametrize(
    "text",
    [
        "",
        " ",
        "\n\t",
        "...",
        "---",
        "___",
        "!@#$%^&*()",
    ],
)
def test_tokenize_returns_empty_tuple_when_there_are_no_tokens(
    text: str,
) -> None:
    assert tokenize(text) == ()


def test_tokenize_returns_tuple() -> None:
    tokens = tokenize("BM25 retrieval")

    assert tokens == ("bm25", "retrieval")
    assert isinstance(tokens, tuple)


def test_tokenize_is_deterministic() -> None:
    text = "BM25 и TF-IDF для sparse retrieval"

    assert tokenize(text) == tokenize(text)