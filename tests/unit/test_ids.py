import pytest

from knowledge_base_assistant.domain.ids import (
    make_chunk_id,
    make_content_hash,
    make_document_id,
    normalize_relative_path,
)


def test_document_id_is_stable() -> None:
    first = make_document_id("road-into-llm", "a/b.md")
    second = make_document_id("road-into-llm", "a/b.md")

    assert first == second


def test_document_id_depends_on_source_name() -> None:
    first = make_document_id("source-a", "a/b.md")
    second = make_document_id("source-b", "a/b.md")

    assert first != second


def test_document_id_does_not_depend_on_content() -> None:
    first = make_document_id("road-into-llm", "a/b.md")
    second = make_document_id("road-into-llm", "a/b.md")

    assert first == second


def test_content_hash_depends_on_content() -> None:
    first = make_content_hash("old content")
    second = make_content_hash("new content")

    assert first != second


def test_chunk_id_depends_on_section_path() -> None:
    first = make_chunk_id("doc-1", ("A", "B"), "content")
    second = make_chunk_id("doc-1", ("A", "C"), "content")

    assert first != second


def test_path_normalization_converts_backslashes() -> None:
    assert normalize_relative_path(r"a\b.md") == "a/b.md"
    
    
def test_path_normalization_rejects_commas() -> None:
    with pytest.raises(ValueError):
        normalize_relative_path(".")


def test_path_normalization_rejects_absolute_path() -> None:
    with pytest.raises(ValueError):
        normalize_relative_path("/a/b.md")


def test_path_normalization_rejects_parent_references() -> None:
    with pytest.raises(ValueError):
        normalize_relative_path("../a/b.md")