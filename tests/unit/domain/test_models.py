from dataclasses import FrozenInstanceError

import pytest

from knowledge_base_assistant.domain.models import Chunk, Document


def test_document_can_be_created() -> None:
    document = Document(
        document_id="doc-1",
        source_name="road-into-llm",
        relative_path="1 Theory/RAG.md",
        content="# RAG\nText",
        content_hash="hash-1",
    )

    assert document.document_id == "doc-1"
    assert document.relative_path == "1 Theory/RAG.md"


def test_document_is_immutable() -> None:
    document = Document(
        document_id="doc-1",
        source_name="road-into-llm",
        relative_path="1 Theory/RAG.md",
        content="# RAG\nText",
        content_hash="hash-1",
    )

    with pytest.raises(FrozenInstanceError):
        document.content = "changed"  # type: ignore[misc]


def test_chunk_can_be_created() -> None:
    chunk = Chunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        source_name="road-into-llm",
        relative_path="1 Theory/RAG.md",
        title="RAG",
        section_path=("RAG", "Chunking"),
        content="Chunk text",
        searchable_text="RAG > Chunking\n\nChunk text",
        chunk_index=0,
        start_line=1,
        end_line=10,
        content_hash="hash-2",
    )

    assert chunk.section_path == ("RAG", "Chunking")
    assert chunk.start_line == 1
    assert chunk.end_line == 10
