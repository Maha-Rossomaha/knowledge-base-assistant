from pathlib import Path

from knowledge_base_assistant.application.lexical_search import (
    search_chunks_file,
)
from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.serialization.jsonl import (
    write_chunks_jsonl,
)


def make_chunk(
    *,
    chunk_id: str,
    searchable_text: str,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
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


def test_search_chunks_file_returns_ranked_results(
    tmp_path: Path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"

    write_chunks_jsonl(
        [
            make_chunk(
                chunk_id="bm25",
                searchable_text=(
                    "BM25 lexical retrieval ranking"
                ),
            ),
            make_chunk(
                chunk_id="docker",
                searchable_text=(
                    "Docker container image"
                ),
            ),
        ],
        chunks_path,
    )

    results = search_chunks_file(
        chunks_path=chunks_path,
        query="BM25 retrieval",
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "bm25"
    assert results[0].rank == 1
    assert results[0].score > 0.0