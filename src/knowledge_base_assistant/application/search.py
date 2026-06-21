from pathlib import Path

from knowledge_base_assistant.retrieval.bm25 import BM25Index
from knowledge_base_assistant.retrieval.models import SearchResult
from knowledge_base_assistant.serialization.jsonl import (
    read_chunks_jsonl,
)


def search_chunks_file(
    *,
    chunks_path: Path,
    query: str,
    top_k: int = 5,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[SearchResult]:
    chunks = read_chunks_jsonl(chunks_path)

    index = BM25Index.build(
        chunks,
        k1=k1,
        b=b,
    )

    return index.search(
        query,
        top_k=top_k,
    )