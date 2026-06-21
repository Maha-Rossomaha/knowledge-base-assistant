from collections.abc import Sequence

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.evaluation.models import (
    GoldenQuery,
    GoldenValidationResult,
)


def validate_golden_queries(
    queries: Sequence[GoldenQuery],
    chunks: Sequence[Chunk],
) -> GoldenValidationResult:
    chunks_by_id = {
        chunk.chunk_id: chunk
        for chunk in chunks
    }

    seen_query_ids: set[str] = set()
    answerable_query_count = 0
    no_answer_query_count = 0
    relevance_judgment_count = 0

    for query in queries:
        _validate_query(
            query=query,
            seen_query_ids=seen_query_ids,
        )

        if query.relevant_chunks:
            answerable_query_count += 1
        else:
            no_answer_query_count += 1

        seen_chunk_ids: set[str] = set()

        for relevant_chunk in query.relevant_chunks:
            if relevant_chunk.relevance not in {1, 2}:
                raise ValueError(
                    f"{query.query_id}: relevance must be 1 or 2, "
                    f"got {relevant_chunk.relevance}"
                )

            if not relevant_chunk.chunk_id.strip():
                raise ValueError(
                    f"{query.query_id}: chunk_id must not be empty"
                )

            if relevant_chunk.chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"{query.query_id}: duplicate chunk_id "
                    f"{relevant_chunk.chunk_id}"
                )

            seen_chunk_ids.add(relevant_chunk.chunk_id)

            actual_chunk = chunks_by_id.get(
                relevant_chunk.chunk_id
            )

            if actual_chunk is None:
                raise ValueError(
                    f"{query.query_id}: unknown chunk_id "
                    f"{relevant_chunk.chunk_id}"
                )

            if (
                relevant_chunk.relative_path
                != actual_chunk.relative_path
            ):
                raise ValueError(
                    f"{query.query_id}: relative_path mismatch "
                    f"for chunk {relevant_chunk.chunk_id}. "
                    f"Expected {actual_chunk.relative_path!r}, "
                    f"got {relevant_chunk.relative_path!r}"
                )

            if (
                relevant_chunk.section_path
                != actual_chunk.section_path
            ):
                raise ValueError(
                    f"{query.query_id}: section_path mismatch "
                    f"for chunk {relevant_chunk.chunk_id}. "
                    f"Expected {actual_chunk.section_path!r}, "
                    f"got {relevant_chunk.section_path!r}"
                )

            relevance_judgment_count += 1

    return GoldenValidationResult(
        query_count=len(queries),
        answerable_query_count=answerable_query_count,
        no_answer_query_count=no_answer_query_count,
        relevance_judgment_count=relevance_judgment_count,
    )


def _validate_query(
    *,
    query: GoldenQuery,
    seen_query_ids: set[str],
) -> None:
    if not query.query_id.strip():
        raise ValueError("query_id must not be empty")

    if not query.query.strip():
        raise ValueError(
            f"{query.query_id}: query must not be empty"
        )

    if query.query_id in seen_query_ids:
        raise ValueError(
            f"Duplicate query_id: {query.query_id}"
        )

    seen_query_ids.add(query.query_id)