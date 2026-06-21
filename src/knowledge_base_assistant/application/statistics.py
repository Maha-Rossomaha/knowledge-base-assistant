from dataclasses import dataclass
from pathlib import Path

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.serialization.jsonl import read_chunks_jsonl


@dataclass(frozen=True, slots=True)
class LargeChunkInfo:
    chunk_id: str
    relative_path: str
    section_path: tuple[str, ...]
    char_count: int
    line_count: int
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class ChunkStatistics:
    chunk_count: int
    document_count: int

    min_char_count: int
    average_char_count: float
    max_char_count: int

    min_line_count: int
    average_line_count: float
    max_line_count: int

    chunks_with_code_fence: int
    chunks_over_max_chars: int
    chunks_over_max_lines: int

    largest_chunks: tuple[LargeChunkInfo, ...]


def calculate_chunk_statistics(
    chunks: list[Chunk],
    *,
    max_chars: int = 3_000,
    max_lines: int = 40,
    largest_limit: int = 10,
) -> ChunkStatistics:
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    if max_lines <= 0:
        raise ValueError("max_lines must be positive")

    if largest_limit < 0:
        raise ValueError("largest_limit must be non-negative")

    if not chunks:
        return ChunkStatistics(
            chunk_count=0,
            document_count=0,
            min_char_count=0,
            average_char_count=0.0,
            max_char_count=0,
            min_line_count=0,
            average_line_count=0.0,
            max_line_count=0,
            chunks_with_code_fence=0,
            chunks_over_max_chars=0,
            chunks_over_max_lines=0,
            largest_chunks=(),
        )

    char_counts = [len(chunk.content) for chunk in chunks]
    line_counts = [_count_lines(chunk.content) for chunk in chunks]

    document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    chunks_with_code_fence = sum(
        _contains_code_fence(chunk.content)
        for chunk in chunks
    )

    chunks_over_max_chars = sum(
        char_count > max_chars
        for char_count in char_counts
    )

    chunks_over_max_lines = sum(
        line_count > max_lines
        for line_count in line_counts
    )

    sorted_chunks = sorted(
        chunks,
        key=lambda chunk: len(chunk.content),
        reverse=True,
    )

    largest_chunks = tuple(
        _make_large_chunk_info(chunk)
        for chunk in sorted_chunks[:largest_limit]
    )

    return ChunkStatistics(
        chunk_count=len(chunks),
        document_count=len(document_ids),
        min_char_count=min(char_counts),
        average_char_count=sum(char_counts) / len(char_counts),
        max_char_count=max(char_counts),
        min_line_count=min(line_counts),
        average_line_count=sum(line_counts) / len(line_counts),
        max_line_count=max(line_counts),
        chunks_with_code_fence=chunks_with_code_fence,
        chunks_over_max_chars=chunks_over_max_chars,
        chunks_over_max_lines=chunks_over_max_lines,
        largest_chunks=largest_chunks,
    )


def calculate_chunk_statistics_from_jsonl(
    path: Path,
    *,
    max_chars: int = 3_000,
    max_lines: int = 40,
    largest_limit: int = 10,
) -> ChunkStatistics:
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks JSONL does not exist: {path}"
        )

    if not path.is_file():
        raise IsADirectoryError(
            f"Chunks JSONL path is not a file: {path}"
        )

    chunks = read_chunks_jsonl(path)

    return calculate_chunk_statistics(
        chunks,
        max_chars=max_chars,
        max_lines=max_lines,
        largest_limit=largest_limit,
    )


def _count_lines(content: str) -> int:
    if not content:
        return 0

    return len(content.splitlines())


def _contains_code_fence(content: str) -> bool:
    return any(
        line.lstrip().startswith(("```", "~~~"))
        for line in content.splitlines()
    )


def _make_large_chunk_info(chunk: Chunk) -> LargeChunkInfo:
    return LargeChunkInfo(
        chunk_id=chunk.chunk_id,
        relative_path=chunk.relative_path,
        section_path=chunk.section_path,
        char_count=len(chunk.content),
        line_count=_count_lines(chunk.content),
        start_line=chunk.start_line,
        end_line=chunk.end_line,
    )