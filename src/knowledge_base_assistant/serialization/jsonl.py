import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.evaluation.models import GoldenQuery, RelevantChunk


def write_chunks_jsonl(
    chunks: list[Chunk],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            data = asdict(chunk)

            file.write(
                json.dumps(
                    data,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
            file.write("\n")


def read_chunks_jsonl(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}") from error

            try:
                chunk = _chunk_from_dict(data)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid chunk at line {line_number}: {error}") from error

            chunks.append(chunk)

    return chunks


def read_golden_queries_jsonl(
    path: Path,
) -> list[GoldenQuery]:
    queries: list[GoldenQuery] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL at line {line_number}") from error

            try:
                query = _golden_query_from_dict(data)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid golden query at line {line_number}: {error}"
                ) from error

            queries.append(query)

    return queries


def _chunk_from_dict(data: Any) -> Chunk:
    if not isinstance(data, dict):
        raise TypeError("JSON value must be an object")

    section_path = data["section_path"]

    if not isinstance(section_path, list):
        raise TypeError("section_path must be a list")

    return Chunk(
        chunk_id=data["chunk_id"],
        document_id=data["document_id"],
        source_name=data["source_name"],
        relative_path=data["relative_path"],
        title=data["title"],
        section_path=tuple(section_path),
        content=data["content"],
        searchable_text=data["searchable_text"],
        chunk_index=data["chunk_index"],
        section_chunk_index=data["section_chunk_index"],
        start_line=data["start_line"],
        end_line=data["end_line"],
        content_hash=data["content_hash"],
    )


def _golden_query_from_dict(data: Any) -> GoldenQuery:
    if not isinstance(data, dict):
        raise TypeError("JSON value must be an object")

    relevant_chunks = data["relevant_chunks"]
    if not isinstance(relevant_chunks, list):
        raise TypeError("relevant_chunks must be a list")

    return GoldenQuery(
        query_id=_required_string(data, "query_id"),
        query=_required_string(data, "query"),
        relevant_chunks=tuple(
            _relevant_chunk_from_dict(relevant_chunk)
            for relevant_chunk in relevant_chunks
        ),
        notes=_required_string(data, "notes"),
    )


def _relevant_chunk_from_dict(data: Any) -> RelevantChunk:
    if not isinstance(data, dict):
        raise TypeError("relevant_chunks item must be an object")

    section_path = data["section_path"]
    if not isinstance(section_path, list):
        raise TypeError("section_path must be a list")
    if not all(isinstance(section, str) for section in section_path):
        raise TypeError("section_path must contain only strings")

    relevance = data["relevance"]
    if not isinstance(relevance, int):
        raise TypeError("relevance must be an integer")

    return RelevantChunk(
        chunk_id=_required_string(data, "chunk_id"),
        relative_path=_required_string(data, "relative_path"),
        section_path=tuple(section_path),
        relevance=relevance,
    )


def _required_string(data: dict[str, Any], field: str) -> str:
    value = data[field]
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    return value
