from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Document:
    document_id: str
    source_name: str
    relative_path: str
    content: str
    content_hash: str
    

@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    source_name: str
    relative_path: str
    title: str | None
    section_path: tuple[str, ...]
    content: str
    searchable_text: str
    chunk_index: int
    section_chunk_index: int
    start_line: int
    end_line: int
    content_hash: str
    

@dataclass(frozen=True, slots=True)
class RelevantChunk:
    chunk_id: str
    relative_path: str
    section_path: tuple[str, ...]
    relevance: int


@dataclass(frozen=True, slots=True)
class GoldenQuery:
    query_id: str
    query: str
    relevant_chunks: tuple[RelevantChunk, ...]
    notes: str


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    document_id: str
    section_path: tuple[str, ...]
    content: str
    start_line: int
    content_start_line: int | None
    end_line: int