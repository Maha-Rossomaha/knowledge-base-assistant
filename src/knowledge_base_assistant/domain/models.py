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
    start_line: int
    end_line: int

    content_hash: str