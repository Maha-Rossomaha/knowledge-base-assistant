from dataclasses import dataclass
from pathlib import Path

from knowledge_base_assistant.ingestion.chunker import (
    ChunkerConfig,
    chunk_sections,
)
from knowledge_base_assistant.ingestion.markdown_parser import (
    parse_markdown_sections,
)
from knowledge_base_assistant.ingestion.scanner import scan_documents
from knowledge_base_assistant.serialization.jsonl import (
    write_chunks_jsonl,
)


@dataclass(frozen=True, slots=True)
class IndexingResult:
    document_count: int
    section_count: int
    chunk_count: int
    output_path: Path
    

def build_chunks(
    repository_path: Path,
    source_name: str,
    output_path: Path,
    chunker_config: ChunkerConfig,
) -> IndexingResult:
    if not repository_path.exists():
        raise FileNotFoundError(
            f"Repository path does not exist: {repository_path}"
        )

    if not repository_path.is_dir():
        raise NotADirectoryError(
            f"Repository path is not a directory: {repository_path}"
        )

    documents = scan_documents(
        repository_path,
        source_name,
    )

    all_chunks = []
    section_count = 0

    for document in documents:
        sections = parse_markdown_sections(document)

        section_count += len(sections)

        document_chunks = chunk_sections(
            document,
            sections,
            chunker_config,
        )

        all_chunks.extend(document_chunks)

    write_chunks_jsonl(
        all_chunks,
        output_path,
    )

    return IndexingResult(
        document_count=len(documents),
        section_count=section_count,
        chunk_count=len(all_chunks),
        output_path=output_path,
    )