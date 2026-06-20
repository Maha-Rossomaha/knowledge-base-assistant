from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from knowledge_base_assistant.domain.ids import (
    make_content_hash,
    make_document_id,
    normalize_relative_path,
)
from knowledge_base_assistant.domain.models import Document


@dataclass(frozen=True, slots=True)
class ScannerConfig:
    included_extensions: frozenset[str]
    excluded_dir_names: frozenset[str]
    excluded_file_patterns: tuple[str, ...]
    

DEFAULT_SCANNER_CONFIG = ScannerConfig(
    included_extensions=frozenset({".md"}),
    excluded_dir_names=frozenset(
        {
            ".git",
            ".venv",
            "venv",
            "env",
            "rec_venv",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            ".ipynb_checkpoints",
            ".runtime",
            ".cache",
            ".clinerules",
            "qdrant_storage",
            "node_modules",
            "data",
            "0 images",
        }
    ),
    excluded_file_patterns=("*_plan*.md",)
)


def scan_documents(
    repository_root: Path,
    source_name: str,
    config: ScannerConfig = DEFAULT_SCANNER_CONFIG,
) -> list[Document]:
    root = repository_root.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Repository root does not exist: {repository_root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Repository root is not a directory: {repository_root}")

    documents: list[Document] = []

    for file_path in _iter_source_files(root, config):
        content = file_path.read_text(encoding="utf-8")
        relative_path = normalize_relative_path(str(file_path.relative_to(root)))

        documents.append(
            Document(
                document_id=make_document_id(source_name, relative_path),
                source_name=source_name,
                relative_path=relative_path,
                content=content,
                content_hash=make_content_hash(content),
            )
        )

    return documents


def _iter_source_files(root: Path, config: ScannerConfig) -> Iterable[Path]:
    paths: list[Path] = []

    for path in root.rglob("*"):
        if _is_inside_excluded_dir(path, root, config.excluded_dir_names):
            continue

        if not path.is_file():
            continue

        if path.suffix.lower() not in config.included_extensions:
            continue
        
        if any(path.match(pattern) for pattern in config.excluded_file_patterns):
            continue

        paths.append(path)

    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _is_inside_excluded_dir(
    path: Path,
    root: Path,
    excluded_dir_names: frozenset[str],
) -> bool:
    relative_parts = path.relative_to(root).parts
    return any(part in excluded_dir_names for part in relative_parts)
