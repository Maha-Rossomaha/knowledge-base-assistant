import hashlib
from pathlib import PurePosixPath

_HASH_SEPARATOR = "\x1f"


def stable_hash(*parts: str) -> str:
    value = _HASH_SEPARATOR.join(parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_relative_path(path: str) -> str:
    normalized = path.replace("\\", "/")
    normalized = str(PurePosixPath(normalized))

    if normalized.startswith("/"):
        raise ValueError("Path must be relative")

    if normalized == ".":
        raise ValueError("Path must not be empty")

    if ".." in PurePosixPath(normalized).parts:
        raise ValueError("Path must not contain '..'")

    return normalized


def make_document_id(source_name: str, relative_path: str) -> str:
    return stable_hash(source_name, normalize_relative_path(relative_path))


def make_content_hash(content: str) -> str:
    return stable_hash(content)


def make_chunk_id(
    document_id: str,
    section_path: tuple[str, ...],
    chunk_index: int,
    content: str,
) -> str:
    return stable_hash(document_id, *section_path, str(chunk_index), content)