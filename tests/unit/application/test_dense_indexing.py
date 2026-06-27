import hashlib

from knowledge_base_assistant.application.dense_indexing import (
    calculate_file_sha256,
)


def test_calculate_file_sha256_returns_expected_hash(
    tmp_path,
) -> None:
    path = tmp_path / "chunks.jsonl"
    content = b"chunk data"

    path.write_bytes(content)

    result = calculate_file_sha256(path)

    assert result == hashlib.sha256(content).hexdigest()


def test_calculate_file_sha256_is_stable_for_unchanged_file(
    tmp_path,
) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_bytes(b"same content")

    first_hash = calculate_file_sha256(path)
    second_hash = calculate_file_sha256(path)

    assert first_hash == second_hash


def test_calculate_file_sha256_changes_when_file_changes(
    tmp_path,
) -> None:
    path = tmp_path / "chunks.jsonl"
    path.write_bytes(b"first content")

    first_hash = calculate_file_sha256(path)

    path.write_bytes(b"second content")

    second_hash = calculate_file_sha256(path)

    assert second_hash != first_hash