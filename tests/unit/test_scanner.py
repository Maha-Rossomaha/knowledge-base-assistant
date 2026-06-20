from pathlib import Path

import pytest

from knowledge_base_assistant.ingestion.scanner import ScannerConfig, scan_documents


def test_scanner_finds_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("# Note\nContent", encoding="utf-8")

    documents = scan_documents(tmp_path, source_name="test-source")

    assert len(documents) == 1
    assert documents[0].relative_path == "note.md"
    assert documents[0].content == "# Note\nContent"


def test_scanner_ignores_non_markdown_files(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("# Note", encoding="utf-8")
    (tmp_path / "script.py").write_text("print('hello')", encoding="utf-8")

    documents = scan_documents(tmp_path, source_name="test-source")

    assert [document.relative_path for document in documents] == ["note.md"]


def test_scanner_ignores_excluded_directories(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "hidden.md").write_text("# Hidden", encoding="utf-8")
    (tmp_path / "visible.md").write_text("# Visible", encoding="utf-8")

    documents = scan_documents(tmp_path, source_name="test-source")

    assert [document.relative_path for document in documents] == ["visible.md"]


def test_scanner_returns_documents_in_sorted_order(tmp_path: Path) -> None:
    (tmp_path / "b.md").write_text("# B", encoding="utf-8")
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")

    documents = scan_documents(tmp_path, source_name="test-source")

    assert [document.relative_path for document in documents] == ["a.md", "b.md"]


def test_scanner_handles_nested_paths(tmp_path: Path) -> None:
    nested_dir = tmp_path / "folder" / "nested"
    nested_dir.mkdir(parents=True)
    (nested_dir / "note.md").write_text("# Nested", encoding="utf-8")

    documents = scan_documents(tmp_path, source_name="test-source")

    assert documents[0].relative_path == "folder/nested/note.md"


def test_scanner_fills_document_ids_and_hashes(tmp_path: Path) -> None:
    (tmp_path / "note.md").write_text("# Note", encoding="utf-8")

    documents = scan_documents(tmp_path, source_name="test-source")
    document = documents[0]

    assert document.document_id
    assert document.content_hash
    assert document.source_name == "test-source"


def test_scanner_rejects_missing_repository_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "missing"

    with pytest.raises(FileNotFoundError):
        scan_documents(missing_root, source_name="test-source")


def test_scanner_rejects_file_as_repository_root(tmp_path: Path) -> None:
    file_path = tmp_path / "file.md"
    file_path.write_text("# File", encoding="utf-8")

    with pytest.raises(NotADirectoryError):
        scan_documents(file_path, source_name="test-source")


def test_custom_config_allows_txt_files(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("Plain text", encoding="utf-8")

    config = ScannerConfig(
        included_extensions=frozenset({".txt"}),
        excluded_dir_names=frozenset(),
    )

    documents = scan_documents(tmp_path, source_name="test-source", config=config)

    assert [document.relative_path for document in documents] == ["note.txt"]
