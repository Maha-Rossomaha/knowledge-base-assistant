from pathlib import Path

import pytest

from knowledge_base_assistant.serialization.jsonl import (
    read_golden_queries_jsonl,
)


def test_reads_golden_queries_jsonl(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        (
            '{"query_id":"q001","query":"What is BM25?",'
            '"relevant_chunks":[{"chunk_id":"chunk-1",'
            '"relative_path":"notes/bm25.md",'
            '"section_path":["BM25"],'
            '"relevance":2}],'
            '"notes":"Definition"}\n'
        ),
        encoding="utf-8",
    )

    queries = read_golden_queries_jsonl(path)

    assert len(queries) == 1
    assert queries[0].query_id == "q001"
    assert queries[0].query == "What is BM25?"
    assert queries[0].notes == "Definition"

    relevant_chunk = queries[0].relevant_chunks[0]

    assert relevant_chunk.chunk_id == "chunk-1"
    assert relevant_chunk.section_path == ("BM25",)
    assert relevant_chunk.relevance == 2


def test_golden_reader_skips_blank_lines(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        (
            "\n"
            '{"query_id":"q001","query":"Query",'
            '"relevant_chunks":[],"notes":""}\n'
            "\n"
        ),
        encoding="utf-8",
    )

    queries = read_golden_queries_jsonl(path)

    assert len(queries) == 1


def test_invalid_golden_json_reports_line_number(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        (
            '{"query_id":"q001","query":"Query",'
            '"relevant_chunks":[],"notes":""}\n'
            "{invalid json}\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid JSONL at line 2",
    ):
        read_golden_queries_jsonl(path)


def test_golden_record_must_be_object(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        '["not", "an", "object"]\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid golden query at line 1",
    ):
        read_golden_queries_jsonl(path)


def test_relevant_chunks_must_be_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        (
            '{"query_id":"q001","query":"Query",'
            '"relevant_chunks":"chunk-1","notes":""}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="relevant_chunks must be a list",
    ):
        read_golden_queries_jsonl(path)


def test_golden_section_path_must_contain_strings(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        (
            '{"query_id":"q001","query":"Query",'
            '"relevant_chunks":[{"chunk_id":"chunk-1",'
            '"relative_path":"notes.md",'
            '"section_path":["Heading",42],'
            '"relevance":2}],'
            '"notes":""}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="section_path must contain only strings",
    ):
        read_golden_queries_jsonl(path)
        
        
def test_relevant_chunk_must_be_object(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        (
            '{"query_id":"q001","query":"Query",'
            '"relevant_chunks":["chunk-1"],'
            '"notes":""}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="relevant_chunks item must be an object",
    ):
        read_golden_queries_jsonl(path)
        
        
def test_golden_section_path_must_be_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        (
            '{"query_id":"q001","query":"Query",'
            '"relevant_chunks":[{"chunk_id":"chunk-1",'
            '"relative_path":"notes.md",'
            '"section_path":"Heading",'
            '"relevance":2}],'
            '"notes":""}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="section_path must be a list",
    ):
        read_golden_queries_jsonl(path)
        
        
def test_golden_relevance_must_be_integer(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        (
            '{"query_id":"q001","query":"Query",'
            '"relevant_chunks":[{"chunk_id":"chunk-1",'
            '"relative_path":"notes.md",'
            '"section_path":["Heading"],'
            '"relevance":"2"}],'
            '"notes":""}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="relevance must be an integer",
    ):
        read_golden_queries_jsonl(path)
        
        
def test_golden_query_id_must_be_string(
    tmp_path: Path,
) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        (
            '{"query_id":1,"query":"Query",'
            '"relevant_chunks":[],"notes":""}\n'
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="query_id must be a string",
    ):
        read_golden_queries_jsonl(path)