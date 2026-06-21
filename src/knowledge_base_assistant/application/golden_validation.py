from pathlib import Path

from knowledge_base_assistant.evaluation.models import (
    GoldenValidationResult,
)
from knowledge_base_assistant.evaluation.validation import (
    validate_golden_queries,
)
from knowledge_base_assistant.serialization.jsonl import (
    read_chunks_jsonl,
    read_golden_queries_jsonl,
)


def validate_golden_files(
    golden_path: Path,
    chunks_path: Path,
) -> GoldenValidationResult:
    queries = read_golden_queries_jsonl(golden_path)
    chunks = read_chunks_jsonl(chunks_path)

    return validate_golden_queries(
        queries,
        chunks,
    )