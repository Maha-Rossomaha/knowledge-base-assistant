import hashlib

import numpy as np

from knowledge_base_assistant.application.dense_indexing import (
    build_dense_index,
)
from knowledge_base_assistant.domain.models import Chunk
from knowledge_base_assistant.retrieval.dense.embedding import EmbeddingModelConfig
from knowledge_base_assistant.retrieval.dense.serialization import (
    read_dense_index_metadata,
)
from knowledge_base_assistant.serialization.jsonl import (
    write_chunks_jsonl,
)


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.received_texts: list[str] | None = None

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def dimension(self) -> int:
        return 3
    
    @property
    def config(self) -> EmbeddingModelConfig:
        return EmbeddingModelConfig(
            provider="fake",
            model_name=self.model_name,
            dimension=self.dimension,
            query_prefix="",
            document_prefix="",
        )

    def embed_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        self.received_texts = texts

        return np.array(
            [
                [3.0, 4.0, 0.0],
                [0.0, 0.0, 2.0],
            ],
            dtype=np.float32,
        )

    def embed_query(
        self,
        query: str,
    ) -> np.ndarray:
        raise NotImplementedError


def test_build_dense_index_creates_expected_artifacts(
    tmp_path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "dense" / "embeddings.npy"
    metadata_path = tmp_path / "dense" / "index_metadata.json"

    chunks = [
        _make_chunk(
            chunk_id="chunk-1",
            searchable_text="First searchable text",
            chunk_index=0,
        ),
        _make_chunk(
            chunk_id="chunk-2",
            searchable_text="Second searchable text",
            chunk_index=1,
        ),
    ]
    write_chunks_jsonl(chunks, chunks_path)

    model = FakeEmbeddingModel()

    result = build_dense_index(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedding_model=model,
    )

    assert embeddings_path.is_file()
    assert metadata_path.is_file()

    assert result.chunk_count == 2
    assert result.dimension == 3
    assert result.embeddings_path == embeddings_path
    assert result.metadata_path == metadata_path

    assert model.received_texts == [
        "First searchable text",
        "Second searchable text",
    ]


def test_build_dense_index_writes_normalized_embeddings(
    tmp_path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "index_metadata.json"

    write_chunks_jsonl(
        [
            _make_chunk(
                chunk_id="chunk-1",
                searchable_text="First",
                chunk_index=0,
            ),
            _make_chunk(
                chunk_id="chunk-2",
                searchable_text="Second",
                chunk_index=1,
            ),
        ],
        chunks_path,
    )

    build_dense_index(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedding_model=FakeEmbeddingModel(),
    )

    embeddings = np.load(
        embeddings_path,
        allow_pickle=False,
    )

    np.testing.assert_allclose(
        embeddings,
        np.array(
            [
                [0.6, 0.8, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        rtol=1e-6,
        atol=1e-6,
    )


def test_build_dense_index_writes_expected_metadata(
    tmp_path,
) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    embeddings_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "index_metadata.json"

    chunks = [
        _make_chunk(
            chunk_id="chunk-1",
            searchable_text="First",
            chunk_index=0,
        ),
        _make_chunk(
            chunk_id="chunk-2",
            searchable_text="Second",
            chunk_index=1,
        ),
    ]
    write_chunks_jsonl(chunks, chunks_path)

    expected_sha256 = hashlib.sha256(
        chunks_path.read_bytes()
    ).hexdigest()

    build_dense_index(
        chunks_path=chunks_path,
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        embedding_model=FakeEmbeddingModel(),
    )

    metadata = read_dense_index_metadata(metadata_path)

    assert metadata.schema_version == 1
    assert metadata.embedding_model.model_name == "fake-model"
    assert metadata.embedding_model.dimension == 3
    assert metadata.normalized is True
    assert metadata.chunks_sha256 == expected_sha256
    assert metadata.chunk_ids == ("chunk-1", "chunk-2")


def _make_chunk(
    *,
    chunk_id: str,
    searchable_text: str,
    chunk_index: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="document-1",
        source_name="test-source",
        relative_path="notes/test.md",
        title="Test",
        section_path=("Test",),
        content="Original content",
        searchable_text=searchable_text,
        chunk_index=chunk_index,
        section_chunk_index=chunk_index,
        start_line=1,
        end_line=2,
        content_hash=f"content-hash-{chunk_index}",
    )