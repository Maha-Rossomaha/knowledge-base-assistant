import numpy as np
import pytest

import knowledge_base_assistant.retrieval.dense.sentence_transformer as sentence_transformer_module
from knowledge_base_assistant.retrieval.dense.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)


class FakeSentenceTransformer:
    def __init__(
        self,
        model_name: str,
        device: str,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.document_calls: list[tuple[list[str], dict[str, object]]] = []
        self.query_calls: list[tuple[str, dict[str, object]]] = []

    def get_embedding_dimension(self) -> int:
        return 3

    def encode_document(
        self,
        texts: list[str],
        **kwargs: object,
    ) -> np.ndarray:
        self.document_calls.append((texts, kwargs))

        return np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        )

    def encode_query(
        self,
        query: str,
        **kwargs: object,
    ) -> np.ndarray:
        self.query_calls.append((query, kwargs))

        return np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        )


def test_sentence_transformer_embedding_model_initializes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sentence_transformer_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    model = SentenceTransformerEmbeddingModel(
        model_name="fake-model",
        batch_size=16,
        device="cpu",
    )

    assert model.model_name == "fake-model"
    assert model.dimension == 3


@pytest.mark.parametrize(
    "model_name",
    [
        "",
        "   ",
    ],
)
def test_sentence_transformer_embedding_model_rejects_empty_model_name(
    model_name: str,
) -> None:
    with pytest.raises(
        ValueError,
        match="model name must not be empty",
    ):
        SentenceTransformerEmbeddingModel(
            model_name=model_name,
            batch_size=16,
            device="cpu",
        )


@pytest.mark.parametrize(
    "batch_size",
    [
        0,
        -1,
    ],
)
def test_sentence_transformer_embedding_model_rejects_invalid_batch_size(
    batch_size: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="Batch size must be at least 1",
    ):
        SentenceTransformerEmbeddingModel(
            model_name="fake-model",
            batch_size=batch_size,
            device="cpu",
        )


def test_embed_documents_returns_embeddings_and_passes_expected_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sentence_transformer_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    model = SentenceTransformerEmbeddingModel(
        model_name="fake-model",
        batch_size=16,
        device="cpu",
    )

    embeddings = model.embed_documents(
        ["first text", "second text"],
    )

    np.testing.assert_array_equal(
        embeddings,
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ],
            dtype=np.float32,
        ),
    )

    fake_model = model._model

    assert fake_model.document_calls == [
        (
            ["first text", "second text"],
            {
                "batch_size": 16,
                "convert_to_numpy": True,
                "normalize_embeddings": False,
                "show_progress_bar": False,
            },
        ),
    ]


def test_embed_query_returns_embedding_and_passes_expected_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sentence_transformer_module,
        "SentenceTransformer",
        FakeSentenceTransformer,
    )

    model = SentenceTransformerEmbeddingModel(
        model_name="fake-model",
        batch_size=16,
        device="cpu",
    )

    embedding = model.embed_query("test query")

    np.testing.assert_array_equal(
        embedding,
        np.array(
            [1.0, 2.0, 3.0],
            dtype=np.float32,
        ),
    )

    fake_model = model._model

    assert fake_model.query_calls == [
        (
            "test query",
            {
                "batch_size": 16,
                "convert_to_numpy": True,
                "normalize_embeddings": False,
                "show_progress_bar": False,
            },
        ),
    ]
    
    
class FakeSentenceTransformerWithInvalidDimension(
    FakeSentenceTransformer
):
    def get_embedding_dimension(self) -> int | None:
        return None
    
    
def test_sentence_transformer_embedding_model_rejects_invalid_dimension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sentence_transformer_module,
        "SentenceTransformer",
        FakeSentenceTransformerWithInvalidDimension,
    )

    with pytest.raises(
        ValueError,
        match="Embedding model dimension must be at least 1",
    ):
        SentenceTransformerEmbeddingModel(
            model_name="fake-model",
            batch_size=16,
            device="cpu",
        )