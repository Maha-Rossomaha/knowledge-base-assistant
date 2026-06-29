import pytest

import knowledge_base_assistant.application.embedding_model_factory as factory_module
from knowledge_base_assistant.retrieval.dense.embedding import (
    EmbeddingModelConfig,
)


class FakeSentenceTransformerEmbeddingModel:
    def __init__(
        self,
        *,
        model_name: str,
        batch_size: int,
        device: str,
        query_prefix: str,
        document_prefix: str,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix


def test_create_sentence_transformer_embedding_model_uses_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        factory_module,
        "SentenceTransformerEmbeddingModel",
        FakeSentenceTransformerEmbeddingModel,
    )

    config = EmbeddingModelConfig(
        provider="sentence-transformers",
        model_name="intfloat/multilingual-e5-base",
        dimension=768,
        query_prefix="query: ",
        document_prefix="passage: ",
    )

    model = (
        factory_module.create_sentence_transformer_embedding_model(
            config,
            batch_size=32,
            device="cpu",
        )
    )

    assert model.model_name == "intfloat/multilingual-e5-base"
    assert model.batch_size == 32
    assert model.device == "cpu"
    assert model.query_prefix == "query: "
    assert model.document_prefix == "passage: "


def test_create_sentence_transformer_embedding_model_rejects_provider() -> None:
    config = EmbeddingModelConfig(
        provider="unsupported-provider",
        model_name="test-model",
        dimension=3,
        query_prefix="",
        document_prefix="",
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unsupported embedding model provider: "
            "unsupported-provider"
        ),
    ):
        factory_module.create_sentence_transformer_embedding_model(
            config,
            batch_size=32,
            device="cpu",
        )