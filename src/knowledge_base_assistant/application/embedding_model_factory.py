from knowledge_base_assistant.retrieval.dense.embedding import (
    EmbeddingModelConfig,
)
from knowledge_base_assistant.retrieval.dense.sentence_transformer import (
    SentenceTransformerEmbeddingModel,
)


def create_sentence_transformer_embedding_model(
    config: EmbeddingModelConfig,
    *,
    batch_size: int,
    device: str,
) -> SentenceTransformerEmbeddingModel:
    if config.provider != "sentence-transformers":
        raise ValueError(
            "Unsupported embedding model provider: "
            f"{config.provider}"
        )

    return SentenceTransformerEmbeddingModel(
        model_name=config.model_name,
        batch_size=batch_size,
        device=device,
        query_prefix=config.query_prefix,
        document_prefix=config.document_prefix,
    )