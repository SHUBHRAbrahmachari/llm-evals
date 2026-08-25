from src.embedding_model_factories.embedding_model_factory import EmbeddingModelFactory
from src.embedding_model_factories.huggingface_embedding_model_factory import HuggingFaceEmbeddingModelFactory
from src.embedding_model_factories.google_embedding_model_factory import GoogleEmbeddingModelFactory

embedding_model_factories: dict[str, EmbeddingModelFactory] = {
    "huggingface": HuggingFaceEmbeddingModelFactory(),
    "google": GoogleEmbeddingModelFactory()
}
