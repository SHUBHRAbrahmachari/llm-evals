from src.reranker_model_factories.reranker_model_factory import RerankerModelFactory
from src.reranker_model_factories.huggingface_reranker_model_factory import HuggingFaceRerankerModelFactory

reranker_model_factories: dict[str, RerankerModelFactory] = {
    "huggingface": HuggingFaceRerankerModelFactory()
}
