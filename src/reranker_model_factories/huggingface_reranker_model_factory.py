from src.reranker_model_factories.reranker_model_factory import RerankerModelFactory
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from typing_extensions import override
from dotenv import load_dotenv
import json


class HuggingFaceRerankerModelFactory(RerankerModelFactory):
    @override
    def load_reranker_model(self):
        load_dotenv(
            dotenv_path=".env",
            verbose=False
        )

        with open("config.json") as f:
            config = json.load(f)

        reranker_model = HuggingFaceCrossEncoder(
            model_name=config.get("reranker_models").get("huggingface")
        )

        return reranker_model
