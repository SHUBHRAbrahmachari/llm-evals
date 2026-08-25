from src.embedding_model_factories.embedding_model_factory import EmbeddingModelFactory
from langchain_huggingface import HuggingFaceEmbeddings
from typing_extensions import override
from dotenv import load_dotenv
import json


class HuggingFaceEmbeddingModelFactory(EmbeddingModelFactory):
    @override
    def load_embedding_model(self):
        load_dotenv(
            dotenv_path=".env",
            verbose=False
        )

        with open("config.json") as f:
            config = json.load(f)

        embedding_model = HuggingFaceEmbeddings(
            model_name=config.get("embedding_models").get("huggingface")
        )

        return embedding_model
