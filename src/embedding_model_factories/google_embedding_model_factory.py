from src.embedding_model_factories.embedding_model_factory import EmbeddingModelFactory
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from typing_extensions import override
import json


class GoogleEmbeddingModelFactory(EmbeddingModelFactory):
    @override
    def load_embedding_model(self):
        load_dotenv(
            dotenv_path=".env",
            verbose=True
        )

        with open("config.json", "r") as f:
            config = json.load(f)

        embedding_model = GoogleGenerativeAIEmbeddings(
            model=config.get("embedding_models").get("google")
        )

        return embedding_model
