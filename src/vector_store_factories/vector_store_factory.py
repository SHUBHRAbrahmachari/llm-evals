from src.embedding_model_factories import embedding_model_factories
from abc import ABC, abstractmethod
import json


class VectorStoreFactory(ABC):
    def __init__(self):
        with open("config.json") as f:
            config = json.load(f)

        self.__embedding_model_factory = embedding_model_factories.get(config.get("embedding_model_provider"))
        self._embedding_model = self.__embedding_model_factory.load_embedding_model()
        
    @abstractmethod
    def load_vector_store(self):
        pass

