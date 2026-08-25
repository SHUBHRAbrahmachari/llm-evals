from abc import ABC, abstractmethod


class EmbeddingModelFactory(ABC):
    @abstractmethod
    def load_embedding_model(self):
        pass
