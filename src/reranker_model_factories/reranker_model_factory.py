from abc import ABC, abstractmethod


class RerankerModelFactory(ABC):
    @abstractmethod
    def load_reranker_model(self):
        pass
