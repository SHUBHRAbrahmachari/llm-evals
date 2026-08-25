from abc import ABC, abstractmethod


class ChatModelFactory(ABC):
    @abstractmethod
    def load_chat_model(self):
        pass
