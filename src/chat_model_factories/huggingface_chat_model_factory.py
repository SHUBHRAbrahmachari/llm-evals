from src.chat_model_factories.chat_model_factory import ChatModelFactory
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing_extensions import override
from dotenv import load_dotenv
import json


class HuggingFaceChatModelFactory(ChatModelFactory):
    @override
    def load_chat_model(self):
        load_dotenv(
            dotenv_path=".env",
            verbose=False
        )

        with open("config.json") as f:
            config = json.load(f)

        llm = HuggingFaceEndpoint(
            model=config.get("chat_models").get("huggingface")
        )

        chat_model = ChatHuggingFace(
            llm=llm
        )

        return chat_model
