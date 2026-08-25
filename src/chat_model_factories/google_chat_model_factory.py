from src.chat_model_factories.chat_model_factory import ChatModelFactory
from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import override
from dotenv import load_dotenv
import json


class GoogleChatModelFactory(ChatModelFactory):
    @override
    def load_chat_model(self):
        load_dotenv(
            dotenv_path=".env",
            verbose=False
        )

        with open("config.json") as f:
            config = json.load(f)

        chat_model = ChatGoogleGenerativeAI(
            model=config.get("chat_models").get("google")
        )

        return chat_model
