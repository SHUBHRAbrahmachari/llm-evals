from src.chat_model_factories.chat_model_factory import ChatModelFactory
from langchain_aws import ChatBedrockConverse
from dotenv import load_dotenv
from typing_extensions import override
import json


class AWSBedrockChatModelFactory(ChatModelFactory):
    @override
    def load_chat_model(self):
        load_dotenv(
            dotenv_path=".env",
            verbose=False
        )

        with open("config.json") as f:
            config = json.load(f)

        model = ChatBedrockConverse(
            model=config.get("chat_models").get("aws_bedrock")
        )

        return model
