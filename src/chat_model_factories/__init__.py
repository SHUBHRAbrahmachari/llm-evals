from src.chat_model_factories.chat_model_factory import ChatModelFactory
from src.chat_model_factories.google_chat_model_factory import GoogleChatModelFactory
from src.chat_model_factories.huggingface_chat_model_factory import HuggingFaceChatModelFactory

chat_model_factories: dict[str, ChatModelFactory] = {
    "huggingface": HuggingFaceChatModelFactory(),
    "google": GoogleChatModelFactory()
}
