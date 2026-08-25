from src.retrievers.reranker_retriever import RerankerRetriever
from src.chat_model_factories import chat_model_factories
from src.utils import generate_context, create_generator_prompt, extract_ai_message_content
from src.document_loaders import document_loaders
from langchain_core.documents import Document
import json


class RAGPipeline:
    def __init__(self):
        with open("config.json") as f:
            config = json.load(f)

        self.__retriever = RerankerRetriever()
        self.__generator = chat_model_factories.get(config.get("chat_model_provider")).load_chat_model()
        self.__document_loader = document_loaders.get(config.get("vector_store_provider"))
        self.__document_loader.load_document_to_vector_store()

    def invoke(self, query: str) -> tuple:
        # FETCH DOCUMENTS
        documents: list[Document] = self.__retriever.fetch_documents(query)

        # EXTRACT THE CONTEXT
        context = generate_context(documents)

        prompt = create_generator_prompt()
        chain = prompt | self.__generator
        ai_response = chain.invoke({
            "query": query,
            "context": context
        })

        answer = extract_ai_message_content(ai_response)

        return documents, answer
