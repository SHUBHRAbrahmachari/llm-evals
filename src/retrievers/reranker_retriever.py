from src.retrievers.simple_retriever import SimpleRetriever
from src.reranker_model_factories import reranker_model_factories
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from typing_extensions import override
import json


class RerankerRetriever(SimpleRetriever):
    def __init__(self):
        super().__init__()

        with open("config.json", "r") as file:
            config = json.load(file)

        reranker_model_provider = config.get("reranker_model_provider")
        reranker_model = reranker_model_factories.get(reranker_model_provider).load_reranker_model()

        cross_encoder = CrossEncoderReranker(
            model=reranker_model,
            top_n=config.get("retriever_config").get("top_k")
        )

        self.__reranker_retriever = ContextualCompressionRetriever(
            base_retriever=self._retriever,
            base_compressor=cross_encoder
        )

    @override
    def fetch_documents(self, query: str) -> list[Document]:
        documents = self.__reranker_retriever.invoke(query)
        return documents

