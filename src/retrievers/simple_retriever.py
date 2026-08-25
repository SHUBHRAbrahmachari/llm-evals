from src.vector_store_factories import vector_store_factories
from langchain_core.documents import Document
import json


class SimpleRetriever:
    def __init__(self):
        with open("config.json") as f:
            config = json.load(f)

        vector_store = vector_store_factories.get(config.get("vector_store_provider")).load_vector_store()

        if config.get("retriever_config").get("search_type") == "similarity":
            self._retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={
                    "fetch_k": config.get("retriever_config").get("fetch_k"),
                    "k": config.get("retriever_config").get("k")
                },
                pre_filter={
                    "doc_name": {
                        "$eq": config.get("doc_name")
                    }
                }
            )

        else:
            self._retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "fetch_k": config.get("retriever_config").get("fetch_k"),
                    "k": config.get("retriever_config").get("k"),
                    "lambda_mult": config.get("retriever_config").get("lambda_mult")
                },
                pre_filter={
                    "doc_name": {
                        "$eq": config.get("doc_name")
                    }
                }
            )

    def fetch_documents(self, query: str) -> list[Document]:
        documents = self._retriever.invoke(query)
        return documents
