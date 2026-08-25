from src.vector_store_factories.vector_store_factory import VectorStoreFactory
from dotenv import load_dotenv
from typing_extensions import override
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from langchain_mongodb import MongoDBAtlasVectorSearch
import json
import os


class MongoDBVectorStoreFactory(VectorStoreFactory):
    def __init__(self):
        super().__init__()

    @override
    def load_vector_store(self):
        load_dotenv(
            dotenv_path=".env",
            verbose=False
        )

        try:
            client = MongoClient(
                host=os.getenv("MONGO_URL")
            )

            with open("config.json") as f:
                config = json.load(f)

            mongodb_vector_store_config = config.get("vector_store_config").get("mongodb")
            database = client[mongodb_vector_store_config.get("database")]
            collection = database[mongodb_vector_store_config.get("collection")]

            index_name = mongodb_vector_store_config.get("index_name")
            text_key = mongodb_vector_store_config.get("text_key")
            embedding_key = mongodb_vector_store_config.get("embedding_key")

            vector_store = MongoDBAtlasVectorSearch(
                collection=collection,
                embedding=self._embedding_model,
                text_key=text_key,
                embedding_key=embedding_key,
                index_name=index_name
            )
            print("CONNECTED TO MONGODB VECTOR STORE SUCCESSFULLY")

            return vector_store

        except PyMongoError as e:
            print("SOMETHING WENT WRONG WHILE TRYING TO CONNECT WITH MONGODB VECTOR STORE")
            print(str(e))
            return None
