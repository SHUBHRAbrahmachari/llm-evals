from src.vector_store_factories.vector_store_factory import VectorStoreFactory
from src.vector_store_factories.mongodb_vector_store_factory import MongoDBVectorStoreFactory

vector_store_factories: dict[str, VectorStoreFactory] = {
    "mongodb": MongoDBVectorStoreFactory()
}
