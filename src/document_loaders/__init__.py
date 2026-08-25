from src.document_loaders.document_loader import DocumentLoader
from src.document_loaders.mongodb_document_loader import MongoDBDocumentLoader

document_loaders: dict[str, DocumentLoader] = {
    "mongodb": MongoDBDocumentLoader()
}
