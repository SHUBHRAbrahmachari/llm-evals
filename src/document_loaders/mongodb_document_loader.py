from src.document_loaders.document_loader import DocumentLoader
from src.vector_store_factories import vector_store_factories
from langchain_community.document_loaders import TextLoader, PDFPlumberLoader, UnstructuredWordDocumentLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import override
import json
import os


class MongoDBDocumentLoader(DocumentLoader):
    def __init__(self):
        super().__init__()

        with open("config.json") as f:
            self.__config = json.load(f)

        self.__vector_store = vector_store_factories.get("mongodb").load_vector_store()
        if self.__vector_store is None:
            print("VECTOR STORE IS NOT FOUND")
            exit(1)

    @override
    def load_document_to_vector_store(self, force_load: bool = False):
        base_path: str = self.__config.get("base_doc_path")
        doc_name = self.__config.get("doc_name")
        file_path: str = base_path + doc_name
        if not os.path.exists(file_path):
            print(f"DOCUMENT {doc_name} WAS NOT FOUND IN BASE PATH {base_path}")
            return

        doc = self.__vector_store.collection.find_one(
            filter={
                "doc_name": doc_name
            }
        )

        if doc is not None:
            print(f"DOCUMENT {doc_name} ALREADY EXISTS IN VECTOR STORE")

            if not force_load:
                return
            else:
                print("force_load HAS BEEN SET TO True, SO EXISTING DOCUMENTS ARE BEING DELETED FIRST")
                self.delete_documents_from_vector_store(doc_name)
                print("EXSITING DOCUMENTS ARE DELETED SUCCESSFULLY")

        if file_path.endswith(".txt"):
            print("TEXT FILE FOUND")
            loader = TextLoader(
                file_path=file_path,
                encoding="utf-8"
            )

        elif file_path.endswith(".pdf"):
            print("PDF FILE FOUND")
            loader = PDFPlumberLoader(
                file_path=file_path
            )

        elif file_path.endswith(".docx"):
            print("DOC FILE FOUND")
            loader = UnstructuredWordDocumentLoader(
                file_path=file_path,
                mode="paged"
            )

        elif file_path.endswith(".pptx"):
            print("PPT FILE FOUND")
            loader = UnstructuredPowerPointLoader(
                file_path=file_path,
                mode="paged"
            )

        else:
            print("DOCUMENT FORMAT IS NOT SUPPORTED!")
            exit(1)

        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True,
            chunk_size=self.__config.get("rag_config").get("chunk_size"),
            chunk_overlap=self.__config.get("rag_config").get("chunk_overlap")
        )

        # ... existing code ...
        chunk_id = 1
        for page in loader.lazy_load():
            chunks = []
            page_chunks = splitter.split_documents([page])

            for chunk in page_chunks:
                chunk.id = str(chunk_id)
                chunk.metadata["doc_name"] = doc_name
                chunk_id += 1
                chunks.append(chunk)

            self.__vector_store.add_documents(chunks)

        print(f"DOCUMENT {doc_name} UPLOADED TO VECTOR STORE SUCCESSFULLY")

    @override
    def delete_documents_from_vector_store(self):
        self.__vector_store.collection.delete_many(
            filter={
                "doc_name": self.__config.get("doc_name")
            }
        )

        return
