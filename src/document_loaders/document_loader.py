from abc import ABC, abstractmethod


class DocumentLoader(ABC):

    @abstractmethod
    def load_document_to_vector_store(self, force_load: bool = False):
        pass

    @abstractmethod
    def delete_documents_from_vector_store(self):
        pass
