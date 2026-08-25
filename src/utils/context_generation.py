from langchain_core.documents import Document
from langsmith import traceable


@traceable
def generate_context(documents: list[Document]) -> str:
    content_list = []
    for doc in documents:
        content_list.append(doc.page_content)

    context = "\n\n".join(content_list)

    return context
