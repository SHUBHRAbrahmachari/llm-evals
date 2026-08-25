from langchain_core.messages import AIMessage
from langsmith import traceable

@traceable
def extract_ai_message_content(message: AIMessage) -> str:
    content = message.content

    if isinstance(content, str) and len(content) > 0:
        return content

    if isinstance(content, list):
        for block in content:
            if block.get("text") is not None:
                return block.get("text")

    return "<NO RESPONSE FOUND>"
