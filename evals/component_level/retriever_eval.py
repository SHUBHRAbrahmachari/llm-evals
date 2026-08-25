"""
The metrics used for retriever eval are--

        1> Contextual Recall
        2> contextual Precision

    Both are supported by deepeval
"""

from src.retrievers.reranker_retriever import RerankerRetriever
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate
from dotenv import load_dotenv
import json

load_dotenv(
    dotenv_path=".env",
    verbose=False
)

DATASET_PATH = "./golden_datasets/golden_dataset.json"

with open(DATASET_PATH, "r") as f:
    dataset = json.load(f)

subset = dataset[5: 8]

gemini_model_1 = GeminiModel(model="gemini-3.5-flash-lite")
gemini_model_2 = GeminiModel(model="gemini-3.1-flash-lite")

metrics = [
    ContextualRecallMetric(
        threshold=0.7,
        model=gemini_model_2,
        include_reason=True,
        async_mode=True,
        verbose_mode=True,
    ),

    ContextualPrecisionMetric(
        threshold=0.7,
        model=gemini_model_1,
        include_reason=True,
        async_mode=True,
        verbose_mode=True
    )
]

test_cases: list[LLMTestCase] = []

retriever = RerankerRetriever()

for block in subset:
    query = block.get("question")
    expected_answer = block.get("expected_answer")

    docs = retriever.fetch_documents(query)

    content_list = []
    for doc in docs:
        content_list.append(doc.page_content)

    test_cases.append(
        LLMTestCase(
            input=query,
            actual_output=None,
            expected_output=expected_answer,
            retrieval_context=content_list
        )
    )


results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
)
