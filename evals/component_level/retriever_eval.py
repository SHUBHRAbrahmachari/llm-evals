"""
The metrics used for retriever eval are--

        1> Contextual Recall
        2> contextual Precision

    Both are supported by deepeval
"""

from src.retrievers.reranker_retriever import RerankerRetriever
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.models import AmazonBedrockModel
from deepeval.test_case import LLMTestCase
from deepeval.evaluate import evaluate
from langsmith import Client
from random import randint
from dotenv import load_dotenv
import json

load_dotenv(
    dotenv_path=".env",
    verbose=False
)

with open("config.json", "r") as f:
    config = json.load(f)

judge = AmazonBedrockModel(
    model=config.get("chat_models").get("aws_bedrock")
)


metrics = [
    ContextualRecallMetric(
        threshold=0.7,
        model=judge,
        include_reason=True,
        async_mode=True,
        verbose_mode=True,
    ),

    ContextualPrecisionMetric(
        threshold=0.7,
        model=judge,
        include_reason=True,
        async_mode=True,
        verbose_mode=True
    )
]

# GET THE RETRIEVER
retriever = RerankerRetriever()

# GET THE CLIENT
client = Client()

test_cases = []

# GET THE EXAMPLES
examples = client.list_examples(
    dataset_name="quality_dataset",
    limit=20
)

for example in examples:

    # WE WILL PICK RANDOMLY
    if randint(1, 100) % 2 == 0:
        query = example.inputs.get("query")
        expected_output = example.outputs.get("expected_answer")

        documents = retriever.fetch_documents(query)
        context = [d.page_content for d in documents]

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=None,
                expected_output=expected_output,
                retrieval_context=context
            )
        )


def run_retriever_eval():
    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    return results
