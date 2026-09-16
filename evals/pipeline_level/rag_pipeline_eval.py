"""
    We have already covered
            1> ContextualRecallMetric
            2> ContextualPrecisionMetric

    to evaluate our Retriever performance. Now we'll checkk how the pipeline performs.

    Retriever -> Generator -> Answer

    Here we have 3 metrics:

                1> Faithfulness (how much the response is grounded with the context?)
                2> Answer Relevance (how much the response is relevant to our actual question?)
                3> Contextual Relevance (how much the context is relevant to the actual question?)

    Anyway since Contextual Relevance might be extremely poor since we're having a chunk size of 1000,
    So better focus on Faithfulness and Answer Relevance. If these two are alread high, then we on't need to worry about
    Contextual Relevance!
"""

from src.pipeline import RAGPipeline
from src.utils import load_eval_config
from deepeval.evaluate import evaluate
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.models import AmazonBedrockModel
from deepeval.test_case import LLMTestCase
from langsmith import Client
from dotenv import load_dotenv
import json

load_dotenv(
    dotenv_path=".env",
    verbose=False
)

with open("config.json") as f:
    config = json.load(f)


judge = AmazonBedrockModel(model=config.get("chat_models").get("aws_bedrock"))

# define the metrics
metrics = [
    FaithfulnessMetric(
        threshold=0.8,
        model=judge,
        include_reason=True,
        async_mode=True,
        verbose_mode=True
    ),

    AnswerRelevancyMetric(
        threshold=0.8,
        model=judge,
        include_reason=True,
        async_mode=True,
        verbose_mode=True
    )
]

# connect to the client
client = Client()

# get the pipeline to trigger
pipeline = RAGPipeline()

# get the test cases
test_cases: list[LLMTestCase] = []

# LOAD THE TEST CASES
records = client.list_examples(
    dataset_name="quality_dataset",
    limit=10
)

for record in records:
    query = record.inputs.get("query")
    expected_answer = record.outputs.get("expected_answer")

    docs, response = pipeline.invoke(query)

    context = [d.page_content for d in docs]

    test_cases.append(
        LLMTestCase(
            input=query,
            actual_output=response,
            expected_output=expected_answer,
            retrieval_context=context
        )
    )

# run the evaluation
result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters=load_eval_config()
)
