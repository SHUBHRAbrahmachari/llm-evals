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
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric
from deepeval.models import GeminiModel
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
import json

load_dotenv(
    dotenv_path=".env",
    verbose=False
)

# load the dataset
with open("./golden_datasets/golden_dataset.json", "r") as d:
    dataset = json.load(d)

# use only a subset of the dataset
dataset = dataset[1: 9: 3]

# get the eval config
hyperparameters = load_eval_config()

# get 3 distinct models as judges
gemini_model_1 = GeminiModel(model="gemini-3.5-flash-lite")
gemini_model_2 = GeminiModel(model="gemini-3.1-flash-lite")

# define the metrics
metrics = [
    FaithfulnessMetric(
        threshold=0.8,
        model=gemini_model_1,
        include_reason=True,
        async_mode=True,
        verbose_mode=True
    ),

    AnswerRelevancyMetric(
        threshold=0.8,
        model=gemini_model_2,
        include_reason=True,
        async_mode=True,
        verbose_mode=True
    ),

    ContextualRelevancyMetric(
        threshold=0.5,
        model=gemini_model_2,
        include_reason=True,
        async_mode=True,
        verbose_mode=True
    )
]

# get the pipeline to trigger
pipeline = RAGPipeline()

# get the test cases
test_cases: list[LLMTestCase] = []

for block in dataset:
    query = block.get("question")
    expected_answer = block.get("expected_answer")

    docs, actual_answer = pipeline.invoke(query)

    retrieval_context = []
    for doc in docs:
        retrieval_context.append(
            doc.page_content
        )

    test_cases.append(
        LLMTestCase(
            input=query,
            expected_output=expected_answer,
            actual_output=actual_answer,
            retrieval_context=retrieval_context
        )
    )

# run the evaluation
result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters=hyperparameters
)
