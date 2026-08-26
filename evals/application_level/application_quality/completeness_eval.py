"""
    Same as we did with correctness eval, we're going to do the same with correctness eval.
    We'll again use GEval for that. The fundamental concept is same!
"""
from networkx.algorithms import threshold
from openai.resources.beta.threads import threads

from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.eval_config_loading import load_eval_config
from deepeval.metrics.g_eval import Rubric
from deepeval.metrics import GEval
from deepeval.evaluate import evaluate
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.models import GeminiModel
from dotenv import load_dotenv
import json

# LOAD THE ENVIRONMENT VARIABLES
load_dotenv(
    dotenv_path=".env",
    verbose=False
)

# LOAD THE DATASET
with open("./golden_datasets/golden_dataset.json", "r") as f:
    dataset = json.load(f)[10: 20: 4]

# DEFINE YOUR EVALUATION PARAMS
evaluation_steps = [
    "Deconstruct the user query in 'input' to identify every specific requirement, sub-question, and constraint asked by the user.",
    "Deconstruct 'actual output' into individual, non-reducible atomic factual claims (facts, entities, numbers, dates, tools).",
    "Cross-reference each extracted atomic claim from 'actual output' directly against 'expected output'.",
    "Mark an atomic claim as 'Fully Correct' if 'expected output' explicitly verifies every detail without missing any qualifiers.",
    "Mark an atomic claim as 'Partially Correct' if the core concept is present in 'expected output' but misses key sub-details or specifics.",
    "Mark an atomic claim as 'False / Hallucinated' if it is completely absent from or contradicts 'expected output'.",
    "Verify whether 'actual output' addresses ALL requirements identified from 'input'. If key parts of the user question are left unanswered despite being present in 'expected output', mark it as incomplete.",
    "Calculate the final score by heavily penalizing false claims and query omissions, moderately penalizing partial claim satisfaction, and awarding full points when all query requirements are exhaustively and accurately answered"
]

# DEFINE THE EVALUATION PARAMS
evaluation_params = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.EXPECTED_OUTPUT


]

# DEFINE THE RUBRICS
rubrics = [
    Rubric(
        score_range=(1, 4),
        expected_outcome="Severe Incompleteness or Hallucinations: 'actual output' contains false/hallucinated claims, or fails to answer core parts of the question in 'input'."
    ),
    Rubric(
        score_range=(5, 7),
        expected_outcome="Partial Satisfaction: 'actual output' answers the main part of 'input' without false claims, but leaves out important secondary details or provides partially satisfied atomic claims."
    ),
    Rubric(
        score_range=(8, 10),
        expected_outcome="Exhaustive Completeness & Precision: Every requirement in 'input' is fully addressed, and every atomic claim in 'actual output' is 100% verified by 'expected output' without omission."
    )
]

# GET THE JUDGE MODEL
judge = GeminiModel(
    model="gemini-3.5-flash-lite"
)

# GET THE METRICS
metrics = [
    GEval(
        name="completeness",
        threshold=0.7,
        evaluation_params=evaluation_params,
        evaluation_steps=evaluation_steps,
        rubric=rubrics,
        verbose_mode=True,
        async_mode=True,
        model=judge
    )
]

# GET THE RAG pipeline
pipeline = RAGPipeline()

test_cases = []
for block in dataset:
    query = block.get("question")
    expected_output = block.get("expected_answer")

    _, actual_output = pipeline.invoke(query=query)

    test_cases.append(
        LLMTestCase(
            input=query,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=None
        )
    )

# RUN THE EVALUATION
result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters=load_eval_config()
)
