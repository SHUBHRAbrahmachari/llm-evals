"""
    Same goes for here as well. Style can be a vast thing to talk about, we need to define what actually we mean by style?
    In our application we majorly focused on two things!

    1> the tone of the response must be friendly and respectful
    2> the language used must be very simple and response must be broken down into meaningful paragraph units instead of using a single large paragraph.
    3> use examples if and only if user asks for example, otherwise skip putting examples

    Now focus here that Style is usually a reference-free eval, but things can get subjective.
"""

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
    dataset = json.load(f)[11: 20: 4]

# DEFINE YOUR EVALUATION PARAMS
evaluation_steps = [
    "Analyze 'actual output' for tone: Verify the response maintains a warm, respectful, polite, and friendly communication style.",
    "Analyze 'actual output' for language simplicity: Ensure vocabulary and sentence structures are simple, direct, and accessible.",
    "Analyze 'actual output' for structural formatting: Confirm the content is broken down using clear structural elements—such as short paragraphs, bullet points, or numbered lists. Heavily penalize if the entire response is presented as a single dense block or wall of text.",
    "Check 'input' for example requests: Determine whether the user explicitly asked for examples in 'input'.",
    "Verify example constraint in 'actual output': If an example was explicitly requested, ensure it is provided. If NO example was requested, ensure 'actual output' contains NO examples.",
    "Calculate the final score based on compliance with tone, simple language, structural scaffolding (paragraphs/bullets/lists), and strict example rules."
]

# DEFINE THE EVALUATION PARAMS
evaluation_params = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT
]

# DEFINE THE RUBRICS
rubrics = [
    Rubric(
        score_range=(1, 4),
        expected_outcome="Unsatisfactory: Tone is cold or abrupt, response is a single dense paragraph block, or includes unrequested examples when 'input' did not ask for them."
    ),
    Rubric(
        score_range=(5, 7),
        expected_outcome="Partially Compliant: Tone is respectful and language is simple, but formatting uses slightly long paragraphs or includes minor unprompted illustrative examples."
    ),
    Rubric(
        score_range=(8, 10),
        expected_outcome="Flawless Style: Tone is warm and friendly, language is very simple, structure is cleanly divided into meaningful paragraph units, and examples are included IF AND ONLY IF explicitly requested."
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
