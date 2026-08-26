"""
    for the metric "Correctness", we do not have a separate CorrectnessMetric, rather we have something called GEval!
    GEval is a reasearch paper that was published in the year of 2023, May 25.

    GEval talked about a different approach of evaluation.

    See, generally what we have done? we'd have written a single prompt to the judge LLM stating that

        " here's the response geenrated. Assign a correctness score to the response against the provided context between 0 to 10"

    But that is not a very good approach. LLMs are probabilistic mathematical models. With the same input, the LLM may generate
    different score itself, and with multiple runs you'll actually see a substantial variance in the score, which is not desired!

    GEval came with a solution! GEval asks us to follow some steps-->

        Step 1> Disintegrate your high level criteria (completeness) into a set of evalution steps.
                (In case you don't proovide it, GEval will generate it by  itself)

        Step 2> Find out the atmost top K  (generally 5) probabilities assigned to the digit tokens by the LLM (LLM is nothing but a transformer only)
        Step 3> Normalize the probabilities such that their sum becomes unity.

                For example, suppose we got top 5 prediction tokens as 7, 6, 8, 9, 5 with probabilities
                        (0.79, 0.0056, 0.00032, 0.000013, 0.0000013) respectively, total 0.7959343

                we normalize these probabilities:

                7 => 0.99
                6 => 0.0070357
                5 => 0.000402043

                rest are simply negligible

                now we simply do this : 7*0.99 + 6*0.007 + 5*0.0004 ~ 69.74

                now since we'are scoring in range [1, 10] (GEval default), so we simply divide the score by 10 => 6.974 (this becomes our score) for that particular test case.
                Variance will be much smaller now and test results will be reliable. Deepeval further divides by 10 to bring it in [0, 1] range using MinMax normalization.

                Although by defining Rubrics you can define your own custom GEval scoring range.
"""

from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.eval_config_loading import load_eval_config
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
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

# LOAD THE DATASET (JUST TAKING AS SIMPLE SUBSET)
with open("./golden_datasets/golden_dataset.json", "r") as f:
    dataset = json.load(f)[5: 14: 4]

# GET THE MODEL
gemini_model = GeminiModel(
    model="gemini-3.5-flash-lite"
)

# DEFINE THE EVALUATION STEPS
evaluation_steps = [
    "Deconstruct 'actual output' into individual, non-reducible atomic factual assertions (e.g., specific names, dates, metrics, technologies, or actions).",
    "Cross-reference every extracted atomic assertion directly against the ground truth provided in 'expected output'.",
    "Classify an assertion as 'Fully Correct' if 'expected output' explicitly validates the assertion along with all its specific details and qualifiers.",
    "Classify an assertion as 'Partially Incorrect' if the underlying statement is generally true, but misquotes numbers, alters timeline details, or drops critical qualifiers.",
    "Classify an assertion as 'Hallucinated / False' if it directly contradicts 'expected output' or introduces factual claims completely unsupported by 'expected output'.",
    "Verify whether 'actual output' answers the query in 'input' with complete factual precision, without introducing misleading or unverified assertions.",
    "Calculate the final score by heavily penalizing 'Hallucinated / False' claims, moderately penalizing 'Partially Incorrect' assertions, and awarding full credit for 'Fully Correct' claims."
]

# DEFINE THE RUBRICS
rubrics = [
    Rubric(
        score_range=(1, 4),
        expected_outcome="Unacceptable / Severe Inaccuracies: 'actual output' contains major hallucinations, false facts, or direct contradictions to 'expected output'."
    ),
    Rubric(
        score_range=(5, 7),
        expected_outcome="Acceptable / Minor Inaccuracies: 'actual output' is mostly factual, but contains minor factual slips, altered qualifiers, or unverified details."
    ),
    Rubric(
        score_range=(8, 10),
        expected_outcome="Flawless Correctness: Every atomic assertion in 'actual output' is 100% accurate, fully supported by 'expected output', and completely free of hallucinations."
    )
]

#DEFINE THE EVALUATION PARAMS
evaluation_params = [
    SingleTurnParams.INPUT,
    SingleTurnParams.ACTUAL_OUTPUT,
    SingleTurnParams.EXPECTED_OUTPUT
]

# WE'LL BE WRITING THE EVALUATION STEPS OURSELVES
metric = GEval(
    name="correctness",
    threshold=0.7,
    evaluation_params=evaluation_params,
    evaluation_steps=evaluation_steps,
    rubric=rubrics,
    async_mode=True,
    verbose_mode=True,
    model=gemini_model
)

metrics = [metric]

# GET THE RAG PIPELINE
pipeline = RAGPipeline()

test_cases = []
for test_case in dataset:
    query = test_case.get("question")
    expected_output = test_case.get("expected_answer")

    # RETRIEVED DOCS WON'T BE REQUIRED HERE
    _, actual_output = pipeline.invoke(query)

    # RETRIEVAL CONTEXT IS NOT REQUIRED HERE
    test_cases.append(
        LLMTestCase(
            input=query,
            expected_output=expected_output,
            actual_output=actual_output,
            retrieval_context=None
        )
    )


result = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    hyperparameters=load_eval_config()
)

