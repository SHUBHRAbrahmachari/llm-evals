from evals.application_level.application_quality import *
from evals.component_level.retriever_eval import run_retriever_eval

# RUN QUALITY EVALS
run_retriever_eval()
run_completeness_eval()
run_correctness_eval()
run_style_eval()
