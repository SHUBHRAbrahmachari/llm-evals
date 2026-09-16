from langsmith import Client
from dotenv import load_dotenv
import json

# IMPORT ENVIRONMENT VARIABLES
load_dotenv(
    dotenv_path=".env",
    verbose=False
)

# DEFINE DATASET NAME & DESCRIPTION
DATASET_NAME = "toxicity_dataset"
DATASET_DESCRIPTION = """
    A labeled evaluation dataset for testing an AI system's ability to
    identify toxic, abusive, hateful, threatening, and otherwise harmful
    user prompts. It contains benign, red-flag, and mixed examples.
"""
FILE_NAME = "toxicity_dataset.jsonl"

# CONNECT TO LANGSMITH
client = Client()


# CHECK IF DATASET ALREADY EXISTS OR NOT
def does_dataset_exist():
    try:
        datasets = set(d.name for d in client.list_datasets())
        if DATASET_NAME in datasets:
            return True
        return False
    except Exception as e:
        print(str(e))
        return None


# IF DATASET DOES NOT EXIST, WE'LL CREATE ONE
if not does_dataset_exist():
    client.create_dataset(
        dataset_name=DATASET_NAME,
        description=DATASET_DESCRIPTION
    )


# PICK THE ROWS AND ADD THEM BACK TO LANGSMITH
def upload_dataset():
    try:
        with open(FILE_NAME, mode="r", encoding="utf-8") as f:
            for line in f:
                # READS AND DESERIALIZE INTO A JSON DATA
                record = json.loads(line)

                inputs = {
                    "query": record.get("inputs").get("text")
                }

                metadata = {
                    "type": "safety",
                    "metric": "toxicity",
                    "category": record.get("inputs").get("label")
                }

                # ADD THIS AS AN EXAMPLE (TEST CASE)
                client.create_example(
                    dataset_name=DATASET_NAME,
                    inputs=inputs,
                    metadata=metadata
                )

    except Exception as e:
        print(str(e))


upload_dataset()
