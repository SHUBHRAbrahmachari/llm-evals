from langsmith import Client
from dotenv import load_dotenv
import json


# LOAD THE ENVIRONMENT VARIBLES
load_dotenv(
    dotenv_path=".env",
    verbose=False
)

# CONNECT TO THE CLIENT
client = Client()

DATASET_ID = "131d9fcc-2841-4e37-8c16-8f4058fbc5a5"
DATASET_NAME = "toxicity_dataset"
FILE_NAME = "toxicity_dataset.jsonl"


# CHECK WHETHER DATASET EXISTS OR NOT
def does_dataset_exist():
    try:
        datasets = set(d.name for d in client.list_datasets())
        if DATASET_NAME in datasets:
            return True

        return False

    except Exception as e:
        print(str(e))
        return None


# DOWNLOAD THE DATASET AND SAVE TO YOUR LOCAL FILE OR JUST USE IT
def download_dataset():
    try:
        if not does_dataset_exist():
            return

        # OPEN YOU FILE AS WRITE/APPEND MODE AS YOU WANT
        with open(FILE_NAME, mode="w", encoding="utf-8") as f:
            # ACQUIRE THE GENERATOR
            examples = client.list_examples(
                dataset_name=DATASET_NAME,
                dataset_id=DATASET_ID
            )

            for example in examples:
                # EXAMPLE OBJECT IS NOT DIRECTLY SERIALIZABLE, SO WE NEED TO CONVERT IT BACK INTO A DICTIONARY
                record = {
                    "id": str(example.id),
                    "inputs": example.inputs,
                    "outputs": example.outputs,
                    "metadata": example.metadata
                }

                content = json.dumps(record)
                f.write(content+"\n")

    except Exception as e:
        print(str(e))


download_dataset()
