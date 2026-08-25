import json


def load_eval_config():
    # load the configuration
    with open("config.json", "r") as f:
        config = json.load(f)

    # create the hyperparameters
    rag_config: dict = {
        "fetch_k": config.get("retriever_config").get("fetch_k"),
        "k": config.get("retriever_config").get("k"),
        "top_k": config.get("retriever_config").get("top_k"),
        "chunk_size": config.get("rag_config").get("chunk_size"),
        "chunk_overlap": config.get("rag_config").get("chunk_overlap"),
        "search_type": config.get("retriever_config").get("search_type")
    }

    if config.get("retriever_config").get("search_type") == "mmr":
        rag_config["lambda_mult"] = config.get("retriever_config").get("lambda_mult")

    return rag_config
