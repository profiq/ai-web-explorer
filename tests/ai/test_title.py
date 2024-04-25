import json
import os

import mlflow
import numpy as np
import openai

from ai_web_explorer import config
from ai_web_explorer import promptrepo

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "hackernews"
)

mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("Title")


def get_text_embedding(client: openai.OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(model="text-embedding-3-small", input=[text])
    return response.data[0].embedding


def cosine_distance(embedding1: list[float], embedding2: list[float]) -> float:
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )


def compare_embeddings(text1: str, text2: str):
    client = openai.OpenAI()
    embedding1 = get_text_embedding(client, text1)
    embedding2 = get_text_embedding(client, text2)
    return cosine_distance(embedding1, embedding2)


def test_title():
    prompt = promptrepo.get_prompt("page_title")
    eval_table = {
        "no": [],
        "title_expected": [],
        "title_generated": [],
        "similarity": [],
    }

    with open(os.path.join(DATA_DIR, "titles_expected.txt")) as f:
        titles_expected = f.read().splitlines()

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".html"):
            continue

        no = int(filename.split(".")[0])
        with open(os.path.join(DATA_DIR, filename)) as f:
            html = f.read()[: config.HTML_PART_LENGTH]
            response = prompt.execute_prompt(openai.OpenAI(), html=html)

        if not response.message.tool_calls or len(response.message.tool_calls) == 0:
            raise ValueError("No tool calls in response when getting page title")

        args_str = response.message.tool_calls[0].function.arguments
        args = json.loads(args_str)
        title_generated = args["title"]
        similarity = compare_embeddings(titles_expected[no - 1], title_generated)
        eval_table["no"].append(no)
        eval_table["title_expected"].append(titles_expected[no - 1])
        eval_table["title_generated"].append(title_generated)
        eval_table["similarity"].append(similarity)

    with mlflow.start_run():
        mlflow.log_metric("similarity_mean", float(np.mean(eval_table["similarity"])))
        mlflow.log_metric("similarity_std", float(np.std(eval_table["similarity"])))
        mlflow.log_param("model", prompt.model)
        mlflow.log_table(data=eval_table, artifact_file="eval_table.json")
