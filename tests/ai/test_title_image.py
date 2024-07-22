import csv
import json
import os

import numpy as np
import pandas as pd
import openai
import mlflow

from ai_web_explorer import config
from ai_web_explorer import promptrepo

TESTS_PATH = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(TESTS_PATH, "data")
TITLES_PATH = os.path.join(DATA_PATH, "title")

openai_client = openai.OpenAI()


def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        model="text-embedding-3-small", input=texts
    )
    return [embedding.embedding for embedding in response.data]


def cosine_similarity(embeddings: list[list[float]]) -> float:
    return np.dot(embeddings[0], embeddings[1]) / (
        np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
    )


def eval_prompt(titles: list[dict], prompt_name, image: bool):
    print("-----")
    print(f"Evaluating prompt {prompt_name}")
    prompt = promptrepo.get_prompt(prompt_name, config.PROMPTS_PATH_TEST)
    similarities = []
    prices = []
    titles_generated = []

    for title in titles:
        screenshot_path = os.path.join(
            TITLES_PATH, "screenshots", f"{title['page_uuid']}_start.png"
        )
        html_path = os.path.join(TITLES_PATH, "htmls", f"{title['page_uuid']}.html")

        if image:
            with open(screenshot_path, "rb") as screenshot_fd:
                screenshot_bytes = screenshot_fd.read()
        else:
            screenshot_bytes = None

        with open(html_path) as html_fd:
            html = html_fd.read()

        response = prompt.execute_prompt(
            openai_client, image_bytes=screenshot_bytes, html=html
        )

        if not response.message.tool_calls or len(response.message.tool_calls) == 0:
            raise ValueError("No tool calls in response when getting page title")

        args_str = response.message.tool_calls[0].function.arguments
        print(args_str)
        args = json.loads(args_str)
        title_generated = args["title"]
        embeddings = get_embeddings([title_generated, title["title"]])

        similarity = cosine_similarity(embeddings)
        similarities.append(similarity)
        price = prompt.get_last_price()
        prices.append(price)
        titles_generated.append(title_generated)

    df = pd.DataFrame(titles)
    df["title_generated"] = titles_generated
    df["similarity"] = similarities
    df["price"] = prices

    similatity_mean = np.mean(similarities)
    similarity_min = np.min(similarities)
    price_total = np.sum(prices)
    similarity_per_price = similatity_mean / price_total

    print("Similarity mean: %.4f" % similatity_mean)
    print("Similarity min: %.4f" % similarity_min)
    print("Price total: %.4f" % price_total)
    print("Similarity per price: %.4f" % similarity_per_price)

    mlflow.log_metric("similarity_mean", float(similatity_mean))
    mlflow.log_metric("similarity_min", similarity_min)
    mlflow.log_metric("price_total", price_total)
    mlflow.log_metric("similarity_per_price", similarity_per_price)
    mlflow.log_table(data=df, artifact_file="outputs.json")


def test_title_image_impact():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Title Input Types")

    title_list_path = os.path.join(TITLES_PATH, "titles.csv")

    with open(title_list_path) as titles_fd:
        titles_reader = csv.DictReader(titles_fd)
        titles = list(titles_reader)
    
    with mlflow.start_run():
        mlflow.log_param("input_type", "both")
        eval_prompt(titles, "page_title_both", image=True)

    with mlflow.start_run():
        mlflow.log_param("input_type", "image_only")
        eval_prompt(titles, "page_title_image_only", image=True)

    with mlflow.start_run():
        mlflow.log_param("input_type", "html_only")
        eval_prompt(titles, "page_title_html_only", image=False)
