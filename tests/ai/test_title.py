import datetime
import json
import os
import time
import typing

from anthropic.types.beta.tools import ToolUseBlock
import mlflow
import numpy as np
import openai
import htmlmin
from ai_web_explorer import config
from ai_web_explorer import promptrepo


PRICES = {
    "gpt-4": {
        "input": 10,
        "output": 30,
    },
    "gpt-3.5": {
        "input": 0.5,
        "output": 1.5,
    },
    "claude-3-sonnet": {
        "input": 3.0,
        "output": 15.0,
    },
    "gemini-1.0": {
        "input": 0.125,
        "output": 0.375,
    },
    "gemini-1.5": {
        "input": 2.5,
        "output": 7.5,
    },
}

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


def _test_model(model_name: str, test_fn: typing.Callable):
    prompt = promptrepo.get_prompt("page_title")
    prompt.model = model_name
    eval_table = {
        "no": [],
        "title_expected": [],
        "title_generated": [],
        "similarity": [],
        "time": [],
        "price": [],
    }

    with open(os.path.join(DATA_DIR, "titles_expected.txt")) as f:
        titles_expected = f.read().splitlines()

    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".html"):
            continue

        no = int(filename.split(".")[0])
        with open(os.path.join(DATA_DIR, filename)) as f:
            html = f.read()[: config.HTML_PART_LENGTH]
            html = htmlmin.minify(html)
            time_start = datetime.datetime.now()
            result = test_fn(prompt, html, titles_expected[no - 1])
            time_end = datetime.datetime.now()
            eval_table["time"].append((time_end - time_start).total_seconds())

        eval_table["no"].append(no)
        eval_table["title_expected"].append(titles_expected[no - 1])
        eval_table["title_generated"].append(result[0])
        eval_table["similarity"].append(result[1])
        eval_table["price"].append(result[2])

    return eval_table, prompt.prompt_text, model_name


def _log_metrics(eval_table: dict, prompt: str, model_type: str):
    mlflow.log_metric("similarity_mean", float(np.mean(eval_table["similarity"])))
    mlflow.log_metric("similarity_std", float(np.std(eval_table["similarity"])))
    mlflow.log_metric("time_mean", float(np.mean(eval_table["time"])))
    mlflow.log_metric("price_mean", float(np.mean(eval_table["price"])))
    mlflow.log_metric(
        "similarity_per_dollar",
        float(np.mean(eval_table["similarity"]) / np.mean(eval_table["price"])),
    )
    mlflow.log_metric(
        "similarity_per_second",
        float(np.mean(eval_table["similarity"]) / np.mean(eval_table["time"])),
    )

    mlflow.log_table(data=eval_table, artifact_file="eval_table.json")
    mlflow.log_param("model", model_type)
    mlflow.log_param("prompt", prompt)


def _test_gpt(prompt: promptrepo.Prompt, html: str, title_expected: str) -> tuple:
    client = openai.OpenAI()
    completion = prompt.execute_prompt(client, html=html)
    response = completion.choices[0]
    args_str = response.message.tool_calls[0].function.arguments
    args = json.loads(args_str)
    title_generated = args["title"]
    similarity = compare_embeddings(title_expected, title_generated)
    model_type = prompt.model.split("-")[0] + "-" + prompt.model.split("-")[1]
    prices = PRICES[model_type]
    price_input = prices["input"] * completion.usage.prompt_tokens / 1_000_000
    price_output = prices["output"] * completion.usage.completion_tokens / 1_000_000
    price_total = price_input + price_output
    return title_generated, similarity, price_total


def _test_claude(prompt: promptrepo.Prompt, html: str, title_expected: str) -> tuple:
    completion = prompt.execute_prompt_anthropic(
        model="claude-3-sonnet-20240229", html=html
    )
    for response in completion.content:
        if type(response) == ToolUseBlock:
            title_generated = response.input["title"]
            similarity = compare_embeddings(title_expected, title_generated)
            break
    else:
        title_generated = ""
        similarity = 0
    model_type = "claude-3-sonnet"
    prices = PRICES[model_type]
    price_input = prices["input"] * completion.usage.input_tokens / 1_000_000
    price_output = prices["output"] * completion.usage.output_tokens / 1_000_000
    price_total = price_input + price_output
    return title_generated, similarity, price_total


def _test_gemini(prompt: promptrepo.Prompt, html: str, title_expected: str) -> tuple:
    time.sleep(5)
    model_type_map = {
        "gemini-1.0": "gemini-1.0-pro-002",
        "gemini-1.5": "gemini-1.5-pro-preview-0409",
    }

    completion = prompt.execute_prompt_gemini(
        model=model_type_map[prompt.model], html=html
    )
    title_generated = (
        completion.candidates[0].content.parts[0].function_call.args["title"]
    )
    similarity = compare_embeddings(title_expected, title_generated)
    prices = PRICES[prompt.model]
    price_input = len(prompt.prompt_with_data(html=html)) * prices["input"] / 1_000_000
    price_output = (
        len(str(completion.candidates[0].content.parts[0]))
        * prices["output"]
        / 1_000_000
    )
    price_total = price_input + price_output
    return title_generated, similarity, price_total


def test_title():
    """
    print("GPT 3.5")
    with mlflow.start_run():
        eval_table, prompt, model_type = _test_model("gpt-3.5-turbo", _test_gpt)
        _log_metrics(eval_table, prompt, model_type)

    print("GPT 4")
    with mlflow.start_run():
        eval_table, prompt, model_type = _test_model("gpt-4-turbo", _test_gpt)
        _log_metrics(eval_table, prompt, model_type)
    
    """
    print("Claude")
    with mlflow.start_run():
        eval_table, prompt, model_type = _test_model("claude-3-sonnet", _test_claude)
        _log_metrics(eval_table, prompt, model_type)

    """
    print("Gemini 1.0")
    with mlflow.start_run():
        eval_table, prompt, model_type = _test_model("gemini-1.0", _test_gemini)
        _log_metrics(eval_table, prompt, model_type)
    """

    print("Gemini 1.5")
    with mlflow.start_run():
        eval_table, prompt, model_type = _test_model("gemini-1.5", _test_gemini)
        _log_metrics(eval_table, prompt, model_type)
