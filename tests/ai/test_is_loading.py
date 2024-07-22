import os

import openai

from ai_web_explorer import promptrepo
from ai_web_explorer import config
import base

LOADING_YES_PATH = os.path.join(base.DATA_PATH, "loading", "yes")
LOADING_NO_PATH = os.path.join(base.DATA_PATH, "loading", "no")

openai_client = openai.OpenAI()


def test_is_loading():
    prompt = promptrepo.get_prompt("is_loading", config.PROMPTS_PATH_TEST)
    num_correct = 0
    total = 0

    for file in os.listdir(LOADING_YES_PATH):
        with open(os.path.join(LOADING_YES_PATH, file), "rb") as fd:
            screenshot_bytes = fd.read()
        response = prompt.execute_prompt(openai_client, image_bytes=screenshot_bytes)

        if response.message.content is None:
            raise ValueError("No content in response when checking if page is loading")

        is_loading = "yes" in response.message.content.lower()

        if is_loading:
            num_correct += 1

        total += 1

    for file in os.listdir(LOADING_NO_PATH):
        with open(os.path.join(LOADING_NO_PATH, file), "rb") as fd:
            screenshot_bytes = fd.read()
        response = prompt.execute_prompt(openai_client, image_bytes=screenshot_bytes)

        if response.message.content is None:
            raise ValueError("No content in response when checking if page is loading")

        is_loading = "yes" in response.message.content.lower()

        if not is_loading:
            num_correct += 1

        total += 1

    accuracy = num_correct / total
    assert accuracy > 0.9
    print(f"Accuracy {accuracy:.4f}")
