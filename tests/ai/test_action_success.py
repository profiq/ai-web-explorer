import os

import mlflow
import openai

from ai_web_explorer import promptrepo
from ai_web_explorer import config
import base

ACTIONS_PATH = os.path.join(base.DATA_PATH, "actions")


def test_action_success():
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Action success")
    actions_list_path = os.path.join(ACTIONS_PATH, "actions.txt")

    actions = []

    with open(actions_list_path) as actions_list_fd:
        for line in actions_list_fd:
            meta, text = line.strip().split("\t")
            meta_split = meta.split(" ")
            success = int(meta_split[0])
            uuid = meta_split[1]
            actions.append({"success": success, "uuid": uuid, "text": text})

    with mlflow.start_run():
        mlflow.log_param("model", "gpt-4o-mini")
        mlflow.log_param("input_type", "verify_action_html")
        eval_prompt(
            actions,
            "verify_action_html",
            image=True,
            same_fail=True,
            model="gpt-4o-mini",
        )

    with mlflow.start_run():
        mlflow.log_param("model", "gpt-4o-mini")
        mlflow.log_param("input_type", "verify_action_images")
        eval_prompt(
            actions,
            "verify_action_images",
            image=True,
            same_fail=True,
            model="gpt-4o-mini",
        )

    with mlflow.start_run():
        mlflow.log_param("model", "gpt-4o")
        mlflow.log_param("input_type", "verify_action_html")
        eval_prompt(
            actions,
            "verify_action_html",
            image=True,
            same_fail=True,
            model="gpt-4o",
        )

    with mlflow.start_run():
        mlflow.log_param("model", "gpt-4o")
        mlflow.log_param("input_type", "verify_action_images")
        eval_prompt(
            actions,
            "verify_action_images",
            image=True,
            same_fail=True,
            model="gpt-4o",
        )


def eval_prompt(
    actions: list[dict], prompt_name, image: bool, same_fail: bool, model="gpt-4o"
):
    print("-----")
    print(f"Evaluating prompt {prompt_name}, model: {model}")

    openai_client = openai.OpenAI()
    screenshots_path = os.path.join(ACTIONS_PATH, "screenshots")
    htmls_path = os.path.join(ACTIONS_PATH, "htmls")
    prompt = promptrepo.get_prompt(prompt_name, config.PROMPTS_PATH_TEST)
    prompt.model = model

    correct_answers = 0
    price_total = 0

    for action in actions:
        print(f"Action: {action['text']}")
        with open(
            os.path.join(screenshots_path, f"{action['uuid']}_before.png"), "rb"
        ) as screenshot_fd:
            screenshot_before = screenshot_fd.read()

        with open(
            os.path.join(screenshots_path, f"{action['uuid']}_after.png"), "rb"
        ) as screenshot_fd:
            screenshot_after = screenshot_fd.read()

        if same_fail:
            if screenshot_before == screenshot_after:
                if action["success"] == 0:
                    correct_answers += 1
                    continue

        with open(os.path.join(htmls_path, f"{action['uuid']}_before.html")) as html_fd:
            html_before = html_fd.read()

        with open(os.path.join(htmls_path, f"{action['uuid']}_after.html")) as html_fd:
            html_after = html_fd.read()

        image_bytes = [screenshot_before, screenshot_after] if image else None

        result = prompt.execute_prompt(
            openai_client,
            image_bytes=image_bytes,
            action=action["text"],
            html_before=html_before,
            html_after=html_after,
        )

        if result.message.content is None:
            action_status = 0
        else:
            action_status = int("success" in result.message.content.lower())

        if action_status == action["success"]:
            correct_answers += 1

        price_total += prompt.get_last_price()

    accuracy = correct_answers / len(actions) * 100
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("price_total", price_total)

    print("Accuracy: ", accuracy)
    print("Total price: ", price_total)
