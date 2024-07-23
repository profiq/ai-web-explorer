import os

import openai

from ai_web_explorer import promptrepo
from ai_web_explorer import config
import base

ACTIONS_PATH = os.path.join(base.DATA_PATH, "actions")


def test_action_success():
    actions_list_path = os.path.join(ACTIONS_PATH, "actions.txt")

    actions = []

    with open(actions_list_path) as actions_list_fd:
        for line in actions_list_fd:
            meta, text = line.strip().split("\t")
            meta_split = meta.split(" ")
            success = int(meta_split[0])
            uuid = meta_split[1]
            actions.append({"success": success, "uuid": uuid, "text": text})

    eval_prompt(actions[:5], "verify_action_images", image=True, same_fail=True)


def eval_prompt(actions: list[dict], prompt_name, image: bool, same_fail: bool):
    openai_client = openai.OpenAI()
    screenshots_path = os.path.join(ACTIONS_PATH, "screenshots")
    prompt = promptrepo.get_prompt(prompt_name, config.PROMPTS_PATH_TEST)

    correct_answers = 0

    for action in actions:
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

        image_bytes = [screenshot_before, screenshot_after] if image else None

        result = prompt.execute_prompt(
            openai_client,
            image_bytes=image_bytes,
            action=action["text"],
        )

        if result.message.content is None:
            action_status = 0
        else:
            action_status = int("success" in result.message.content.lower())

        if action_status == action["success"]:
            correct_answers += 1

    print(correct_answers / len(actions) * 100)
