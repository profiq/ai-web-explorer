import json
import logging

import openai
import playwright.sync_api

from . import webstate
from . import promptrepo
from . import html


class Executor:

    def __init__(self, page: playwright.sync_api.Page, client: openai.OpenAI):
        self._page = page
        self._client = client

    def execute(self, action: webstate.Action) -> bool:
        prompt = promptrepo.get_prompt("execute_action")
        logging.info(f"Executing action: {action.description}")

        tool_calls = []
        html_full = html.get_full_html(self._page)

        for i, html_part in enumerate(html.iterate_html(self._page)):
            if i != action.part:
                continue
            response = prompt.execute_prompt(
                self._client,
                action=action.description,
                html=html_part,
            )

            if not response.message.tool_calls or len(response.message.tool_calls) == 0:
                raise ValueError("No tool calls in response when executing action")

            tool_calls = response.message.tool_calls
            break

        for tool_call in tool_calls:
            args = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "click_element":
                selector = args["selector"]
                self._page.click(selector)
            elif tool_call.function.name == "fill_text_input":
                selector = args["selector"]
                text = args["text"]
                self._page.fill(selector, text)
            else:
                raise ValueError(f"Unknown function {tool_call.function.name}")

        return html.get_full_html(self._page) != html_full
