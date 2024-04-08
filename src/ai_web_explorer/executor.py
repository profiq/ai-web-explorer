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
                logging.error("No tool calls in response when executing action")
                return False

            tool_calls = response.message.tool_calls
            break

        for tool_call in tool_calls:
            try:
                self._execute_tool_call(tool_call)
            except playwright.sync_api.TimeoutError:
                logging.error("Timeout error when executing action")
                logging.error(f"Tool call: {tool_call}")
                return False
        return html.get_full_html(self._page) != html_full

    def _execute_tool_call(self, tool_call):
        args = json.loads(tool_call.function.arguments)
        logging.info(f"Executing tool call: {tool_call.function.name}")
        logging.info(f"Arguments: {args}")
        if tool_call.function.name == "click_element":
            selector = args["selector"]
            self._page.click(selector)
        elif tool_call.function.name == "fill_text_input":
            selector = args["selector"]
            text = args["text"]
            self._page.fill(selector, text)
        else:
            raise ValueError(f"Unknown function {tool_call.function.name}")
