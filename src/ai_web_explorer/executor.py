import json
import logging

import openai
from openai.types import chat
import playwright.sync_api

from . import webstate
from . import promptrepo
from . import html
from . import config


class Executor:

    def __init__(self, page: playwright.sync_api.Page, client: openai.OpenAI):
        self._page = page
        self._client = client

    def execute(self, action: webstate.Action) -> bool:
        prompt = promptrepo.get_prompt("execute_action")
        prompt_verify = promptrepo.get_prompt("verify_action")

        logging.info(f"Executing action: {action.description}")
        html_full = html.get_full_html(self._page)
        messages: list[chat.ChatCompletionMessageParam] = []

        for i, html_part in enumerate(html.iterate_html(self._page)):
            if i != action.part:
                continue

            messages.append(
                {
                    "role": "user",
                    "content": prompt.prompt_with_data(
                        action=action.description, html=html_part
                    ),
                }
            )

        if len(messages) == 0:
            logging.error("No messages to send in response")
            return False

        for _ in range(config.ACTION_MAX_TRIES):
            response = self._client.chat.completions.create(
                model=prompt.model,
                messages=messages,
                temperature=prompt.temperature,
                tools=prompt.tools,  # type: ignore
            ).choices[0]

            messages.append(response.message)  # type: ignore
            
            if not response.message.tool_calls or len(response.message.tool_calls) == 0:
                if (
                    response.message.content
                    and "success" in response.message.content.lower()
                ):
                    logging.info("Action completed successfully")
                elif (
                    response.message.content
                    and "failure" in response.message.content.lower()
                ):
                    logging.info("Action failed")
                else:
                    logging.error("No tool calls in response when executing action")
                break

            tool_calls = response.message.tool_calls

            for tool_call in tool_calls:
                try:
                    response_message = self._execute_tool_call(tool_call)
                except playwright.sync_api.TimeoutError:
                    logging.error("Timeout error when executing action")
                    logging.error(f"Tool call: {tool_call}")
                    response_message = {
                        "role": "tool",
                        "content": "Timeout error",
                        "tool_call_id": tool_call.id,
                    }
                except playwright.sync_api.Error as e:
                    logging.error("Error when executing action")
                    logging.error(f"Tool call: {tool_call}")
                    logging.error(e)
                    response_message = {
                        "role": "tool",
                        "content": f"Error: {e}",
                        "tool_call_id": tool_call.id,
                    }
                messages.append(response_message)  # type: ignore
            messages.append({"role": "user", "content": prompt_verify.prompt_text})

        return html.get_full_html(self._page) != html_full

    def _execute_tool_call(self, tool_call):
        args = json.loads(tool_call.function.arguments)
        selector = args["selector"]
        self._page.locator(selector).scroll_into_view_if_needed(timeout=config.PLAYWRIGHT_TIMEOUT)
        logging.info(f"Executing tool call: {tool_call.function.name}")
        logging.info(f"Arguments: {args}")
        if tool_call.function.name == "click_element":
            self._page.click(selector, timeout=config.PLAYWRIGHT_TIMEOUT)
        elif tool_call.function.name == "fill_text_input":
            text = args["text"]
            self._page.fill(selector, text, timeout=config.PLAYWRIGHT_TIMEOUT)
        else:
            raise ValueError(f"Unknown function {tool_call.function.name}")
        return {"role": "tool", "content": "OK", "tool_call_id": tool_call.id}
