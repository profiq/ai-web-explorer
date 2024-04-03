import json
import logging

import openai
import playwright.sync_api
import yaml

from . import promptrepo
from . import html
from . import config
from . import webstate


class Describer:

    def __init__(self, page: playwright.sync_api.Page, client: openai.OpenAI):
        self._page = page
        self._client = client

    def get_title(self, confirm: bool = True, store_title: bool = True) -> str:
        page_html = html.get_full_html(self._page)[: config.HTML_PART_LENGTH * 2]
        prompt = promptrepo.get_prompt("page_title")
        logging.info(f"Getting title for webpage")
        response = prompt.execute_prompt(self._client, html=page_html)

        if not response.message.tool_calls or len(response.message.tool_calls) == 0:
            raise ValueError("No tool calls in response when getting page title")

        args_str = response.message.tool_calls[0].function.arguments
        args = json.loads(args_str)
        title = args["title"]

        while confirm:
            print("The page will have the following title:")
            print(title)
            user_input = input("Is this title correct? (y/n) ")
            if user_input.lower() != "y":
                title = input("Please enter the correct title: ")
            else:
                confirm = False

        if store_title:
            with open(config.TITLES_PATH, "a") as f:
                f.write(
                    json.dumps(
                        {"url": self._page.url, "html": page_html, "title": title}
                    )
                    + "\n"
                )

        logging.info(f"Title for webpage: {title}")
        return title

    def get_title_embedding(self, title: str) -> list[float]:
        logging.info(f"Getting embedding for title: {title}")
        response = self._client.embeddings.create(
            model="text-embedding-3-small", input=[title]
        )
        return response.data[0].embedding

    def get_description(self) -> list[dict]:
        description = []
        prompt = promptrepo.get_prompt("describe_html")
        logging.info(f"Getting description for webpage")

        for part in html.iterate_html(self._page):
            response = prompt.execute_prompt(self._client, html_part=part)

            if not response.message.tool_calls or len(response.message.tool_calls) == 0:
                raise ValueError("No tool calls in response when getting description")

            args_str = response.message.tool_calls[0].function.arguments
            args = json.loads(args_str)
            description.append(args)

        return description

    def get_actions(self, description: list[dict]) -> list[webstate.Action]:
        description_str = "\n\n".join(
            [
                "PART " + str(i) + ":\n" + yaml.dump(part)
                for i, part in enumerate(description)
            ]
        )

        prompt = promptrepo.get_prompt("suggest_actions")
        logging.info(f"Getting actions for webpage")

        response = prompt.execute_prompt(
            self._client,
            description=description_str,
            url=self._page.url,
            title=self._page.title,
        )

        if not response.message.tool_calls or len(response.message.tool_calls) == 0:
            raise ValueError("No tool calls in response when getting actions")

        args_str = response.message.tool_calls[0].function.arguments
        args = json.loads(args_str)

        if not "actions" in args:
            raise ValueError("No actions in response when getting actions")

        actions = [webstate.Action(**action) for action in args["actions"]]
        return actions

