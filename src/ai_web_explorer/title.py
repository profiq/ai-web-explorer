import logging
import openai
import json
import playwright.sync_api

from . import promptrepo
from . import html
from . import config


def get_title_for_webpage(
    page: playwright.sync_api.Page,
    client: openai.Client,
    confirm: bool = True,
    store_title: bool = True,
) -> str:
    page_html = html.get_full_html(page)[: config.HTML_PART_LENGTH * 2]
    prompt = promptrepo.get_prompt("page_title")
    logging.info(f"Getting title for webpage")
    response = prompt.execute_prompt(client, html=page_html)

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
            f.write(json.dumps({"url": page.url, "html": page_html, "title": title}) + "\n")

    logging.info(f"Title for webpage: {title}")
    return title
