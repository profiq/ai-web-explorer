import openai
import playwright.sync_api

from . import promptrepo
from . import html


def accept_cookies_if_present(
    client: openai.Client, page: playwright.sync_api.Page
) -> bool:
    prompt_search_cookies = promptrepo.get_prompt("search_cookies")
    has_cookies = False
    for html_part in html.iterate_html(page):
        response = prompt_search_cookies.execute_prompt(client, html_part=html_part)
        if response.message.content and "yes" in response.message.content.lower():
            has_cookies = True
            break

    if not has_cookies:
        return False

    prompt_accept_selector = promptrepo.get_prompt("accept_cookies_selector")
    response = prompt_accept_selector.execute_prompt(client, html_part=html_part)
    
    if not response.message.tool_calls or len(response.message.tool_calls) == 0:
        return False

    args = response.message.tool_calls[0].function.arguments
    print(args)


