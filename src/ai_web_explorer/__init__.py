import argparse
import logging

import openai
import playwright.sync_api

from . import cookies
from . import loop
from . import html

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Explore a website")
parser.add_argument("domain", help="The domain to start the exploration from")
parser.add_argument(
    "--iterations",
    "-i",
    type=int,
    help="The number of iterations to explore",
    default=None,
)
parser.add_argument(
    "--store-titles",
    "-t",
    action="store_true",
    help="Store the titles of the webpages",
    default=False,
)


def main():
    args = parser.parse_args()
    domain = args.domain
    url = f"http://{domain}"
    openai_client = openai.Client()

    logging.info(f"Exploring {domain}")

    # Launch the browser and navigate to the URL
    pw = playwright.sync_api.sync_playwright().start()
    browser = pw.chromium.launch(headless=False)
    page = browser.new_page()
    page.add_init_script(html.JS_FUNCTIONS)
    page.goto(url)

    # Accept cookies if a cookie banner is present
    cookies.accept_cookies_if_present(openai_client, page)

    loop_config = loop.LoopConfig(
        iterations=args.iterations,
        store_titles=args.store_titles,
        confirm_titles=args.store_titles,
    )

    explore_loop = loop.ExploreLoop(domain, url, page, openai_client, loop_config)
    explore_loop.start()
    explore_loop.print_graph()

    pw.stop()
