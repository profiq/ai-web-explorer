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


    # Accept cookies if a cookie banner is present

    loop_config = loop.LoopConfig(
        iterations=args.iterations,
        store_titles=args.store_titles,
        confirm_titles=args.store_titles,
    )

    explore_loop = loop.ExploreLoop(domain, url, openai_client, loop_config)
    explore_loop.start()
    explore_loop.print_graph()
    explore_loop.stop()
