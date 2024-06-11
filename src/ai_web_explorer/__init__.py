import argparse
import logging
import time

import openai
import playwright.sync_api

from . import loop

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

parser.add_argument(
    "--login",
    "-l",
    type=str,
    help="The username and password to use for login on the website separated by a colon",
    default=None,
)

parser.add_argument(
    "--additional-info",
    "-a",
    type=str,
    help="Additional information to be used for exploration",
    default="----- No additional information was provided -----",
)

parser.add_argument(
    "--output",
    "-o",
    type=str,
    help='Output format - "jsonemb" (includes title embedding), "json" or "digraph". JSON is more detailed, digraph can be easily visualized',
    default="digraph",
)


def main():
    args = parser.parse_args()
    domain = args.domain
    url = f"http://{domain}"
    openai_client = openai.Client()

    logging.info(f"Exploring {domain}")

    credentials = args.login.split(":") if args.login else [None, None]

    loop_config = loop.LoopConfig(
        iterations=args.iterations,
        store_titles=args.store_titles,
        confirm_titles=args.store_titles,
        username=credentials[0],
        password=credentials[1],
        additional_info=args.additional_info,
    )

    explore_loop = loop.ExploreLoop(domain, url, openai_client, loop_config)
    explore_loop.start()
    if args.output == "jsonemb":
        explore_loop.print_json(True)
    elif args.output == "json":
        explore_loop.print_json(False)
    elif args.output == "digraph":
        explore_loop.print_graph()
    explore_loop.stop()
