import argparse
import logging
import time
import openai
import playwright.sync_api

from . import cookies


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Explore a website")
parser.add_argument("domain", help="The domain to start the exploration from")


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
    page.goto(url)

    # Accept cookies if a cookie banner is present
    cookies.accept_cookies_if_present(openai_client, page)

    pw.stop()
