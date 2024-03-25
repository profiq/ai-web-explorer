import argparse
import logging
import playwright.sync_api


logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Explore a website")
parser.add_argument("domain", help="The domain to start the exploration from")


def main():
    args = parser.parse_args()
    domain = args.domain
    logging.info(f"Exploring {domain}")
    
    pw = playwright.sync_api.sync_playwright().start()
    browser = pw.chromium.launch()
    page = browser.new_page()
    page.goto(domain)

    


    page.close()
    browser.close()
    pw.stop()
