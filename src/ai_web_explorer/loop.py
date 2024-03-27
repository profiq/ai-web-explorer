from . import webstate
from . import title
import openai
import playwright.sync_api
import logging


class ExploreLoop:

    def __init__(
        self,
        domain: str,
        url: str,
        page: playwright.sync_api.Page,
        openai_client: openai.OpenAI,
        iterations: int | None = None,
    ):
        self._page = page
        self._domain = domain
        self._url = url
        self._openai_client = openai_client
        self._iterations = iterations
        self._webstates: list[webstate.WebState] = []

    def start(self):
        while self._iterations is None or len(self._webstates) < self._iterations:
            self._explore()
            break

    def _explore(self):
        page_title = title.get_title_for_webpage(self._page, self._openai_client)
        embedding = title.get_title_embedding(page_title, self._openai_client)
        print(embedding)

        for ws in self._webstates:
            distance = ws.cosine_distance(embedding)
            if distance > 0.95:
                logging.info(f"Found similar state: {ws.title}")
