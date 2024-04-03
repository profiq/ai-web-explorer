import dataclasses
import logging

import openai
import playwright.sync_api

from . import webstate
from . import describer
from . import executor


@dataclasses.dataclass
class LoopConfig:
    iterations: int | None = dataclasses.field(default=None)
    confirm_titles: bool = dataclasses.field(default=False)
    store_titles: bool = dataclasses.field(default=False)


class ExploreLoop:

    def __init__(
        self,
        domain: str,
        url: str,
        page: playwright.sync_api.Page,
        openai_client: openai.OpenAI,
        config: LoopConfig,
    ):
        self._page = page
        self._domain = domain
        self._url = url
        self._openai_client = openai_client
        self._config = config

        self._webstates: list[webstate.WebState] = []
        self._webstate_current: webstate.WebState | None = None
        self._action_current: webstate.Action | None = None
        self._describer = describer.Describer(page, openai_client)
        self._executor = executor.Executor(page, openai_client)

    def start(self):
        i = 0

        while self._config.iterations is None or i < self._config.iterations:
            logging.info(f"Iteration {i}")
            self._explore()
            i += 1
            break

    def _explore(self):
        logging.info(f"Current URL: {self._url}")
        ws = self._get_webstate()

        if self._webstate_current and self._action_current:
            self._webstate_current.transitions.append(
                webstate.StateTransition(action=self._action_current, state_new=ws)
            )

        self._webstate_current = ws
        self._action_current = ws.random_action

        if not self._action_current:
            logging.info("No more actions to take")
            return

        logging.info(f"Randomly selected action: {self._action_current.description}")
        action_result = self._executor.execute(self._action_current)
        logging.info(f"Action result: {action_result}")

    def _get_webstate(self) -> webstate.WebState:
        page_title = self._describer.get_title(
            self._config.confirm_titles, self._config.store_titles
        )
        embedding = self._describer.get_title_embedding(page_title)

        for ws in self._webstates:
            distance = ws.cosine_distance(embedding)
            if distance > 0.95:
                logging.info(f"Found similar state: {ws.title}")
                if self._url not in ws.urls:
                    ws.urls.append(self._url)
                return ws

        logging.info(f"Creating new state: {page_title}")
        ws = webstate.WebState(
            title=page_title,
            title_embedding=embedding,
            urls=[self._url],
            description=self._describer.get_description(),
            actions=[],
            transitions=[],
        )

        ws.actions = self._describer.get_actions(ws.description)
        self._webstates.append(ws)
        return ws
