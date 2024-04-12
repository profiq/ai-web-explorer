import dataclasses
import logging

import openai
import playwright.sync_api

from . import webstate
from . import describer
from . import executor
from . import config
from . import cookies
from . import html


@dataclasses.dataclass
class LoopConfig:
    iterations: int | None = dataclasses.field(default=None)
    confirm_titles: bool = dataclasses.field(default=False)
    store_titles: bool = dataclasses.field(default=False)
    username: str | None = dataclasses.field(default=None)
    password: str | None = dataclasses.field(default=None)


WebStateBacktrack = tuple[
    webstate.WebState, int | None, webstate.StateTransition | None
]


class ExploreLoop:

    def __init__(
        self,
        domain: str,
        url: str,
        openai_client: openai.OpenAI,
        config: LoopConfig,
    ):
        self._domain = domain
        self._url = url
        self._openai_client = openai_client
        self._config = config

        self._webstates: list[webstate.WebState] = []
        self._webstate_current: webstate.WebState | None = None
        self._action_current: webstate.Action | None = None
        self._pw, self._page = self._init_browser(self._url)
        self._describer = describer.Describer(self._page, openai_client)
        self._executor = executor.Executor(
            self._page, openai_client, config.username, config.password
        )

    def start(self):
        i = 0

        while self._config.iterations is None or i < self._config.iterations:
            logging.info(f"Iteration {i}")
            self._explore()
            i += 1

        self._explore(True)

    def _explore(self, finish=False):
        logging.info(f"Current URL: {self._page.url}")
        ws = self._get_webstate()

        if (
            self._webstate_current
            and self._action_current
            and self._action_current.status == "success"
        ):
            self._webstate_current.transitions.append(
                webstate.StateTransition(action=self._action_current, state_new=ws)
            )

        if finish:
            return

        self._webstate_current = ws
        self._action_current = ws.random_action

        if not self._action_current:
            logging.info("No more actions to take on this page")
            self._action_current = self._search_next_available_action()
            if not self._action_current:
                logging.info("No more actions to take in the domain")
                return

        logging.info(f"Randomly selected action: {self._action_current.description}")
        action_result, tool_calls = self._executor.execute(self._action_current)
        self._action_current.function_calls = tool_calls

        if self._domain not in self._page.url:
            self._back_to_domain()
            self._action_current.status = "failure"
        else:
            self._action_current.status = "success" if action_result else "failure"

        logging.info(f"Action result: {self._action_current.status}")

    def _get_webstate(self) -> webstate.WebState:
        page_title = self._describer.get_title(
            self._config.confirm_titles, self._config.store_titles
        )
        embedding = self._describer.get_title_embedding(page_title)

        for ws in self._webstates:
            distance = ws.cosine_distance(embedding)
            if distance > config.TITLE_SIMILARITY_THRESHOLD:
                logging.info(f"Found similar state: {ws.title}")
                if self._url not in ws.urls:
                    ws.urls.append(self._url)
                return ws

        logging.info(f"Creating new state: {page_title}")
        ws = webstate.WebState(
            title=page_title,
            title_embedding=embedding,
            urls=[self._page.url],
            description=self._describer.get_description(),
            actions=[],
            transitions=[],
        )

        ws.actions = self._describer.get_actions(ws.title, ws.description)
        self._webstates.append(ws)
        return ws

    def print_graph(self):
        print("digraph G {")

        for ws in self._webstates:
            ws_id = str(ws.ws_id).replace("-", "")
            print(f'ws_{ws_id} [label="{ws.title}"]')

        for ws in self._webstates:
            ws_id = str(ws.ws_id).replace("-", "")
            for transition in ws.transitions:
                ws_new = transition.state_new
                ws_new_id = str(ws_new.ws_id).replace("-", "")
                print(
                    f'ws_{ws_id} -> ws_{ws_new_id} [label="{transition.action.description}"]'
                )

        print("}")

    def stop(self):
        self._pw.stop()

    def _init_browser(
        self, url: str
    ) -> tuple[playwright.sync_api.Playwright, playwright.sync_api.Page]:
        pw = playwright.sync_api.sync_playwright().start()
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()
        page.add_init_script(html.JS_FUNCTIONS)
        page.goto(url)
        cookies.accept_cookies_if_present(self._openai_client, page)
        return pw, page

    def _back_to_domain(self):
        logging.info("Navigated away from domain, going back to domain")
        while self._domain not in self._page.url:
            self._page.go_back()

    def _search_next_available_action(self) -> webstate.Action | None:
        states_visiting: list[WebStateBacktrack] = [(self._webstates[0], None, None)]
        states_visited = []

        while len(states_visiting) > 0:
            ws = states_visiting.pop()
            states_visited.append(ws)

            for action in ws[0].actions:
                if action.status == "none":
                    logging.info(
                        f"Found next available action: {action.description} in state: {ws[0].title}"
                    )
                    transitions = []
                    back: WebStateBacktrack = ws
                    while type(back[1]) == int:
                        transitions.append(back[2])
                        back = states_visited[back[1]]  # type: ignore
                    transitions.reverse()
                    logging.info(f"Transitions to get to state: {ws[0].title}")
                    for transition in transitions:
                        logging.info(f"{transition.action.description}")
                    self._perform_transitions(transitions)
                    return action

            for transition in ws[0].transitions:
                if transition.state_new not in [state[0] for state in states_visited]:
                    states_visiting.append(
                        (transition.state_new, len(states_visited) - 1, transition)
                    )

        return None

    def _perform_transitions(self, transitions: list[webstate.StateTransition]):
        self.stop()
        self._pw, self._page = self._init_browser(self._url)
        self._executor = executor.Executor(self._page, self._openai_client)
        self._describer = describer.Describer(self._page, self._openai_client)
        self._page.goto(self._url)
        for transition in transitions:
            self._executor.replicate_tool_calls(transition.action.function_calls)
            self._webstate_current = transition.state_new
