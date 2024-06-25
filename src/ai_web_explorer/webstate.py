import dataclasses
import json
import typing
import uuid
from openai.types.chat import ChatCompletionMessageToolCall
import numpy as np

ActionStatus = typing.Literal["none", "success", "failure"]


@dataclasses.dataclass
class Action:
    description: str
    part: int
    priority: int
    status: ActionStatus = dataclasses.field(default="none")
    function_calls: list[dict] = dataclasses.field(default_factory=list)

    def dict(self, simple=False):

        if simple:
            return self.description

        return {
            "description": self.description,
            "part": self.part,
            "priority": self.priority,
            "status": self.status,
            "function_calls": [f.model_dump() for f in self.function_calls],
        }

    @staticmethod
    def from_dict(action_raw) -> "Action":
        return Action(
            description=action_raw["description"],
            part=action_raw["part"],
            priority=action_raw["priority"],
            status=action_raw["status"],
            function_calls=[ChatCompletionMessageToolCall(**f) for f in action_raw["function_calls"]],  # type: ignore
        )


@dataclasses.dataclass
class StateTransition:
    action: Action
    state_new: "WebState"

    def dict(self, simple=False):
        return {
            "action": self.action.dict(simple),
            "state_new": str(self.state_new.ws_id),
        }


@dataclasses.dataclass
class WebState:
    title: str
    title_embedding: list[float]
    urls: list[str]
    description: list[dict]
    actions: list[Action]
    transitions: list[StateTransition]
    ws_id: uuid.UUID = dataclasses.field(default_factory=uuid.uuid4)

    @property
    def random_action(self) -> None | Action:
        candidates_obvious = [
            action
            for action in self.actions
            if action.status == "none" and action.priority >= 11
        ]

        if candidates_obvious:
            candidates_obvious = sorted(
                candidates_obvious, key=lambda x: x.priority, reverse=True
            )
            return candidates_obvious[0]
        candidates = [action for action in self.actions if action.status == "none"]
        probabilities = np.array([action.priority for action in candidates])
        probabilities = probabilities / probabilities.sum()

        if candidates:
            return np.random.choice(np.array(candidates), p=probabilities)  # type: ignore

        return None

    def cosine_distance(self, embedding: list[float]):
        return np.dot(self.title_embedding, embedding) / (
            np.linalg.norm(self.title_embedding) * np.linalg.norm(embedding)
        )

    def dict(self, simple=False):
        d = {
            "ws_id": str(self.ws_id),
            "title": self.title,
            "urls": self.urls,
            "description": self.description,
            "actions": [a.dict(simple) for a in self.actions],
            "transitions": [t.dict(simple) for t in self.transitions],
        }

        if not simple:
            d["title_embedding"] = self.title_embedding

        return d


def load_states_from_file(file_path: str) -> list[WebState]:
    with open(file_path, "r") as f:
        json_string = f.read()
        print(json_string)
        webstates_raw = json.loads(json_string)

    webstates = []
    transitions_raw = []

    for ws_raw in webstates_raw:
        ws = WebState(
            title=ws_raw["title"],
            title_embedding=ws_raw["title_embedding"],
            urls=ws_raw["urls"],
            description=ws_raw["description"],
            actions=[Action.from_dict(a) for a in ws_raw["actions"]],
            transitions=[],
            ws_id=uuid.UUID(ws_raw["ws_id"]),
        )
        webstates.append(ws)
        transitions_raw.extend((uuid.UUID(ws_raw["ws_id"]), t) for t in ws_raw["transitions"] if t)

    for state_id, transition_raw in transitions_raw:
        state_current = next(ws for ws in webstates if ws.ws_id == state_id)
        action = Action.from_dict(transition_raw["action"])
        state_new = next(
            ws for ws in webstates if ws.ws_id == uuid.UUID(transition_raw["state_new"])
        )
        state_current.transitions.append(StateTransition(action, state_new))

    return webstates
