import dataclasses
import typing
import uuid

import numpy as np

ActionStatus = typing.Literal["none", "success", "failure"]


@dataclasses.dataclass
class Action:
    description: str
    part: int
    priority: int
    status: ActionStatus = dataclasses.field(default="none")
    function_calls: list[dict] = dataclasses.field(default_factory=list)

    def dict(self):
        return {
            "description": self.description,
            "part": self.part,
            "priority": self.priority,
            "status": self.status,
            "function_calls": [f.model_dump() for f in self.function_calls],
        }


@dataclasses.dataclass
class StateTransition:
    action: Action
    state_new: "WebState"

    def dict(self):
        return {
            "action": self.action.dict(),
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

    def dict(self, embedding=True):
        d =  {
            "ws_id": str(self.ws_id),
            "title": self.title,
            "urls": self.urls,
            "description": self.description,
            "actions": [a.dict() for a in self.actions],
            "transitions": [t.dict() for t in self.transitions],
        }

        if embedding:
            d["title_embedding"] = self.title_embedding

        return d
