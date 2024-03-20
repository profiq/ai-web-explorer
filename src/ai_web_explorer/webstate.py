import dataclasses
import typing
import random


@dataclasses.dataclass
class Action:
    description: str
    status: typing.Literal["none", "success", "failure"]


@dataclasses.dataclass
class StateTransition:
    action: Action
    state_new: "WebState"


@dataclasses.dataclass
class WebState:
    title: str
    urls: list[str]
    description: str
    actions: list[Action]
    transitions: list[StateTransition]

    @property
    def random_action(self) -> None | Action:
        candidates = [action for action in self.actions if action.status == "none"]
        if candidates:
            return random.choice(candidates)
        return None
