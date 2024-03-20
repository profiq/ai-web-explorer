import dataclasses

@dataclasses.dataclass
class Prompt:
    prompt_text: str
    functions: list[dict]


def get_prompt(name: str) -> Prompt | None:
    pass
