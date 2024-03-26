import dataclasses

import openai
from openai.types import chat, shared_params
import yaml

from . import config


@dataclasses.dataclass
class Prompt:
    prompt_text: str
    functions: list[shared_params.FunctionDefinition]
    temperature: float

    def prompt_with_data(self, **data) -> str:
        return self.prompt_text.format(**data)

    def execute_prompt(
        self, client: openai.Client, **data
    ) -> chat.chat_completion.Choice:
        prompt_text = self.prompt_with_data(**data)
        message: chat.ChatCompletionMessageParam = {
            "role": "user",
            "content": prompt_text,
        }
        tools: list[chat.ChatCompletionToolParam] | openai.NotGiven = (
            [{"type": "function", "function": f} for f in self.functions]
            if len(self.functions) > 0
            else openai.NOT_GIVEN
        )

        tool_choice: chat.ChatCompletionToolChoiceOptionParam | openai.NotGiven = (
            {"type": "function", "function": {"name": self.functions[0]["name"]}}
            if len(self.functions) == 1
            else openai.NOT_GIVEN
        )
        completion = client.chat.completions.create(
            model=config.MODEL_DEFAULT,
            messages=[message],
            temperature=self.temperature,
            tools=tools,
            tool_choice=tool_choice,
        )
        response = completion.choices[0]
        return response


def get_prompt(name: str) -> Prompt:
    with open(config.PROMPTS_PATH) as f:
        prompts = yaml.safe_load(f)
    if name not in prompts:
        raise ValueError(f"Prompt {name} not found")
    prompt_raw = prompts[name]
    if "prompt" not in prompt_raw:
        raise ValueError(f"Prompt {name} is missing a prompt")
    prompt = Prompt(
        prompt_raw["prompt"],
        prompt_raw.get("functions", []),
        prompt_raw.get("temperature", 0.1),
    )
    return prompt
