import base64
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
    model: str
    max_tokens: int

    @property
    def tools(self) -> list[chat.ChatCompletionToolParam]:
        return [{"type": "function", "function": f} for f in self.functions]

    def prompt_with_data(self, **data) -> str:
        return self.prompt_text.format(**data)

    def execute_prompt(
        self, client: openai.Client, image_bytes: bytes | None, **data
    ) -> chat.chat_completion.Choice:
        prompt_text = self.prompt_with_data(**data)
        content: list[dict] = [{"type": "text", "text": prompt_text}]

        if image_bytes is not None:
            image_encoded = base64.b64encode(image_bytes).decode("utf-8")
            image_encoded = f"data:image/png;base64,{image_encoded}"
            content.append({"type": "image_url", "image_url": {"url": image_encoded}})

        message: chat.ChatCompletionMessageParam = {
            "role": "user",
            "content": content,  # type: ignore
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
            model=self.model,
            messages=[message],
            temperature=self.temperature,
            tools=tools,
            tool_choice=tool_choice,
            max_tokens=self.max_tokens,
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
        prompt_raw.get("temperature", config.TEMPERATURE_DEFAULT),
        prompt_raw.get("model", config.MODEL_DEFAULT),
        prompt_raw.get("max_tokens", config.MAX_TOKENS_DEFAULT),
    )
    return prompt
