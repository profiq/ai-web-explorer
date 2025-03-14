import base64
import dataclasses

import openai
from openai.types import chat, shared_params
import yaml
import datetime
import json

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
        self, client: openai.Client, image_bytes: bytes | list[bytes] | None, **data
    ) -> chat.chat_completion.Choice:
        prompt_text = self.prompt_with_data(**data)
        content: list[dict] = [{"type": "text", "text": prompt_text}]

        if image_bytes is not None:
            if not isinstance(image_bytes, list):
                image_bytes = [image_bytes]

            for image in image_bytes:
                image_encoded = base64.b64encode(image).decode("utf-8")
                image_encoded = f"data:image/png;base64,{image_encoded}"
                content.append(
                    {"type": "image_url", "image_url": {"url": image_encoded}}
                )

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

        self._last_completion = completion
        response = completion.choices[0]

        if config.PROMPT_LOGGING_ENABLED:
            self.log_prompt([message], response)

        return response

    def log_prompt(self, messages, response) -> None:
        for i, message in enumerate(messages):
            if not isinstance(message, dict):
                message = message.model_dump()
            if isinstance(message["content"], list):
                message["content"] = message["content"][0]
            messages[i] = message

        with open(config.PROMPT_LOGS_PATH, "a") as f:
            log_record = {
                "timestamp": datetime.datetime.now().isoformat(),
                "messages": messages,
                "response": response.model_dump(),
                "functions": self.functions,
                "temperature": self.temperature,
                "model": self.model,
            }

            f.write(json.dumps(log_record) + "\n")

    def get_last_price(self) -> float:
        if self.model == "gpt-4o":
            unit_price_input = 5.0
            unit_price_output = 15.0
        else:
            unit_price_input = 0.15
            unit_price_output = 0.60


        if not hasattr(self, "_last_completion"):
            raise ValueError("No last completion to get price from")

        if not self._last_completion.usage:
            return 0.0

        input_tokens = self._last_completion.usage.prompt_tokens
        output_tokens = self._last_completion.usage.completion_tokens

        input_price = input_tokens * unit_price_input / 1_000_000
        output_price = output_tokens * unit_price_output / 1_000_000

        return input_price + output_price


def get_prompt(name: str, prompts_path: str = config.PROMPTS_PATH) -> Prompt:
    with open(prompts_path) as f:
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
