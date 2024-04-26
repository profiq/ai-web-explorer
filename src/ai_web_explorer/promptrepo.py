import dataclasses

import anthropic
import openai
from openai.types import chat, shared_params
import yaml
import copy

from . import config


@dataclasses.dataclass
class Prompt:
    prompt_text: str
    functions: list[shared_params.FunctionDefinition]
    temperature: float
    model: str

    @property
    def tools(self) -> list[chat.ChatCompletionToolParam]:
        return [{"type": "function", "function": f} for f in self.functions]

    def prompt_with_data(self, **data) -> str:
        return self.prompt_text.format(**data)

    def execute_prompt(
        self, client: openai.Client, **data
    ) -> chat.chat_completion.ChatCompletion:
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
            model=self.model,
            messages=[message],
            temperature=self.temperature,
            tools=tools,
            tool_choice=tool_choice,
        )
        return completion

    def execute_prompt_anthropic(self, model: str, **data):
        prompt_text = self.prompt_with_data(**data)
        
        tools = copy.deepcopy(self.functions)
        for tool in tools:
            tool["input_schema"] = tool["parameters"]
            del tool["parameters"]

        response = anthropic.Anthropic().beta.tools.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt_text}],
            temperature=self.temperature,
            max_tokens=2048,
            tools=tools,
        )

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
    )
    return prompt
