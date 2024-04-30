import copy
import dataclasses

import anthropic
import openai
from openai.types import chat, shared_params
import vertexai
import vertexai.generative_models
from vertexai.preview import generative_models
import yaml

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

    def execute_prompt_gemini(self, model: str, **data):
        prompt_text = self.prompt_with_data(**data)
        prompt_text = prompt_text + "\n\n" + "Always call a function!"
        functions = []

        for fn in self.functions:
            t = vertexai.generative_models.FunctionDeclaration(
                name=fn["name"],
                description=fn.get("description", ""),
                parameters=fn.get("parameters", {}),
            )
            functions.append(t)

        tool = vertexai.generative_models.Tool(function_declarations=functions)

        vertexai.init(project="vebiss-1129")
        m = vertexai.generative_models.GenerativeModel(model)
        response = m.generate_content(
            [prompt_text],
            generation_config={"temperature": 0},
            tools=[tool],
            safety_settings={
                generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            },
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
