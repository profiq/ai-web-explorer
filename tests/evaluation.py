import os

import mlflow
import openai
import pandas as pd
from mlflow.models import signature
import mlflow.types
from ai_web_explorer import promptrepo
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "hackernews")


mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("Description")


class TitleGpt(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.client = openai.OpenAI()
        self.prompt = promptrepo.get_prompt("page_title")

    def predict(self, context, model_input):
        html = model_input["html"]
        response = self.prompt.execute_prompt(self.client, html=html)
        function_call = response.choices[0].message.tool_calls[0].function
        args = function_call.arguments
        print(args)
        title = json.loads(args)["title"]
        return {"title": title}


with mlflow.start_run():
    prompt = promptrepo.get_prompt("page_title")

    model_signature = signature.ModelSignature(
        inputs=mlflow.types.Schema(
            [mlflow.types.ColSpec(mlflow.types.DataType.string, "html")],
        ),
        outputs=mlflow.types.Schema(
            [mlflow.types.ColSpec(mlflow.types.DataType.string, "title")],
        ),
    )

    logged_model_info = mlflow.pyfunc.log_model(
        "title-gpt", python_model=TitleGpt(), signature=model_signature
    )

    """
    logged_model_info = mlflow.openai.log_model(
        model="gpt-4-turbo",
        task=openai.chat.completions,
        artifact_path="model",
        messages=[{"role": "user", "content": prompt.prompt_text}],
    )
    """

    with open(os.path.join(DATA_DIR, "1.html")) as f:
        html = f.read()

    eval_data = pd.DataFrame({"html": [html]})

    results = mlflow.evaluate(
        logged_model_info.model_uri,
        eval_data,
        extra_metrics=[mlflow.metrics.token_count()],
        model_type=None,
    )

    mlflow.log_param("prompt", prompt.prompt_text)
