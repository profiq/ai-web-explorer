import os

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESOURCES_PATH = os.path.join(BASE_PATH, "resources")
PROMPTS_PATH = os.path.join(RESOURCES_PATH, "prompts.yaml")

MODEL_DEFAULT = "gpt-3.5-turbo"
TEMPERATURE_DEFAULT = 0.1

HTML_PART_LENGTH = 10000
