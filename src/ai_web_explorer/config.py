import os

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESOURCES_PATH = os.path.join(BASE_PATH, "resources")
DATA_PATH = os.path.join(BASE_PATH, "data")
PROMPTS_PATH = os.path.join(RESOURCES_PATH, "prompts.yaml")
TITLES_PATH = os.path.join(DATA_PATH, "titles.jsonl")

MODEL_DEFAULT = "gpt-4o"
TEMPERATURE_DEFAULT = 0.1

HTML_PART_LENGTH = 15000
PLAYWRIGHT_TIMEOUT = 5000

ENSURE_LOADED_MAX_TRIES = 3
ENSURE_LOADED_SLEEP_TIME = 2

TITLE_SIMILARITY_THRESHOLD = 0.92
ACTION_MAX_TRIES = 5
ACTION_SLEEP_TIME = 1
