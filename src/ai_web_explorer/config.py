import os

# Basic path definitions
BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
RESOURCES_PATH = os.path.join(BASE_PATH, "resources")
DATA_PATH = os.path.join(BASE_PATH, "data")

# Title settings
TITLES_PATH = os.path.join(DATA_PATH, "titles.jsonl")
HTMLS_PATH = os.path.join(DATA_PATH, "htmls")
SCREENSHOTS_PATH = os.path.join(DATA_PATH, "screenshots")

# GPT settings
MODEL_DEFAULT = "gpt-4o"
MAX_TOKENS_DEFAULT = 512
TEMPERATURE_DEFAULT = 0.1
PROMPTS_PATH = os.path.join(RESOURCES_PATH, "prompts.yaml")
PROMPT_LOGGING_ENABLED = True
PROMPT_LOGS_PATH = os.path.join(DATA_PATH, "prompt_logs.jsonl")

# Browser settings
BROWSER_SIZE = (1024, 768)
HTML_PART_LENGTH = 40000
PLAYWRIGHT_TIMEOUT = 5000

# Time to wait to check if the page is loaded
ENSURE_LOADED_SLEEP_TIME = 2

# Number of times to check if the page is loaded if the previous check failed
ENSURE_LOADED_MAX_TRIES = 3

# Minimum cosine similarity between two titles to consider them to be the same
TITLE_SIMILARITY_THRESHOLD = 0.92

# If performing a browser action fails, how many times to retry
ACTION_MAX_TRIES = 5

# Time to wait between browser actions
ACTION_SLEEP_TIME = 1
