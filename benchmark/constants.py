from pathlib import Path

from data_models import Prompt

INPUT = Path("input")
RESULTS = Path("results")

ALL_RESULTS = RESULTS / "all"
ALL_VAL = RESULTS / "validation"

OPEN_AI_BATCH_INPUT_FILES = INPUT / "batch_input_files" / "open_ai"

OPEN_AI_RESULTS = RESULTS / "open_ai"
OPEN_AI_RAW_RESULTS = OPEN_AI_RESULTS / "raw"
OPEN_AI_ERRORS = OPEN_AI_RESULTS / "errors"
OPEN_AI_MISSING_CUSTOM_IDS = OPEN_AI_RESULTS / "missing_custom_ids.json"
OPEN_AI_VAL = OPEN_AI_RESULTS / "validation"

ANTHROPIC_RESULTS = RESULTS / "anthropic"

GOOGLE_BATCH_INPUT_FILES = Path("input") / "batch_input_files" / "google"

GOOGLE_RESULTS = RESULTS / "google"
GOOGLE_RAW_RESULTS = GOOGLE_RESULTS / "raw"

VAL = RESULTS / "validation"

TEMPERATURE = 0.0
O_MODELS_TEMPERATURE = 1.0

SEED = 42

OPEN_AI_MODELS = [
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4o-2024-08-06",
    "gpt-4o-mini-2024-07-18",
    "o3-2025-04-16",
    "o4-mini-2025-04-16",
]

GOOGLE_MODEL_CONFIGS: list[dict[str, str | int]] = [
    {
        "model": "gemini-2.5-flash",
        "temperature": 0,
        "thinking_budget": 0,
        "max_tokens": 2048,
        "media_resolution": "MEDIA_RESOLUTION_MEDIUM",
    },
    {
        "model": "gemini-2.5-flash-lite",
        "temperature": 0,
        "thinking_budget": 0,
        "max_tokens": 2048,
        "media_resolution": "MEDIA_RESOLUTION_MEDIUM",
    },
    {
        "model": "gemini-2.5-flash-lite",
        "temperature": 0,
        "thinking_budget": 8192,
        "max_tokens": 10240,
        "media_resolution": "MEDIA_RESOLUTION_MEDIUM",
    },
]

CLUSTER_COUNT_PROMPT_TEXT = (
    "How many clusters are there in the scatterplot? Answer with a number in curly brackets, e.g., {4}."
)
CLUSTER_BBOXES_PROMPT_TEXT = 'Detect all clusters in the scatterplot. For each detected cluster, provide its bounding box in normalized coordinates (x1, y1, x2, y2), where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the image. Answer with a JSON object, e.g., {"clusters": [[200, 30, 300, 50]]}.'
CLUSTER_POINTS_PROMPT_TEXT = 'Identify each cluster in the scatterplot. For each, provide its center point in normalized coordinates (x, y), where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the image. Answer with a JSON object, e.g., {"cluster_centers": [[250, 40], [700, 500]]}.'

OUTLIER_COUNT_PROMPT_TEXT = (
    "How many outliers are there in the scatterplot? Answer with a number in curly brackets, e.g., {3}."
)
OUTLIER_POINTS_PROMPT_TEXT = 'Identify each outlier in the scatterplot. For each, provide its location in normalized coordinates (x, y), where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the image. Answer with a JSON object, e.g., {"outliers": [[150, 900], [820, 150]]}.'

PROMPTS: list[Prompt] = [
    {
        "prompt_id": "cluster_count",
        "prompt": CLUSTER_COUNT_PROMPT_TEXT,
    },
    {
        "prompt_id": "cluster_bboxes",
        "prompt": CLUSTER_BBOXES_PROMPT_TEXT,
    },
    {
        "prompt_id": "cluster_points",
        "prompt": CLUSTER_POINTS_PROMPT_TEXT,
    },
    {
        "prompt_id": "outlier_count",
        "prompt": OUTLIER_COUNT_PROMPT_TEXT,
    },
    {
        "prompt_id": "outlier_points",
        "prompt": OUTLIER_POINTS_PROMPT_TEXT,
    },
]

# Source:
# - https://ai.google.dev/gemini-api/docs/openai#thinking
# - https://ai.google.dev/gemini-api/docs/thinking#set-budget
# - https://openrouter.ai/docs/use-cases/reasoning-tokens#reasoning-max-tokens-for-anthropic-models
NO_THINKING_BUDGET = 0
REASONING_EFFORT_TO_THINKING_BUDGET = {
    "low": 1_024,
    "medium": 8_192,
    "high": 24_576,
}

# Source:
# - https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/get-token-count#pricing_and_quota
# - https://firebase.google.com/docs/ai-logic/count-tokens#pricing
GOOGLE_MAX_RPM = 3_000

OPEN_AI_COLOR = "#000000"

# Source: https://gemini.google.com/app
GOOGLE_COLOR = "#0b57d0"

ICONS = Path("input/icons")

PROMPT_ID_TO_ICON: dict[str, Path] = {
    "cluster_count": ICONS / "cluster_counting.svg",
    "outlier_count": ICONS / "outlier_counting.svg",
    "cluster_bboxes": ICONS / "cluster_detection.svg",
    "cluster_points": ICONS / "cluster_identification.svg",
    "outlier_points": ICONS / "outlier_identification.svg",
}

OPEN_AI_MODEL_TO_LABEL: dict[str, str] = {
    "gpt-4.1-2025-04-14": "GPT-4.1",
    "gpt-4.1-mini-2025-04-14": "GPT-4.1 mini",
    "gpt-4.1-nano-2025-04-14": "GPT-4.1 nano",
    "gpt-4o-2024-08-06": "GPT-4o",
    "gpt-4o-mini-2024-07-18": "GPT-4o mini",
    "o3-2025-04-16": "o3",
    "o4-mini-2025-04-16": "o4-mini",
}

GOOGLE_MODEL_TO_LABEL: dict[str, str] = {
    "gemini-2.5-flash": "Flash",
    "gemini-2.5-flash-lite": "Flash-Lite",
    "gemini-2.5-flash-lite-reasoning": "Flash-Lite (Thinking)",
}

MODEL_TO_LABEL = {**OPEN_AI_MODEL_TO_LABEL, **GOOGLE_MODEL_TO_LABEL}

PROMPT_STRATEGY_TO_LABEL: dict[str, str] = {
    "zero_shot": "zero-shot prompt",
    "one_shot": "one-shot",
    "few_shot": "few-shot",
}

METRIC_ID_TO_LABEL: dict[str, str] = {
    "accuracy": "Accuracy",
    "precision_0_75": "Precision @ IoU75",
    "recall_0_75": "Recall @ IoU75",
    "precision_10": "Precision @ 10px",
    "recall_10": "Recall @ 10px",
}

PROMPT_ID_TO_PROMPT_STRATEGY_COLORS: dict[str, list[str]] = {
    "cluster_count": ["#fb923c", "#ea580c", "#9a3412"],
    "outlier_count": ["#fb923c", "#ea580c", "#9a3412"],
    "cluster_bboxes": ["#38bdf8", "#0284c7", "#075985"],
    "cluster_points": ["#e879f9", "#c026d3", "#86198f"],
    "outlier_points": ["#e879f9", "#c026d3", "#86198f"],
}

PROMPT_ID_TO_LABEL: dict[str, str] = {
    "cluster_count": "cluster counting",
    "outlier_count": "outlier counting",
    "cluster_bboxes": "cluster detection",
    "cluster_points": "cluster identification",
    "outlier_points": "outlier identification",
}
