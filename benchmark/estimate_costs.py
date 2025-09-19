from pathlib import Path

from openai.types.chat import ChatCompletionMessageParam
from praicing.openai import estimate_costs_for_messages

from prompts import Q1_COUNT_CLUSTERS_PROMPT, VC_TASKS_NAIVE_3_ALT_HBB_CLUSTERS_PROMPT
from utils import encode_image

if __name__ == "__main__":
    short_prompt: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image(Path("input") / "fuzzy_datasets_d1+default.png"),
                        "detail": "high",
                    },
                },
                {"type": "text", "text": Q1_COUNT_CLUSTERS_PROMPT},
            ],
        }
    ]

    long_prompt: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encode_image(Path("input") / "fuzzy_datasets_d1+default.png"),
                        "detail": "high",
                    },
                },
                {"type": "text", "text": VC_TASKS_NAIVE_3_ALT_HBB_CLUSTERS_PROMPT},
            ],
        }
    ]

    short_single_cost = estimate_costs_for_messages(short_prompt, "gpt-4o-2024-08-06", "batch")
    long_single_cost = estimate_costs_for_messages(long_prompt, "gpt-4o-2024-08-06", "batch")

    short_cheap_single_cost = estimate_costs_for_messages(short_prompt, "gpt-4.1-mini-2025-04-14", "batch")
    long_cheap_single_cost = estimate_costs_for_messages(long_prompt, "gpt-4.1-mini-2025-04-14", "batch")

    DATASETS = 115
    CHART_DESIGNS = 15
    IMAGE_SIZE = 1
    TASKS = 5
    PROMPTS = 3
    MODELS = 5

    short_total_cost = short_single_cost * DATASETS * CHART_DESIGNS * IMAGE_SIZE * TASKS * PROMPTS * MODELS
    long_total_cost = long_single_cost * DATASETS * CHART_DESIGNS * IMAGE_SIZE * TASKS * PROMPTS * MODELS

    print(f"W/ short prompt(s): {short_total_cost}")
    print(f"W/ long prompt(s): {long_total_cost}")

    short_cheap_total_cost = short_cheap_single_cost * DATASETS * CHART_DESIGNS * IMAGE_SIZE * TASKS * PROMPTS * MODELS
    long_cheap_total_cost = long_cheap_single_cost * DATASETS * CHART_DESIGNS * IMAGE_SIZE * TASKS * PROMPTS * MODELS

    print(f"W/ short prompt(s) and cheap model: {short_cheap_total_cost}")
    print(f"W/ long prompt(s) and cheap model: {long_cheap_total_cost}")
