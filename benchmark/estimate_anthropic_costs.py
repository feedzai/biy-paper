import base64
import os
from mimetypes import guess_type
from typing import Literal, TypeGuard
from urllib.request import urlopen

import anthropic
import numpy as np
import pandas as pd
from anthropic.types import MessageParam
from dotenv import load_dotenv
from gaveta.files import ensure_dir
from loguru import logger

from constants import (
    ANTHROPIC_RESULTS,
    INPUT,
    OPEN_AI_RESULTS,
    PROMPTS,
    REASONING_EFFORT_TO_THINKING_BUDGET,
    SEED,
)
from data_models import Prompt
from utils import format_prompt_answer, split_dataset

ANTHROPIC_MODELS = ["claude-sonnet-4-20250514", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022"]

# Source:
# - https://www.anthropic.com/pricing#api
# - https://aws.amazon.com/bedrock/pricing/
# - https://docs.anthropic.com/en/docs/about-claude/models/overview#model-pricing
# - https://docs.anthropic.com/en/docs/build-with-claude/batch-processing#pricing
COSTS_MTOK: dict[str, dict[str, dict[str, float]]] = {
    "claude-sonnet-4-20250514": {
        "base": {
            "input": 3,
            "output": 15,
        },
        "batch": {
            "input": 1.5,
            "output": 7.5,
        },
    },
    "claude-3-5-sonnet-20241022": {
        "base": {
            "input": 3,
            "output": 15,
        },
        "batch": {
            "input": 1.5,
            "output": 7.5,
        },
    },
    "claude-3-5-haiku-20241022": {
        "base": {
            "input": 0.8,
            "output": 4,
        },
        "batch": {
            "input": 0.4,
            "output": 2,
        },
    },
}


def estimate_cost_for_tokens(
    input_tokens: int,
    output_tokens: int,
    model: str,
    pricing: Literal["base", "batch"] = "base",
) -> float:
    input_cost = (COSTS_MTOK[model][pricing]["input"] * input_tokens) / 1_000_000
    output_cost = (COSTS_MTOK[model][pricing]["output"] * output_tokens) / 1_000_000

    return input_cost + output_cost


def is_supported_mime_type(
    mime_type: str | None,
) -> TypeGuard[Literal["image/jpeg", "image/png", "image/gif", "image/webp"]]:
    return mime_type in ["image/jpeg", "image/png", "image/gif", "image/webp"]


def image_url_to_b64(image_url: str) -> tuple[str, Literal["image/jpeg", "image/png", "image/gif", "image/webp"]]:
    # Documentation:
    # - https://docs.anthropic.com/en/docs/build-with-claude/vision#base64-encoded-image-example
    # - https://docs.anthropic.com/en/docs/build-with-claude/token-counting#count-tokens-in-messages-with-images

    with urlopen(image_url) as response:
        image_b64: str = base64.standard_b64encode(response.read()).decode("utf-8")

    mime_type, _ = guess_type(image_url, strict=True)

    if is_supported_mime_type(mime_type):
        return image_b64, mime_type

    raise ValueError("Unsupported MIME type")


def generate_zero_shot_anthropic_messages(prompt: Prompt, image_url: str) -> list[MessageParam]:
    image_b64, mime_type = image_url_to_b64(image_url)

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt["prompt"]},
            ],
        }
    ]


def generate_n_shot_anthropic_messages(dataset: pd.DataFrame, prompt: Prompt, image_url: str) -> list[MessageParam]:
    # Documentation: https://docs.anthropic.com/en/api/messages-examples#multiple-conversational-turns

    messages: list[MessageParam] = []

    for example in dataset.itertuples(index=False):
        formatted_answer = format_prompt_answer(example, prompt)
        image_b64, mime_type = image_url_to_b64(str(example.image))

        message: list[MessageParam] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt["prompt"]},
                ],
            },
            {"role": "assistant", "content": formatted_answer},
        ]

        messages.extend(message)

    image_b64, mime_type = image_url_to_b64(image_url)

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": image_b64,
                    },
                },
                {"type": "text", "text": prompt["prompt"]},
            ],
        }
    )

    return messages


if __name__ == "__main__":
    # Documentation:
    # - https://docs.anthropic.com/en/docs/build-with-claude/token-counting
    # - https://docs.anthropic.com/en/docs/build-with-claude/vision#url-based-image-example

    ensure_dir(ANTHROPIC_RESULTS)
    load_dotenv()

    n_shot_dataset, one_shot_dataset, test_dataset = split_dataset(INPUT / "dataset.parquet")

    n_datasets = test_dataset["dataset_id"].nunique()
    chart_design_images = test_dataset.groupby("chart_design").sample(n=1, random_state=SEED)["image"].to_list()

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    model_token_counts: dict[str, dict[str, int]] = {}

    logger.info("Total requests: {n_requests}", n_requests=n_datasets * len(chart_design_images) * 5 * 3)

    # Input tokens:
    for model in ANTHROPIC_MODELS:
        model_token_counts[model] = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

        for prompt in PROMPTS:
            for chart_design_image in chart_design_images:
                zero_shot = generate_zero_shot_anthropic_messages(prompt, chart_design_image)
                one_shot = generate_n_shot_anthropic_messages(one_shot_dataset, prompt, chart_design_image)
                few_shot = generate_n_shot_anthropic_messages(n_shot_dataset, prompt, chart_design_image)

                zero_shot_prompt_tokens = client.messages.count_tokens(
                    model=model,
                    messages=zero_shot,
                )
                one_shot_prompt_tokens = client.messages.count_tokens(
                    model=model,
                    messages=one_shot,
                )
                few_shot_prompt_tokens = client.messages.count_tokens(
                    model=model,
                    messages=few_shot,
                )

                zero_shot_input_tokens = zero_shot_prompt_tokens.input_tokens
                one_shot_input_tokens = one_shot_prompt_tokens.input_tokens
                few_shot_input_tokens = few_shot_prompt_tokens.input_tokens

                logger.info("Number of zero-shot tokens: {n_tokens}", n_tokens=zero_shot_input_tokens)
                logger.info("Number of one-shot tokens: {n_tokens}", n_tokens=one_shot_input_tokens)
                logger.info("Number of few-shot tokens: {n_tokens}", n_tokens=few_shot_input_tokens)

                model_token_counts[model]["input_tokens"] += (
                    zero_shot_input_tokens * n_datasets
                    + one_shot_input_tokens * n_datasets
                    + few_shot_input_tokens * n_datasets
                )

    # Output tokens:
    open_ai_usage = pd.read_parquet(OPEN_AI_RESULTS / "usage.parquet", engine="pyarrow")

    # Alternative: "median"
    max_usage_per_prompt_id = open_ai_usage.groupby(["prompt_id"], as_index=False, dropna=False).agg(
        max_output_tokens=("output_tokens", "mean")
    )
    max_usage_per_prompt_id["max_output_tokens"] = np.ceil(max_usage_per_prompt_id["max_output_tokens"])

    ref_output_tokens = int(max_usage_per_prompt_id["max_output_tokens"].sum())

    for model in ANTHROPIC_MODELS:
        model_token_counts[model]["output_tokens"] = ref_output_tokens * n_datasets * len(chart_design_images) * 5 * 3

    # Reasoning (output) tokens:
    model_token_counts["claude-sonnet-4-20250514"]["output_tokens"] += (
        REASONING_EFFORT_TO_THINKING_BUDGET["medium"] * n_datasets * len(chart_design_images) * 5 * 3
    )

    # Output:
    cost_df = pd.DataFrame.from_dict(model_token_counts, orient="index").reset_index(names="model")

    cost_df["estimated_cost"] = cost_df.apply(
        lambda row: estimate_cost_for_tokens(
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            model=row["model"],
            pricing="base" if row["model"] == "claude-sonnet-4-20250514" else "batch",
        ),
        axis=1,
    )
    cost_df.insert(0, "provider", "Anthropic")

    cost_df.to_csv(ANTHROPIC_RESULTS / "estimated_costs.csv", index=False)
    logger.info("Estimated cost file generated")
