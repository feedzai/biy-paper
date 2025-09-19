from mimetypes import guess_type
from typing import Literal
from urllib.request import urlopen

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from gaveta.files import ensure_dir
from google import genai
from google.genai.types import (
    Content,
    ContentOrDict,
    CountTokensConfig,
    GenerationConfig,
    HttpOptions,
    MediaResolution,
    Part,
)
from loguru import logger

from constants import GOOGLE_RESULTS, INPUT, OPEN_AI_RESULTS, PROMPTS, SEED
from data_models import Prompt
from utils import format_prompt_answer, split_dataset

# Source: https://cloud.google.com/vertex-ai/generative-ai/pricing
COSTS_MTOK: dict[str, dict[str, dict[str, float]]] = {
    "gemini-2.5-pro": {
        "base_200K": {
            "input": 1.25,
            "output": 10,
        },
        "base_plus200K": {
            "input": 2.5,
            "output": 15,
        },
        "batch_200K": {
            "input": 0.625,
            "output": 5,
        },
        "batch_plus200K": {
            "input": 1.25,
            "output": 7.5,
        },
    },
    "gemini-2.5-flash": {
        "base_200K": {
            "input": 0.3,
            "output": 2.5,
        },
        "base_plus200K": {
            "input": 0.3,
            "output": 2.5,
        },
        "batch_200K": {
            "input": 0.15,
            "output": 1.25,
        },
        "batch_plus200K": {
            "input": 0.15,
            "output": 1.25,
        },
    },
    "gemini-2.5-flash-lite": {
        "base_200K": {
            "input": 0.1,
            "output": 0.4,
        },
        "base_plus200K": {
            "input": 0.1,
            "output": 0.4,
        },
        "batch_200K": {
            "input": 0.05,
            "output": 0.2,
        },
        "batch_plus200K": {
            "input": 0.05,
            "output": 0.2,
        },
    },
}

GOOGLE_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]


def estimate_cost_for_tokens(
    input_tokens: int,
    output_tokens: int,
    model: str,
    pricing: Literal["base", "batch"] = "base",
) -> float:
    pricing_suffix = "200K" if input_tokens <= 200_000 else "plus200K"
    pricing_id = f"{pricing}_{pricing_suffix}"

    input_cost = (COSTS_MTOK[model][pricing_id]["input"] * input_tokens) / 1_000_000
    output_cost = (COSTS_MTOK[model][pricing_id]["output"] * output_tokens) / 1_000_000

    return input_cost + output_cost


def simple_estimate_cost_for_tokens(
    input_tokens: int,
    output_tokens: int,
    model: str,
    pricing: Literal["base", "batch"] = "base",
) -> float:
    pricing_suffix = "200K"
    pricing_id = f"{pricing}_{pricing_suffix}"

    actual_model = model.removesuffix("+reasoning")

    input_cost = (COSTS_MTOK[actual_model][pricing_id]["input"] * input_tokens) / 1_000_000
    output_cost = (COSTS_MTOK[actual_model][pricing_id]["output"] * output_tokens) / 1_000_000

    return input_cost + output_cost


def image_url_to_bytes(image_url: str) -> tuple[bytes, str]:
    # Documentation:
    # - https://ai.google.dev/gemini-api/docs/image-understanding#inline-image
    # - https://github.com/python/typeshed/blob/3f08a4ed10b321c378107c236a06a33584869a9b/stdlib/urllib/request.pyi#L52

    with urlopen(image_url) as response:
        image_bytes: bytes = response.read()

    mime_type, _ = guess_type(image_url, strict=True)

    if mime_type is None:
        raise ValueError("Unknown MIME type")

    return image_bytes, mime_type


def generate_zero_shot_google_messages(
    client: genai.Client, model: str, prompt: Prompt, image_url: str
) -> list[Content]:
    image_bytes, mime_type = image_url_to_bytes(image_url)

    chat = client.chats.create(
        model=model,
        history=[
            Content(
                role="user",
                parts=[
                    Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    ),
                    Part.from_text(text=prompt["prompt"]),
                ],
            ),
        ],
    )

    return chat.get_history()


def generate_n_shot_google_messages(
    dataset: pd.DataFrame, client: genai.Client, model: str, prompt: Prompt, image_url: str
) -> list[Content]:
    messages: list[ContentOrDict] = []

    for example in dataset.itertuples(index=False):
        formatted_answer = format_prompt_answer(example, prompt)
        image_bytes, mime_type = image_url_to_bytes(str(example.image))

        message: list[Content] = [
            Content(
                role="user",
                parts=[
                    Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    ),
                    Part.from_text(text=prompt["prompt"]),
                ],
            ),
            Content(role="model", parts=[Part.from_text(text=formatted_answer)]),
        ]

        messages.extend(message)

    image_bytes, mime_type = image_url_to_bytes(image_url)

    messages.append(
        Content(
            role="user",
            parts=[
                Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type,
                ),
                Part.from_text(text=prompt["prompt"]),
            ],
        )
    )

    chat = client.chats.create(
        model=model,
        history=messages,
    )

    return chat.get_history()


if __name__ == "__main__":
    # Documentation:
    # - https://developers.googleblog.com/en/gemini-2-5-models-now-support-implicit-caching/
    # - https://firebase.google.com/docs/ai-logic/count-tokens#sample-count-multimodal-tokens-images
    # - https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/get-token-count
    # - https://ai.google.dev/gemini-api/docs/tokens?lang=python#multi-turn-tokens
    # - https://ai.google.dev/gemini-api/docs/image-understanding#inline-image
    # - https://ai.google.dev/api/generate-content#chat

    ensure_dir(GOOGLE_RESULTS)
    load_dotenv()

    n_shot_dataset, one_shot_dataset, test_dataset = split_dataset(INPUT / "dataset.parquet")

    n_datasets = test_dataset["dataset_id"].nunique()
    chart_design_images = test_dataset.groupby("chart_design").sample(n=1, random_state=SEED)["image"].to_list()

    client = genai.Client(http_options=HttpOptions(api_version="v1"))

    model_token_counts: dict[str, dict[str, int]] = {}

    logger.info("Total requests: {n_requests}", n_requests=n_datasets * len(chart_design_images) * 5 * 3)

    # Input tokens:
    for model in GOOGLE_MODELS:
        model_token_counts[model] = {
            "input_tokens": 0,
            "output_tokens": 0,
        }

        for prompt in PROMPTS:
            for chart_design_image in chart_design_images:
                zero_shot = generate_zero_shot_google_messages(client, model, prompt, chart_design_image)
                one_shot = generate_n_shot_google_messages(one_shot_dataset, client, model, prompt, chart_design_image)
                few_shot = generate_n_shot_google_messages(n_shot_dataset, client, model, prompt, chart_design_image)

                # Note: In multi-turn conversations with multiple images, the MEDIA_RESOLUTION_HIGH setting doesn't seem to work, defaulting to MEDIA_RESOLUTION_MEDIUM instead.
                # To standardize all prompting strategies, MEDIA_RESOLUTION_MEDIUM has been explicitly adopted.
                # It might just be an issue with the Count Tokens API and not affect the inference API.
                # Issue: https://github.com/googleapis/python-genai/issues/1198
                zero_shot_prompt_tokens = client.models.count_tokens(
                    model=model,
                    contents=zero_shot,
                    config=CountTokensConfig(
                        generation_config=GenerationConfig(media_resolution=MediaResolution.MEDIA_RESOLUTION_MEDIUM)
                    ),
                )
                one_shot_prompt_tokens = client.models.count_tokens(
                    model=model,
                    contents=one_shot,
                    config=CountTokensConfig(
                        generation_config=GenerationConfig(media_resolution=MediaResolution.MEDIA_RESOLUTION_MEDIUM)
                    ),
                )
                few_shot_prompt_tokens = client.models.count_tokens(
                    model=model,
                    contents=few_shot,
                    config=CountTokensConfig(
                        generation_config=GenerationConfig(media_resolution=MediaResolution.MEDIA_RESOLUTION_MEDIUM)
                    ),
                )

                zero_shot_input_tokens = (
                    zero_shot_prompt_tokens.total_tokens if zero_shot_prompt_tokens.total_tokens else 0
                )
                one_shot_input_tokens = (
                    one_shot_prompt_tokens.total_tokens if one_shot_prompt_tokens.total_tokens else 0
                )
                few_shot_input_tokens = (
                    few_shot_prompt_tokens.total_tokens if few_shot_prompt_tokens.total_tokens else 0
                )

                logger.info("Number of zero-shot tokens: {n_tokens}", n_tokens=zero_shot_input_tokens)
                logger.info("Number of one-shot tokens: {n_tokens}", n_tokens=one_shot_input_tokens)
                logger.info(
                    "Number of one-shot cached tokens: {n_tokens}",
                    n_tokens=one_shot_prompt_tokens.cached_content_token_count
                    if one_shot_prompt_tokens.cached_content_token_count
                    else 0,
                )
                logger.info("Number of few-shot tokens: {n_tokens}", n_tokens=few_shot_input_tokens)
                logger.info(
                    "Number of few-shot cached tokens: {n_tokens}",
                    n_tokens=few_shot_prompt_tokens.cached_content_token_count
                    if few_shot_prompt_tokens.cached_content_token_count
                    else 0,
                )

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

    for model in GOOGLE_MODELS:
        model_token_counts[model]["output_tokens"] = ref_output_tokens * n_datasets * len(chart_design_images) * 5 * 3

    # Reasoning (output) tokens:
    # ref_reasoning_tokens = REASONING_EFFORT_TO_THINKING_BUDGET["medium"]
    ref_reasoning_tokens = 5735

    total_reasoning_tokens = ref_reasoning_tokens * n_datasets * len(chart_design_images) * 5 * 3

    model_token_counts["gemini-2.5-pro"]["output_tokens"] += total_reasoning_tokens

    model_token_counts["gemini-2.5-flash+reasoning"] = {
        "input_tokens": model_token_counts["gemini-2.5-flash"]["input_tokens"],
        "output_tokens": model_token_counts["gemini-2.5-flash"]["output_tokens"] + total_reasoning_tokens,
    }

    model_token_counts["gemini-2.5-flash-lite+reasoning"] = {
        "input_tokens": model_token_counts["gemini-2.5-flash-lite"]["input_tokens"],
        "output_tokens": model_token_counts["gemini-2.5-flash-lite"]["output_tokens"] + total_reasoning_tokens,
    }

    # Output:
    cost_df = pd.DataFrame.from_dict(model_token_counts, orient="index").reset_index(names="model")

    cost_df["estimated_cost"] = cost_df.apply(
        lambda row: simple_estimate_cost_for_tokens(
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            model=row["model"],
            pricing="batch",
        ),
        axis=1,
    )
    cost_df.insert(0, "provider", "Google")

    cost_df.to_csv(GOOGLE_RESULTS / "estimated_costs.csv", index=False)
    logger.info("Estimated cost file generated")
