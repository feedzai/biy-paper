"""
Script to prepare batch input files for Google Gemini API batch processing.

This script:
1. Splits a dataset into training (n-shot, one-shot) and test sets
2. Generates different types of prompts (zero-shot, one-shot, few-shot) for each test example
3. Creates batch request files for each model configuration that can be submitted to Google's Gemini API batch mode
4. Handles file size limits by splitting large batch files into parts
5. Formats requests according to Google's Gemini API batch mode format with key-based identification

"""

import math
from typing import TypedDict

import humanize
from gaveta.files import ensure_dir
from loguru import logger
from more_itertools import divide

from constants import GOOGLE_BATCH_INPUT_FILES, GOOGLE_MODEL_CONFIGS, INPUT, PROMPTS
from utils import (
    generate_n_shot_google_messages,
    generate_zero_shot_google_messages,
    read_jsonl,
    split_dataset,
    write_jsonl,
)

# Constants for file size management
BASE = 1000  # or 1024
# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini#quotas_and_limits
MAX_FILE_SIZE = 1 * BASE * BASE * BASE  # 1GB - Google's batch file size limit


# Data model for batch input files
# Based on Google Vertex AI batch prediction format and batch mode format:
# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-from-cloud-storage
# https://ai.google.dev/gemini-api/docs/batch-mode#input-file


class GoogleBatchInputFile(TypedDict):
    key: str  # User-defined key to identify each request
    request: dict[str, object]  # The actual GenerateContentRequest object with contents


if __name__ == "__main__":
    # Ensure the output directory exists
    ensure_dir(GOOGLE_BATCH_INPUT_FILES)

    # Split the dataset into different subsets for different prompting strategies
    # - n_shot_dataset: Multiple examples for few-shot learning
    # - one_shot_dataset: Single example for one-shot prompting
    # - test_dataset: Examples to be evaluated
    n_shot_dataset, one_shot_dataset, test_dataset = split_dataset(INPUT / "dataset.parquet")

    # Documentation references:
    # - https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-from-cloud-storage
    # - https://ai.google.dev/gemini-api/docs/batch-mode#input-file
    # - https://googleapis.github.io/python-genai/genai.html#genai.batches.Batches.create

    # Process each model configuration separately to create individual batch files
    for model_config in GOOGLE_MODEL_CONFIGS:
        logger.info(
            "Processing model configuration: {model} with temperature {temperature} and thinking_budget {thinking_budget}",
            model=model_config["model"],
            temperature=model_config["temperature"],
            thinking_budget=model_config["thinking_budget"],
        )

        batch_requests: list[GoogleBatchInputFile] = []
        # Create filename based on model and thinking budget configuration
        config_suffix = "with_thinking" if model_config["thinking_budget"] > 0 else "without_thinking"
        batch_file = GOOGLE_BATCH_INPUT_FILES / f"{model_config['model']}+{config_suffix}.jsonl"

        # For each prompt template and test example, generate three types of requests:
        # 1. Zero-shot: No examples provided
        # 2. One-shot: Single example provided
        # 3. Few-shot: Multiple examples provided
        for prompt in PROMPTS:
            for test_example in test_dataset.itertuples(index=False):
                # Generate messages for each prompting strategy
                zero_shot = generate_zero_shot_google_messages(prompt, str(test_example.image))
                one_shot = generate_n_shot_google_messages(one_shot_dataset, prompt, str(test_example.image))
                few_shot = generate_n_shot_google_messages(n_shot_dataset, prompt, str(test_example.image))

                # Create batch request entries for each prompting strategy
                # Google's format: {"key": "user-defined-key", "request": {"contents": [...]}}
                # https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1/GenerationConfig#Modality
                batch_requests.extend(
                    [
                        {
                            "key": "+".join(
                                [
                                    model_config["model"],
                                    config_suffix,
                                    prompt["prompt_id"],
                                    "zero_shot",
                                    str(test_example.id),
                                ]
                            ),
                            "request": {
                                "contents": zero_shot,
                                "generationConfig": {
                                    "thinkingConfig": {
                                        "thinkingBudget": model_config["thinking_budget"],
                                    },
                                    "temperature": model_config["temperature"],
                                    "maxOutputTokens": model_config["max_tokens"],
                                    "mediaResolution": model_config["media_resolution"],
                                },
                            },
                        },
                        {
                            "key": "+".join(
                                [
                                    model_config["model"],
                                    config_suffix,
                                    prompt["prompt_id"],
                                    "one_shot",
                                    str(test_example.id),
                                ]
                            ),
                            "request": {
                                "contents": one_shot,
                                "generationConfig": {
                                    "thinkingConfig": {
                                        "thinkingBudget": model_config["thinking_budget"],
                                    },
                                    "temperature": model_config["temperature"],
                                    "maxOutputTokens": model_config["max_tokens"],
                                    "mediaResolution": model_config["media_resolution"],
                                },
                            },
                        },
                        {
                            "key": "+".join(
                                [
                                    model_config["model"],
                                    config_suffix,
                                    prompt["prompt_id"],
                                    "few_shot",
                                    str(test_example.id),
                                ]
                            ),
                            "request": {
                                "contents": few_shot,
                                "generationConfig": {
                                    "thinkingConfig": {
                                        "thinkingBudget": model_config["thinking_budget"],
                                    },
                                    "temperature": model_config["temperature"],
                                    "maxOutputTokens": model_config["max_tokens"],
                                    "mediaResolution": model_config["media_resolution"],
                                },
                            },
                        },
                    ]
                )

        # Write all batch requests to a JSONL file
        write_jsonl(batch_requests, batch_file)
        logger.info("Wrote batch file: {batch_file}", batch_file=batch_file.name)

        # Check if the batch file exceeds Google's size limit (1GB)
        # https://cloud.google.com/document-ai/limits
        batch_file_size = batch_file.stat().st_size
        logger.info("Number of requests: {n_requests}", n_requests=len(batch_requests))
        logger.info("Batch file size: {file_size}", file_size=humanize.naturalsize(batch_file_size))

        # If file is too large, split it into multiple parts
        if batch_file_size >= MAX_FILE_SIZE:
            # Calculate how many parts we need
            n_parts = math.ceil(batch_file_size / MAX_FILE_SIZE)

            # Split the batch requests into equal parts
            for index, part in enumerate(divide(n_parts, batch_requests), start=1):
                batch_file_part = GOOGLE_BATCH_INPUT_FILES / f"{batch_file.stem}+part{index}.jsonl"

                # Write each part to a separate file
                write_jsonl(part, batch_file_part)

                batch_file_part_size = batch_file_part.stat().st_size
                logger.info("Batch file part size: {file_size}", file_size=humanize.naturalsize(batch_file_part_size))

            # Verify the total number of requests across all parts
            n_requests = sum(
                [
                    len(read_jsonl(batch_file_part))
                    for batch_file_part in GOOGLE_BATCH_INPUT_FILES.glob(f"{batch_file.stem}+part*.jsonl")
                ]
            )
            logger.info("Number of total requests: {n_requests}", n_requests=n_requests)

            # Remove the original oversized file since we've split it into parts
            batch_file.unlink(missing_ok=False)
