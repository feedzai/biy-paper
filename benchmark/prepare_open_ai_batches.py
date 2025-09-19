"""
Script to prepare batch input files for OpenAI API batch processing.

This script:
1. Splits a dataset into training (n-shot, one-shot) and test sets
2. Generates different types of prompts (zero-shot, one-shot, few-shot) for each test example
3. Creates batch request files for each model that can be submitted to OpenAI's batch API
4. Handles file size limits by splitting large batch files into parts
"""

import math
from typing import cast

import humanize
from gaveta.files import ensure_dir
from gaveta.json import write_json
from loguru import logger
from more_itertools import divide
from openai.types.chat import CompletionCreateParams

from constants import INPUT, O_MODELS_TEMPERATURE, OPEN_AI_BATCH_INPUT_FILES, OPEN_AI_MODELS, PROMPTS, TEMPERATURE
from data_models import BatchInputFile
from utils import (
    generate_custom_id,
    generate_n_shot_open_ai_messages,
    generate_zero_shot_open_ai_messages,
    read_jsonl,
    split_dataset,
    write_jsonl,
)

# Constants for file size management
BASE = 1000  # or 1024
MAX_FILE_SIZE = 200 * BASE * BASE  # 200MB - OpenAI's batch file size limit

if __name__ == "__main__":
    # Ensure the output directory exists
    ensure_dir(OPEN_AI_BATCH_INPUT_FILES)

    # Split the dataset into different subsets for different prompting strategies
    # - n_shot_dataset: Multiple examples for few-shot learning
    # - one_shot_dataset: Single example for one-shot prompting
    # - test_dataset: Examples to be evaluated
    n_shot_dataset, one_shot_dataset, test_dataset = split_dataset(INPUT / "dataset.parquet")

    # Documentation:
    # - https://platform.openai.com/docs/guides/batch
    # - https://cookbook.openai.com/examples/batch_processing

    all_custom_ids: list[str] = []

    # Process each model separately to create individual batch files
    for model in OPEN_AI_MODELS:
        batch_requests: list[BatchInputFile] = []
        batch_file = OPEN_AI_BATCH_INPUT_FILES / f"{model}.jsonl"

        # Documentation: https://platform.openai.com/docs/guides/reasoning/how-reasoning-works#get-started-with-reasoning
        extra_body = (
            {"temperature": O_MODELS_TEMPERATURE, "reasoning_effort": "medium"}
            if model.startswith("o")
            else {"temperature": TEMPERATURE}
        )

        # For each prompt template and test example, generate three types of requests:
        # 1. Zero-shot: No examples provided
        # 2. One-shot: Single example provided
        # 3. Few-shot: Multiple examples provided
        for prompt in PROMPTS:
            for test_example in test_dataset.itertuples(index=False):
                # custom_id format: "model+prompt_id+strategy+example_id"
                zero_shot_id = generate_custom_id(model, prompt["prompt_id"], "zero_shot", str(test_example.id))
                one_shot_id = generate_custom_id(model, prompt["prompt_id"], "one_shot", str(test_example.id))
                few_shot_id = generate_custom_id(model, prompt["prompt_id"], "few_shot", str(test_example.id))

                all_custom_ids.extend([zero_shot_id, one_shot_id, few_shot_id])

                # Generate messages for each prompting strategy
                zero_shot = generate_zero_shot_open_ai_messages(prompt, str(test_example.image))
                one_shot = generate_n_shot_open_ai_messages(one_shot_dataset, prompt, str(test_example.image))
                few_shot = generate_n_shot_open_ai_messages(n_shot_dataset, prompt, str(test_example.image))

                # Create batch request entries for each prompting strategy
                zero_shot_body = cast("CompletionCreateParams", {"model": model, "messages": zero_shot} | extra_body)
                one_shot_body = cast("CompletionCreateParams", {"model": model, "messages": one_shot} | extra_body)
                few_shot_body = cast("CompletionCreateParams", {"model": model, "messages": few_shot} | extra_body)

                batch_requests.extend(
                    [
                        {
                            "custom_id": zero_shot_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": zero_shot_body,
                        },
                        {
                            "custom_id": one_shot_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": one_shot_body,
                        },
                        {
                            "custom_id": few_shot_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": few_shot_body,
                        },
                    ]
                )

        # Write all batch requests to a JSONL file
        write_jsonl(batch_requests, batch_file)

        # Check if the batch file exceeds OpenAI's size limit (200MB)
        # https://platform.openai.com/docs/guides/batch#rate-limits
        batch_file_size = batch_file.stat().st_size
        logger.info("Number of requests: {n_requests}", n_requests=len(batch_requests))
        logger.info("Batch file size: {file_size}", file_size=humanize.naturalsize(batch_file_size))

        # If file is too large, split it into multiple parts
        if batch_file_size >= MAX_FILE_SIZE:
            # Calculate how many parts we need
            n_parts = math.ceil(batch_file_size / MAX_FILE_SIZE) + 1

            # Split the batch requests into equal parts
            for index, part in enumerate(divide(n_parts, batch_requests), start=1):
                batch_file_part = OPEN_AI_BATCH_INPUT_FILES / f"{batch_file.stem}+part{index}.jsonl"

                # Write each part to a separate file
                write_jsonl(part, batch_file_part)

                batch_file_part_size = batch_file_part.stat().st_size
                logger.info("Batch file part size: {file_size}", file_size=humanize.naturalsize(batch_file_part_size))

            # Verify the total number of requests across all parts
            n_requests = sum(
                [
                    len(read_jsonl(batch_file_part))
                    for batch_file_part in OPEN_AI_BATCH_INPUT_FILES.glob(f"{batch_file.stem}+part*.jsonl")
                ]
            )
            logger.info("Total number of requests: {n_requests}", n_requests=n_requests)

            # Remove the original oversized file since we've split it into parts
            batch_file.unlink(missing_ok=False)

    write_json(all_custom_ids, OPEN_AI_BATCH_INPUT_FILES / "custom_ids.json")
    logger.info("Total number of custom IDs: {n_custom_ids}", n_custom_ids=len(all_custom_ids))
    logger.info("Number of unique custom IDs: {n_custom_ids}", n_custom_ids=len(set(all_custom_ids)))
