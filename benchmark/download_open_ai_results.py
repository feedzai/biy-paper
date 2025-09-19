"""
Script to download and save results from completed OpenAI batch jobs.

This script:
1. Loads environment variables and initializes OpenAI client
2. Iterates through all batch job metadata files to get the relevant IDs for retrieval (*_job_metadata.json)
3. Checks the status of each batch job
4. Downloads completed batch results to local storage
5. Logs the download status for each batch job
"""

import os

import pandas as pd
from dotenv import load_dotenv
from gaveta.files import ensure_dir
from gaveta.json import read_json, write_json
from loguru import logger
from openai import OpenAI
from openai.types import Batch
from praicing.openai import estimate_costs_for_tokens

from constants import OPEN_AI_BATCH_INPUT_FILES, OPEN_AI_MISSING_CUSTOM_IDS, OPEN_AI_RAW_RESULTS, OPEN_AI_RESULTS
from data_models import BatchOutputFile

if __name__ == "__main__":
    # Ensure the results directory exists for storing downloaded files
    ensure_dir(OPEN_AI_RAW_RESULTS)

    # Load environment variables from .env file (including OPENAI_API_KEY)
    load_dotenv()

    # Initialize OpenAI client with API key from environment
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Process each batch job metadata file that was created by run_open_ai_batches.py
    for batch_job_metadata in OPEN_AI_BATCH_INPUT_FILES.glob("*_job_metadata.json"):
        # Parse the batch job metadata from the JSON file
        metadata = Batch.model_validate_json(batch_job_metadata.read_text())

        # Retrieve the current status of the batch job from OpenAI
        batch = client.batches.retrieve(metadata.id)

        # Check if the batch job has completed and has output results
        if batch.output_file_id:
            # Download the results file from OpenAI's file storage
            model_responses = client.files.content(batch.output_file_id)

            # Save the actual results/responses + metadata to a local JSONL file using the output file ID as filename
            model_responses.write_to_file(OPEN_AI_RAW_RESULTS / f"{batch.output_file_id}.jsonl")

            logger.info("{id} downloaded", id=batch.output_file_id)
        else:
            # Log if the batch job hasn't completed yet or doesn't have results
            logger.info("{id} results are not available", id=metadata.id)

    model_token_counts: dict[str, dict[str, int]] = {}
    custom_ids: list[str] = []

    for raw_results in OPEN_AI_RAW_RESULTS.glob("*.jsonl"):
        results = [BatchOutputFile.model_validate_json(line) for line in raw_results.read_text().splitlines()]

        for result in results:
            custom_ids.append(result.custom_id)

            if result.response.body.usage:
                model = result.response.body.model

                stats = model_token_counts.get(model, {"Input tokens": 0, "Output tokens": 0})

                stats["Input tokens"] += result.response.body.usage.prompt_tokens
                stats["Output tokens"] += result.response.body.usage.completion_tokens

                model_token_counts[model] = stats

    cost_df = pd.DataFrame.from_dict(model_token_counts, orient="index").reset_index(names="Model")

    cost_df["Cost"] = cost_df.apply(
        lambda row: estimate_costs_for_tokens(
            input_tokens=row["Input tokens"],
            output_tokens=row["Output tokens"],
            model=row["Model"],
            pricing="batch",
        ),
        axis=1,
    )
    cost_df.insert(0, "Provider", "OpenAI")

    cost_df.to_csv(OPEN_AI_RESULTS / "costs.csv", index=False)
    logger.info("Cost file generated")

    expected_custom_ids = read_json(OPEN_AI_BATCH_INPUT_FILES / "custom_ids.json")

    missing_custom_ids = list(set(expected_custom_ids).difference(custom_ids))
    missing_pct = len(missing_custom_ids) / len(expected_custom_ids)

    logger.info("All custom IDs are unique: {is_unique}", is_unique=len(custom_ids) == len(set(custom_ids)))

    write_json(missing_custom_ids, OPEN_AI_MISSING_CUSTOM_IDS)

    if missing_pct > 0:
        logger.info(
            "{missing_pct:.2%} of expected custom IDs missing. List of missing custom IDs generated.",
            missing_pct=missing_pct,
        )
    else:
        logger.info("All custom IDs verified")
