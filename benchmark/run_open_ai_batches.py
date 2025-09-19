"""
Script to submit batch input files to OpenAI's batch API for processing.

This script:
1. Loads environment variables (including OpenAI API key)
2. Iterates through all prepared batch input files (*.jsonl)
3. Uploads each file to OpenAI's file storage
4. Creates batch jobs for processing the uploaded files
5. Tracks the status of each batch job
6. Saves metadata for both files and jobs for later reference
"""

import os

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

from constants import OPEN_AI_BATCH_INPUT_FILES
from utils import write_model_json

if __name__ == "__main__":
    # Load environment variables from .env file (including OPENAI_API_KEY)
    load_dotenv()

    # Initialize OpenAI client with API key from environment
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Process each batch input file that was prepared by prepare_open_ai_batches.py
    for batch_input_file in OPEN_AI_BATCH_INPUT_FILES.glob("*.jsonl"):
        # Extract model name from filename (e.g., "gpt-4" from "gpt-4.jsonl")
        model = batch_input_file.stem

        # Step 1: Upload the batch input file to OpenAI's file storage
        # Documentation:
        # - https://platform.openai.com/docs/guides/batch#2-upload-your-batch-input-file
        # - https://platform.openai.com/docs/api-reference/files
        metadata = client.files.create(file=batch_input_file.open(mode="rb"), purpose="batch")

        # Save the file metadata for later reference
        write_model_json(metadata, OPEN_AI_BATCH_INPUT_FILES / f"{model}_metadata.json")
        logger.info("Batch file ID: {id}", id=metadata.id)

        # Verify the file was uploaded correctly by retrieving its metadata
        retrieved_metadata = client.files.retrieve(metadata.id)
        logger.info(retrieved_metadata)

        # Step 2: Create a batch job to process the uploaded file
        batch_job = client.batches.create(
            input_file_id=metadata.id,  # Use the uploaded file ID
            endpoint="/v1/chat/completions",  # API endpoint for chat completions
            completion_window="24h",  # Time limit for job completion
            metadata={"description": "Experiment for the BIY paper."},  # Job description
        )

        # Save the batch job metadata for tracking
        write_model_json(batch_job, OPEN_AI_BATCH_INPUT_FILES / f"{model}_job_metadata.json")

        # Retrieve the latest status of the batch job
        batch_job = client.batches.retrieve(batch_job.id)
        logger.info("Batch job ID: {id}", id=batch_job.id)
        logger.info("Batch job status: {status}", status=batch_job.status)

    # Provide a link to monitor all batch jobs on OpenAI's platform
    logger.info("All batches: https://platform.openai.com/batches")
