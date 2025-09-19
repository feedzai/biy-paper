"""
Script to submit batch input files to Google Vertex AI batch API for processing.

This script:
1. Loads environment variables (including Google Cloud credentials)
2. Iterates through all prepared batch input files (*.jsonl)
3. Creates batch jobs for processing the files from Cloud Storage
4. Saves metadata for jobs for later reference
"""

import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions
from loguru import logger

from constants import GOOGLE_BATCH_INPUT_FILES, GOOGLE_RESULTS
from utils import write_model_json

# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini
# https://googleapis.github.io/python-genai/genai.html#genai.batches.Batches.create
# https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/batch-prediction/intro_batch_prediction.ipynb
if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Get configuration from environment
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    output_bucket = os.getenv("GOOGLE_CLOUD_OUTPUT_BUCKET")

    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable must be set")
    if not output_bucket:
        raise ValueError("GOOGLE_CLOUD_OUTPUT_BUCKET environment variable must be set")

    # Set environment variable for Vertex AI
    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    # Initialize Google Gen AI client
    client = genai.Client(project=project_id, location=location, http_options=HttpOptions(api_version="v1"))

    # Ensure results directory exists
    GOOGLE_RESULTS.mkdir(parents=True, exist_ok=True)

    # Process each batch input file that was prepared by prepare_google_batches.py
    for batch_input_file in GOOGLE_BATCH_INPUT_FILES.glob("*.jsonl"):
        # Extract model name from filename (e.g. "gemini-2.5-flash" from "gemini-2.5-flash.jsonl")
        # Handle part files by extracting the base model name
        model = batch_input_file.stem
        if "+part" in model:
            # Extract base model name from part files (e.g., "gemini-2.5-flash" from "gemini-2.5-flash+part1")
            model = model.split("+part")[0]
            part_suffix = batch_input_file.stem.split("+part")[1]
            logger.info("Processing part file for model: {model}, part: {part}", model=model, part=part_suffix)
        else:
            logger.info("Processing batch file for model: {model}", model=model)

        # Construct Cloud Storage paths
        input_bucket_name = output_bucket.replace("gs://", "").split("/")[0]
        gs_input_path = f"gs://{input_bucket_name}/batch-inputs/{batch_input_file.name}"

        # Include part information in output path if it's a part file
        if "+part" in batch_input_file.stem:
            part_suffix = batch_input_file.stem.split("+part")[1]
            gs_output_path = f"{output_bucket}/{model}/part{part_suffix}"
        else:
            gs_output_path = f"{output_bucket}/{model}"

        try:
            # Create batch job using Google's API
            job = client.batches.create(
                model=model,
                src=gs_input_path,
                config=CreateBatchJobConfig(dest=gs_output_path),
            )

            # Save the batch job metadata for tracking
            job_metadata_file = GOOGLE_RESULTS / f"{model}_job_metadata.json"
            write_model_json(job, job_metadata_file)

            logger.info("Batch job ID: {job_name}", job_name=job.name)
            logger.info("Batch job status: {job_state}", job_state=job.state)

        except Exception as e:
            logger.error("Error creating batch job for model {model}: {error}", model=model, error=str(e))
            raise
