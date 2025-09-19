"""

This script uploads the prepared batch input files to a Google Cloud Storage bucket
so they can be accessed by Google Vertex AI batch prediction jobs.

Prerequisites:
1. Google Cloud SDK installed and configured
2. Appropriate permissions to upload to the target bucket
3. Environment variables set (GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_OUTPUT_BUCKET)
"""

import os

from dotenv import load_dotenv
from google.cloud import storage
from loguru import logger

from constants import GOOGLE_BATCH_INPUT_FILES

if __name__ == "__main__":
    load_dotenv()

    output_bucket_uri = os.getenv("GOOGLE_CLOUD_OUTPUT_BUCKET")

    if not output_bucket_uri:
        raise ValueError("GOOGLE_CLOUD_OUTPUT_BUCKET environment variable must be set")

    # Extract bucket name
    bucket_name = output_bucket_uri.replace("gs://", "").split("/")[0]
    # Define the prefix for batch inputs within the bucket
    destination_blob_prefix = "batch-inputs/"

    logger.info(
        "Uploading batch files to bucket: {bucket} with prefix: {prefix}",
        bucket=bucket_name,
        prefix=destination_blob_prefix,
    )

    try:
        # Initialize a Google Cloud Storage client
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)

        # Upload all JSONL files from the batch input directory
        for batch_file_path in GOOGLE_BATCH_INPUT_FILES.glob("*.jsonl"):
            # Construct the destination path (blob name) in GCS
            destination_blob_name = f"{destination_blob_prefix}{batch_file_path.name}"

            logger.info(
                "Uploading {file} to gs://{bucket}/{destination}",
                file=batch_file_path.name,
                bucket=bucket_name,
                destination=destination_blob_name,
            )

            # Upload the file
            blob = bucket.blob(destination_blob_name)
            blob.upload_from_filename(batch_file_path)

            logger.info("Successfully uploaded {file}", file=batch_file_path.name)

    except Exception as e:
        logger.error("An error occurred during upload: {error}", error=e)
        raise

    logger.info("All batch files uploaded successfully to Google Cloud Storage.")
    logger.info("Finally, run 'run_google_batches.py' to submit the batch jobs.")
