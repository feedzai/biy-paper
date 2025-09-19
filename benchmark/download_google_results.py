"""
Script to download Google Vertex AI batch prediction results from Cloud Storage (GCS).

This script:
1. Downloads all result files from the specified bucket/prefix.
2. Organizes them by model in the local results directory, maintaining subfolder structure.
3. Provides progress logging for the download process.
"""

import os

from dotenv import load_dotenv
from gaveta.files import ensure_dir
from google.cloud import storage
from loguru import logger

from constants import GOOGLE_RESULTS  # Assuming GOOGLE_RESULTS is a Path object for your local results directory


def download_results_from_gcs(model_name: str | None = None) -> None:
    output_bucket_uri = os.getenv("GOOGLE_CLOUD_OUTPUT_BUCKET")

    if not output_bucket_uri:
        raise ValueError(
            "GOOGLE_CLOUD_OUTPUT_BUCKET environment variable must be set "
            "(e.g., gs://your-bucket/my-batch-results-folder)"
        )

    # Parse the bucket name and the base path within the bucket from the URI
    bucket_parts = output_bucket_uri.replace("gs://", "").split("/", 1)
    bucket_name = bucket_parts[0]

    # The prefix in GCS for listing objects
    # This will be "results_data/" or "" if no subfolder was specified in the URI
    base_gcs_path = bucket_parts[1] + "/" if len(bucket_parts) > 1 else ""

    # Construct the actual prefix for listing blobs
    if model_name:
        # If downloading for a specific model, append model_name to the base path
        list_prefix = f"{base_gcs_path}{model_name}/"
        logger.info(
            "Downloading results for model: {model!r} from gs://{bucket}/{prefix}",
            model=model_name,
            bucket=bucket_name,
            prefix=list_prefix,
        )
    else:
        # If downloading all results, use the base path directly
        list_prefix = base_gcs_path
        logger.info(
            "Downloading all results from gs://{bucket}/{prefix}",
            bucket=bucket_name,
            prefix=list_prefix if list_prefix else "[root]",
        )

    try:
        # Initialize Google Cloud Storage client
        storage_client = storage.Client()
        bucket = storage_client.get_bucket(bucket_name)

        # Ensure local results directory exists
        ensure_dir(GOOGLE_RESULTS)

        # List all blobs under the determined prefix
        blobs = bucket.list_blobs(prefix=list_prefix)

        downloaded_files = []

        for blob in blobs:
            # Skip blobs that exactly match the prefix
            if blob.name == list_prefix:
                continue

            # Construct the local relative path by removing the base GCS prefix
            relative_path = blob.name.replace(base_gcs_path, "", 1)  # Use count=1 to replace only the first occurrence

            local_file_path = GOOGLE_RESULTS / relative_path

            # Ensure the local directory structure exists for the file
            ensure_dir(local_file_path.parent)

            logger.info(
                "Downloading GCS object '{blob_name}' to local path '{local_path}'",
                blob_name=blob.name,
                local_path=local_file_path,
            )

            # Download the file
            blob.download_to_filename(local_file_path)

            downloaded_files.append(local_file_path)
            logger.success("Successfully downloaded '{file_name}'", file_name=local_file_path.name)

        if downloaded_files:
            logger.info(
                "Download completed! Downloaded {count} files to: {results_dir}",
                count=len(downloaded_files),
                results_dir=GOOGLE_RESULTS,
            )
            logger.info("Downloaded files summary:")
            for file_path in downloaded_files:
                file_size = file_path.stat().st_size
                logger.info("  - {file} ({size} bytes)", file=file_path.relative_to(GOOGLE_RESULTS), size=file_size)
        else:
            logger.warning("No files found to download under prefix: {prefix}", prefix=list_prefix)

    except Exception as e:
        logger.error("An error occurred during download: {error}", error=e)
        raise


if __name__ == "__main__":
    import sys

    # Load environment variables from .env file
    load_dotenv()

    # Example Usage:
    # To download all results: python your_script_name.py
    # To download results for a specific model: python your_script_name.py gemini-2.5-flash

    if len(sys.argv) > 1:
        # Download specific model results
        model_name_arg = sys.argv[1]
        download_results_from_gcs(model_name=model_name_arg)
    else:
        # Download all results
        download_results_from_gcs()
