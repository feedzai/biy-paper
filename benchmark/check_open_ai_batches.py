import os

from dotenv import load_dotenv
from gaveta.files import ensure_dir
from openai import OpenAI
from openai.types import Batch
from rich.console import Console
from rich.table import Table
from rich.text import Text

from constants import OPEN_AI_BATCH_INPUT_FILES, OPEN_AI_ERRORS

if __name__ == "__main__":
    ensure_dir(OPEN_AI_ERRORS)

    load_dotenv()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    table = Table(title="OpenAI batches")
    table.add_column("Batch ID")
    table.add_column("File ID")
    table.add_column("Status")
    table.add_column("Completed", justify="right")
    table.add_column("Failed", justify="right")
    table.add_column("Errors")

    for batch_job_metadata in OPEN_AI_BATCH_INPUT_FILES.glob("*_job_metadata.json"):
        metadata = Batch.model_validate_json(batch_job_metadata.read_text())
        batch = client.batches.retrieve(metadata.id)

        # Note: When the status is "validating", the batch.request_counts.total is 0.
        completed_progress = (
            f"{batch.request_counts.completed / batch.request_counts.total:.2%}"
            if batch.request_counts and batch.request_counts.total > 0
            else None
        )
        failed_progress = (
            f"{batch.request_counts.failed / batch.request_counts.total:.2%}"
            if batch.request_counts and batch.request_counts.total > 0
            else None
        )

        if batch.status == "completed" and batch.error_file_id:
            error_file = OPEN_AI_ERRORS / f"{batch.id}_errors.jsonl"

            errors = client.files.content(batch.error_file_id)
            errors.write_to_file(error_file)

            status = f"[bold magenta] {batch.status}"
            table.add_row(
                batch.id,
                batch.output_file_id,
                Text.assemble((batch.status, "bold red")),
                Text.assemble((completed_progress, "green"))
                if isinstance(completed_progress, str) and completed_progress != "0.00%"
                else completed_progress,
                Text.assemble((failed_progress, "red"))
                if isinstance(failed_progress, str) and failed_progress != "0.00%"
                else failed_progress,
                error_file.name,
            )
        else:
            table.add_row(
                batch.id,
                batch.output_file_id,
                Text.assemble((batch.status, "bold green")) if batch.status == "completed" else batch.status,
                Text.assemble((completed_progress, "green"))
                if isinstance(completed_progress, str) and completed_progress != "0.00%"
                else completed_progress,
                Text.assemble((failed_progress, "red"))
                if isinstance(failed_progress, str) and failed_progress != "0.00%"
                else failed_progress,
                None,
            )

    console = Console()
    console.print(table)
