from typing import Literal, TypedDict

from openai.types.chat import ChatCompletion, CompletionCreateParams
from pydantic import BaseModel
from typing_extensions import NotRequired


class Result(TypedDict):
    model: str
    revision: NotRequired[str]
    provider: str
    timestamp: NotRequired[str]
    prompt_id: str
    prompt_strategy: str
    example_id: str
    response: str | None


class Prompt(TypedDict):
    prompt_id: str
    prompt: str


# Workaround:
# - https://github.com/openai/openai-python/issues/1937
# - https://platform.openai.com/docs/guides/batch#1-prepare-your-batch-file
# - https://github.com/openai/openai-python/blob/v1.97.0/src/openai/types/chat/completion_create_params.py
class BatchInputFile(TypedDict):
    custom_id: str
    method: Literal["POST"]
    url: Literal["/v1/chat/completions"]
    body: CompletionCreateParams


# Workaround:
# - https://github.com/openai/openai-python/issues/1937
# - https://platform.openai.com/docs/api-reference/batch/request-output
class BatchOutputFileResponse(BaseModel):
    status_code: Literal[200]
    request_id: str
    body: ChatCompletion


class BatchOutputFile(BaseModel):
    id: str
    custom_id: str
    response: BatchOutputFileResponse
    error: None


class CustomId(TypedDict):
    model: str
    prompt_id: str
    prompt_strategy: str
    example_id: str


class ExampleId(TypedDict):
    dataset_gen: str
    chart_design: str
    scale_factor: str
