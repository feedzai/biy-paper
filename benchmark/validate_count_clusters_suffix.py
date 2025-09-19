import re
import subprocess

import openai
from gaveta.files import ensure_dir
from gaveta.json import write_json
from gaveta.time import now_iso
from loguru import logger

from constants import INPUT, TEMPERATURE, VAL
from data_models import Prompt, Result
from utils import encode_image

QUESTION = "How many clusters are there in the scatterplot?"

PROMPTS: list[Prompt] = [
    {
        "prompt_id": "curly_example",
        "prompt": f"{QUESTION} Answer with a number in curly brackets, e.g., {9}.",
    },
    {
        "prompt_id": "curly",
        "prompt": f"{QUESTION} Answer with a number in curly brackets.",
    },
    {
        "prompt_id": "round_example",
        "prompt": f"{QUESTION} Answer with a number in round brackets, e.g., (9).",
    },
    {
        "prompt_id": "round",
        "prompt": f"{QUESTION} Answer with a number in round brackets.",
    },
    {
        "prompt_id": "square_example",
        "prompt": f"{QUESTION} Answer with a number in square brackets, e.g., [9].",
    },
    {
        "prompt_id": "square",
        "prompt": f"{QUESTION} Answer with a number in square brackets.",
    },
    {
        "prompt_id": "angle_example",
        "prompt": f"{QUESTION} Answer with a number in angle brackets, e.g., <9>.",
    },
    {
        "prompt_id": "angle",
        "prompt": f"{QUESTION} Answer with a number in angle brackets.",
    },
    {"prompt_id": "single_number", "prompt": f"{QUESTION} Please give the answer in a single number."},
    {"prompt_id": "boxed", "prompt": f"{QUESTION} Put your final answer in \\boxed{{}} using a single number."},
]

MODEL = "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M"


def get_provider() -> str:
    version_output = subprocess.run(["llama-server", "--version"], capture_output=True, check=False).stderr.decode(
        "utf-8"
    )
    match = re.search(r"version:\s(\d+)", version_output)

    if match:
        version_number = match.group(1)
        return f"llama.cpp v{version_number}"

    raise ValueError(f"Version number not found: {version_output!r}")


if __name__ == "__main__":
    ensure_dir(VAL)

    model_results: list[Result] = []
    provider = get_provider()

    client = openai.OpenAI(base_url="http://localhost:8080/v1", api_key="llama-cpp")
    logger.info("Total models: {n_models}", n_models=len(client.models.list().data))
    logger.info("Model ID: {model_id}", model_id=client.models.list().data[0].id)

    for chart in INPUT.glob("*.png"):
        encoded_chart = encode_image(chart)

        for prompt in PROMPTS:
            completion = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt["prompt"]},
                            {
                                "type": "image_url",
                                "image_url": {"url": encoded_chart},
                            },
                        ],
                    }
                ],
                temperature=TEMPERATURE,
            )

            response = completion.choices[0].message.content
            logger.info("{model} responded to {prompt}", model=MODEL, prompt=prompt["prompt_id"])

            model_results.append(
                {
                    "model": MODEL,
                    "provider": provider,
                    "timestamp": now_iso(),
                    "prompt_id": prompt["prompt_id"],
                    "prompt_strategy": "zero_shot",
                    "example_id": chart.stem,
                    "response": response,
                }
            )

    write_json(model_results, VAL / f"{MODEL.replace('/', '_').replace(':', '_')}")
