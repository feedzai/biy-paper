import base64
import json
import mimetypes
import shutil
import statistics
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TypeGuard
from urllib.request import urlopen

import altair as alt
import lap
import numpy as np
import pandas as pd
import torch
from gaveta.files import ensure_dir
from loguru import logger
from num2words import num2words
from openai.types.chat import ChatCompletionMessageParam
from pandas.io.formats.style_render import _escape_latex
from PIL import Image
from pydantic import BaseModel
from torchvision.ops import box_iou

from constants import (
    ALL_RESULTS,
    GOOGLE_COLOR,
    METRIC_ID_TO_LABEL,
    MODEL_TO_LABEL,
    OPEN_AI_COLOR,
    OPEN_AI_MODEL_TO_LABEL,
    PROMPT_ID_TO_ICON,
    PROMPT_ID_TO_LABEL,
    PROMPT_ID_TO_PROMPT_STRATEGY_COLORS,
    PROMPT_STRATEGY_TO_LABEL,
    SEED,
)
from data_models import CustomId, ExampleId, Prompt


def ensure_clean_dir(folder: Path) -> None:
    try:
        shutil.rmtree(folder)
        ensure_dir(folder)
    except FileNotFoundError:
        ensure_dir(folder)


def is_str(val: object) -> TypeGuard[str]:
    return isinstance(val, str)


def encode_image(image: Path) -> str:
    media_type = mimetypes.types_map[image.suffix]
    base64_image = base64.b64encode(image.read_bytes()).decode("utf-8")

    return f"data:{media_type};base64,{base64_image}"


def split_dataset(dataset: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    chart_design_ignore = ["random_shapes", "square"]

    full_dataset = (
        pd.read_parquet(dataset, engine="pyarrow")
        .query("dataset_gen != 'shapes' & scale_factor == 2 & chart_design not in @chart_design_ignore")
        .reset_index(drop=True)
    )

    n_shot_examples_per_dataset = [
        (1, "gaussian_blobs"),
        (1, "gaussian_blobs_noise"),
        (2, "single_gaussian_blob_outliers"),
        (1, "relationship"),
        (1, "random"),
    ]

    n_shot_dataset = pd.concat(
        [
            full_dataset.loc[
                (full_dataset["chart_design"] == "default") & (full_dataset["dataset_gen"] == config[1])
            ].sample(config[0], random_state=SEED, replace=False)
            for config in n_shot_examples_per_dataset
        ],
        ignore_index=True,
    ).sample(frac=1, random_state=0, replace=False, ignore_index=True)

    one_shot_dataset = n_shot_dataset.query("dataset_gen == 'gaussian_blobs'").sample(
        1, random_state=SEED, replace=False, ignore_index=True
    )

    test_examples_per_dataset = [
        (25, "gaussian_blobs"),
        (25, "gaussian_blobs_noise"),
        (50, "single_gaussian_blob_outliers"),
        (8, "relationship"),
        (7, "random"),
    ]

    test_dataset_ids = pd.concat(
        [
            full_dataset.loc[
                (~full_dataset["dataset_id"].isin(n_shot_dataset["dataset_id"]))
                & (full_dataset["dataset_gen"] == config[1]),
                ["dataset_id"],
            ]
            .drop_duplicates()
            .sample(config[0], random_state=SEED, replace=False)
            for config in test_examples_per_dataset
        ],
        ignore_index=True,
    )["dataset_id"]

    test_dataset = full_dataset.query("dataset_id in @test_dataset_ids").reset_index(drop=True)

    return n_shot_dataset, one_shot_dataset, test_dataset


def normalize_xyxy_bbox(encoded_image: str, bbox: list[float]) -> list[int]:
    with Image.open(urlopen(encoded_image)) as im:
        width, height = im.size

    return [
        int(1_000 * (bbox[0] / width)),
        int(1_000 * (bbox[1] / height)),
        int(1_000 * (bbox[2] / width)),
        int(1_000 * (bbox[3] / height)),
    ]


def denormalize_xyxy_bbox(encoded_image: str, bbox: list[float]) -> list[int]:
    with Image.open(urlopen(encoded_image)) as im:
        width, height = im.size

    return [
        int(width * (bbox[0] / 1_000)),
        int(height * (bbox[1] / 1_000)),
        int(width * (bbox[2] / 1_000)),
        int(height * (bbox[3] / 1_000)),
    ]


def process_xyxy_bbox(encoded_image: str, bbox: list[float]) -> list[int]:
    x1, y1, x2, y2 = denormalize_xyxy_bbox(encoded_image, bbox)

    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def normalize_xy_point(encoded_image: str, point: list[float]) -> list[int]:
    with Image.open(urlopen(encoded_image)) as im:
        width, height = im.size

    return [
        int(1_000 * (point[0] / width)),
        int(1_000 * (point[1] / height)),
    ]


def denormalize_xy_point(encoded_image: str, point: list[float]) -> list[int]:
    with Image.open(urlopen(encoded_image)) as im:
        width, height = im.size

    return [
        int(width * (point[0] / 1_000)),
        int(height * (point[1] / 1_000)),
    ]


def format_prompt_answer(example: Any, prompt: Prompt) -> str:
    if prompt["prompt_id"] == "cluster_count":
        return f"{{{example.cluster_count}}}"
    if prompt["prompt_id"] == "outlier_count":
        return f"{{{example.outlier_count}}}"
    if prompt["prompt_id"] == "cluster_bboxes":
        answer = {"clusters": [normalize_xyxy_bbox(example.image, bbox.tolist()) for bbox in example.cluster_bboxes]}
        return json.dumps(answer, ensure_ascii=False)
    if prompt["prompt_id"] == "cluster_points":
        answer = {
            "cluster_centers": [normalize_xy_point(example.image, point.tolist()) for point in example.cluster_points]
        }
        return json.dumps(answer, ensure_ascii=False)
    if prompt["prompt_id"] == "outlier_points":
        answer = {"outliers": [normalize_xy_point(example.image, point.tolist()) for point in example.outlier_points]}
        return json.dumps(answer, ensure_ascii=False)

    msg = f"Unsupported prompt: {prompt['prompt_id']}"
    raise ValueError(msg)


def generate_n_shot_open_ai_messages(
    dataset: pd.DataFrame, prompt: Prompt, encoded_image: str
) -> list[ChatCompletionMessageParam]:
    messages: list[ChatCompletionMessageParam] = []

    for example in dataset.itertuples(index=False):
        formatted_answer = format_prompt_answer(example, prompt)

        message: list[ChatCompletionMessageParam] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(example.image),
                            "detail": "high",
                        },
                    },
                    {"type": "text", "text": prompt["prompt"]},
                ],
            },
            {"role": "assistant", "content": formatted_answer},
        ]

        messages.extend(message)

    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encoded_image,
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt["prompt"]},
            ],
        }
    )

    return messages


def generate_zero_shot_open_ai_messages(prompt: Prompt, encoded_image: str) -> list[ChatCompletionMessageParam]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": encoded_image,
                        "detail": "high",
                    },
                },
                {"type": "text", "text": prompt["prompt"]},
            ],
        }
    ]


def write_jsonl(data: Iterable[Any], output_path: Path) -> None:
    with output_path.open(mode="w", encoding="utf-8") as f:
        for datum in data:
            f.write(json.dumps(datum, ensure_ascii=False) + "\n")


def read_jsonl(input_path: Path) -> list[Any]:
    with input_path.open(mode="r") as f:
        return [json.loads(f_line) for f_line in f]


def write_model_json(model: BaseModel, output_path: Path) -> None:
    with output_path.open(mode="w", encoding="utf-8") as f:
        f.write(model.model_dump_json(indent=2))
        f.write("\n")


def write_text(text: str, output_path: Path) -> None:
    with output_path.open(mode="w", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def generate_custom_id(model: str, prompt_id: str, prompt_strategy: str, example_id: str) -> str:
    return "+".join([model, prompt_id, prompt_strategy, example_id])


def parse_custom_id(custom_id: str) -> CustomId:
    model, prompt_id, prompt_strategy, example_id = custom_id.split("+", maxsplit=3)

    return {"model": model, "prompt_id": prompt_id, "prompt_strategy": prompt_strategy, "example_id": example_id}


def parse_example_id(example_id: str) -> ExampleId:
    dataset_gen, _, chart_design, scale_factor = example_id.split("+", maxsplit=3)

    return {"dataset_gen": dataset_gen, "chart_design": chart_design, "scale_factor": scale_factor}


def compute_bbox_metrics(
    pred_bboxes: list[list[int]], target_bboxes: list[list[int]], iou_threshold: float = 0.5
) -> dict[str, float]:
    suffix = str(iou_threshold).replace(".", "_")

    if len(pred_bboxes) == 0 and len(target_bboxes) == 0:
        return {f"precision_{suffix}": 1.0, f"recall_{suffix}": 1.0, f"mean_iou_{suffix}": 1.0}

    if len(pred_bboxes) == 0 or len(target_bboxes) == 0:
        return {f"precision_{suffix}": 0.0, f"recall_{suffix}": 0.0, f"mean_iou_{suffix}": 0.0}

    iou_matrix = box_iou(
        torch.tensor(pred_bboxes),
        torch.tensor(target_bboxes),
    )

    # Based on https://github.com/ultralytics/ultralytics/blob/v8.3.177/ultralytics/trackers/utils/matching.py#L64-L101
    cost_matrix = (1 - iou_matrix).numpy()  # higher IoU = lower cost
    cost_threshold = 1 - iou_threshold

    _, _, target_indices = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=cost_threshold)

    matched_iou_values = []

    for target_idx, pred_idx in enumerate(target_indices):
        if pred_idx != -1:
            matched_iou_values.append(iou_matrix[pred_idx, target_idx].item())

    tp = len(matched_iou_values)
    fn = len(target_bboxes) - tp
    fp = len(pred_bboxes) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    mean_iou = statistics.mean(matched_iou_values) if len(matched_iou_values) > 0 else 0.0

    return {
        f"precision_{suffix}": precision,
        f"recall_{suffix}": recall,
        f"mean_iou_{suffix}": mean_iou,
    }


# 10 pixels as in Point, Detect, Count: Multi-Task Medical Image Understanding with Instruction-Tuned Vision-Language Models (https://arxiv.org/abs/2505.16647)
def compute_point_metrics(
    pred_points: list[list[int]], target_points: list[list[int]], distance_threshold: int = 10
) -> dict[str, float]:
    if len(pred_points) == 0 and len(target_points) == 0:
        return {
            f"precision_{distance_threshold}": 1.0,
            f"recall_{distance_threshold}": 1.0,
        }

    if len(pred_points) == 0 or len(target_points) == 0:
        return {
            f"precision_{distance_threshold}": 0.0,
            f"recall_{distance_threshold}": 0.0,
        }

    distance_matrix = np.linalg.norm(
        np.array(pred_points)[:, np.newaxis, :] - np.array(target_points)[np.newaxis, :, :], axis=2
    )

    _, _, target_indices = lap.lapjv(distance_matrix, extend_cost=True, cost_limit=distance_threshold)

    matched_distance_values = []

    for target_idx, pred_idx in enumerate(target_indices):
        if pred_idx != -1:
            matched_distance_values.append(distance_matrix[pred_idx, target_idx])

    tp = len(matched_distance_values)
    fn = len(target_points) - tp
    fp = len(pred_points) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return {
        f"precision_{distance_threshold}": precision,
        f"recall_{distance_threshold}": recall,
    }


def _format_bar_value(dataset: pd.DataFrame, metric_col: str, group: str, bar: str) -> str:
    n_decimals = 2

    if metric_col == "mae":
        return str(
            round(
                dataset.loc[(dataset["model"] == group) & (dataset["prompt_strategy"] == bar), metric_col].item(),
                n_decimals,
            )
        )

    return rf"{round(dataset.loc[(dataset['model'] == group) & (dataset['prompt_strategy'] == bar), metric_col].item() * 100, n_decimals)}%"


def generate_grouped_bar_chart_alt_text(dataset: pd.DataFrame, prompt_id: str, metric_col: str) -> str:
    # Based on Accessible Bar Charts Through Textual Description Templates (https://journals-sol.sbc.org.br/index.php/jbcs/article/view/2301):

    title = f"performance for the {PROMPT_ID_TO_LABEL[prompt_id]} task"
    y_axis_label = f"{METRIC_ID_TO_LABEL[metric_col]}"
    x_axis_label = "model"
    alt_text_template = f"This is a grouped vertical bar chart. Its title is {title}. The y-axis legend is {y_axis_label}. The x-axis legend is {x_axis_label}."

    groups = list(MODEL_TO_LABEL.values())
    groups_value = num2words(len(groups), lang="en", to="cardinal") if len(groups) < 10 else len(groups)
    alt_text_template = (
        f"{alt_text_template} The chart is made up of {groups_value} groups of bars: {', '.join(groups)}."
    )

    bars = list(PROMPT_STRATEGY_TO_LABEL.values())
    bars_value = num2words(len(bars), lang="en", to="cardinal") if len(bars) < 10 else len(bars)
    alt_text_template = f"{alt_text_template} Each group contains {bars_value} bars: {', '.join(bars)}, which will be presented in that order."

    for index, group in enumerate(groups, start=1):
        values = ", ".join([_format_bar_value(dataset, metric_col, group, bar) for bar in bars])
        alt_text_template = f"{alt_text_template} The {num2words(index, lang='en', to='ordinal')} group of bars is {group} and has values {values}."

    return alt_text_template


def generate_grouped_bar_chart(dataset: pd.DataFrame, prompt_id: str, metric_col: str) -> None:
    height = 150
    label_padding = 4
    manual_legend_x = 307

    y_domain: tuple[int, int] = (0, 1) if metric_col != "mae" else (0, 20)
    y_values: list[float] = [0, 0.25, 0.5, 0.75, 1] if metric_col != "mae" else [0, 5, 10, 15, 20]
    y_format = "%" if metric_col != "mae" else "s"

    chart_filename = f"{prompt_id}_{metric_col}"

    dataset_to_plot = dataset.query("prompt_id == @prompt_id").assign(
        prompt_strategy=dataset["prompt_strategy"].map(PROMPT_STRATEGY_TO_LABEL),
        model=dataset["model"].map(MODEL_TO_LABEL),
    )

    label_color_cond = " || ".join([f'datum.label == "{label}"' for label in OPEN_AI_MODEL_TO_LABEL.values()])

    base = alt.Chart(
        dataset_to_plot,
        width=alt.Step(12),
    ).encode(
        x=alt.X("model:N")
        .axis(
            title=None,
            zindex=1,
            # labels=prompt_id == "outlier_count" or metric_col.startswith("recall"),
            # labels=prompt_id == "outlier_count" or prompt_id == "outlier_points",
            labelAngle=0,
            ticks=False,
            labelExpr="split(datum.label, ' ')",
            labelColor=alt.when(label_color_cond).then(alt.value(OPEN_AI_COLOR)).otherwise(alt.value(GOOGLE_COLOR)),
            labelPadding=label_padding,
            labelFontSize=10,
        )
        .sort(list(MODEL_TO_LABEL.values())),
        y=alt.Y(f"{metric_col}:Q")
        .scale(domain=y_domain)
        .axis(
            title=None,
            format=y_format,
            tickCount=len(y_values),
            values=y_values,
            labelExpr=f"datum.value == {y_domain[1]} ? [datum.label, '{METRIC_ID_TO_LABEL[metric_col]}', '({prompt_id.split('_')[0]}s)'] : datum.label"
            if metric_col in {"accuracy", "mae"}
            else f"datum.value == {y_domain[1]} ? [datum.label, '{METRIC_ID_TO_LABEL[metric_col].split(' ', 1)[0]}', '{METRIC_ID_TO_LABEL[metric_col].split(' ', 1)[1]}'] : datum.label",
            labelFontSize=8,
        ),
        xOffset=alt.XOffset("prompt_strategy:N", sort=list(PROMPT_STRATEGY_TO_LABEL.values())).scale(paddingOuter=0.25),
    )

    (
        alt.layer(
            base.mark_bar(size=20, stroke="white", fillOpacity=0.9).encode(
                color=alt.Color(
                    "prompt_strategy:N",
                    legend=alt.Legend(
                        title=None,
                        direction="horizontal",
                        orient="none",
                        legendY=-12 - label_padding,
                        legendX=manual_legend_x,
                        symbolOpacity=0.9,
                        labelFontSize=10,
                        labelOffset=2,
                        labelBaseline="middle",
                        symbolSize=75,
                    ),
                    # if prompt_id == "cluster_count" or metric_col.startswith("precision")
                    # if prompt_id == "cluster_count" or prompt_id == "cluster_bboxes"
                    # else None,
                ).scale(
                    domain=list(PROMPT_STRATEGY_TO_LABEL.values()), range=PROMPT_ID_TO_PROMPT_STRATEGY_COLORS[prompt_id]
                )
            ),
        )
        + alt.Chart()
        .mark_image(width=12, height=12, x=12 / 2 + 2, y=12 / 2 + 2, baseline="middle")
        .encode(
            url=alt.value(encode_image(PROMPT_ID_TO_ICON[prompt_id])),
        )
    ).configure(font="Inter").configure_axis(
        gridColor="#e2e8f0", domainColor="#64748b", tickColor="#64748b", labelColor="#0f172a"
    ).configure_legend(labelColor="#0f172a", titleColor="#0f172a").configure_view(stroke=None).properties(
        height=height
    ).save(ALL_RESULTS / f"{chart_filename}.png", scale_factor=8)

    alt_text = generate_grouped_bar_chart_alt_text(
        dataset_to_plot,
        prompt_id,
        metric_col,
    )

    logger.info(alt_text)
    write_text(alt_text, ALL_RESULTS / f"{chart_filename}.txt")
    write_text(_escape_latex(alt_text), ALL_RESULTS / f"{chart_filename}_latex.txt")
