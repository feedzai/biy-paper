import json
from pathlib import Path

import pandas as pd
from loguru import logger

from constants import ALL_RESULTS
from utils import compute_bbox_metrics, generate_grouped_bar_chart


def compute_metrics(single_result: pd.Series) -> pd.Series:
    pred_bboxes = json.loads(single_result["pred_response"])
    target_bboxes = [[int(coordinate) for coordinate in bbox] for bbox in json.loads(single_result["response"])]

    bbox_metrics_0_5 = compute_bbox_metrics(pred_bboxes, target_bboxes, iou_threshold=0.5)
    bbox_metrics_0_75 = compute_bbox_metrics(pred_bboxes, target_bboxes, iou_threshold=0.75)

    return pd.Series(
        {
            "model": single_result["model"],
            "prompt_id": single_result["prompt_id"],
            "prompt_strategy": single_result["prompt_strategy"],
            "example_id": single_result["example_id"],
            "chart_design": single_result["chart_design"],
            "dataset_gen": single_result["dataset_gen"],
            **bbox_metrics_0_5,
            **bbox_metrics_0_75,
        }
    )


if __name__ == "__main__":
    results = pd.read_parquet(ALL_RESULTS / "results.parquet", engine="pyarrow")
    results = results.dropna(subset=["pred_response"]).query("prompt_id == 'cluster_bboxes'")

    metrics = results.apply(compute_metrics, axis=1)
    metrics.to_csv(ALL_RESULTS / "cluster_bboxes.csv", index=False)

    # Based on https://deepforest.readthedocs.io/en/v1.5.2/user_guide/12_evaluation.html#how-to-average-evaluation-metrics-across-images
    overall = metrics.groupby("model", as_index=False).mean(numeric_only=True)
    logger.info("Overall:\n{results}", results=overall)

    good_examples = (
        metrics.query("dataset_gen == 'gaussian_blobs'").sort_values(by=["recall_0_5"], ascending=False).head(10)
    )
    logger.info("Good examples:\n{examples}", examples=good_examples)

    for good_example in good_examples.itertuples(index=False):
        logger.info(
            Path("results/open_ai/validation/cluster_bboxes")
            / f"{good_example.model}+{good_example.prompt_strategy}+{good_example.example_id}.png"
        )

    bad_examples = metrics.sort_values(by=["precision_0_5"], ascending=False).tail(5)
    logger.info("Bad examples:\n{examples}", examples=bad_examples)

    for bad_example in bad_examples.itertuples(index=False):
        logger.info(
            Path("results/open_ai/validation/cluster_bboxes")
            / f"{bad_example.model}+{bad_example.prompt_strategy}+{bad_example.example_id}.png"
        )

    prompt_strategy = (
        metrics.groupby(["prompt_id", "model", "prompt_strategy"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(by=["prompt_id", "precision_0_75"], ascending=False)
    )
    logger.info("Precision and Recall:\n{results}", results=prompt_strategy)
    prompt_strategy.to_csv(ALL_RESULTS / "detection_precision_recall.csv", index=False)

    generate_grouped_bar_chart(prompt_strategy, "cluster_bboxes", "precision_0_75")
    generate_grouped_bar_chart(prompt_strategy, "cluster_bboxes", "recall_0_75")

    chart_design = metrics.groupby(["model", "prompt_strategy", "chart_design"], as_index=False).mean(numeric_only=True)
    logger.info("Precision and Recall (chart design):\n{results}", results=chart_design)
    chart_design.to_csv(ALL_RESULTS / "detection_precision_recall_chart_design.csv", index=False)

    no_patterns = results.query("dataset_gen == 'random' | dataset_gen == 'relationship'")
    no_patterns = no_patterns.assign(is_empty=no_patterns["pred_response"] == no_patterns["response"])
    no_patterns = no_patterns.groupby(["model", "prompt_strategy"], as_index=False).agg(is_empty=("is_empty", "mean"))
    no_patterns = no_patterns.assign(is_empty_report=(no_patterns["is_empty"] * 100).round(2)).sort_values(
        "is_empty", ascending=False
    )
    logger.info("Precision and Recall (no patterns):\n{results}", results=no_patterns)
    no_patterns.to_csv(ALL_RESULTS / "detection_precision_recall_no_patterns.csv", index=False)
