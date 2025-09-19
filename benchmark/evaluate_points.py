import json
from pathlib import Path

import pandas as pd
from loguru import logger

from constants import ALL_RESULTS
from utils import compute_point_metrics, generate_grouped_bar_chart


def compute_metrics(single_result: pd.Series) -> pd.Series:
    pred_points = json.loads(single_result["pred_response"])
    target_points = [[int(coordinate) for coordinate in point] for point in json.loads(single_result["response"])]

    point_metrics_10 = compute_point_metrics(pred_points, target_points, distance_threshold=10)
    point_metrics_20 = compute_point_metrics(pred_points, target_points, distance_threshold=20)

    return pd.Series(
        {
            "model": single_result["model"],
            "prompt_id": single_result["prompt_id"],
            "prompt_strategy": single_result["prompt_strategy"],
            "example_id": single_result["example_id"],
            "chart_design": single_result["chart_design"],
            "dataset_gen": single_result["dataset_gen"],
            **point_metrics_10,
            **point_metrics_20,
        }
    )


if __name__ == "__main__":
    results = pd.read_parquet(ALL_RESULTS / "results.parquet", engine="pyarrow")
    results = results.dropna(subset=["pred_response"]).query(
        "(prompt_id == 'cluster_points') | (prompt_id == 'outlier_points')"
    )

    metrics = results.apply(compute_metrics, axis=1)

    overall = metrics.groupby(["prompt_id", "model"], as_index=False).mean(numeric_only=True)
    logger.info("Overall:\n{results}", results=overall)

    good_examples = (
        metrics.query("(prompt_id == 'outlier_points') & (dataset_gen == 'single_gaussian_blob_outliers')")
        .sort_values(by=["recall_10"], ascending=False)
        .head(5)
    )
    logger.info("Good examples:\n{examples}", examples=good_examples)

    for good_example in good_examples.itertuples(index=False):
        logger.info(
            Path("results/open_ai/validation/outlier_points")
            / f"{good_example.model}+{good_example.prompt_strategy}+{good_example.example_id}.png"
        )

    bad_examples = (
        metrics.query("(prompt_id == 'outlier_points') & (dataset_gen == 'single_gaussian_blob_outliers')")
        .sort_values(by=["precision_10"], ascending=False)
        .tail(5)
    )
    logger.info("Bad examples:\n{examples}", examples=bad_examples)

    for bad_example in bad_examples.itertuples(index=False):
        logger.info(
            Path("results/open_ai/validation/outlier_points")
            / f"{bad_example.model}+{bad_example.prompt_strategy}+{bad_example.example_id}.png"
        )

    prompt_strategy = (
        metrics.groupby(["prompt_id", "model", "prompt_strategy"], as_index=False)
        .mean(numeric_only=True)
        .sort_values(by=["prompt_id", "precision_10"], ascending=False)
    )
    logger.info("Precision and Recall:\n{results}", results=prompt_strategy)
    prompt_strategy.to_csv(ALL_RESULTS / "identification_precision_recall.csv", index=False)

    generate_grouped_bar_chart(prompt_strategy, "cluster_points", "precision_10")
    generate_grouped_bar_chart(prompt_strategy, "cluster_points", "recall_10")

    generate_grouped_bar_chart(prompt_strategy, "outlier_points", "precision_10")
    generate_grouped_bar_chart(prompt_strategy, "outlier_points", "recall_10")

    chart_design = metrics.groupby(["prompt_id", "model", "prompt_strategy", "chart_design"], as_index=False).mean(
        numeric_only=True
    )
    logger.info("Precision and Recall (chart design):\n{results}", results=chart_design)
    chart_design.to_csv(ALL_RESULTS / "identification_precision_recall_chart_design.csv", index=False)

    no_patterns = results.query("dataset_gen == 'random' | dataset_gen == 'relationship'")
    no_patterns = no_patterns.assign(is_empty=no_patterns["pred_response"] == no_patterns["response"])

    logger.info(
        no_patterns.loc[no_patterns["pred_response"] != no_patterns["response"], "pred_response"].value_counts(
            dropna=False
        )
    )

    no_patterns = no_patterns.groupby(["prompt_id", "model", "prompt_strategy"], as_index=False).agg(
        is_empty=("is_empty", "mean")
    )
    no_patterns = no_patterns.assign(is_empty_report=(no_patterns["is_empty"] * 100).round(2)).sort_values(
        ["prompt_id", "is_empty"], ascending=False
    )
    logger.info("Precision and Recall (no patterns):\n{results}", results=no_patterns)
    no_patterns.to_csv(ALL_RESULTS / "identification_precision_recall_no_patterns.csv", index=False)

    with_outliers = (
        metrics.query("dataset_gen == 'single_gaussian_blob_outliers'")
        .groupby(["prompt_id", "model", "prompt_strategy"], as_index=False)
        .mean(numeric_only=True)
    )
    logger.info("Precision and Recall (single_gaussian_blob_outliers):\n{results}", results=with_outliers)
    with_outliers.to_csv(ALL_RESULTS / "identification_precision_recall_with_outliers.csv", index=False)
