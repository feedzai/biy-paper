import json

import pandas as pd
from loguru import logger

from constants import ALL_RESULTS


def extract_count(response: str) -> int:
    try:
        return int(response)
    except ValueError:
        parsed_response = json.loads(response)
        return len(parsed_response)


if __name__ == "__main__":
    results = pd.read_parquet(ALL_RESULTS / "results.parquet", engine="pyarrow").dropna(subset=["pred_response"])

    results = results.assign(count=results["pred_response"].apply(extract_count))

    wide_results = (
        results.reset_index()
        .pivot_table(index=["example_id", "model", "prompt_strategy"], columns="prompt_id", values="count")
        .reset_index()
    )
    logger.info("Wide results:\n{results}", results=wide_results)

    clusters = wide_results[["model", "prompt_strategy", "cluster_bboxes", "cluster_count", "cluster_points"]].dropna(
        subset=["cluster_bboxes", "cluster_count", "cluster_points"]
    )
    clusters = clusters.assign(
        is_count_equal=clusters[["cluster_bboxes", "cluster_count", "cluster_points"]].nunique(axis=1) == 1
    )
    logger.info(clusters["is_count_equal"].mean())

    clusters = clusters.groupby(["model", "prompt_strategy"], as_index=False).agg(
        is_count_equal=("is_count_equal", "mean")
    )
    clusters = clusters.assign(is_count_equal_report=(clusters["is_count_equal"] * 100).round(2)).sort_values(
        ["is_count_equal"], ascending=False
    )
    logger.info("Clusters:\n{results}", results=clusters)
    clusters.to_csv(ALL_RESULTS / "counting_consistency_clusters.csv", index=False)

    outliers = wide_results[["model", "prompt_strategy", "outlier_count", "outlier_points"]].dropna(
        subset=["outlier_count", "outlier_points"]
    )
    outliers = outliers.assign(is_count_equal=outliers[["outlier_count", "outlier_points"]].nunique(axis=1) == 1)
    logger.info(outliers["is_count_equal"].mean())

    outliers = outliers.groupby(["model", "prompt_strategy"], as_index=False).agg(
        is_count_equal=("is_count_equal", "mean")
    )
    outliers = outliers.assign(is_count_equal_report=(outliers["is_count_equal"] * 100).round(2)).sort_values(
        ["is_count_equal"], ascending=False
    )
    logger.info("Outliers:\n{results}", results=outliers)
    outliers.to_csv(ALL_RESULTS / "counting_consistency_outliers.csv", index=False)

    logger.info(clusters.shape[0])
    logger.info((clusters["is_count_equal"] < 0.5).sum())
    logger.info((outliers["is_count_equal"] < 0.5).sum())
