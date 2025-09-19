import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error

from constants import ALL_RESULTS
from utils import generate_grouped_bar_chart

if __name__ == "__main__":
    results = pd.read_parquet(ALL_RESULTS / "results.parquet", engine="pyarrow")
    results = results.dropna(subset=["pred_response"]).query(
        "(prompt_id == 'cluster_count') | (prompt_id == 'outlier_count')"
    )

    results = results.assign(is_correct=results["pred_response"] == results["response"])

    prompt_strategy = results.groupby(["prompt_id", "model", "prompt_strategy"], as_index=False).agg(
        accuracy=("is_correct", "mean")
    )

    generate_grouped_bar_chart(prompt_strategy, "cluster_count", "accuracy")
    generate_grouped_bar_chart(prompt_strategy, "outlier_count", "accuracy")

    prompt_strategy = prompt_strategy.assign(accuracy_report=(prompt_strategy["accuracy"] * 100).round(2)).sort_values(
        ["prompt_id", "accuracy"], ascending=False
    )
    logger.info("Accuracy:\n{results}", results=prompt_strategy)
    prompt_strategy.to_csv(ALL_RESULTS / "counting_accuracy.csv", index=False)

    prompt_strategy_mae = (
        results.groupby(["prompt_id", "model", "prompt_strategy"])[["response", "pred_response"]]
        .apply(lambda group: mean_absolute_error(group["response"], group["pred_response"]))
        .reset_index(name="mae")
        .sort_values(["prompt_id", "mae"], ascending=True)
    )
    logger.info("MAE:\n{results}", results=prompt_strategy_mae)
    prompt_strategy_mae.to_csv(ALL_RESULTS / "counting_mae.csv", index=False)

    no_patterns = results.query("dataset_gen == 'random' | dataset_gen == 'relationship'").assign(
        is_zero=results["pred_response"] == results["response"]
    )
    no_patterns = no_patterns.groupby(["prompt_id", "model", "prompt_strategy"], as_index=False).agg(
        is_zero=("is_zero", "mean")
    )
    no_patterns = no_patterns.assign(is_zero_report=(no_patterns["is_zero"] * 100).round(2)).sort_values(
        ["prompt_id", "is_zero"], ascending=False
    )
    logger.info("Accuracy (no patterns):\n{results}", results=no_patterns)
    no_patterns.to_csv(ALL_RESULTS / "counting_accuracy_no_patterns.csv", index=False)

    chart_design = results.groupby(["prompt_id", "model", "prompt_strategy", "chart_design"], as_index=False).agg(
        accuracy=("is_correct", "mean")
    )
    chart_design = chart_design.assign(accuracy_report=(chart_design["accuracy"] * 100).round(2)).sort_values(
        ["prompt_id", "prompt_strategy", "model", "accuracy"], ascending=False
    )
    logger.info("Accuracy (chart design):\n{results}", results=chart_design)
    chart_design.to_csv(ALL_RESULTS / "counting_accuracy_chart_design.csv", index=False)
