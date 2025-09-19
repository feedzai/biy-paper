import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import expit

from constants import ALL_RESULTS

if __name__ == "__main__":
    results = pd.read_parquet(ALL_RESULTS / "results.parquet", engine="pyarrow")
    results = results.dropna(subset=["pred_response"]).query("prompt_id == 'cluster_count'")

    results = results.assign(is_correct=(results["pred_response"] == results["response"]).astype(int))

    model = smf.logit(formula="is_correct ~ C(chart_design, Treatment('default'))", data=results).fit()
    print(model.summary())
    # print(np.exp(model.params))

    with (ALL_RESULTS / "logistic_regression_chart_design.html").open(mode="w", encoding="utf-8") as f:
        f.write(model.summary().as_html())

    # Source: https://heds.nz/posts/logistic-regression-python/
    conf = model.conf_int()
    conf.columns = ["Lower CI", "Upper CI"]
    conf["OR"] = model.params
    print(np.exp(conf))

    intercept = 0.7055
    print("Baseline:", expit([intercept]))
    print("half_opacity:", expit([intercept + 0.1114]))
    print("16_9:", expit([intercept - 0.1104]))
    print("21_9:", expit([intercept - 0.2017]))
    print("random_colors:", expit([intercept - 0.2155]))

    # model = smf.logit(formula="is_correct ~ C(model, Treatment('gpt-4.1-2025-04-14'))", data=results).fit()
    # print(model.summary())

    # model = smf.logit(
    #     formula="is_correct ~ C(chart_design, Treatment('default')) + C(model, Treatment('gpt-4.1-2025-04-14'))",
    #     data=results,
    # ).fit()
    # print(model.summary())

    # interaction_model = smf.logit(
    #     formula="is_correct ~ C(chart_design, Treatment('default')) : C(model)", data=results
    # ).fit()
    # print(interaction_model.summary())

    # model = smf.logit(
    #     formula="is_correct ~ C(chart_design, Treatment('default')) + C(prompt_strategy, Treatment('zero_shot'))",
    #     data=results,
    # ).fit()
    # print(model.summary())

    # interaction_model = smf.logit(
    #     formula="is_correct ~ C(chart_design, Treatment('default')) : C(prompt_strategy, Treatment('zero_shot'))",
    #     data=results,
    # ).fit()
    # print(interaction_model.summary())
