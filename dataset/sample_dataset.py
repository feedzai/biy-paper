from pathlib import Path

import pandas as pd

DATASET = Path("output/dataset.parquet")
DATASET_ID = "gaussian_blobs_noise+500_3_0_4"
SHORT_DESC = (
    "A scatter plot with three well-separated clusters and several randomly scattered points throughout the chart."
)
SCALE_FACTOR = 2

if __name__ == "__main__":
    dataset = pd.read_parquet(DATASET, engine="pyarrow")
    sample = dataset.query("dataset_id == @DATASET_ID & scale_factor == @SCALE_FACTOR")

    sample = sample.assign(
        image=f"![{SHORT_DESC}]"
        + "(dataset/input/images/"
        + sample["dataset_id"]
        + "+"
        + sample["chart_design"]
        + "+"
        + sample["scale_factor"].astype(int).astype(str)
        + ".png)"
    )

    print(sample.to_markdown(index=False, tablefmt="github"))
