import json
from urllib.request import urlopen

import pandas as pd
from PIL import Image, ImageDraw

from constants import INPUT
from utils import ensure_clean_dir

FIGMA = INPUT / "figma"

IDS = [
    "gaussian_blobs+2000_6_0_6+full_opacity+2",
    "gaussian_blobs_noise+100_4_0_2+21_9+2",
    "single_gaussian_blob_outliers+3_0_004_90+dark+2",
    "gaussian_blobs+2000_3_0_4+half_opacity+2",
    "relationship+exponential_0_2+9_16+2",
    "gaussian_blobs_noise+1000_2_0_6+colors+2",
]


def read_dataset() -> pd.DataFrame:
    chart_design_ignore = ["random_shapes", "square"]

    return (
        pd.read_parquet(INPUT / "dataset.parquet", engine="pyarrow")
        .query("dataset_gen != 'shapes' & scale_factor == 2 & chart_design not in @chart_design_ignore")
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    ensure_clean_dir(FIGMA)

    dataset = read_dataset()

    # chart_designs = pd.concat(
    #     [
    #         dataset.groupby(["chart_design", "dataset_gen"]).sample(n=1, random_state=1),
    #         dataset.groupby(["chart_design", "dataset_gen"]).sample(n=1, random_state=42),
    #     ],
    #     ignore_index=True,
    # )

    for chart_design in dataset.itertuples(index=False):
        if chart_design.id in IDS:
            with (
                urlopen(str(chart_design.image)) as input_f,
                (FIGMA / f"{chart_design.id}.png").open(mode="wb") as output_f,
            ):
                output_f.write(input_f.read())

    for image in FIGMA.glob("*.png"):
        parsed_id = image.stem.split("+")

        color = "#e2e8f0" if "dark" in parsed_id else "#0f172a"

        with Image.open(image) as im:
            draw = ImageDraw.Draw(im)

            cluster_bboxes = json.loads(dataset.loc[dataset["id"] == image.stem, "cluster_bboxes"].item())
            cluster_points = json.loads(dataset.loc[dataset["id"] == image.stem, "cluster_points"].item())
            outlier_points = json.loads(dataset.loc[dataset["id"] == image.stem, "outlier_points"].item())

            for cluster_bbox in cluster_bboxes:
                draw.rectangle(list(cluster_bbox), outline=color, width=4)

            for cluster_point in cluster_points:
                draw.circle(list(cluster_point), radius=10, fill=color, outline="white", width=2)

            for outlier_point in outlier_points:
                draw.circle(list(outlier_point), radius=10, fill=color, outline="white", width=2)

            im.save(FIGMA / f"{image.stem}_val.png")
