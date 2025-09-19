import json
import threading
from string import Template
from urllib.request import urlopen

import numpy as np
import pandas as pd
from fastcore.xml import FT
from fasthtml.common import Div, Main, Script, fast_app, serve
from gaveta.json import write_json
from playwright.sync_api import ConsoleMessage, sync_playwright
from sklearn.neighbors import NearestCentroid

from charts import CHART_DESIGNS
from constants import DATASETS, DEFAULT_PADDING, IMAGES, METADATA, SCALE_FACTORS
from utils import ensure_clean_dir

VL_SCRIPT = Template("""
vegaEmbed("#chart", $vl_spec).then((result) => {
  const [xOrigin, yOrigin] = result.view.origin();
  const padding = $padding;

  const xOffset = xOrigin + padding;
  const yOffset = yOrigin + padding;

  // const [xMin, xMax] = result.view.scale("x").range();
  // const [yMin, yMax] = result.view.scale("y").range();

  const boundingBoxes = $bounding_boxes;
  const centroids = $centroids;

  const outliers = $outliers;

  const scaledBoundingBoxes = boundingBoxes.map((boundingBox) => {
    const xMinPx = result.view.scale("x")(boundingBox.x_min);
    const xMaxPx = result.view.scale("x")(boundingBox.x_max);

    const yMaxPx = result.view.scale("y")(boundingBox.y_min);
    const yMinPx = result.view.scale("y")(boundingBox.y_max);

    return [
      (xMinPx + xOffset) * $scale_factor,
      (yMinPx + yOffset) * $scale_factor,
      (xMaxPx + xOffset) * $scale_factor,
      (yMaxPx + yOffset) * $scale_factor,
    ];
  });

  const scaledCentroids = centroids.map((centroid) => {
    const xPx = result.view.scale("x")(centroid[0]);
    const yPx = result.view.scale("y")(centroid[1]);

    return [(xPx + xOffset) * $scale_factor, (yPx + yOffset) * $scale_factor];
  });

  const scaledOutliers = outliers.map((outlier) => {
    const xPx = result.view.scale("x")(outlier[0]);
    const yPx = result.view.scale("y")(outlier[1]);

    return [(xPx + xOffset) * $scale_factor, (yPx + yOffset) * $scale_factor];
  });

  result.view.toImageURL("png", $scale_factor).then((data_url) => {
    console.log(data_url, {
      bbox: scaledBoundingBoxes,
      centroid: scaledCentroids,
      outlier: scaledOutliers,
    });
  });
});
""")


def compute_cluster_bounding_boxes(dataset: pd.DataFrame) -> str:
    cluster_dataset = dataset.query("cluster > -1")

    bounding_boxes = cluster_dataset.groupby("cluster", as_index=False).agg(
        x_min=("x", "min"), x_max=("x", "max"), y_min=("y", "min"), y_max=("y", "max")
    )

    return bounding_boxes.to_json(orient="records")


def compute_cluster_centroids(dataset: pd.DataFrame) -> str:
    cluster_dataset = dataset.query("cluster > -1")

    n_clusters = cluster_dataset["cluster"].nunique()

    if n_clusters == 0:
        return json.dumps([], ensure_ascii=False)
    if n_clusters == 1:
        return json.dumps([np.mean(cluster_dataset[["x", "y"]], axis=0).tolist()], ensure_ascii=False)

    clf = NearestCentroid()
    clf.fit(cluster_dataset[["x", "y"]], cluster_dataset["cluster"])

    return json.dumps(clf.centroids_.tolist(), ensure_ascii=False)


def compute_outliers(dataset: pd.DataFrame) -> str:
    outlier_dataset = dataset.query("outlier > -1")

    return json.dumps(outlier_dataset[["x", "y"]].to_dict(orient="split")["data"], ensure_ascii=False)


app, _ = fast_app(
    pico=False,
    live=False,
    hdrs=(
        Script(src="https://cdn.jsdelivr.net/npm/vega@6.1.2"),
        Script(src="https://cdn.jsdelivr.net/npm/vega-lite@6.1.0"),
        Script(src="https://cdn.jsdelivr.net/npm/vega-embed@7.0.2"),
    ),
)


@app.get("/{dataset_id}/{chart_design}/{scale_factor}")  # type: ignore[misc]
def home(dataset_id: str, chart_design: str, scale_factor: float) -> FT:
    dataset = pd.read_json(DATASETS / f"{dataset_id}.json")

    bounding_boxes = compute_cluster_bounding_boxes(dataset)
    centroids = compute_cluster_centroids(dataset)

    outliers = compute_outliers(dataset)

    vl_spec = CHART_DESIGNS[chart_design](dataset)

    return Main(
        Div(id="chart"),
        Script(
            VL_SCRIPT.safe_substitute(
                vl_spec=vl_spec,
                bounding_boxes=bounding_boxes,
                centroids=centroids,
                outliers=outliers,
                scale_factor=scale_factor,
                padding=DEFAULT_PADDING,
            )
        ),
    )


def handle_msg(msg: ConsoleMessage, dataset_id: str, chart_design: str, scale_factor: float) -> None:
    image = msg.args[0].json_value()
    metadata = msg.args[1].json_value()

    extra_metadata = {
        "dataset_id": dataset_id,
        "chart_design": chart_design,
        "scale_factor": scale_factor,
    }

    final_id = f"{dataset_id}+{chart_design}+{str(scale_factor).replace('.', '_')}"

    with (
        urlopen(image) as i,
        (IMAGES / f"{final_id}.png").open(mode="wb") as f,
    ):
        f.write(i.read())

    write_json({**metadata, **extra_metadata}, METADATA / f"{final_id}.json")


if __name__ == "__main__":
    ensure_clean_dir(IMAGES)
    ensure_clean_dir(METADATA)

    dataset_ids = [dataset.stem for dataset in DATASETS.glob("*.json")]

    server_thread = threading.Thread(target=lambda: serve(reload=False), daemon=True)
    server_thread.start()

    with sync_playwright() as playwright:
        chromium = playwright.chromium
        browser = chromium.launch(headless=False)
        page = browser.new_page()

        for dataset_id in dataset_ids:
            for chart_design in CHART_DESIGNS:
                for scale_factor in SCALE_FACTORS:
                    with page.expect_console_message() as msg_info:
                        page.goto(f"http://localhost:5001/{dataset_id}/{chart_design}/{scale_factor}")
                        msg = msg_info.value

                    handle_msg(msg, dataset_id, chart_design, scale_factor)

        browser.close()
