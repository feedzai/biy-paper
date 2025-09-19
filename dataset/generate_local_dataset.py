import json
from typing import Any

from datasets import Dataset
from gaveta.files import ensure_dir
from gaveta.json import read_json

from constants import IMAGES, METADATA, OUTPUT
from utils import encode_image

if __name__ == "__main__":
    ensure_dir(OUTPUT)

    all_metadata: list[Any] = []

    for metadata in METADATA.glob("*.json"):
        raw_metadata = read_json(metadata)
        instance_id = metadata.stem

        dataset_gen = raw_metadata["dataset_id"].split("+", maxsplit=1)[0]

        final_metadata = {
            "id": instance_id,
            "image": encode_image(IMAGES / f"{instance_id}.png"),
            "dataset_gen": dataset_gen,
            "dataset_id": raw_metadata["dataset_id"],
            "chart_design": raw_metadata["chart_design"],
            "scale_factor": raw_metadata["scale_factor"],
            "cluster_count": len(raw_metadata["bbox"]),
            "cluster_bboxes": json.dumps(raw_metadata["bbox"], ensure_ascii=False),
            "cluster_points": json.dumps(raw_metadata["centroid"], ensure_ascii=False),
            "outlier_count": len(raw_metadata["outlier"]),
            "outlier_points": json.dumps(raw_metadata["outlier"], ensure_ascii=False),
        }

        all_metadata.append(final_metadata)

    dataset = Dataset.from_list(all_metadata)

    dataset.to_parquet(OUTPUT / "dataset.parquet")
