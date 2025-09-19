import json

import lap
import pandas as pd
import torch
from gaveta.files import ensure_dir
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from torchvision.ops import box_iou

from constants import ALL_RESULTS, OPEN_AI_VAL
from utils import parse_custom_id

EXAMPLES = [
    "gpt-4o-2024-08-06+outlier_points+one_shot+single_gaussian_blob_outliers+3_0_005_0+21_9+2",
    "o3-2025-04-16+cluster_bboxes+zero_shot+gaussian_blobs+100_3_0_4+random_colors+2",
]

OUTLINE_COLOR = "white"


def concatenate_images_horizontal(img1: Image.Image, img2: Image.Image) -> Image.Image:
    width1, height1 = img1.size
    width2, height2 = img2.size

    target_height = min(height1, height2)

    # Resize if needed
    if height1 != target_height:
        aspect_ratio1 = width1 / height1
        new_width1 = int(target_height * aspect_ratio1)
        img1 = img1.resize((new_width1, target_height), Image.Resampling.LANCZOS)

    if height2 != target_height:
        aspect_ratio2 = width2 / height2
        new_width2 = int(target_height * aspect_ratio2)
        img2 = img2.resize((new_width2, target_height), Image.Resampling.LANCZOS)

    final_width1, _ = img1.size
    final_width2, _ = img2.size
    combined_width = final_width1 + final_width2

    combined_image = Image.new("RGB", (combined_width, target_height))
    combined_image.paste(img1, (0, 0))
    combined_image.paste(img2, (final_width1, 0))

    return combined_image


if __name__ == "__main__":
    ensure_dir(OPEN_AI_VAL)

    results = pd.read_parquet(ALL_RESULTS / "results.parquet", engine="pyarrow")

    for example in EXAMPLES:
        custom_id = parse_custom_id(example)

        row = results[
            (results["model"] == custom_id["model"])
            & (results["prompt_id"] == custom_id["prompt_id"])
            & (results["prompt_strategy"] == custom_id["prompt_strategy"])
            & (results["example_id"] == custom_id["example_id"])
        ].iloc[0]

        if custom_id["prompt_id"] == "outlier_points":
            with Image.open(OPEN_AI_VAL / f"{custom_id['example_id'].removesuffix('2')}8.png") as im:
                draw = ImageDraw.Draw(im)

                pred_points = json.loads(row["pred_response"])
                target_points = json.loads(row["response"])

                for point in target_points:
                    scaled_point = [value * 4 for value in point]
                    draw.circle(scaled_point, radius=10 * 4, fill="#0f172a", outline=OUTLINE_COLOR, width=2 * 4)

                for point in pred_points:
                    scaled_point = [value * 4 for value in point]
                    x, y = scaled_point
                    half_size = (15 / 2) * 4
                    width = 5 * 4
                    outline_width = 1 * 4

                    draw.line(
                        [
                            (x - half_size - outline_width, y - half_size - outline_width),
                            (x + half_size + outline_width, y + half_size + outline_width),
                        ],
                        fill=OUTLINE_COLOR,
                        width=width + (outline_width * 2),
                    )
                    draw.line(
                        [
                            (x + half_size + outline_width, y - half_size - outline_width),
                            (x - half_size - outline_width, y + half_size + outline_width),
                        ],
                        fill=OUTLINE_COLOR,
                        width=width + (outline_width * 2),
                    )

                    draw.line(
                        [(x - half_size, y - half_size), (x + half_size, y + half_size)], fill="#be185d", width=width
                    )
                    draw.line(
                        [(x + half_size, y - half_size), (x - half_size, y + half_size)], fill="#be185d", width=width
                    )

                im.save(OPEN_AI_VAL / f"{example}.png")
        elif custom_id["prompt_id"] == "cluster_bboxes":
            with Image.open(OPEN_AI_VAL / f"{custom_id['example_id'].removesuffix('2')}8.png") as im:
                new = Image.new("RGBA", im.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(new)

                pred_bboxes = json.loads(row["pred_response"])
                target_bboxes = json.loads(row["response"])

                iou_matrix = box_iou(
                    torch.tensor(pred_bboxes),
                    torch.tensor(target_bboxes),
                )
                cost_matrix = (1 - iou_matrix).numpy()

                _, _, target_indices = lap.lapjv(cost_matrix, extend_cost=True)

                for target_idx, pred_idx in enumerate(target_indices):
                    scaled_target_bbox = [value * 4 for value in target_bboxes[target_idx]]
                    scaled_pred_bbox = [value * 4 for value in pred_bboxes[pred_idx]]

                    x_min, y_min, x_max, y_max = scaled_target_bbox
                    target_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

                    x_min, y_min, x_max, y_max = scaled_pred_bbox
                    pred_polygon = Polygon([(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)])

                    intersection = target_polygon.intersection(pred_polygon)

                    # https://gist.github.com/lopspower/03fb1cc0ac9f32ef38f4
                    draw.rectangle(intersection.bounds, fill="#d1fae5" + "80")

                    draw.rectangle(scaled_target_bbox, outline="#0f172a", width=4 * 4)
                    draw.rectangle(scaled_pred_bbox, outline="#0f766e", width=4 * 4)

                Image.alpha_composite(im.convert("RGBA"), new).save(OPEN_AI_VAL / f"{example}.png")

    with (
        Image.open(
            OPEN_AI_VAL / "o3-2025-04-16+cluster_bboxes+zero_shot+gaussian_blobs+100_3_0_4+random_colors+2.png"
        ) as im1,
        Image.open(
            OPEN_AI_VAL / "gpt-4o-2024-08-06+outlier_points+one_shot+single_gaussian_blob_outliers+3_0_005_0+21_9+2.png"
        ) as im2,
    ):
        concatenate_images_horizontal(im1, im2).save(ALL_RESULTS / "examples.png")
