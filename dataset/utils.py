import base64
import json
import mimetypes
import shutil
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gaveta.files import ensure_dir


def ensure_clean_dir(folder: Path) -> None:
    try:
        shutil.rmtree(folder)
        ensure_dir(folder)
    except FileNotFoundError:
        ensure_dir(folder)


def get_height_from_aspect_ratio(width: float, aspect_ratio_width: int, aspect_ratio_height: int) -> float:
    return (width / aspect_ratio_width) * aspect_ratio_height


def rotate_coordinates_around_center(df: pd.DataFrame, angle_degrees: int) -> pd.DataFrame:
    angle_rad = np.radians(angle_degrees)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    center_x = (df["x"].min() + df["x"].max()) / 2
    center_y = (df["y"].min() + df["y"].max()) / 2

    translated_x = df["x"] - center_x
    translated_y = df["y"] - center_y

    rotated_x = translated_x * cos_angle - translated_y * sin_angle
    rotated_y = translated_x * sin_angle + translated_y * cos_angle

    df["x"] = rotated_x + center_x
    df["y"] = rotated_y + center_y

    return df


def write_jsonl(data: Iterable[Any], output_path: Path) -> None:
    with output_path.open(mode="w", encoding="utf-8") as f:
        for datum in data:
            f.write(json.dumps(datum, ensure_ascii=False) + "\n")


def encode_image(image: Path) -> str:
    media_type = mimetypes.types_map[image.suffix]
    base64_image = base64.b64encode(image.read_bytes()).decode("utf-8")

    return f"data:{media_type};base64,{base64_image}"
