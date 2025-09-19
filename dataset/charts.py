import json
from collections.abc import Callable

import pandas as pd

from constants import (
    DARK_THEME_CONFIG,
    DEFAULT_OPACITY,
    DEFAULT_PADDING,
    DEFAULT_POINT_SIZE,
    DEFAULT_WIDTH,
    TABLEAU_10_COLOR_SCHEME,
)
from utils import get_height_from_aspect_ratio


def generate_3_4_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
            "config": {
                "view": {
                    "continuousWidth": DEFAULT_WIDTH,
                    "continuousHeight": get_height_from_aspect_ratio(DEFAULT_WIDTH, 3, 4),
                }
            },
        },
        ensure_ascii=False,
    )


def generate_4_3_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
            "config": {
                "view": {
                    "continuousWidth": DEFAULT_WIDTH,
                    "continuousHeight": get_height_from_aspect_ratio(DEFAULT_WIDTH, 4, 3),
                }
            },
        },
        ensure_ascii=False,
    )


def generate_9_16_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
            "config": {
                "view": {
                    "continuousWidth": DEFAULT_WIDTH,
                    "continuousHeight": get_height_from_aspect_ratio(DEFAULT_WIDTH, 9, 16),
                }
            },
        },
        ensure_ascii=False,
    )


def generate_16_9_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
            "config": {
                "view": {
                    "continuousWidth": DEFAULT_WIDTH,
                    "continuousHeight": get_height_from_aspect_ratio(DEFAULT_WIDTH, 16, 9),
                }
            },
        },
        ensure_ascii=False,
    )


def generate_21_9_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
            "config": {
                "view": {
                    "continuousWidth": DEFAULT_WIDTH,
                    "continuousHeight": get_height_from_aspect_ratio(DEFAULT_WIDTH, 21, 9),
                }
            },
        },
        ensure_ascii=False,
    )


def generate_colors_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "color": {
                    "field": "cluster",
                    "type": "nominal",
                    "scale": {
                        "domain": [-1] + list(range(len(TABLEAU_10_COLOR_SCHEME))),
                        "range": ["gray"] + TABLEAU_10_COLOR_SCHEME,
                    },
                    "legend": None,
                },
            },
        },
        ensure_ascii=False,
    )


def generate_dark_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
            "config": DARK_THEME_CONFIG,
        },
        ensure_ascii=False,
    )


def generate_default_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
        },
        ensure_ascii=False,
    )


def generate_full_opacity_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {"type": "point", "filled": True, "opacity": 1},
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
        },
        ensure_ascii=False,
    )


def generate_half_opacity_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {"type": "point", "filled": True, "opacity": DEFAULT_OPACITY / 2},
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
        },
        ensure_ascii=False,
    )


def generate_half_point_size_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
                "size": DEFAULT_POINT_SIZE / 2,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
        },
        ensure_ascii=False,
    )


def generate_points_only_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {"type": "point", "filled": True},
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": None,
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": None,
                    "scale": {"zero": False},
                },
            },
            "config": {"view": {"stroke": None}},
        },
        ensure_ascii=False,
    )


def generate_random_colors_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
            },
            "transform": [
                {"window": [{"op": "row_number", "as": "index"}]},
                {"calculate": "datum.index % 10", "as": "color_index"},
            ],
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "color": {"field": "color_index", "type": "nominal", "scale": {"scheme": "tableau10"}, "legend": None},
            },
        },
        ensure_ascii=False,
    )


def generate_random_shapes_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
            },
            "transform": [
                {"window": [{"op": "row_number", "as": "index"}]},
                {"calculate": "datum.index % 8", "as": "shape_index"},
            ],
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "shape": {"field": "shape_index", "type": "nominal", "legend": None},
            },
        },
        ensure_ascii=False,
    )


def generate_square_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
                "shape": "square",
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
        },
        ensure_ascii=False,
    )


def generate_x2_point_size_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {
                "type": "point",
                "filled": True,
                "size": DEFAULT_POINT_SIZE * 2,
            },
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
        },
        ensure_ascii=False,
    )


def generate_yaxis_only_spec(dataset: pd.DataFrame) -> str:
    return json.dumps(
        {
            "padding": DEFAULT_PADDING,
            "data": {"values": dataset.to_dict(orient="records")},
            "mark": {"type": "point", "filled": True},
            "encoding": {
                "x": {
                    "field": "x",
                    "type": "quantitative",
                    "axis": None,
                    "scale": {"zero": False},
                },
                "y": {
                    "field": "y",
                    "type": "quantitative",
                    "axis": {"title": None},
                    "scale": {"zero": False},
                },
            },
            "config": {"view": {"stroke": None}},
        },
        ensure_ascii=False,
    )


CHART_DESIGNS: dict[str, Callable[[pd.DataFrame], str]] = {
    "3_4": generate_3_4_spec,
    "4_3": generate_4_3_spec,
    "9_16": generate_9_16_spec,
    "16_9": generate_16_9_spec,
    "21_9": generate_21_9_spec,
    "colors": generate_colors_spec,
    "dark": generate_dark_spec,
    "default": generate_default_spec,
    "full_opacity": generate_full_opacity_spec,
    "half_opacity": generate_half_opacity_spec,
    "half_point_size": generate_half_point_size_spec,
    "points_only": generate_points_only_spec,
    "random_colors": generate_random_colors_spec,
    "random_shapes": generate_random_shapes_spec,
    "square": generate_square_spec,
    "x2_point_size": generate_x2_point_size_spec,
    "yaxis_only": generate_yaxis_only_spec,
}
