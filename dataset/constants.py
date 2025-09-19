from pathlib import Path

INPUT = Path("input")
OUTPUT = Path("output")
DATASETS = INPUT / "datasets"
IMAGES = INPUT / "images"
METADATA = INPUT / "metadata"
VALIDATION = INPUT / "validation"

SCALE_FACTORS = [0.5, 1, 2]

# https://vega.github.io/vega-lite/docs/spec.html#config
DEFAULT_WIDTH = 300
DEFAULT_HEIGHT = 300
DEFAULT_PADDING = 5
DEFAULT_OPACITY = 0.7

# https://vega.github.io/vega-lite/docs/point.html
DEFAULT_POINT_SIZE = 30

NONE_LABEL = -1

# Source: https://github.com/vega/vega-themes/blob/v3.0.0/src/theme-dark.ts
DARK_THEME_CONFIG = {
    "background": "#333",
    "view": {
        "stroke": "#888",
    },
    "title": {
        "color": "#fff",
        "subtitleColor": "#fff",
    },
    "style": {
        "guide-label": {
            "fill": "#fff",
        },
        "guide-title": {
            "fill": "#fff",
        },
    },
    "axis": {
        "domainColor": "#fff",
        "gridColor": "#888",
        "tickColor": "#fff",
    },
}

# Source: https://github.com/vega/vega/blob/v6.1.2/packages/vega-scale/src/palettes.js#L92
TABLEAU_10_COLOR_SCHEME: list[str] = [
    "#4c78a8",
    "#f58518",
    "#e45756",
    "#72b7b2",
    "#54a24b",
    "#eeca3b",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
    "#bab0ac",
]
