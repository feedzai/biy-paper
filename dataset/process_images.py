import shutil
import subprocess

import humanize
from loguru import logger

from constants import IMAGES, INPUT

if __name__ == "__main__":
    total_size = sum([scatterplot.stat().st_size for scatterplot in IMAGES.glob("*.png")])
    logger.info(
        "Total file size (before Oxipng): {total_size}",
        total_size=humanize.naturalsize(total_size),
    )

    subprocess.run(
        ["oxipng", "-o", "4", "--strip", "safe", "--alpha", "--recursive", str(IMAGES)],
        check=False,
    )

    total_size = sum([scatterplot.stat().st_size for scatterplot in IMAGES.glob("*.png")])
    logger.info(
        "Total file size (after Oxipng): {total_size}",
        total_size=humanize.naturalsize(total_size),
    )

    shutil.make_archive(
        base_name=str(INPUT / IMAGES.stem),
        format="zip",
        root_dir=INPUT,
        base_dir=IMAGES.stem,
    )
