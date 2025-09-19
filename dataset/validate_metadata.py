from gaveta.json import read_json
from loguru import logger
from PIL import Image, ImageDraw

from constants import IMAGES, METADATA, VALIDATION
from utils import ensure_clean_dir

if __name__ == "__main__":
    ensure_clean_dir(VALIDATION)

    for metadata in METADATA.glob("**/*.json"):
        dataset = metadata.stem
        image = IMAGES / f"{dataset}.png"

        data = read_json(metadata)

        with Image.open(image) as im:
            val_im = im.convert("RGB") if im.mode == "P" else im

            draw = ImageDraw.Draw(val_im)

            for bbox in data["bbox"]:
                draw.rectangle(bbox, outline="red")

            for centroid in data["centroid"]:
                draw.circle(centroid, radius=5, fill="red", outline="white")

            for outlier in data["outlier"]:
                draw.circle(outlier, radius=5, fill="green", outline="white")

            val_im.save(VALIDATION / f"{dataset}.png")
            logger.info("{dataset} validation image generated", dataset=dataset)
