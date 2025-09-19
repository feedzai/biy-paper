from gaveta.files import ensure_dir
from PIL import Image, ImageDraw

from constants import IMAGES, VALIDATION

R = 10

if __name__ == "__main__":
    ensure_dir(VALIDATION)

    image = next(IMAGES.glob("*+2.png"))

    with Image.open(image) as im:
        width, height = im.size
        val_im = im.convert("RGB") if im.mode == "P" else im

        draw = ImageDraw.Draw(val_im)

        # "(...) normalized coordinates (x1, y1, x2, y2), where (0, 0) is the top-left corner and (1000, 1000) is the bottom-right corner of the image (...)"

        draw.circle((0, 0), radius=R, fill="red", outline="white")
        draw.text((0 + R, 0 + R), "(0, 0)", fill="red", stroke_width=1, stroke_fill="white", font_size=16)

        draw.circle((200, 200), radius=R, fill="green", outline="white")
        draw.text((200 + R, 200 + R), "(200, 200)", fill="green", stroke_width=1, stroke_fill="white", font_size=16)

        draw.circle((width, height), radius=R, fill="blue", outline="white")
        draw.text(
            (width - R, height - R),
            f"({width}, {height})",
            fill="blue",
            stroke_width=1,
            stroke_fill="white",
            font_size=16,
            anchor="rs",
        )

        val_im.save(VALIDATION / "_coordinates.png")
