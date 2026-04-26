import os
import cv2
import numpy as np
from pathlib import Path


def load_image(path):

    img=cv2.imread(str(path))

    if img is None:
        raise ValueError(
            f"Cannot read {path}"
        )

    img=cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    return img


def save_image(path,image,quality=85):

    bgr=cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    ext=Path(path).suffix.lower()

    if ext in [".jpg",".jpeg"]:

        cv2.imwrite(
            str(path),
            bgr,
            [
              cv2.IMWRITE_JPEG_QUALITY,
              quality
            ]
        )
    else:
        cv2.imwrite(str(path),bgr)


def file_size(path):
    return os.path.getsize(path)/(1024*1024)


def make_output_name(
        input_path,
        clusters
):
    p=Path(input_path)

    return str(
        p.parent.parent /
        "output" /
        "compressed" /
        f"{p.stem}_{clusters}colors.jpg"
    )
