from typing import Union

import numpy as np

Number = Union[int, float]


def crop_roi(
    image: np.ndarray,
    x: Number,
    y: Number,
    width: Number,
    height: Number,
    clamp: bool = True,
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Image cannot be None or empty")
    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")
    x_int = int(round(float(x)))
    y_int = int(round(float(y)))
    width_int = int(round(float(width)))
    height_int = int(round(float(height)))
    h, w = image.shape[:2]
    x1, y1 = x_int, y_int
    x2, y2 = x_int + width_int, y_int + height_int
    if clamp:
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
    else:
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            raise ValueError(f"ROI out of bounds: {x1}:{x2} x {y1}:{y2} in {w}x{h}")
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid ROI: {x1}:{x2} x {y1}:{y2}")
    return image[y1:y2, x1:x2, :] if len(image.shape) == 3 else image[y1:y2, x1:x2]
