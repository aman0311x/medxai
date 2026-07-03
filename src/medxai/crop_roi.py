"""
ROI cropping utility for medical images.
Handles floating-point coordinates safely.
"""

from typing import Tuple, Union

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
    """
    Crop a region of interest (ROI) from an image with safe coordinate handling.

    Args:
        image: Input image as numpy array (H, W) or (H, W, C)
        x: X-coordinate (can be float)
        y: Y-coordinate (can be float)
        width: Width of ROI (can be float)
        height: Height of ROI (can be float)
        clamp: Whether to clamp coordinates to image bounds

    Returns:
        Cropped image as numpy array
    """
    if image is None or image.size == 0:
        raise ValueError("Image cannot be None or empty")

    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive")

    # Convert to integers safely
    x_int = int(round(float(x)))
    y_int = int(round(float(y)))
    width_int = int(round(float(width)))
    height_int = int(round(float(height)))

    # Get image dimensions
    h, w = image.shape[:2]

    # Calculate boundaries
    x1 = x_int
    y1 = y_int
    x2 = x_int + width_int
    y2 = y_int + height_int

    # Clamp coordinates if requested
    if clamp:
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
    else:
        # Validate coordinates are within bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            raise ValueError(
                f"ROI coordinates out of bounds: "
                f"x1={x1}, y1={y1}, x2={x2}, y2={y2}, "
                f"image dimensions: {w}x{h}"
            )

    # Ensure we have valid dimensions
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid ROI dimensions: {x1}:{x2} x {y1}:{y2}")

    # Crop the image
    if len(image.shape) == 3:
        return image[y1:y2, x1:x2, :]
    else:
        return image[y1:y2, x1:x2]
