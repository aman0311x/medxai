from typing import Tuple
import cv2
import numpy as np

def clahe(image: np.ndarray, clip_limit: float = 2.0, tile_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Applies Adaptive Histogram Equalization safely handling multi-channel images."""
    if not isinstance(image, np.ndarray):
        raise TypeError("Input image must be a numpy array")
    
    if len(image.shape) == 3:
        # Convert to LAB, apply CLAHE to L channel, convert back
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        cl = clahe_obj.apply(l_channel)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        return clahe_obj.apply(image)

def normalize(image: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """Min-Max normalization safeguarding against zero variance."""
    img_float = image.astype(np.float32)
    img_min, img_max = img_float.min(), img_float.max()
    if img_max == img_min:
        return np.zeros_like(img_float)
    return (img_float - img_min) / (img_max - img_min) * (max_val - min_val) + min_val

def resize(image: np.ndarray, size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Resizes image using anti-aliasing interpolation for high fidelity."""
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def roi_crop(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Crops the region of interest based on mask bounding boxes."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return image  # Return original if mask is empty
    y_min, x_min = coords.min(axis=0)[:2]
    y_max, x_max = coords.max(axis=0)[:2]
    return image[y_min:y_max+1, x_min:x_max+1]