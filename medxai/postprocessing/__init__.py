import numpy as np
from skimage.morphology import remove_small_objects


def clean_noisy_mask(mask: np.ndarray, min_size: int = 64) -> np.ndarray:
    """
    Removes small disconnected false-positive artifacts from a predicted binary mask.
    Highly useful in medical imaging to clean up network predictions.

    Args:
        mask (np.ndarray): Binary segmentation mask (boolean or 0/1 integers).
        min_size (int): Minimum size (in pixels) of the objects to keep.

    Returns:
        np.ndarray: Cleaned binary mask.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError("Input mask must be a NumPy array.")

    is_boolean = mask.dtype == bool
    mask_bool = mask.astype(bool)

    cleaned = remove_small_objects(mask_bool, min_size=min_size)

    return cleaned if is_boolean else cleaned.astype(mask.dtype)
