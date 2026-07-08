import numpy as np
import pytest

from medxai.preprocessing import crop_roi


def test_crop_roi_with_float_coordinates():
    """Ensures crop_roi handles float bbox coordinates without raising TypeError."""
    image = np.zeros((256, 256, 3), dtype=np.float32)
    bbox = [10.5, 20.2, 100.7, 150.9]

    crop = crop_roi(image, bbox)

    assert crop.shape == (91, 131, 3)


def test_crop_roi_with_int_coordinates():
    """Ensures crop_roi still works correctly with plain integer coordinates."""
    image = np.zeros((100, 100, 3), dtype=np.float32)
    bbox = [10, 10, 50, 50]

    crop = crop_roi(image, bbox)

    assert crop.shape == (40, 40, 3)


def test_crop_roi_reversed_coordinates():
    """Ensures crop_roi handles reversed (ymax < ymin) coordinates gracefully."""
    image = np.zeros((100, 100, 3), dtype=np.float32)
    bbox = [50, 50, 10, 10]

    crop = crop_roi(image, bbox)

    assert crop.shape == (40, 40, 3)


def test_crop_roi_out_of_bounds_clips_to_image():
    """Ensures crop_roi clips coordinates that exceed image boundaries."""
    image = np.zeros((100, 100, 3), dtype=np.float32)
    bbox = [-10, -10, 200, 200]

    crop = crop_roi(image, bbox)

    assert crop.shape == (100, 100, 3)


def test_crop_roi_invalid_bbox_length_raises():
    """Ensures crop_roi raises ValueError when bbox doesn't have 4 values."""
    image = np.zeros((100, 100, 3), dtype=np.float32)
    bbox = [10, 10, 50]  # only 3 values

    with pytest.raises(ValueError):
        crop_roi(image, bbox)


def test_crop_roi_empty_result_raises():
    """Ensures crop_roi raises ValueError when the resulting crop is empty."""
    image = np.zeros((100, 100, 3), dtype=np.float32)
    bbox = [50, 50, 50, 50]  # zero-size box

    with pytest.raises(ValueError):
        crop_roi(image, bbox)
