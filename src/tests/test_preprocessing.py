import numpy as np
import pytest
from medxai.preprocessing import crop_roi

def test_crop_roi_integer_coords():
    image = np.zeros((256, 256, 3))
    bbox = [10, 20, 100, 150]
    crop = crop_roi(image, bbox)
    assert crop.shape == (90, 130, 3)

def test_crop_roi_float_coords():
    image = np.zeros((256, 256, 3))
    bbox = [10.5, 20.2, 100.7, 150.9]
    crop = crop_roi(image, bbox)
    assert crop.shape == (90, 131, 3)  # Redondeo: 11 a 101, 20 a 151

def test_crop_roi_out_of_bounds():
    image = np.zeros((256, 256, 3))
    bbox = [-5, -5, 300, 300]
    crop = crop_roi(image, bbox)
    assert crop.shape == (256, 256, 3)  # Clamp a los límites de la imagen

def test_crop_roi_invalid_bbox():
    image = np.zeros((256, 256, 3))
    with pytest.raises(ValueError):
        crop_roi(image, [10, 20, 30])  # Solo 3 elementos

def test_crop_roi_ymin_greater_than_ymax():
    image = np.zeros((256, 256, 3))
    with pytest.raises(ValueError):
        crop_roi(image, [100, 20, 10, 150])  # ymin > ymax