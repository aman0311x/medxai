import numpy as np
import torch

from medxai.losses import FocalTverskyLoss
from medxai.models import UNet
from medxai.postprocessing import clean_noisy_mask


def test_unet_forward():
    model = UNet(n_channels=1, n_classes=2)
    # Batch size 1, 1 channel, 64x64 image
    dummy_input = torch.randn(1, 1, 64, 64)
    output = model(dummy_input)
    # Output should be (1, 2, 64, 64)
    assert output.shape == (1, 2, 64, 64)


def test_clean_noisy_mask():
    # Create a blank 50x50 mask
    mask = np.zeros((50, 50), dtype=np.uint8)
    # Add a large valid tumor (10x10 = 100 pixels)
    mask[10:20, 10:20] = 1
    # Add a tiny noise artifact (3x3 = 9 pixels)
    mask[40:43, 40:43] = 1

    # Remove objects smaller than 50 pixels
    cleaned = clean_noisy_mask(mask, min_size=50)

    assert cleaned[15, 15] == 1  # Large tumor should remain
    assert cleaned[41, 41] == 0  # Small noise should be removed


def test_focal_tversky_loss():
    loss_fn = FocalTverskyLoss()
    preds = torch.tensor([0.9, 0.1, 0.8, 0.2])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    loss = loss_fn(preds, targets)
    assert isinstance(loss.item(), float)
    assert loss >= 0
