import pytest
import torch
from medxai.metrics import dice_score, iou_score

def test_metrics_perfect_overlap():
    pred = torch.ones(1, 1, 32, 32)
    target = torch.ones(1, 1, 32, 32)
    assert dice_score(pred, target).item() > 0.999
    assert iou_score(pred, target).item() > 0.999

def test_metrics_shape_mismatch():
    pred = torch.ones(1, 1, 32, 32)
    target = torch.ones(1, 1, 64, 64)
    with pytest.raises(ValueError):
        dice_score(pred, target)

def test_metrics_device_mismatch():
    if torch.cuda.is_available():
        pred = torch.ones(1, 1, 32, 32).cuda()
        target = torch.ones(1, 1, 32, 32).cpu()
        with pytest.raises(ValueError):
            dice_score(pred, target)