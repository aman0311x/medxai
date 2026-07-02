import pytest
import torch

from medxai.metrics import dice_score, hausdorff_distance_95, iou_score,  average_surface_distance, average_symmetric_surface_distance


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


def test_hd95_perfect_overlap():
    mask = torch.zeros(20, 20)
    mask[5:15, 5:15] = 1
    hd = hausdorff_distance_95(mask, mask.clone())
    assert hd.item() == pytest.approx(0.0, abs=1e-4)


def test_hd95_known_offset():
    pred = torch.zeros(30, 30)
    pred[5:15, 5:15] = 1
    target = torch.zeros(30, 30)
    target[5:15, 8:18] = 1  # shifted 3 pixels
    hd = hausdorff_distance_95(pred, target)
    assert hd.item() == pytest.approx(3.0, abs=1e-4)


def test_hd95_3d_with_spacing():
    mask = torch.zeros(10, 10, 10)
    mask[2:8, 2:8, 2:8] = 1
    hd = hausdorff_distance_95(mask, mask.clone(), spacing=(2.0, 1.0, 1.0))
    assert hd.item() == pytest.approx(0.0, abs=1e-4)


def test_hd95_shape_mismatch():
    pred = torch.ones(5, 5)
    target = torch.ones(6, 6)
    with pytest.raises(ValueError):
        hausdorff_distance_95(pred, target)


def test_hd95_empty_mask_raises():
    pred = torch.zeros(5, 5)
    target = torch.ones(5, 5)
    with pytest.raises(ValueError):
        hausdorff_distance_95(pred, target)

def test_asd_perfect_overlap():
    mask = torch.zeros(20, 20)
    mask[5:15, 5:15] = 1
    asd = average_surface_distance(mask, mask.clone())
    assert asd.item() == pytest.approx(0.0, abs=1e-4)


def test_assd_perfect_overlap():
    mask = torch.zeros(20, 20)
    mask[5:15, 5:15] = 1
    assd = average_symmetric_surface_distance(mask, mask.clone())
    assert assd.item() == pytest.approx(0.0, abs=1e-4)


def test_assd_known_offset():
    pred = torch.zeros(30, 30)
    pred[5:15, 5:15] = 1
    target = torch.zeros(30, 30)
    target[5:15, 8:18] = 1  # shifted 3 pixels
    assd = average_symmetric_surface_distance(pred, target)
    hd = hausdorff_distance_95(pred, target)
    assert assd.item() < hd.item()
    assert assd.item() > 0.0


def test_assd_is_symmetric():
    pred = torch.zeros(30, 30)
    pred[5:15, 5:15] = 1
    target = torch.zeros(30, 30)
    target[5:15, 8:18] = 1
    forward = average_symmetric_surface_distance(pred, target)
    backward = average_symmetric_surface_distance(target, pred)
    assert forward.item() == pytest.approx(backward.item(), abs=1e-4)


def test_asd_3d_with_spacing():
    mask = torch.zeros(10, 10, 10)
    mask[2:8, 2:8, 2:8] = 1
    asd = average_surface_distance(mask, mask.clone(), spacing=(2.0, 1.0, 1.0))
    assert asd.item() == pytest.approx(0.0, abs=1e-4)


def test_asd_shape_mismatch():
    pred = torch.ones(5, 5)
    target = torch.ones(6, 6)
    with pytest.raises(ValueError):
        average_surface_distance(pred, target)


def test_assd_shape_mismatch():
    pred = torch.ones(5, 5)
    target = torch.ones(6, 6)
    with pytest.raises(ValueError):
        average_symmetric_surface_distance(pred, target)


def test_asd_empty_mask_raises():
    pred = torch.zeros(5, 5)
    target = torch.ones(5, 5)
    with pytest.raises(ValueError):
        average_surface_distance(pred, target)


def test_assd_empty_mask_raises():
    pred = torch.zeros(5, 5)
    target = torch.ones(5, 5)
    with pytest.raises(ValueError):
        average_symmetric_surface_distance(pred, target)