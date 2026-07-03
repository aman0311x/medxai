import pytest
import torch
import torch.nn.functional as F

from medxai.losses import GeneralizedDiceLoss


def test_gdl_perfect_prediction_class_index_target():
    torch.manual_seed(0)
    n, c, h, w = 2, 3, 8, 8
    target = torch.randint(0, c, (n, h, w))
    one_hot = F.one_hot(target, c).permute(0, 3, 1, 2).float()
    logits = (one_hot * 20) - 10  # confident logits matching target

    loss = GeneralizedDiceLoss()(logits, target)
    assert loss.item() < 1e-3


def test_gdl_perfect_prediction_one_hot_target():
    torch.manual_seed(0)
    n, c, h, w = 2, 3, 8, 8
    target = torch.randint(0, c, (n, h, w))
    one_hot = F.one_hot(target, c).permute(0, 3, 1, 2).float()
    logits = (one_hot * 20) - 10

    loss = GeneralizedDiceLoss()(logits, one_hot)
    assert loss.item() < 1e-3


def test_gdl_random_prediction_in_valid_range():
    torch.manual_seed(0)
    n, c, h, w = 2, 3, 8, 8
    target = torch.randint(0, c, (n, h, w))
    logits = torch.randn(n, c, h, w)

    loss = GeneralizedDiceLoss()(logits, target)
    assert 0.0 <= loss.item() <= 1.0


def test_gdl_shape_mismatch():
    logits = torch.randn(2, 3, 8, 8)
    target = torch.randint(0, 3, (2, 8, 9))
    with pytest.raises(ValueError):
        GeneralizedDiceLoss()(logits, target)


def test_gdl_gradient_flow():
    n, c, h, w = 1, 3, 4, 4
    logits = torch.randn(n, c, h, w, requires_grad=True)
    target = torch.randint(0, c, (n, h, w))

    loss = GeneralizedDiceLoss()(logits, target)
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
