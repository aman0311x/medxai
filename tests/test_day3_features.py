import numpy as np
import torch

from medxai.inference import sliding_window_inference_2d
from medxai.losses import DiceBCELoss
from medxai.metrics import compute_clinical_metrics
from medxai.models import UNet
from medxai.models.layers import AttentionGate2d
from medxai.preprocessing import apply_ct_window


def test_ct_windowing():
    # Fake raw CT data (Hounsfield units usually range from -1000 to 2000)
    raw_ct = np.array([-1000, 0, 100, 400, 2000], dtype=np.float32)
    # Apply soft tissue window (Center: 40, Width: 400 => Min: -160, Max: 240)
    normalized = apply_ct_window(raw_ct, window_center=40, window_width=400)
    assert normalized[0] == 0.0  # -1000 should be clipped to min and normalized to 0
    assert normalized[-1] == 1.0  # 2000 should be clipped to max and normalized to 1


def test_attention_gate():
    gate = AttentionGate2d(F_g=32, F_l=32, F_int=16)
    g = torch.randn(1, 32, 16, 16)
    x = torch.randn(1, 32, 16, 16)
    out = gate(g, x)
    assert out.shape == x.shape


def test_dice_bce_loss():
    loss_fn = DiceBCELoss()
    logits = torch.randn(2, 1, 32, 32)
    targets = torch.randint(0, 2, (2, 1, 32, 32)).float()
    loss = loss_fn(logits, targets)
    assert loss.item() > 0


def test_sliding_window():
    model = UNet(n_channels=1, n_classes=1)
    large_image = torch.randn(1, 1, 128, 128)
    output = sliding_window_inference_2d(
        model, large_image, patch_size=(64, 64), stride=(32, 32)
    )
    assert output.shape == (1, 1, 128, 128)


def test_clinical_metrics():
    preds = torch.tensor([2.0, -2.0, 3.0, -3.0])  # Logits
    targets = torch.tensor([1.0, 0.0, 1.0, 1.0])
    metrics = compute_clinical_metrics(preds, targets)
    assert "sensitivity" in metrics
    assert "specificity" in metrics
