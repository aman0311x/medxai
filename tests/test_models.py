import torch

from medxai.models import UNet3D


def test_unet3d_output_shape():
    model = UNet3D(n_channels=1, n_classes=1)
    x = torch.randn(1, 1, 16, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 16, 64, 64)


def test_unet3d_multiclass_output_shape():
    model = UNet3D(n_channels=1, n_classes=3)
    x = torch.randn(1, 1, 16, 64, 64)
    out = model(x)
    assert out.shape == (1, 3, 16, 64, 64)


def test_unet3d_multichannel_input():
    model = UNet3D(n_channels=2, n_classes=1)
    x = torch.randn(1, 2, 16, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 16, 64, 64)
