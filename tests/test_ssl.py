import torch

from medxai.models.ssl import MedicalMAE
from medxai.models.ssl_transfer import load_pretrained_encoder
from medxai.models.unet import UNet
from medxai.models.unet3d import UNet3D
from medxai.preprocessing.masked_patch_generator import MaskedPatchGenerator


# --------------------------------------------------------------------------
# MaskedPatchGenerator
# --------------------------------------------------------------------------
def test_masked_patch_generator_2d_shapes_and_ratio():
    gen = MaskedPatchGenerator(patch_size=8, mask_ratio=0.5, mode="2d")
    image = torch.rand(1, 64, 64)
    out = gen(image)

    assert out["grid_shape"] == (8, 8)
    assert out["mask"].shape == (8, 8)
    assert torch.isclose(out["mask"].sum(), torch.tensor(32.0))  # 50% of 64 patches


def test_masked_patch_generator_3d_shapes_and_ratio():
    gen = MaskedPatchGenerator(patch_size=8, mask_ratio=0.25, mode="3d")
    volume = torch.rand(1, 32, 32, 32)
    out = gen(volume)

    assert out["grid_shape"] == (4, 4, 4)
    assert out["mask"].shape == (4, 4, 4)
    assert torch.isclose(out["mask"].sum(), torch.tensor(16.0))  # 25% of 64 patches


def test_masked_patch_generator_rejects_non_divisible_size():
    gen = MaskedPatchGenerator(patch_size=8, mask_ratio=0.5, mode="2d")
    image = torch.rand(1, 63, 64)
    try:
        gen(image)
        assert False, "expected ValueError for non-divisible spatial size"
    except ValueError:
        pass


# --------------------------------------------------------------------------
# MedicalMAE (2D, using the real UNet encoder)
# --------------------------------------------------------------------------
def test_medical_mae_2d_forward_and_backward():
    model = UNet(n_channels=1, n_classes=2)
    mae = MedicalMAE(
        encoder=model.encoder,
        encoder_out_channels=512,
        patch_size=8,
        in_channels=1,
        mode="2d",
        mask_ratio=0.5,
    )

    x = torch.rand(2, 1, 64, 64)
    loss, pred, pixel_mask = mae(x)

    assert loss.dim() == 0
    assert loss.item() >= 0
    assert pixel_mask.shape == (2, 1, 64, 64)

    loss.backward()
    grad_found = any(p.grad is not None for p in mae.encoder.parameters())
    assert grad_found, "gradients should flow back into the encoder"


# --------------------------------------------------------------------------
# MedicalMAE (3D, using the real UNet3D encoder)
# --------------------------------------------------------------------------
def test_medical_mae_3d_forward_and_backward():
    model = UNet3D(n_channels=1, n_classes=2)
    mae = MedicalMAE(
        encoder=model.encoder,
        encoder_out_channels=512,
        patch_size=8,
        in_channels=1,
        mode="3d",
        mask_ratio=0.5,
    )

    x = torch.rand(1, 1, 32, 32, 32)
    loss, pred, pixel_mask = mae(x)

    assert loss.dim() == 0
    assert pixel_mask.shape == (1, 1, 32, 32, 32)

    loss.backward()


# --------------------------------------------------------------------------
# Transfer learning utility
# --------------------------------------------------------------------------
def test_load_pretrained_encoder_transfers_weights(tmp_path):
    model = UNet(n_channels=1, n_classes=2)
    mae = MedicalMAE(
        encoder=model.encoder,
        encoder_out_channels=512,
        patch_size=8,
        in_channels=1,
        mode="2d",
        mask_ratio=0.5,
    )

    ckpt_path = tmp_path / "mae.pt"
    torch.save({"state_dict": mae.state_dict()}, ckpt_path)

    fresh_model = UNet(n_channels=1, n_classes=2)
    fresh_model = load_pretrained_encoder(
        fresh_model, str(ckpt_path), encoder_attr="encoder", strict=True
    )

    # weights should now match the pretrained encoder exactly
    for p1, p2 in zip(model.encoder.parameters(), fresh_model.encoder.parameters()):
        assert torch.allclose(p1, p2)

    # sanity check the model still runs end-to-end for segmentation
    x = torch.rand(1, 1, 64, 64)
    out = fresh_model(x)
    assert out.shape == (1, 2, 64, 64)