"""
medxai/utils/ssl_transfer.py

Utility for transferring MedicalMAE-pretrained encoder weights into a
downstream UNet / UNet3D model for fine-tuning on labeled segmentation data.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def load_pretrained_encoder(
    model: nn.Module,
    mae_checkpoint_path: str,
    encoder_attr: str = "encoder",
    strict: bool = False,
    map_location: str = "cpu",
) -> nn.Module:
    """
    Loads encoder weights from a MedicalMAE checkpoint into `model.<encoder_attr>`.

    Args:
        model: a UNet / UNet3D instance whose `encoder_attr` submodule has the
            SAME architecture as the encoder used during MAE pre-training.
        mae_checkpoint_path: path to a checkpoint saved from `MedicalMAE`
            (either a raw state_dict, or a dict containing a "state_dict" key).
        encoder_attr: name of the encoder submodule on `model`
            (defaults to "encoder"; change if your UNet exposes it differently,
            e.g. "down_path" or "backbone").
        strict: forwarded to `nn.Module.load_state_dict`. Kept False by default
            since decoder / mask-token weights from the MAE checkpoint must be
            dropped, and some architectures may have partially-shared keys.
        map_location: device mapping for `torch.load`.

    Returns:
        The same `model`, with `encoder_attr` weights loaded in-place.

    Raises:
        AttributeError: if `model` has no `encoder_attr` submodule.
        RuntimeError: if `strict=True` and keys mismatch.
    """
    ckpt = torch.load(mae_checkpoint_path, map_location=map_location)
    state_dict = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )

    prefix = "encoder."
    encoder_state = {
        k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)
    }

    if not encoder_state:
        raise ValueError(
            "No keys starting with 'encoder.' were found in the checkpoint. "
            "Make sure this checkpoint was produced by MedicalMAE.state_dict()."
        )

    if not hasattr(model, encoder_attr):
        raise AttributeError(
            f"'{type(model).__name__}' has no attribute '{encoder_attr}'. "
            f"Pass encoder_attr= matching how your UNet/UNet3D exposes its "
            f"encoder submodule."
        )

    encoder = getattr(model, encoder_attr)
    result = encoder.load_state_dict(encoder_state, strict=strict)

    if getattr(result, "missing_keys", None):
        print(
            f"[load_pretrained_encoder] Missing keys (not loaded): {result.missing_keys}"
        )
    if getattr(result, "unexpected_keys", None):
        print(
            f"[load_pretrained_encoder] Unexpected keys (ignored): {result.unexpected_keys}"
        )

    return model


def freeze_encoder(model: nn.Module, encoder_attr: str = "encoder") -> nn.Module:
    """Optionally freeze the transferred encoder for linear-probe style fine-tuning."""
    encoder = getattr(model, encoder_attr)
    for p in encoder.parameters():
        p.requires_grad = False
    return model
