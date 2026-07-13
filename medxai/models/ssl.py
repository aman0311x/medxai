"""
medxai/models/ssl.py

Self-Supervised Pre-training for MedXAI encoders (SimMIM / MAE style).

`MedicalMAE` wraps an arbitrary convolutional encoder (e.g. the encoder half
of `UNet` / `UNet3D`) and adds:
    1. A learnable mask token that replaces masked input patches
       (SimMIM-style "masking at the input", so the encoder's own
       downsampling/skip-connection structure needs no modification).
    2. A lightweight 1x1 (or 1x1x1) convolutional decoder head that projects
       encoder features back to pixel/voxel space, one vector per patch.
    3. A masked-patch L1 reconstruction loss:

        L_rec = (1/|M|) * sum_{i in M} || I_i - I_hat_i ||_1

       where M is the set of masked patches.

Works for both 2D (X-ray, single-slice CT/MRI) and 3D (volumetric CT/MRI)
inputs by passing mode="2d" or mode="3d".

Typical usage
-------------
    from medxai.models.unet import UNet
    from medxai.models.ssl import MedicalMAE

    backbone = UNet(in_channels=1, out_channels=2)
    mae = MedicalMAE(
        encoder=backbone.encoder,          # must output a single feature map
        encoder_out_channels=backbone.encoder_out_channels,
        patch_size=16,
        in_channels=1,
        mode="2d",
        mask_ratio=0.5,
    )

    for images in dataloader:          # images: (B, C, H, W)
        loss, pred, mask = mae(images)
        loss.backward()
        optimizer.step()

    torch.save({"state_dict": mae.state_dict()}, "mae_pretrained.pt")
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MedicalMAE(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoder_out_channels: int,
        patch_size: int = 16,
        in_channels: int = 1,
        mode: str = "2d",
        mask_ratio: float = 0.5,
    ):
        super().__init__()
        if mode not in ("2d", "3d"):
            raise ValueError(f"mode must be '2d' or '3d', got {mode!r}")
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

        self.encoder = encoder
        self.mode = mode
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.mask_ratio = mask_ratio
        self.n_spatial_dims = 2 if mode == "2d" else 3

        # learnable token that replaces masked-out input regions
        token_shape = (1, in_channels) + (1,) * self.n_spatial_dims
        self.mask_token = nn.Parameter(torch.zeros(token_shape))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # one flattened target vector per patch: C * patch_size^{2 or 3}
        self.out_dim = in_channels * (patch_size**self.n_spatial_dims)
        conv_cls = nn.Conv2d if mode == "2d" else nn.Conv3d
        self.decoder = conv_cls(encoder_out_channels, self.out_dim, kernel_size=1)

    # ---------------------------------------------------------------- utils

    def _grid_shape(self, spatial_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        ps = self.patch_size
        for s in spatial_shape:
            if s % ps != 0:
                raise ValueError(f"Spatial size {s} not divisible by patch_size={ps}.")
        return tuple(s // ps for s in spatial_shape)

    def _random_mask(
        self, batch_size: int, grid_shape: Tuple[int, ...], device
    ) -> torch.Tensor:
        """Returns (B, num_patches) binary mask, 1 = masked."""
        num_patches = 1
        for g in grid_shape:
            num_patches *= g
        num_mask = int(round(num_patches * self.mask_ratio))

        mask = torch.zeros(batch_size, num_patches, device=device)
        for b in range(batch_size):
            idx = torch.randperm(num_patches, device=device)[:num_mask]
            mask[b, idx] = 1.0
        return mask

    def _pixel_mask(
        self, mask_flat: torch.Tensor, grid_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Upsamples (B, num_patches) -> (B, 1, *pixel_shape)."""
        ps = self.patch_size
        B = mask_flat.shape[0]
        m = mask_flat.view(B, 1, *grid_shape)
        for dim in range(2, m.dim()):
            m = m.repeat_interleave(ps, dim=dim)
        return m

    def _to_patches(self, x: torch.Tensor, grid_shape: Tuple[int, ...]) -> torch.Tensor:
        """(B,C,*spatial) -> (B, num_patches, C*ps^n), non-overlapping patches."""
        ps = self.patch_size
        B, C = x.shape[0], x.shape[1]
        if self.mode == "2d":
            gh, gw = grid_shape
            x = x.view(B, C, gh, ps, gw, ps)
            x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
            x = x.view(B, gh * gw, C * ps * ps)
        else:
            gd, gh, gw = grid_shape
            x = x.view(B, C, gd, ps, gh, ps, gw, ps)
            x = x.permute(0, 2, 4, 6, 1, 3, 5, 7).contiguous()
            x = x.view(B, gd * gh * gw, C * ps * ps * ps)
        return x

    def _pred_to_patches(
        self, pred: torch.Tensor, grid_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """(B, out_dim, *grid_shape) -> (B, num_patches, out_dim)."""
        B = pred.shape[0]
        if self.mode == "2d":
            gh, gw = grid_shape
            pred = pred.permute(0, 2, 3, 1).contiguous().view(B, gh * gw, self.out_dim)
        else:
            gd, gh, gw = grid_shape
            pred = (
                pred.permute(0, 2, 3, 4, 1)
                .contiguous()
                .view(B, gd * gh * gw, self.out_dim)
            )
        return pred

    # -------------------------------------------------------------- forward

    def forward(self, images: torch.Tensor):
        """
        Args:
            images: (B, C, H, W) if mode="2d", or (B, C, D, H, W) if mode="3d".

        Returns:
            loss: scalar masked-patch L1 reconstruction loss
            pred_patches: (B, out_dim, *grid_shape) raw decoder output (for viz/debug)
            pixel_mask: (B, 1, *spatial) binary mask upsampled to input resolution
        """
        spatial_shape = images.shape[2:]
        grid_shape = self._grid_shape(spatial_shape)
        device = images.device

        mask_flat = self._random_mask(
            images.shape[0], grid_shape, device
        )  # (B, num_patches)
        pixel_mask = self._pixel_mask(mask_flat, grid_shape)  # (B,1,*spatial)

        mask_token = self.mask_token.expand_as(images)
        masked_images = images * (1 - pixel_mask) + mask_token * pixel_mask

        features = self.encoder(masked_images)
        if isinstance(features, (list, tuple)):
            # if the encoder returns multi-scale feature maps (common in
            # UNet-style encoders), use the deepest/last one for reconstruction
            features = features[-1]

        pred_patches = self.decoder(features)

        # decoder output spatial size must match the patch grid; if the
        # encoder downsamples further than patch_size, upsample back
        if pred_patches.shape[2:] != grid_shape:
            pred_patches = F.interpolate(pred_patches, size=grid_shape, mode="nearest")

        target = self._to_patches(images, grid_shape)  # (B, N, out_dim)
        pred = self._pred_to_patches(pred_patches, grid_shape)  # (B, N, out_dim)

        per_patch_l1 = F.l1_loss(pred, target, reduction="none").mean(dim=-1)  # (B, N)
        loss = (per_patch_l1 * mask_flat).sum() / (mask_flat.sum() + 1e-8)

        return loss, pred_patches, pixel_mask
