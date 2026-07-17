"""
medxai/transforms/masked_patch_generator.py

Generates random, patch-wise binary masks for Masked Image Modeling
(MAE / SimMIM style) pre-training, for both 2D (X-ray) and 3D (CT/MRI) volumes.

The mask itself is generated at the *patch grid* resolution and can be
upsampled to full pixel/voxel resolution on demand. The generator does
NOT modify the image itself (no zeroing / replacing pixels) -- that is
left to the model (see medxai/models/ssl.py), since MAE/SimMIM implementations
usually inject a *learnable* mask token rather than a static value, and doing
this inside the model keeps the transform reusable across architectures.
"""

from __future__ import annotations

import torch


class MaskedPatchGenerator:
    """
    Callable transform that produces a random patch-level binary mask
    for an input image / volume.

    Args:
        patch_size: size of one (square/cubic) patch along each spatial dim.
        mask_ratio: fraction of patches to mask (e.g. 0.5 -> 50%).
        mode: "2d" for (C, H, W) inputs, "3d" for (C, D, H, W) inputs.

    Returns (via __call__):
        A dict with:
            "image": the original, UNCHANGED image tensor
            "mask": binary tensor at patch-grid resolution, shape (gh, gw) or
                    (gd, gh, gw). 1 = masked, 0 = visible.
            "grid_shape": tuple of ints, the patch grid shape.
            "patch_size": the patch size used.
    """

    def __init__(self, patch_size: int = 16, mask_ratio: float = 0.5, mode: str = "2d"):
        if mode not in ("2d", "3d"):
            raise ValueError(f"mode must be '2d' or '3d', got {mode!r}")
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")

        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.mode = mode

    def _grid_shape(self, image: torch.Tensor):
        ps = self.patch_size
        spatial = image.shape[1:]  # drop channel dim
        for s in spatial:
            if s % ps != 0:
                raise ValueError(
                    f"Spatial dim {s} is not divisible by patch_size={ps}. "
                    f"Pad/crop the input first."
                )
        return tuple(s // ps for s in spatial)

    def __call__(self, image: torch.Tensor) -> dict:
        if image.dim() not in (3, 4):
            raise ValueError(
                "Expected image shaped (C,H,W) for mode='2d' or (C,D,H,W) for "
                f"mode='3d'; got tensor with shape {tuple(image.shape)}"
            )

        grid_shape = self._grid_shape(image)
        num_patches = 1
        for g in grid_shape:
            num_patches *= g
        num_mask = int(round(num_patches * self.mask_ratio))

        # random permutation -> first num_mask indices are "masked"
        perm = torch.randperm(num_patches)
        mask_flat = torch.zeros(num_patches, dtype=torch.float32)
        mask_flat[perm[:num_mask]] = 1.0
        mask = mask_flat.view(*grid_shape)

        return {
            "image": image,
            "mask": mask,
            "grid_shape": grid_shape,
            "patch_size": self.patch_size,
        }

    @staticmethod
    def upsample_mask(mask: torch.Tensor, patch_size: int) -> torch.Tensor:
        """Upsample a (gh,gw) or (gd,gh,gw) patch-mask to pixel/voxel resolution.
        Returns a tensor with a leading channel dim of size 1, e.g. (1,H,W)."""
        m = mask.unsqueeze(0)  # add channel dim
        for dim in range(1, m.dim()):
            m = m.repeat_interleave(patch_size, dim=dim)
        return m
