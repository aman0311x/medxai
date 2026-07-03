from typing import Optional

import numpy as np
import torch
from scipy import ndimage


def _validate_inputs(pred: torch.Tensor, target: torch.Tensor) -> None:
    """Validates tensor shapes and devices."""
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch: pred shape {pred.shape} must match target shape {target.shape}"
        )
    if pred.device != target.device:
        raise ValueError(
            f"Device mismatch: pred is on {pred.device} but target is on {target.device}"
        )


def dice_score(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """
    Computes the Dice Coefficient for binary segmentation.

    Parameters:
        pred (torch.Tensor): Predicted binary mask or thresholded probabilities.
        target (torch.Tensor): Ground truth binary mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Scalar Dice score tensor.
    """
    _validate_inputs(pred, target)
    p = pred.float().flatten()
    t = target.float().flatten()
    intersection = (p * t).sum()
    return (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)


def iou_score(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """Computes the Intersection over Union (Jaccard Index)."""
    _validate_inputs(pred, target)
    p = pred.float().flatten()
    t = target.float().flatten()
    intersection = (p * t).sum()
    union = p.sum() + t.sum() - intersection
    return (intersection + smooth) / (union + smooth)


def precision(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """Computes Precision (Positive Predictive Value)."""
    _validate_inputs(pred, target)
    p = pred.float().flatten()
    t = target.float().flatten()
    tp = (p * t).sum()
    fp = (p * (1.0 - t)).sum()
    return (tp + smooth) / (tp + fp + smooth)


def recall(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """Computes Recall / Sensitivity (True Positive Rate)."""
    _validate_inputs(pred, target)
    p = pred.float().flatten()
    t = target.float().flatten()
    tp = (p * t).sum()
    fn = ((1.0 - p) * t).sum()
    return (tp + smooth) / (tp + fn + smooth)


def sensitivity(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """Alias for recall."""
    return recall(pred, target, smooth)


def specificity(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """Computes Specificity (True Negative Rate)."""
    _validate_inputs(pred, target)
    p = pred.float().flatten()
    t = target.float().flatten()
    tn = ((1.0 - p) * (1.0 - t)).sum()
    fp = (p * (1.0 - t)).sum()
    return (tn + smooth) / (tn + fp + smooth)


def hausdorff_distance_95(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing: Optional[tuple] = None,
) -> torch.Tensor:
    """
    Computes the 95th percentile Hausdorff Distance (HD95) between a
    predicted and a ground-truth binary mask.

    HD95 measures boundary/contour agreement rather than region overlap,
    making it a useful complement to metrics like Dice or IoU for tasks
    such as organ or tumor delineation.

    Args:
        pred:
            Predicted binary mask of shape (H, W) or (D, H, W).
        target:
            Ground truth binary mask, same shape as pred.
        spacing:
            Optional physical voxel spacing, one value per spatial
            dimension, used to scale distances to real-world units.
            Defaults to isotropic unit spacing.

    Returns:
        Scalar HD95 distance tensor.
    """
    _validate_inputs(pred, target)

    if pred.dim() not in (2, 3):
        raise ValueError(
            "hausdorff_distance_95 expects a single 2D or 3D mask "
            f"(H, W) or (D, H, W), got shape {tuple(pred.shape)}"
        )

    pred_np = pred.detach().cpu().numpy().astype(bool)
    target_np = target.detach().cpu().numpy().astype(bool)

    if not pred_np.any() or not target_np.any():
        raise ValueError("hausdorff_distance_95 is undefined when either mask is empty")

    pred_surface = pred_np ^ ndimage.binary_erosion(pred_np)
    target_surface = target_np ^ ndimage.binary_erosion(target_np)

    target_dt = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    pred_dt = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

    pred_to_target = target_dt[pred_surface]
    target_to_pred = pred_dt[target_surface]

    hd95 = max(
        float(np.percentile(pred_to_target, 95)),
        float(np.percentile(target_to_pred, 95)),
    )
    return torch.tensor(hd95, dtype=torch.float32, device=pred.device)


def _surface_distances(
    pred_np: np.ndarray,
    target_np: np.ndarray,
    spacing: Optional[tuple] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes one-directional surface distances in both directions between
    a predicted and a ground-truth binary mask.

    Shared helper used by average_surface_distance and
    average_symmetric_surface_distance to avoid duplicating the surface
    extraction / distance transform logic already used in
    hausdorff_distance_95.

    Args:
        pred_np:
            Predicted binary mask as a numpy boolean array.
        target_np:
            Ground truth binary mask, same shape as pred_np.
        spacing:
            Optional physical voxel spacing, one value per spatial
            dimension. Defaults to isotropic unit spacing.

    Returns:
        Tuple of (pred_to_target, target_to_pred) 1D distance arrays.
    """
    if not pred_np.any() or not target_np.any():
        raise ValueError(
            "Surface distance metrics are undefined when either mask is empty"
        )

    pred_surface = pred_np ^ ndimage.binary_erosion(pred_np)
    target_surface = target_np ^ ndimage.binary_erosion(target_np)

    target_dt = ndimage.distance_transform_edt(~target_surface, sampling=spacing)
    pred_dt = ndimage.distance_transform_edt(~pred_surface, sampling=spacing)

    pred_to_target = target_dt[pred_surface]
    target_to_pred = pred_dt[target_surface]

    return pred_to_target, target_to_pred


def average_surface_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing: Optional[tuple] = None,
) -> torch.Tensor:
    """
    Computes the Average Surface Distance (ASD) between a predicted and a
    ground-truth binary mask.

    ASD measures the mean one-directional distance from the surface of
    pred to the nearest surface voxel of target. Unlike HD95, it is not
    symmetric and is not robust to outliers by design — it reflects
    average boundary error rather than worst-case error.

    Args:
        pred:
            Predicted binary mask of shape (H, W) or (D, H, W).
        target:
            Ground truth binary mask, same shape as pred.
        spacing:
            Optional physical voxel spacing, one value per spatial
            dimension, used to scale distances to real-world units.
            Defaults to isotropic unit spacing.

    Returns:
        Scalar ASD distance tensor.
    """
    _validate_inputs(pred, target)

    if pred.dim() not in (2, 3):
        raise ValueError(
            "average_surface_distance expects a single 2D or 3D mask "
            f"(H, W) or (D, H, W), got shape {tuple(pred.shape)}"
        )

    pred_np = pred.detach().cpu().numpy().astype(bool)
    target_np = target.detach().cpu().numpy().astype(bool)

    pred_to_target, _ = _surface_distances(pred_np, target_np, spacing)

    asd = float(pred_to_target.mean())
    return torch.tensor(asd, dtype=torch.float32, device=pred.device)


def average_symmetric_surface_distance(
    pred: torch.Tensor,
    target: torch.Tensor,
    spacing: Optional[tuple] = None,
) -> torch.Tensor:
    """
    Computes the Average Symmetric Surface Distance (ASSD) between a
    predicted and a ground-truth binary mask.

    ASSD averages the surface distances in both directions (pred ->
    target and target -> pred), giving a symmetric metric that
    complements HD95: HD95 captures worst-case boundary error, while
    ASSD captures average boundary error.

    Args:
        pred:
            Predicted binary mask of shape (H, W) or (D, H, W).
        target:
            Ground truth binary mask, same shape as pred.
        spacing:
            Optional physical voxel spacing, one value per spatial
            dimension, used to scale distances to real-world units.
            Defaults to isotropic unit spacing.

    Returns:
        Scalar ASSD distance tensor.
    """
    _validate_inputs(pred, target)

    if pred.dim() not in (2, 3):
        raise ValueError(
            "average_symmetric_surface_distance expects a single 2D or 3D mask "
            f"(H, W) or (D, H, W), got shape {tuple(pred.shape)}"
        )

    pred_np = pred.detach().cpu().numpy().astype(bool)
    target_np = target.detach().cpu().numpy().astype(bool)

    pred_to_target, target_to_pred = _surface_distances(pred_np, target_np, spacing)

    assd = float(
        (pred_to_target.sum() + target_to_pred.sum())
        / (pred_to_target.size + target_to_pred.size)
    )
    return torch.tensor(assd, dtype=torch.float32, device=pred.device)


def compute_clinical_metrics(
    preds: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5
):
    """
    Computes Sensitivity (Recall) and Specificity for medical segmentation masks.
    """
    preds_bin = (torch.sigmoid(preds) > threshold).float().view(-1)
    targets_bin = targets.float().view(-1)

    TP = (preds_bin * targets_bin).sum()
    TN = ((1 - preds_bin) * (1 - targets_bin)).sum()
    FP = (preds_bin * (1 - targets_bin)).sum()
    FN = ((1 - preds_bin) * targets_bin).sum()

    sensitivity = (TP + 1e-6) / (TP + FN + 1e-6)
    specificity = (TN + 1e-6) / (TN + FP + 1e-6)

    return {"sensitivity": sensitivity.item(), "specificity": specificity.item()}
