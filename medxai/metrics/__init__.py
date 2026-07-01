from typing import Optional

import torch


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
