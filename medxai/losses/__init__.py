import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Robust Dice Loss function for binary semantic segmentation."""
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch in DiceLoss: {pred.shape} vs {target.shape}")
        pred = torch.sigmoid(pred).flatten()
        target = target.float().flatten()
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        return 1.0 - dice

class TverskyLoss(nn.Module):
    """Tversky Loss to optimize for data imbalance (FN vs FP weighting)."""
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred).flatten()
        target = target.float().flatten()
        tp = (pred * target).sum()
        fp = (pred * (1.0 - target)).sum()
        fn = ((1.0 - pred) * target).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky

class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for hard-to-segment boundaries."""
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 0.75, smooth: float = 1e-6) -> None:
        super().__init__()
        self.tversky = TverskyLoss(alpha, beta, smooth)
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tversky_out = self.tversky(pred, target)
        return torch.pow(tversky_out, self.gamma)

class BCEDiceLoss(nn.Module):
    """Combination of Binary Cross Entropy and Dice Loss."""
    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.dice = DiceLoss(smooth)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(pred, target.float())
        dice = self.dice(pred, target)
        return bce + dice