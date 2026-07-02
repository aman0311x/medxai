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
            raise ValueError(
                f"Shape mismatch in DiceLoss: {pred.shape} vs {target.shape}"
            )
        pred = torch.sigmoid(pred).flatten()
        target = target.float().flatten()
        intersection = (pred * target).sum()
        dice = (2.0 * intersection + self.smooth) / (
            pred.sum() + target.sum() + self.smooth
        )
        return 1.0 - dice


class TverskyLoss(nn.Module):
    """Tversky Loss to optimize for data imbalance (FN vs FP weighting)."""

    def __init__(
        self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6
    ) -> None:
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
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss for hard-to-segment boundaries."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1e-6,
    ) -> None:
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


class GeneralizedDiceLoss(nn.Module):
    """Generalized Dice Loss (GDL) for multi-class semantic segmentation.

    Weights each class by the inverse squared volume of its ground-truth
    region, which makes the loss robust to severe class imbalance between
    foreground structures of very different sizes.

    Reference:
        Sudre et al., "Generalised Dice overlap as a deep learning loss
        function for highly unbalanced segmentations", 2017.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:
                Raw logits of shape (N, C, ...) where C is the number of
                classes.
            target:
                Either class-index labels of shape (N, ...) with integer
                values in [0, C - 1], or a one-hot tensor of shape
                (N, C, ...).

        Returns:
            Scalar Generalized Dice loss tensor.
        """
        if pred.dim() < 2:
            raise ValueError(
                f"Expected pred of shape (N, C, ...), got {tuple(pred.shape)}"
            )

        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)

        if target.dim() == pred.dim() - 1:
            target = F.one_hot(target.long(), num_classes)
            target = target.permute(0, -1, *range(1, target.dim() - 1))
        target = target.float().contiguous()

        if pred.shape != target.shape:
            raise ValueError(
                f"Shape mismatch in GeneralizedDiceLoss: {pred.shape} vs {target.shape}"
            )

        # (N, C, ...) -> (C, N * ...) so each class is reduced over everything else
        pred = pred.transpose(0, 1).flatten(1)
        target = target.transpose(0, 1).flatten(1)

        class_weights = 1.0 / (target.sum(dim=1).pow(2) + self.smooth)
        intersection = (pred * target).sum(dim=1)
        cardinality = pred.sum(dim=1) + target.sum(dim=1)

        numerator = (class_weights * intersection).sum()
        denominator = (class_weights * cardinality).sum()

        gdl = (2.0 * numerator + self.smooth) / (denominator + self.smooth)
        return 1.0 - gdl
