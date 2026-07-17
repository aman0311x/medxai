from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from medxai.losses import BCEDiceLoss
from medxai.metrics import dice_score, sensitivity, specificity
from medxai.models.unet import UNet
from medxai.models.unet3d import UNet3D

MODEL_REGISTRY = {
    "unet": UNet,
    "unet3d": UNet3D,
}


# --------------------------------------------------------------------------
# Distributed setup / teardown
# --------------------------------------------------------------------------
def setup_distributed(backend: str):
    """
    Reads torchrun-provided env vars and initializes the process group if
    running with more than one process. Returns (rank, world_size,
    local_rank, device, is_distributed).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed:
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return rank, world_size, local_rank, device, is_distributed


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def reduce_mean(
    tensor: torch.Tensor, world_size: int, is_distributed: bool
) -> torch.Tensor:
    """All-reduces (SUM) a scalar metric tensor across ranks, then averages.

    NOTE: this assumes each rank's DistributedSampler shard contributes a
    (roughly) equal number of samples/batches, which holds for
    `DistributedSampler`'s default even-padding behavior. For exact
    sample-weighted aggregation, gather (sum, count) pairs instead.
    """
    if not is_distributed:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= world_size
    return tensor


# --------------------------------------------------------------------------
# Mock dataset (for CI / smoke-testing the training loop end-to-end)
# --------------------------------------------------------------------------
class MockSegmentationDataset(Dataset):
    """Synthetic random images + binary masks, just to exercise the
    train/val loop's shapes, AMP, DDP, and metrics-aggregation code paths
    without requiring real medical imaging data."""

    def __init__(self, n_samples=16, mode="2d", n_channels=1, size=64):
        self.n_samples = n_samples
        self.mode = mode
        self.n_channels = n_channels
        self.size = size

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        if self.mode == "2d":
            image = torch.rand(self.n_channels, self.size, self.size)
            mask = torch.randint(0, 2, (1, self.size, self.size)).float()
        else:
            image = torch.rand(self.n_channels, self.size, self.size, self.size)
            mask = torch.randint(0, 2, (1, self.size, self.size, self.size)).float()
        return image, mask


def build_dataloaders(args, is_distributed: bool, rank: int, world_size: int):
    if args.mock:
        train_ds = MockSegmentationDataset(
            n_samples=32,
            mode=args.mode,
            n_channels=args.n_channels,
            size=args.mock_size,
        )
        val_ds = MockSegmentationDataset(
            n_samples=8, mode=args.mode, n_channels=args.n_channels, size=args.mock_size
        )
    else:
        raise NotImplementedError(
            "Plug in your real dataset here (e.g. reading from --data-dir). "
            "Run with --mock to smoke-test the training loop without real data."
        )

    train_sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if is_distributed
        else None
    )
    val_sampler = (
        DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        if is_distributed
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
    )
    return train_loader, val_loader, train_sampler


# --------------------------------------------------------------------------
# Model construction (DDP + SyncBatchNorm)
# --------------------------------------------------------------------------
def build_model(args, device, is_distributed: bool, local_rank: int):
    if args.model not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown --model '{args.model}'. Available: {list(MODEL_REGISTRY)}"
        )
    model = MODEL_REGISTRY[args.model](
        n_channels=args.n_channels, n_classes=args.n_classes
    )
    model.to(device)

    if is_distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_kwargs = {"device_ids": [local_rank]} if device.type == "cuda" else {}
        model = nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    return model


# --------------------------------------------------------------------------
# Train / validate for one epoch
# --------------------------------------------------------------------------
def train_one_epoch(
    model, loader, optimizer, criterion, scaler, device, args, epoch, rank
):
    model.train()
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    running_loss = torch.zeros(1, device=device)
    n_batches = 0

    for step, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=autocast_device, enabled=args.amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.detach()
        n_batches += 1

        if rank == 0 and step % max(1, len(loader) // 5 or 1) == 0:
            print(f"[epoch {epoch}] step {step}/{len(loader)} loss={loss.item():.4f}")

        if args.steps and n_batches >= args.steps:
            break

    return (running_loss / max(n_batches, 1)).item()


@torch.no_grad()
def validate(model, loader, device, args, world_size, is_distributed):
    model.eval()
    dice_sum = torch.zeros(1, device=device)
    sens_sum = torch.zeros(1, device=device)
    spec_sum = torch.zeros(1, device=device)
    n_batches = 0

    for step, (images, masks) in enumerate(loader):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).float()

        dice_sum += dice_score(preds, masks)
        sens_sum += sensitivity(preds, masks)
        spec_sum += specificity(preds, masks)
        n_batches += 1

        if args.steps and n_batches >= args.steps:
            break

    n_batches = max(n_batches, 1)
    local_metrics = {
        "dice": dice_sum / n_batches,
        "sensitivity": sens_sum / n_batches,
        "specificity": spec_sum / n_batches,
    }

    # aggregate metrics across all ranks (see reduce_mean docstring for the
    # even-sharding assumption)
    global_metrics = {
        k: reduce_mean(v, world_size, is_distributed).item()
        for k, v in local_metrics.items()
    }
    return global_metrics


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MedXAI distributed (DDP) + AMP training")
    p.add_argument("--model", default="unet", choices=list(MODEL_REGISTRY))
    p.add_argument("--mode", default="2d", choices=["2d", "3d"])
    p.add_argument("--n-channels", type=int, default=1)
    p.add_argument("--n-classes", type=int, default=1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--amp", action="store_true", help="Enable Automatic Mixed Precision training"
    )
    p.add_argument(
        "--backend",
        default="nccl",
        choices=["nccl", "gloo"],
        help="Distributed backend. Use 'gloo' for CPU-only runs (e.g. CI).",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use a synthetic in-memory dataset instead of --data-dir (for CI/smoke tests)",
    )
    p.add_argument(
        "--mock-size", type=int, default=64, help="Spatial size for --mock data"
    )
    p.add_argument(
        "--data-dir", default=None, help="Path to real training data (TODO: wire up)"
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Limit number of steps per epoch (useful for quick CI smoke tests)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    rank, world_size, local_rank, device, is_distributed = setup_distributed(
        args.backend
    )

    try:
        model = build_model(args, device, is_distributed, local_rank)
        train_loader, val_loader, train_sampler = build_dataloaders(
            args, is_distributed, rank, world_size
        )

        criterion = BCEDiceLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scaler = torch.amp.GradScaler(
            device.type, enabled=(args.amp and device.type == "cuda")
        )

        for epoch in range(args.epochs):
            if is_distributed and train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                scaler,
                device,
                args,
                epoch,
                rank,
            )
            metrics = validate(
                model, val_loader, device, args, world_size, is_distributed
            )

            if rank == 0:
                print(
                    f"[epoch {epoch}] train_loss={train_loss:.4f} "
                    f"dice={metrics['dice']:.4f} "
                    f"sensitivity={metrics['sensitivity']:.4f} "
                    f"specificity={metrics['specificity']:.4f}"
                )
    finally:
        cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()
