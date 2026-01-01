from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class EpochResult:
    loss: float
    acc: float


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return float(correct) / float(total) if total else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = False,
) -> EpochResult:
    model.train()

    use_amp = bool(amp and device.type == "cuda")
    scaler: Optional[torch.amp.GradScaler] = torch.amp.GradScaler("cuda") if use_amp else None
    autocast_ctx = torch.amp.autocast(device_type="cuda") if use_amp else nullcontext()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=False)
        yb = yb.to(device, non_blocking=False)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            logits = model(xb)
            loss = criterion(logits, yb)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_seen += int(bs)

    avg_loss = total_loss / total_seen if total_seen else 0.0
    avg_acc = total_correct / total_seen if total_seen else 0.0
    return EpochResult(loss=float(avg_loss), acc=float(avg_acc))


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for xb, yb in loader:
        xb = xb.to(device, non_blocking=False)
        yb = yb.to(device, non_blocking=False)

        logits = model(xb)
        loss = criterion(logits, yb)

        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_correct += int((logits.argmax(dim=1) == yb).sum().item())
        total_seen += int(bs)

    avg_loss = total_loss / total_seen if total_seen else 0.0
    avg_acc = total_correct / total_seen if total_seen else 0.0
    return EpochResult(loss=float(avg_loss), acc=float(avg_acc))
