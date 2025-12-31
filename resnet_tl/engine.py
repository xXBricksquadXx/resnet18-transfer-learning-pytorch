from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import AverageMeter, accuracy_top1


@dataclass(frozen=True)
class EpochResult:
    loss: float
    acc: float


def _amp_autocast(device: torch.device, amp: bool):
    if amp and device.type == "cuda":
        return torch.cuda.amp.autocast()
    # no-op context manager
    from contextlib import nullcontext
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool = False,
) -> EpochResult:
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc="train", leave=False)
    for xb, yb in pbar:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)

        with _amp_autocast(device, amp):
            logits = model(xb)
            loss = criterion(logits, yb)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = yb.size(0)
        acc = accuracy_top1(logits.detach(), yb)

        loss_meter = loss_meter.update(float(loss.item()), bs)
        acc_meter = acc_meter.update(acc, bs)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return EpochResult(loss=loss_meter.avg, acc=acc_meter.avg)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    pbar = tqdm(loader, desc="val", leave=False)
    for xb, yb in pbar:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        bs = yb.size(0)
        acc = accuracy_top1(logits, yb)

        loss_meter = loss_meter.update(float(loss.item()), bs)
        acc_meter = acc_meter.update(acc, bs)

        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{acc_meter.avg:.4f}")

    return EpochResult(loss=loss_meter.avg, acc=acc_meter.avg)
