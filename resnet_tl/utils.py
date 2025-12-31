from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass(frozen=True)
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int) -> "AverageMeter":
        return AverageMeter(self.total + value * n, self.count + n)

    @property
    def avg(self) -> float:
        return self.total / max(1, self.count)


def set_seed(seed: Optional[int], deterministic: bool = True) -> None:
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: str) -> torch.device:
    device = device.lower().strip()
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Device 'cuda' requested but CUDA is not available.")
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be one of: cpu, cuda")


@torch.no_grad()
def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return float(correct) / float(total) if total else 0.0


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
