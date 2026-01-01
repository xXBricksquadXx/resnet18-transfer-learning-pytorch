from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from torchvision.models import ResNet18_Weights


@dataclass(frozen=True)
class DataSpec:
    train_dir: Path
    val_dir: Path
    class_names: List[str]
    train_count: int
    val_count: int


def _mean_std_from_weights(weights: ResNet18_Weights) -> tuple[list[float], list[float]]:
    """
    torchvision version differences:
    - some versions expose mean/std via weights.meta
    - others expose mean/std via weights.transforms() preset (preferred)
    Fallback to standard ImageNet mean/std if we can't find them.
    """
    # 1) Try weights.meta if present
    try:
        meta = getattr(weights, "meta", {}) or {}
        mean = meta.get("mean", None)
        std = meta.get("std", None)
        if mean is not None and std is not None:
            return list(mean), list(std)
    except Exception:
        pass

    # 2) Try weights.transforms() preset attributes (common in torchvision 0.15+)
    try:
        preset = weights.transforms()
        if hasattr(preset, "mean") and hasattr(preset, "std"):
            return list(preset.mean), list(preset.std)
    except Exception:
        pass

    # 3) Try to locate a Normalize transform inside a Compose
    try:
        preset = weights.transforms()
        transforms_list = getattr(preset, "transforms", None)
        if transforms_list is None and isinstance(preset, T.Compose):
            transforms_list = preset.transforms
        if transforms_list:
            for tr in transforms_list:
                if isinstance(tr, T.Normalize):
                    return list(tr.mean), list(tr.std)
    except Exception:
        pass

    # 4) Fallback: standard ImageNet normalization
    return [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def build_transforms(weights: ResNet18_Weights, train: bool):
    mean, std = _mean_std_from_weights(weights)

    if train:
        return T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    # Validation: use the preset transform stack from the weights (most compatible)
    return weights.transforms()


def make_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int,
    weights: ResNet18_Weights,
) -> Tuple[DataLoader, DataLoader, DataSpec]:
    root = Path(data_dir)
    train_dir = root / "train"
    val_dir = root / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"Missing train dir: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Missing val dir: {val_dir}")

    train_tfms = build_transforms(weights, train=True)
    val_tfms = build_transforms(weights, train=False)

    train_ds = ImageFolder(train_dir, transform=train_tfms)
    val_ds = ImageFolder(val_dir, transform=val_tfms)

    if val_ds.class_to_idx != train_ds.class_to_idx:
        raise RuntimeError(
            f"class_to_idx mismatch between train and val.\n"
            f"train: {train_ds.class_to_idx}\n"
            f"val:   {val_ds.class_to_idx}"
        )

    common_loader_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=False,
    )
    if num_workers > 0:
        common_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, shuffle=True, **common_loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **common_loader_kwargs)

    spec = DataSpec(
        train_dir=train_dir,
        val_dir=val_dir,
        class_names=train_ds.classes,
        train_count=len(train_ds),
        val_count=len(val_ds),
    )
    return train_loader, val_loader, spec
