from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

from torchvision import transforms
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


def build_transforms(weights: ResNet18_Weights, train: bool) -> transforms.Compose:
    # ResNet18 default is typically 224x224; use weights meta for mean/std.
    mean = weights.meta["mean"]
    std = weights.meta["std"]

    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    # Validation: deterministic transforms similar to weights.transforms()
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


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

    # Safety: ensure val has same class->idx mapping
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
