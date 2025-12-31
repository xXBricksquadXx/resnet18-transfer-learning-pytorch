from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def build_resnet18_base(
    weights: ResNet18_Weights = ResNet18_Weights.DEFAULT,
) -> nn.Module:
    """
    Pretrained ResNet-18 exactly as torchvision ships it:
    - conv1 / bn1 / relu / maxpool
    - layer1..layer4 residual stages
    - avgpool
    - fc: Linear(..., out_features=1000)  # ImageNet head
    """
    return resnet18(weights=weights)


def _replace_fc(model: nn.Module, num_classes: int, dropout: float = 0.0) -> None:
    in_features = model.fc.in_features  # type: ignore[attr-defined]
    if dropout and dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=float(dropout)), nn.Linear(in_features, num_classes))  # type: ignore[attr-defined]
    else:
        model.fc = nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]


def set_trainable_policy(
    model: nn.Module,
    freeze_backbone: bool,
    unfreeze_last: bool,
    unfreeze_all: bool,
) -> None:
    # Default: everything trainable unless policy freezes it.
    for p in model.parameters():
        p.requires_grad = True

    if unfreeze_all:
        return

    if freeze_backbone:
        # Train only the classifier head by default.
        for name, p in model.named_parameters():
            if name.startswith("fc."):
                p.requires_grad = True
            else:
                p.requires_grad = False

        if unfreeze_last:
            # Unfreeze layer4 + fc (common TL step-up).
            for name, p in model.named_parameters():
                if name.startswith("layer4.") or name.startswith("fc."):
                    p.requires_grad = True


def set_batchnorm_eval(model: nn.Module) -> None:
    # Common TL practice when freezing most layers: keep BN running stats fixed.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def build_resnet18_tl(
    num_classes: int = 2,
    weights: ResNet18_Weights = ResNet18_Weights.DEFAULT,
    freeze_backbone: bool = True,
    unfreeze_last: bool = False,
    unfreeze_all: bool = False,
    dropout: float = 0.0,
) -> nn.Module:
    model = build_resnet18_base(weights=weights)
    _replace_fc(model, num_classes=num_classes, dropout=dropout)
    set_trainable_policy(
        model=model,
        freeze_backbone=freeze_backbone,
        unfreeze_last=unfreeze_last,
        unfreeze_all=unfreeze_all,
    )
    return model


def print_resnet18_highlights(model: nn.Module) -> None:
    # Full structure:
    print(model)

    # Highlights for the chapter demo:
    print("\nRESNET-18 HIGHLIGHTS:")
    for key in ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "fc"]:
        if hasattr(model, key):
            print(f"  {key}: {getattr(model, key)}")
