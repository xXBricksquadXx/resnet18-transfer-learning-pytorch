from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn

from torchvision.models import ResNet18_Weights

from resnet_tl.data import make_dataloaders
from resnet_tl.engine import train_one_epoch, evaluate
from resnet_tl.io import save_checkpoint
from resnet_tl.models import (
    build_resnet18_tl,
    build_resnet18_base,
    print_resnet18_highlights,
    set_batchnorm_eval,
)
from resnet_tl.utils import set_seed, get_device, count_trainable_params


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transfer learning with pretrained ResNet-18 (torchvision).")

    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)

    p.add_argument("--freeze-backbone", type=int, choices=[0, 1], default=1)
    p.add_argument("--unfreeze-last", type=int, choices=[0, 1], default=0)
    p.add_argument("--unfreeze-all", type=int, choices=[0, 1], default=0)

    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--amp", type=int, choices=[0, 1], default=0)

    p.add_argument("--run-name", type=str, default="resnet18")
    # When enabled, prints BOTH:
    # 1) The base pretrained ResNet-18 (fc out_features=1000)
    # 2) The transfer-learning model (fc replaced for your classes)
    p.add_argument("--print-model", type=int, choices=[0, 1], default=0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    set_seed(args.seed, deterministic=True)
    device = get_device(args.device)

    weights = ResNet18_Weights.DEFAULT

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{args.run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Make RUN_DIR easy to parse for the PowerShell script
    print(f"RUN_DIR: {run_dir.resolve()}")

    train_loader, val_loader, spec = make_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        weights=weights,
    )

    print("\nDATASET:")
    print(f"  train: {spec.train_count} images, classes={spec.class_names}")
    print(f"  val:   {spec.val_count} images, classes={spec.class_names}")

    # Build TL model (head replaced)
    model = build_resnet18_tl(
        num_classes=len(spec.class_names),
        weights=weights,
        freeze_backbone=bool(args.freeze_backbone),
        unfreeze_last=bool(args.unfreeze_last),
        unfreeze_all=bool(args.unfreeze_all),
        dropout=float(args.dropout),
    ).to(device)

    print("\nMODEL:")
    print("  torchvision: resnet18(weights=ResNet18_Weights.DEFAULT)")
    print(f"  head: fc -> num_classes={len(spec.class_names)} (dropout={args.dropout})")
    print(f"  freeze_backbone={args.freeze_backbone} unfreeze_last={args.unfreeze_last} unfreeze_all={args.unfreeze_all}")
    print(f"  trainable_params={count_trainable_params(model):,}")

    if args.print_model:
        # Print the exact base structure you pasted (fc=1000) AND the TL variant.
        print("\n=== PRETRAINED RESNET-18 (BASE, IMAGENET HEAD fc=1000) ===")
        base = build_resnet18_base(weights=weights).cpu()
        print_resnet18_highlights(base)

        print("\n=== TRANSFER-LEARNING RESNET-18 (HEAD REPLACED) ===")
        print_resnet18_highlights(model)

    # If we're freezing most of the model, keep BatchNorm stats fixed (common TL practice)
    freeze_like = bool(args.freeze_backbone) and not bool(args.unfreeze_all)
    if freeze_like:
        set_batchnorm_eval(model)

    criterion = nn.CrossEntropyLoss()

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    amp = bool(args.amp)
    if amp and device.type != "cuda":
        print("NOTE: --amp 1 requested on CPU; disabling AMP.")
        amp = False

    # Save args for the run
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    best_val_acc = -1.0
    best_acc_epoch = -1
    best_val_loss = float("inf")
    best_loss_epoch = -1

    history = []

    for epoch in range(1, args.epochs + 1):
        if freeze_like:
            set_batchnorm_eval(model)

        train_res = train_one_epoch(model, train_loader, criterion, optimizer, device=device, amp=amp)
        val_res = evaluate(model, val_loader, criterion, device=device)

        lr_now = optimizer.param_groups[0]["lr"]
        line = (
            f"EPOCH {epoch}/{args.epochs}  "
            f"train_loss={train_res.loss:.4f} train_acc={train_res.acc:.4f}  "
            f"val_loss={val_res.loss:.4f} val_acc={val_res.acc:.4f}  "
            f"lr={lr_now:.6g}"
        )
        print(line)

        history.append(
            dict(
                epoch=epoch,
                train_loss=train_res.loss,
                train_acc=train_res.acc,
                val_loss=val_res.loss,
                val_acc=val_res.acc,
                lr=lr_now,
            )
        )

        ckpt: Dict[str, Any] = dict(
            epoch=epoch,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            best_val_acc=best_val_acc,
            best_val_loss=best_val_loss,
            class_names=spec.class_names,
            weights=str(weights),
            args=vars(args),
            history=history,
        )

        save_checkpoint(run_dir / "latest.pt", {**ckpt, "best_epoch": best_acc_epoch})

        if val_res.acc > best_val_acc:
            best_val_acc = val_res.acc
            best_acc_epoch = epoch
            save_checkpoint(run_dir / "best_acc.pt", {**ckpt, "best_epoch": best_acc_epoch, "best_val_acc": best_val_acc})

        if val_res.loss < best_val_loss:
            best_val_loss = val_res.loss
            best_loss_epoch = epoch
            save_checkpoint(run_dir / "best_loss.pt", {**ckpt, "best_epoch": best_loss_epoch, "best_val_loss": best_val_loss})

    print("\nBEST:")
    print(f"  best_acc={best_val_acc:.4f} at epoch={best_acc_epoch}")
    print(f"  best_loss={best_val_loss:.4f} at epoch={best_loss_epoch}")
    print(f"  best_acc_ckpt:  {(run_dir / 'best_acc.pt').resolve()}")
    print(f"  best_loss_ckpt: {(run_dir / 'best_loss.pt').resolve()}")


if __name__ == "__main__":
    main()
