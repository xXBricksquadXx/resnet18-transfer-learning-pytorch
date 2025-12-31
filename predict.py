from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from torchvision.models import ResNet18_Weights

from resnet_tl.io import load_checkpoint
from resnet_tl.models import build_resnet18_tl
from resnet_tl.utils import get_device


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict with a trained ResNet-18 transfer learning checkpoint.")
    p.add_argument("--image", type=str, required=True, help="Path to an image file OR a folder (folder mode).")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint (best_acc.pt recommended).")
    p.add_argument("--device", type=str, choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--random", type=int, choices=[0, 1], default=0, help="If --image is a folder, sample random images.")
    p.add_argument("--n", type=int, default=1, help="If folder mode, number of images to predict.")
    return p.parse_args()


def list_images(folder: Path) -> List[Path]:
    files = [p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS and p.is_file()]
    return sorted(files)


def pick_images(path: Path, random_mode: bool, n: int) -> List[Path]:
    if path.is_file():
        return [path]

    imgs = list_images(path)
    if not imgs:
        raise FileNotFoundError(f"No images found under: {path}")

    n = max(1, min(n, len(imgs)))
    if random_mode:
        return random.sample(imgs, k=n)
    return imgs[:n]


@torch.no_grad()
def predict_one(
    model: torch.nn.Module,
    img_path: Path,
    device: torch.device,
    weights: ResNet18_Weights,
    class_names: List[str],
    topk: int,
) -> List[Tuple[str, float]]:
    tfm = weights.transforms()

    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)

    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0)

    k = min(topk, probs.numel())
    vals, idxs = torch.topk(probs, k=k)

    out = []
    for p, i in zip(vals.tolist(), idxs.tolist()):
        name = class_names[i] if i < len(class_names) else str(i)
        out.append((name, float(p)))
    return out


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    ckpt = load_checkpoint(args.checkpoint, device=str(device))
    class_names = ckpt.get("class_names") or ["class0", "class1"]

    weights = ResNet18_Weights.DEFAULT
    model = build_resnet18_tl(
        num_classes=len(class_names),
        weights=weights,
        freeze_backbone=False,   # irrelevant for inference
        unfreeze_last=False,
        unfreeze_all=True,
        dropout=float(ckpt.get("args", {}).get("dropout", 0.0)),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    path = Path(args.image)
    imgs = pick_images(path, random_mode=bool(args.random), n=int(args.n))

    print(f"CHECKPOINT: {Path(args.checkpoint).resolve()}")
    print(f"  best_epoch: {ckpt.get('best_epoch')}")
    print(f"  best_val_acc: {ckpt.get('best_val_acc')}")
    print(f"  class_names: {class_names}")
    print("")

    for img_path in imgs:
        top = predict_one(model, img_path, device=device, weights=weights, class_names=class_names, topk=args.topk)
        print(f"IMAGE: {img_path}")
        print("TOPK:")
        for rank, (name, p) in enumerate(top, start=1):
            print(f"  {rank}) {name}  p={p:.4f}")
        print("")


if __name__ == "__main__":
    main()
