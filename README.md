## ![Header](assets/header-banner.png)

# ResNet-18 Transfer Learning (PyTorch) — pretrained models, fine-tuning, prediction, saving/loading

A compact, practical reference for this chapter’s workflow:

- load a **pretrained ResNet-18** from `torchvision.models` (one-stop shopping)
- examine the model structure (spot **BatchNorm** + **MaxPool**)
- perform **transfer learning** (replace `fc` for **2 classes**: `cat`, `fish`)
- run a clean **train → val** loop with metrics each epoch
- run **single-image** + **folder** predictions
- save and restore models:

  - **state_dict** (weights-only)
  - **checkpoint dict** (resume + inference; recommended)
  - **best-checkpoint** selection (recommended)
  - **full-model demo** (brittle; optional)

This repo is intentionally small so you can iterate on:

- freeze vs unfreeze policy (head-only → `layer4` → full fine-tune)
- learning rate + weight decay
- batch size
- dropout in the head (optional)
- data augmentation strength

---

## Baseline demo (screen recording)

▶ **Baseline video:** [assets/test_run.mp4](assets/test_run.mp4)

[![Watch the video](https://img.shields.io/badge/▶_Watch-Baseline_Video-blue?style=for-the-badge)](https://github.com/user-attachments/assets/84002fcb-69d6-42b0-ab9b-260f2c57e308)

<div align="center">
  <a href="https://github.com/user-attachments/assets/84002fcb-69d6-42b0-ab9b-260f2c57e308">
  </a>
</div>

Notes:

- GitHub may not autoplay MP4 inside the README; the link should open/download the file.

---

## Repo layout

```
resnet18-transfer-learning-pytorch/
  assets/                  # visuals only
  data/                    # training-only images (ImageFolder)
    train/
      cat/
      fish/
    val/
      cat/
      fish/
  logs/                    # transcripts (gitignored)
  runs/                    # outputs/checkpoints (gitignored)
  scripts/
    dataset_counts.py      # dataset headcount helper
    test_run.ps1           # one-shot demo: train + predict + transcript to logs/
  resnet_tl/               # package
    __init__.py
    data.py
    engine.py
    io.py
    models.py
    utils.py
  train.py                 # CLI entrypoint
  predict.py               # CLI entrypoint
  requirements.txt
  .gitignore
```

Notes:

- `data/` is for model training only.
- `assets/` is for banners/icons/screenshots/video so they never leak into training.

---

## 1) Setup

```bash
python -m venv .venv

# Windows (PowerShell)
./.venv/Scripts/Activate.ps1

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Sanity:

```bash
python -c "import torch, torchvision; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('cuda:', torch.cuda.is_available())"
```

---

## 2) Dataset (ImageFolder)

This uses `torchvision.datasets.ImageFolder`.

Expected folder structure:

```
data/
  train/
    cat/*.png
    fish/*.png
  val/
    cat/*.png
    fish/*.png
```

Convention:

- `data/train/*` = **clean** examples
- `data/val/*` = **challenge** examples (harder / different distribution)

### Current dataset size

- train/cat: **20**
- train/fish: **20**
- val/cat: **16**
- val/fish: **16**

Total validation images: **32**

---

## 3) Examine the pretrained model structure (the exact one we preserve)

This chapter demo preserves the canonical torchvision ResNet-18 backbone:

- `conv1`
- `bn1` (**BatchNorm**)
- `relu`
- `maxpool` (**MaxPool**)
- residual stages: `layer1`, `layer2`, `layer3`, `layer4`
- `avgpool`
- `fc` (ImageNet head is **1000** outputs)

Print both the base pretrained structure **and** the transfer-learning version:

```powershell
python train.py --data-dir data --device cpu --epochs 1 --batch-size 8 --num-workers 0 --seed 1337 --print-model 1
```

What you should see (base): `fc: Linear(in_features=512, out_features=1000, bias=True)`.

---

## 4) Baseline run (ResNet-18 TL) — recommended config

This is the simplest, most demo-friendly baseline:

- freeze the backbone (train only the new `fc` head)
- track best checkpoint by validation accuracy
- keep everything deterministic with a seed

### One-shot demo (recommended for screen recording)

```powershell
./scripts/test_run.ps1
```

### Manual train (same settings as the script)

```powershell
python train.py `
  --data-dir data `
  --device cpu `
  --epochs 10 `
  --batch-size 16 `
  --lr 0.001 `
  --weight-decay 0.0001 `
  --freeze-backbone 1 `
  --unfreeze-last 0 `
  --unfreeze-all 0 `
  --seed 1337 `
  --num-workers 0 `
  --amp 0
```

### Outputs

Training creates a timestamped run folder:

- `runs/<run_name>_<timestamp>/latest.pt` (last epoch checkpoint)
- `runs/<run_name>_<timestamp>/best_acc.pt` (**best** checkpoint by val accuracy; recommended for inference)
- `runs/<run_name>_<timestamp>/best_loss.pt` (best by val loss)
- `runs/<run_name>_<timestamp>/args.json` (config snapshot)

---

## 5) Predict

Single file:

```powershell
python predict.py `
  --image "data/val/cat/cat-challenge-001.png" `
  --checkpoint "runs/<...>/best_acc.pt" `
  --device cpu `
  --topk 2
```

Directory input (picks the first image in the folder):

```powershell
python predict.py `
  --image "data/val/cat" `
  --checkpoint "runs/<...>/best_acc.pt" `
  --device cpu `
  --topk 2
```

Random sampling mode:

```powershell
python predict.py `
  --image "data/val" `
  --checkpoint "runs/<...>/best_acc.pt" `
  --device cpu `
  --topk 2 `
  --random 1 `
  --n 5
```

---

## 6) Why BEST checkpoint matters

Small datasets often show **late-epoch drift / overfit**.

Even if train accuracy stays high, validation loss/accuracy can become volatile.

For demos + inference, prefer:

- `best_acc.pt`

Baseline evidence:

- **Best checkpoint (by val accuracy):** epoch **6** → val_acc **1.0000** (32/32)
- **Last epoch (epoch 10):** val_acc **0.9688** (31/32)

So for prediction demos, prefer:

- `runs/resnet18_20251231_191000/best_acc.pt`

---

## 7) Final results (baseline run: head-only transfer learning)

Environment:

- PyTorch: `2.9.1+cpu`
- torchvision: `0.24.1+cpu`
- CUDA available: `False` (CPU training)

Dataset:

- train/cat: **20**
- train/fish: **20**
- val/cat: **16**
- val/fish: **16**

One-shot demo command:

```powershell
./scripts/test_run.ps1
```

Logs + outputs:

- Transcript: `logs/transcript_20251231_190948.txt`
- Train log: `logs/train_20251231_190948.txt`
- `RUN_DIR: runs/resnet18_20251231_191000`

Final training log highlights:

- **Peak validation accuracy:** **1.0000** at **epoch 6** (32/32 correct)
- **Best val loss:** **0.1739** at **epoch 9**
- **Last epoch:** epoch 10 → val_acc **0.9688** (31/32)

Epoch lines (verbatim):

```text
EPOCH 1/10  train_loss=0.8240 train_acc=0.4250  val_loss=0.6595 val_acc=0.5625  lr=0.001
EPOCH 2/10  train_loss=0.5720 train_acc=0.7000  val_loss=0.5295 val_acc=0.7500  lr=0.001
EPOCH 3/10  train_loss=0.4206 train_acc=1.0000  val_loss=0.4720 val_acc=0.7812  lr=0.001
EPOCH 4/10  train_loss=0.3831 train_acc=0.8750  val_loss=0.4586 val_acc=0.6875  lr=0.001
EPOCH 5/10  train_loss=0.2418 train_acc=0.9500  val_loss=0.3252 val_acc=0.9062  lr=0.001
EPOCH 6/10  train_loss=0.1858 train_acc=1.0000  val_loss=0.2504 val_acc=1.0000  lr=0.001
EPOCH 7/10  train_loss=0.1544 train_acc=1.0000  val_loss=0.2097 val_acc=1.0000  lr=0.001
EPOCH 8/10  train_loss=0.1408 train_acc=1.0000  val_loss=0.1882 val_acc=1.0000  lr=0.001
EPOCH 9/10  train_loss=0.1637 train_acc=0.9750  val_loss=0.1739 val_acc=1.0000  lr=0.001
EPOCH 10/10  train_loss=0.1215 train_acc=1.0000  val_loss=0.1864 val_acc=0.9688  lr=0.001
```

Checkpoint summary:

- best_acc: **1.0000** at epoch **6** → `runs/resnet18_20251231_191000/best_acc.pt`
- best_loss: **0.1739** at epoch **9** → `runs/resnet18_20251231_191000/best_loss.pt`

Predictions (random sample from `data/val` using `best_acc.pt`):

```text
IMAGE: data/val/fish/fish-challenge-002.png
TOPK:
  1) fish  p=0.7005
  2) cat   p=0.2995

IMAGE: data/val/fish/fish-challenge-013.png
TOPK:
  1) fish  p=0.6674
  2) cat   p=0.3326

IMAGE: data/val/fish/fish-challenge-007.png
TOPK:
  1) fish  p=0.7594
  2) cat   p=0.2406

IMAGE: data/val/cat/cat-challenge-001.png
TOPK:
  1) cat   p=0.7898
  2) fish  p=0.2102

IMAGE: data/val/fish/fish-challenge-010.png
TOPK:
  1) fish  p=0.6416
  2) cat   p=0.3584
```

Takeaway: pretrained ResNet-18 features are strong enough that a tiny head-only fine-tune converges quickly on CPU.

---

## 8) Saving & loading (what to remember)

### A) Full model object (works, but brittle)

```py
import torch

# SAVE
torch.save(model, "runs/resnet18_full_model.pt")

# LOAD
model = torch.load("runs/resnet18_full_model.pt", map_location=device)
```

### B) Weights only (recommended)

```py
import torch

# SAVE
torch.save(model.state_dict(), "runs/resnet18_state_dict.pt")

# LOAD (rebuild architecture first)
model = ...  # build_resnet18_tl(...)
model.load_state_dict(torch.load("runs/resnet18_state_dict.pt", map_location=device))
```

### C) Checkpoint dict (recommended for real work)

Includes:

- model weights
- optimizer state
- epoch
- class names
- run metadata (args, weights enum)

This is what `predict.py` uses.

---

## 9) BatchNorm / MaxPool (mapped to ResNet-18)

- **BatchNorm**: `bn1` (and BNs inside each residual `BasicBlock`)

  - stabilizes training by normalizing activations
  - in transfer learning, it’s common to **freeze** most pretrained layers and keep BN behavior stable

- **MaxPool**: `maxpool`

  - down-samples early feature maps
  - reduces compute and increases receptive field quickly

---

## 10) torch.hub.load(...)

`torchvision.models` is the primary, stable “one-stop shopping” interface.

An alternative is loading via Torch Hub:

```py
import torch

model = torch.hub.load(
    "pytorch/vision",
    "resnet18",
    weights="ResNet18_Weights.IMAGENET1K_V1",
)
```

For this chapter repo, we keep the primary path:

- `from torchvision.models import resnet18, ResNet18_Weights`

---

## Troubleshooting

If training is slow on CPU:

- lower epochs (5–10) for a clean demo
- reduce batch size if memory is tight

If validation behavior is noisy:

- that’s expected on small data; use `best_acc.pt`
- try mild augmentation or a slightly higher dropout in the head

---

## Close-out analysis

- pretrained backbone (ResNet-18) provides strong features; the main learning happens in the `fc` head first.
- best checkpoint selection is essential on small datasets (validation drift is normal).
- next iteration path:

  - Phase 1: freeze backbone (baseline)
  - Phase 2: unfreeze `layer4` + `fc` (lower LR)
  - Phase 3: full fine-tune (very low LR; careful)
