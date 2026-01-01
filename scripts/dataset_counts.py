from pathlib import Path
from torchvision.datasets import ImageFolder


def report(split: str) -> None:
    d = Path("data") / split
    if not d.exists():
        print(f"{split}: MISSING ({d})")
        return

    ds = ImageFolder(d)
    per = {c: 0 for c in ds.classes}
    for _, y in ds.samples:
        per[ds.classes[y]] += 1

    print(f"{split}: {len(ds)} images | classes={ds.classes} | per_class={per}")


if __name__ == "__main__":
    report("train")
    report("val")
