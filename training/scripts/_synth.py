"""A tiny synthetic dataset in the exact layout datasets/scripts/09_build_yolo_ds.py emits.

Used by smoke_test.py and the tests to exercise train, evaluate, sweep, export and the hard-set
mapping end to end without the real corpus. The images are coloured rectangles on noise: enough
for a 2-epoch fine-tune to learn something and for every code path to run. NOTHING measured on
this data may be reported as a result; it exists to prove the tooling runs.

Layout produced under `root`:

    images/{train,val}/<source>_<raw stem>_<visible|lwir>.{jpg,png}
    labels/{train,val}/<same stem>.txt
    data.yaml                 (path/train/val/nc/names, no test key, as 09 writes it)
    manifest.tsv              (split, relative image path, label count)
    index/split.jsonl         (one record per image: source, modality, set, sequence_key, image)
    hard_set.txt              (raw image paths from split.jsonl, the form 06_split.py writes)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

NAMES = ["person", "two_wheeler", "car", "truck", "cart"]
_FILL = {0: (60, 60, 230), 1: (60, 200, 60), 2: (230, 80, 60), 3: (60, 220, 220)}  # BGR


def _background(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    base = rng.integers(70, 130, size=(h // 8 + 1, w // 8 + 1, 3), dtype=np.uint8)
    import cv2

    img = cv2.resize(base, (w, h), interpolation=cv2.INTER_CUBIC)
    noise = rng.normal(0, 6, size=img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _draw(img: np.ndarray, cls: int, cx: int, cy: int, bw: int, bh: int, ir: bool) -> None:
    import cv2

    x1, y1, x2, y2 = cx - bw // 2, cy - bh // 2, cx + bw // 2, cy + bh // 2
    color = (235, 235, 235) if ir else _FILL[cls]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)
    if cls == 0 and bh > 12:
        cv2.circle(img, (cx, y1 + max(2, bw // 3)), max(2, bw // 3), (255, 255, 255) if ir else (200, 200, 250), -1)


def make_dataset(
    root: str | Path,
    n_train: int = 48,
    n_val: int = 24,
    seed: int = 0,
    size: tuple[int, int] = (320, 256),
) -> dict:
    """Create the dataset. Returns a summary dict (counts per split/modality)."""
    import cv2

    root = Path(root)
    rng = np.random.default_rng(seed)
    w, h = size
    records: list[dict] = []
    summary: dict[str, int] = {}

    # (split, source, modality, kaist set) plan. Val holds both a daylight and a night KAIST set,
    # LLVIP and IDD frames, so every slice and every lighting sub-slice is populated.
    def plan(split: str, n: int):
        out = []
        for i in range(n):
            r = i % 10
            if split == "train":
                kind = ["idd", "idd", "kaist03", "kaist03", "llvip", "kaist00", "idd", "llvip", "kaist04", "kaist00"][r]
            else:
                kind = ["kaist06v", "kaist06i", "kaist09i", "kaist09v", "llvip_i", "idd", "llvip_v", "kaist06i", "kaist09i", "kaist06v"][r]
            out.append((split, i, kind))
        return out

    for split, n in (("train", n_train), ("val", n_val)):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for split_, i, kind in plan(split, n):
            if kind.startswith("kaist"):
                set_name = "set" + kind[5:7]
                modality = "lwir" if kind.endswith("i") or (kind in ("kaist03", "kaist04")) else "visible"
                if kind in ("kaist00",):
                    modality = "visible"
                video = f"V{i % 3:03d}"
                raw = f"{set_name}_{video}_I{i:05d}"
                source, seq = "kaist", f"kaist/{set_name}/{video}"
            elif kind.startswith("llvip"):
                modality = "lwir" if kind == "llvip_i" or (kind == "llvip" and i % 2 == 0) else "visible"
                raw = f"{(i % 4) + 1:02d}{i:04d}"
                source, set_name, seq = "llvip", None, f"llvip/{(i % 4) + 1:02d}"
            else:
                modality, raw = "visible", f"{i:07d}_leftImg8bit"
                source, set_name, seq = "idd", None, f"idd/drive{i % 3}"

            ir = modality == "lwir"
            img = _background(rng, w, h)
            lines = []
            n_obj = int(rng.integers(0, 5))
            classes = [0] if (ir or source in ("kaist", "llvip")) else [0, 0, 1, 2, 3]
            for _ in range(n_obj):
                c = int(rng.choice(classes))
                bh = int(rng.choice([10, 14, 20, 28, 40, 60, 90])) if c == 0 else int(rng.integers(24, 70))
                bw = max(4, bh // 2) if c == 0 else int(bh * rng.uniform(1.2, 2.2))
                if bw >= w - 4 or bh >= h - 4:
                    continue
                cx = int(rng.integers(bw // 2 + 2, w - bw // 2 - 2))
                cy = int(rng.integers(bh // 2 + 2, h - bh // 2 - 2))
                _draw(img, c, cx, cy, bw, bh, ir)
                lines.append(f"{c} {cx / w:.6f} {cy / h:.6f} {bw / w:.6f} {bh / h:.6f}")
            if ir:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = np.repeat(gray[..., None], 3, axis=2)  # B = G = R, as 04_ir_to_3ch.py writes

            stem = f"{source}_{raw}_{modality}"
            suffix = ".png" if ir else ".jpg"
            cv2.imwrite(str(root / "images" / split / f"{stem}{suffix}"), img)
            (root / "labels" / split / f"{stem}.txt").write_text("".join(f"{ln}\n" for ln in lines), encoding="utf-8")
            record = {
                "image": f"/synthetic/processed/{source}/{raw}{suffix}",
                "source": source,
                "modality": modality,
                "sequence_key": seq,
                "split": split,
            }
            if set_name:
                record["set"] = set_name
            records.append(record)
            summary[f"{split}/{modality}"] = summary.get(f"{split}/{modality}", 0) + 1

    (root / "index").mkdir(exist_ok=True)
    (root / "index" / "split.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), encoding="utf-8")
    val_records = [r for r in records if r["split"] == "val"]
    (root / "hard_set.txt").write_text("".join(r["image"] + "\n" for r in val_records[: max(4, len(val_records) // 2)]), encoding="utf-8")
    (root / "manifest.tsv").write_text(
        "".join(
            f"{s}\timages/{s}/{p.name}\t{len(_lines(root / 'labels' / s / (p.stem + '.txt')))}\n"
            for s in ("train", "val")
            for p in sorted((root / "images" / s).iterdir())
        ),
        encoding="utf-8",
    )
    (root / "data.yaml").write_text(
        "# synthetic fixture in the layout datasets/scripts/09_build_yolo_ds.py emits\n"
        f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnc: {len(NAMES)}\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)),
        encoding="utf-8",
    )
    return summary


def _lines(path: Path) -> list[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root")
    ap.add_argument("--train", type=int, default=48)
    ap.add_argument("--val", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(json.dumps(make_dataset(a.root, a.train, a.val, a.seed), indent=2))
