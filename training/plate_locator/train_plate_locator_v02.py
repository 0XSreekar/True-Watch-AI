"""Fine-tune the v0.1 plate locator into v0.2 locally on Apple MPS (ultralytics, AGPL: training/ only).

  python train_plate_locator_v02.py --weights <v0.1 best.pt> --data <build_v02_data.py out>/ds/data.yaml \
      --work <dir> [--epochs 40] [--smoke]
Writes <work>/runs/v02/weights/best.pt and <work>/plate_locator_v0.2.onnx (opset 17, dynamic, imgsz 640,
same interface as v0.1: input `images`, output `output0` (batch, 5, anchors)).
"""
import argparse
import shutil
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--work", required=True)
ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--resume", action="store_true", help="continue from <work>/runs/v02/weights/last.pt")
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
if not torch.backends.mps.is_available():
    raise SystemExit("MPS unavailable: not training on CPU")
work = Path(a.work).resolve()
t0 = time.time()
model = None if a.resume else YOLO(a.weights)
kw = dict(fraction=0.02, epochs=1) if a.smoke else dict(epochs=a.epochs)
if a.resume:
    model = YOLO(str(work / "runs" / "v02" / "weights" / "last.pt"))
    model.train(resume=True)
else:
  model.train(data=a.data, imgsz=640, batch=a.batch, device="mps", workers=4, patience=10, optimizer="AdamW",
              lr0=0.0005, lrf=0.1, cos_lr=True, warmup_epochs=1, mosaic=1.0, close_mosaic=5, hsv_v=0.6, hsv_s=0.5,
              seed=0, single_cls=True, project=str(work / "runs"), name="v02", exist_ok=True, plots=False, **kw)
best = work / "runs" / "v02" / "weights" / "best.pt"
print("train minutes %.1f" % ((time.time() - t0) / 60), flush=True)
m = YOLO(str(best)).val(data=a.data, split="test", imgsz=640, device="mps", plots=False,
                        project=str(work / "runs"), name="v02_test", exist_ok=True)
print("test mAP50 %.4f R %.4f P %.4f" % (m.box.map50, m.box.mr, m.box.mp), flush=True)
onnx = YOLO(str(best)).export(format="onnx", opset=17, dynamic=True, imgsz=640, device="cpu")
shutil.copy(onnx, work / "plate_locator_v0.2.onnx")
