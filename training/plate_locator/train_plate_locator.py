"""Train a single-class YOLO11n licence-plate detector on Kaggle.

Kaggle mounts the dataset under /kaggle/input (either /kaggle/input/<slug>/ or
/kaggle/input/datasets/<owner>/<slug>/) and auto-extracts the tar. This script
finds data.yaml, rewrites its root, trains, evaluates on the held-out TEST split,
exports ONNX and copies the artefacts to /kaggle/working/results/.

Local smoke test:  python plate_det.py --smoke --input <dir with data.yaml> --work <out dir>
(2 epochs on CPU, 200-image subset.)
"""
import argparse, glob, json, os, random, shutil, subprocess, sys, tarfile, time

ULTRA_VERSION = "8.4.155"

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true", default=os.environ.get("PLATEDET_SMOKE") == "1")
ap.add_argument("--input", default="/kaggle/input")
ap.add_argument("--work", default="/kaggle/working")
ap.add_argument("--weights", default="yolo11n.pt")
args, _ = ap.parse_known_args()

try:
    import ultralytics  # noqa: F401
    if ultralytics.__version__ != ULTRA_VERSION and not args.smoke:
        raise ImportError
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"ultralytics=={ULTRA_VERSION}"])
import ultralytics
from ultralytics import YOLO

WORK = os.path.abspath(args.work)
RES = os.path.join(WORK, "results")
os.makedirs(RES, exist_ok=True)
print("ultralytics", ultralytics.__version__, "| input", args.input, "| smoke", args.smoke, flush=True)


def find_root(base):
    """Return the folder holding data.yaml + images/train (tar may or may not be extracted)."""
    cands = []
    for dp, dn, fn in os.walk(base, followlinks=True):
        if "data.yaml" in fn and os.path.isdir(os.path.join(dp, "images", "train")):
            cands.append(dp)
    if cands:
        cands.sort(key=len)
        return cands[0]
    tars = glob.glob(os.path.join(base, "**", "*.tar"), recursive=True)
    for t in tars:  # not auto-extracted: unpack into working dir
        dst = os.path.join(WORK, "data_extracted")
        print("extracting", t, "->", dst, flush=True)
        with tarfile.open(t) as tf:
            tf.extractall(dst)
        return find_root(dst)
    sys.exit(f"data.yaml with images/train not found under {base}")


root = find_root(args.input)
print("dataset root:", root, flush=True)
for s in ("train", "val", "test"):
    print(f"  {s}: {len(os.listdir(os.path.join(root, 'images', s)))} images", flush=True)

if args.smoke:
    # 200-image subset (160 train / 20 val / 20 test) copied into the work dir
    sub = os.path.join(WORK, "smoke_data")
    shutil.rmtree(sub, ignore_errors=True)
    rnd = random.Random(0)
    for s, n in (("train", 160), ("val", 20), ("test", 20)):
        ims = sorted(os.listdir(os.path.join(root, "images", s)))
        os.makedirs(os.path.join(sub, "images", s)); os.makedirs(os.path.join(sub, "labels", s))
        for f in rnd.sample(ims, min(n, len(ims))):
            shutil.copy(os.path.join(root, "images", s, f), os.path.join(sub, "images", s, f))
            lab = os.path.splitext(f)[0] + ".txt"
            shutil.copy(os.path.join(root, "labels", s, lab), os.path.join(sub, "labels", s, lab))
    root = sub

data_yaml = os.path.join(WORK, "data_patched.yaml")
with open(data_yaml, "w") as fh:
    fh.write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\n"
             "nc: 1\nnames: [\"plate\"]\n")
print(open(data_yaml).read(), flush=True)

if args.smoke:
    cfg = dict(imgsz=320, epochs=2, patience=20, batch=8, device="cpu", workers=0)
else:
    import torch
    if not torch.cuda.is_available():
        sys.exit("No GPU visible: enable a GPU accelerator for this notebook")
    print("GPU:", torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0), flush=True)
    cfg = dict(imgsz=640, epochs=80, patience=20, batch=32, device=0, workers=4)

t0 = time.time()
model = YOLO(args.weights)
model.train(data=data_yaml, cos_lr=True, mosaic=1.0, project=os.path.join(WORK, "runs"),
            name="platedet", exist_ok=True, seed=0, plots=True, single_cls=True, **cfg)
run_dir = os.path.join(WORK, "runs", "platedet")
best = os.path.join(run_dir, "weights", "best.pt")
print("train done in %.1f min; best=%s" % ((time.time() - t0) / 60, best), flush=True)

# held-out TEST split
m = YOLO(best).val(data=data_yaml, split="test", imgsz=cfg["imgsz"], batch=cfg["batch"],
                   device=cfg["device"], project=os.path.join(WORK, "runs"), name="platedet_test",
                   exist_ok=True, plots=True)
metrics = {
    "split": "test",
    "images": len(os.listdir(os.path.join(root, "images", "test"))),
    "mAP50": float(m.box.map50),
    "mAP50-95": float(m.box.map),
    "P": float(m.box.mp),
    "R": float(m.box.mr),
    "imgsz": cfg["imgsz"],
    "epochs_trained": None,
    "smoke": args.smoke,
    "ultralytics": ultralytics.__version__,
}
res_csv = os.path.join(run_dir, "results.csv")
if os.path.exists(res_csv):
    metrics["epochs_trained"] = sum(1 for _ in open(res_csv)) - 1
with open(os.path.join(RES, "platedet_test.json"), "w") as fh:
    json.dump(metrics, fh, indent=2)
print(json.dumps(metrics, indent=2), flush=True)

for f in (best, res_csv):
    if os.path.exists(f):
        shutil.copy(f, RES)
try:
    onnx = YOLO(best).export(format="onnx", opset=17, dynamic=True, imgsz=cfg["imgsz"], device="cpu")
    shutil.copy(onnx, os.path.join(RES, "best.onnx"))
except Exception as e:  # keep the metrics/weights even if export fails
    print("ONNX export failed:", repr(e), flush=True)
print("results:", sorted(os.listdir(RES)), flush=True)
