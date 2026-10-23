# Round G7 of the PP-OCRv5 Devanagari line recogniser on a T4: continue from round G6 on a mix of
#  - lines cut from REAL training photographs by the ONNX plate locator + rectify.prepare_lines (inference-style crops),
#    with the lines the G6 reader got wrong repeated more, and
#  - replay of the G6 training data (every real set in full, a sample of the synthetic corpora).
# Same PaddleOCR (v3.3.0) and Paddle (3.3.1) as the earlier rounds.
import glob, os, random, re, shutil, subprocess, time
from collections import Counter
from pathlib import Path

def sh(cmd, cwd=None):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

WORK, TMP = Path("/kaggle/working"), Path("/kaggle/temp")
TMP.mkdir(exist_ok=True)

def one(pattern, must=""):
    hits = sorted((h for h in glob.glob(f"/kaggle/input/**/{pattern}", recursive=True) if must in h), key=len)
    print(pattern, "->", hits[:3], flush=True)
    return Path(hits[0])

root6 = one("train_list.txt", "lines6").parent
root7 = one("lines7_list.txt", "lines7").parent
config_template = one("round_gpu.yml", "lines7")
init_weights = one("output/model/latest.pdparams", "gpu6").with_suffix("")  # round G6 final checkpoint

# --- build the G7 mix in a symlink farm so every list path stays relative to one data root -------------------------
farm = TMP / "g7data"
farm.mkdir()
for d in root6.iterdir():
    if d.is_dir():
        (farm / d.name).symlink_to(d, target_is_directory=True)
(farm / "lines7").symlink_to(root7 / "lines7", target_is_directory=True)

rng = random.Random(7)
REAL = ("hfreal2", "hfreal7", "pseudo", "real7", "r2", "rf_college", "rf_prajwal", "rf_sushant")
SYN_KEEP = 0.25
rows6 = [l for l in (root6 / "train_list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
replay = []
for l in rows6:
    top = l.split("/", 1)[0]
    if top in REAL or rng.random() < SYN_KEEP:
        replay.append(l)
wrong = set(l.strip() for l in (root7 / "lines7_wrong.txt").read_text(encoding="utf-8").splitlines() if l.strip())
REPEAT = {"hf_train": 8, "realplates": 4, "rf_college": 1, "rf_prajwal": 1, "rf_sushant": 1}
new = []
for l in (root7 / "lines7_list.txt").read_text(encoding="utf-8").splitlines():
    if not l.strip():
        continue
    path = l.split("\t")[0]
    src = path.split("/")[1]
    n = REPEAT[src] + (2 if path in wrong else 0)
    new.extend([l] * n)
rows = replay + new
rng.shuffle(rows)
(farm / "train_list.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
shutil.copy(root6 / "val_list.txt", farm / "val_list.txt")
print("replay", len(replay), "new(with repeats)", len(new), "total", len(rows), flush=True)
print(Counter(l.split("/", 1)[0] for l in rows).most_common(), flush=True)
data_root = farm

sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
for index in ("cu126", "cu118", "cu126"):
    rc = subprocess.run(f"pip install -q --default-timeout=600 --retries 10 paddlepaddle-gpu==3.3.1 "
                        f"-i https://www.paddlepaddle.org.cn/packages/stable/{index}/", shell=True).returncode
    print("paddle install via", index, "->", rc, flush=True)
    if rc == 0:
        break
else:
    raise RuntimeError("paddlepaddle-gpu could not be installed")
sh("python -c \"import paddle; paddle.utils.run_check(); print(paddle.__version__)\"")
sh(f"git clone --depth 1 --branch v3.3.0 https://github.com/PaddlePaddle/PaddleOCR.git {TMP}/PaddleOCR")
sh("pip install -q -r requirements.txt", cwd=f"{TMP}/PaddleOCR")
out = WORK / "output"
cfg = config_template.read_text()
cfg = (cfg.replace("@DATA@", str(data_root)).replace("@OUT@", str(out / "model"))
          .replace("@PRETRAINED@", str(init_weights)).replace("@DICT@", f"{TMP}/PaddleOCR/ppocr/utils/dict/ppocrv5_devanagari_dict.txt"))
cfg = re.sub(r"batch_size_per_card: \d+", "batch_size_per_card: 64", cfg)
cfg = re.sub(r"learning_rate: [0-9.e-]+", "learning_rate: 0.00015", cfg)  # G6 used 0.0003
iters = -(-len(rows) // 64)
cfg = re.sub(r"eval_batch_step: .*", f"eval_batch_step: [0, {iters}]", cfg)
(TMP / "round_gpu.yml").write_text(cfg)
started = time.time()
sh(f"python tools/train.py -c {TMP}/round_gpu.yml", cwd=f"{TMP}/PaddleOCR")
print(f"training took {time.time() - started:.0f} s", flush=True)
for ck in ("best_accuracy", "latest"):
    if (out / "model" / f"{ck}.pdparams").exists():
        sh(f"python tools/export_model.py -c {TMP}/round_gpu.yml -o Global.pretrained_model={out}/model/{ck} "
           f"Global.save_inference_dir={out}/infer_{ck}", cwd=f"{TMP}/PaddleOCR")
for p in (out / "model").glob("*.pdopt"):
    p.unlink()
print(sorted(str(p.relative_to(WORK)) for p in out.rglob("*") if p.is_file())[:40])
