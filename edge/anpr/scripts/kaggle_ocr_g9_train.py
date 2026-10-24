# Round G8 of the PP-OCRv5 Devanagari line recogniser on a T4: continue from round G6 on a mix of
#  - lines cut from double-transcribed street-photo plates (locator box + rectify.prepare_lines), repeated,
#  - the G7 locator-cut real lines (same repeats as G7), and
#  - replay of the G6 training data (every real set in full, a sample of the synthetic corpora).
# VARIANT "a" keeps the G6/G7 augmentation; "b" lowers RecAug (tia/crop/noise/jitter/blur/hsv 0.1-0.2, no colour
# inversion), the cheap try against dropped and inserted serial digits.
# Same PaddleOCR (v3.3.0) and Paddle (3.3.1) as the earlier rounds.
import glob, os, random, re, shutil, subprocess, time
from collections import Counter
from pathlib import Path

def sh(cmd, cwd=None):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)

VARIANT = "a"
WORK, TMP = Path("/kaggle/working"), Path("/kaggle/temp")
TMP.mkdir(exist_ok=True)

def one(pattern, must=""):
    hits = sorted((h for h in glob.glob(f"/kaggle/input/**/{pattern}", recursive=True) if must in h), key=len)
    print(pattern, "->", hits[:3], flush=True)
    return Path(hits[0])

root6 = one("train_list.txt", "lines6").parent
root9 = one("lines9_list.txt", "lines9").parent
config_template = one("round_gpu.yml", "lines7")
init_weights = one("output/model/latest.pdparams", "gpu6").with_suffix("")  # round G6 final checkpoint

farm = TMP / "g9data"
farm.mkdir()
for d in root6.iterdir():
    if d.is_dir():
        (farm / d.name).symlink_to(d, target_is_directory=True)
(farm / "lines9").symlink_to(root9 / "lines9", target_is_directory=True)

rng = random.Random(9)
REAL = ("hfreal2", "hfreal7", "pseudo", "real7", "r2", "rf_college", "rf_prajwal", "rf_sushant")
SYN_KEEP = 0.25
rows6 = [l for l in (root6 / "train_list.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
replay = [l for l in rows6 if l.split("/", 1)[0] in REAL or rng.random() < SYN_KEEP]
wrong9 = set(l.strip() for l in (root9 / "lines9_wrong.txt").read_text(encoding="utf-8").splitlines() if l.strip())
REPEAT = {"hf_g9": 6, "nivkarki": 2, "crimson": 2, "xyz": 2}
new9 = []
for l in (root9 / "lines9_list.txt").read_text(encoding="utf-8").splitlines():
    if l.strip():
        p = l.split("\t")[0]
        new9.extend([l] * (REPEAT[p.split("/")[1]] + (2 if p in wrong9 else 0)))
rows = replay + new9
rng.shuffle(rows)
(farm / "train_list.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
shutil.copy(root6 / "val_list.txt", farm / "val_list.txt")
print("replay", len(replay), "g9 new(with repeats)", len(new9), "total", len(rows), flush=True)
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
if VARIANT == "b":
    cfg, n = re.subn(r"^(\s*)- RecAug: null$", lambda m: m.group(1) + "- RecAug:\n" + "".join(
        f"{m.group(1)}    {k}: {v}\n" for k, v in (("tia_prob", 0.1), ("crop_prob", 0.1), ("reverse_prob", 0.0),
        ("noise_prob", 0.2), ("jitter_prob", 0.2), ("blur_prob", 0.2), ("hsv_aug_prob", 0.2))).rstrip("\n"), cfg, flags=re.M)
    assert n == 1, "RecAug line not found in the config template"
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
