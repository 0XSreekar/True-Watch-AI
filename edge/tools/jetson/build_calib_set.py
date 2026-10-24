"""HOST side: pick the INT8 calibration images and pack them for the Jetson.

Picks N images (default 500) from the val split, half visible ("day") and half LWIR ("ir"), spread
round-robin over the source datasets inside each half, deterministically (a hash of seed+name decides
the order, so the same listing and seed always give the same files). Writes, OUTSIDE the repository:

  calib_list.txt        one image name per line (+ modality and source)
  calib_manifest.json   seed, counts, sha256 of the list, and the licence of every source used
  calib_<n>.tar.gz      the images themselves, flat

Images are never committed. The licence of every source is read from datasets/config/sources.yaml; most
are `open_question` (the original publisher's terms govern, redistribution assumed forbidden), so the
tarball is for calibrating on YOUR board, not for sharing. The script says so in the manifest.

Calibration uses val images only because the INT8 accuracy check (accuracy_int8.py) reads a DIFFERENT
subset of val: it excludes every calibration image, so INT8 accuracy is not measured on its own
calibration data.

  python edge/tools/jetson/build_calib_set.py --data-root <yolo dataset root> --out-dir ~/truewatch-jetson
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from tools.jetson.common import REPO_ROOT, KitError  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}
_MODALITY = re.compile(r"_(visible|lwir)$")
SLICE_OF = {"visible": "day", "lwir": "ir"}
SOURCES_YAML = REPO_ROOT / "datasets" / "config" / "sources.yaml"


def modality_of(stem: str) -> str | None:
    m = _MODALITY.search(stem)
    return m.group(1) if m else None


def source_of(stem: str) -> str:
    """Files are named f"{source}_{raw_stem}_{modality}" (datasets 09_build_yolo_ds.py)."""
    return stem.split("_", 1)[0]


def _key(seed: int, name: str) -> str:
    return hashlib.sha256(f"{seed}:{name}".encode()).hexdigest()


def pick(names: list[str], n: int, ir_fraction: float = 0.5, seed: int = 0, exclude: frozenset[str] = frozenset()) -> list[dict]:
    """Deterministic stratified pick. Returns [{name, modality, slice, source}] sorted by name.

    Stratified by modality (ir_fraction of n from LWIR), then round-robin over sources inside each
    modality, each source ordered by hash(seed, name). Independent of the order of `names`.
    """
    by: dict[str, dict[str, list[str]]] = {"visible": {}, "lwir": {}}
    for name in sorted(set(names) - exclude):
        stem = Path(name).stem
        mod = modality_of(stem)
        if mod is None:
            continue
        by[mod].setdefault(source_of(stem), []).append(name)
    quota = {"lwir": round(n * ir_fraction)}
    quota["visible"] = n - quota["lwir"]
    chosen: list[dict] = []
    for mod, groups in by.items():
        queues = {s: sorted(v, key=lambda x: _key(seed, x)) for s, v in sorted(groups.items())}
        taken = 0
        while taken < quota[mod] and any(queues.values()):
            for s in sorted(queues):
                if taken >= quota[mod]:
                    break
                if queues[s]:
                    name = queues[s].pop(0)
                    chosen.append({"name": name, "modality": mod, "slice": SLICE_OF[mod], "source": s})
                    taken += 1
        if taken < quota[mod]:
            raise KitError(f"only {taken} {mod} images available, wanted {quota[mod]}")
    return sorted(chosen, key=lambda r: r["name"])


def list_text(rows: list[dict]) -> str:
    return "".join(f"{r['name']}\t{r['slice']}\t{r['source']}\n" for r in rows)


def licences(sources: set[str], yaml_path: Path = SOURCES_YAML) -> dict[str, str]:
    """Licence string per source from sources.yaml; 'not recorded' if the file or entry is missing."""
    found: dict[str, str] = {}
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        for name, entry in (data.get("sources") or {}).items():
            found[name] = str(entry.get("licence", "not recorded"))
    except Exception:
        pass
    return {s: found.get(s, "not recorded") for s in sorted(sources)}


def inside_git_repo(path: Path) -> bool:
    p = path.resolve()
    probe = p if p.exists() else p.parent
    while not probe.exists():
        probe = probe.parent
    r = subprocess.run(["git", "-C", str(probe), "rev-parse", "--is-inside-work-tree"], capture_output=True, text=True)
    return r.returncode == 0 and r.stdout.strip() == "true"


def build_manifest(rows: list[dict], seed: int, ir_fraction: float, lic: dict[str, str]) -> dict:
    text = list_text(rows)
    return {
        "schema": "truewatch.jetson_calib.v1",
        "seed": seed,
        "n": len(rows),
        "ir_fraction": ir_fraction,
        "by_slice": {s: sum(1 for r in rows if r["slice"] == s) for s in ("day", "ir")},
        "by_source": {s: sum(1 for r in rows if r["source"] == s) for s in sorted({r["source"] for r in rows})},
        "list_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "licences": lic,
        "licence_note": "Entries marked open_question are governed by the original publisher's terms and are assumed "
                        "non-redistributable. This set is for calibrating on your own board; do not publish the tarball.",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, required=True, help="directory holding images/val and labels/val")
    ap.add_argument("--out-dir", type=Path, required=True, help="must be OUTSIDE any git repository")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--ir-fraction", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if inside_git_repo(args.out_dir):
        raise SystemExit(f"--out-dir {args.out_dir} is inside a git repository; images must never be committed")
    folder = args.data_root / "images" / "val"
    if not folder.is_dir():
        raise SystemExit(f"no images/val under {args.data_root}")
    names = sorted(p.name for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    rows = pick(names, args.n, args.ir_fraction, args.seed)
    man = build_manifest(rows, args.seed, args.ir_fraction, licences({r["source"] for r in rows}))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "calib_list.txt").write_text(list_text(rows), encoding="utf-8")
    (args.out_dir / "calib_manifest.json").write_text(json.dumps(man, indent=2) + "\n", encoding="utf-8")
    tar = args.out_dir / f"calib_{len(rows)}.tar.gz"
    with tarfile.open(tar, "w:gz") as t:
        for r in rows:
            t.add(folder / r["name"], arcname=r["name"])
    print(f"wrote {tar.name} ({man['by_slice']}), calib_list.txt, calib_manifest.json in {args.out_dir}")
    for s, l in man["licences"].items():
        print(f"  licence {s}: {l}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
