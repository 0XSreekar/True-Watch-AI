#!/usr/bin/env python3
"""Locate or fetch every enabled source, verify its counts, record where it lives.

Three ways to give this script a source, in order of precedence:

  --root NAME=PATH        an existing directory, used where it is: never copied, never
                          extracted, never written to. Read-only mounts are fine.
  --mode kaggle_input     search --input-base (default /kaggle/input) for each source's
                          marker (sources.yaml `marker`), e.g. /kaggle/input/<slug>/... or
                          /kaggle/input/datasets/<owner>/<slug>/...
  --mode download         (default) download the Kaggle mirror with the kaggle CLI into
                          <raw>/<name>, resumable: a finished download is never repeated.

Whatever the mode, the resolved root of each source is written to
<processed>/_state/source_roots.json, which every converter reads, so later steps never need
to be told again. Disabled sources (sources.yaml enabled: false) are skipped and say why.

Counts are verified against sources.yaml `expected`. A mismatch is logged and counted; it is a
warning, not a failure, because a deliberately partial fixture (the local smoke run) must pass
through the same code.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    REPORT_DIR,
    Counters,
    Logger,
    base_parser,
    load_sources,
    load_state,
    locate_root,
    read_json,
    require_dirs,
    roots_state_path,
    run,
    save_state,
    write_json,
)

SCRIPT = "00_fetch"


# --------------------------------------------------------------------------- counting


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def count_files(directory: Path, suffixes: tuple[str, ...]) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for entry in os.scandir(directory) if entry.is_file() and Path(entry.name).suffix.lower() in suffixes)


def observed_counts(name: str, root: Path, spec: dict) -> dict[str, int]:
    """Cheap, file-system-only counts that correspond to sources.yaml `expected`."""
    images = (".jpg", ".jpeg", ".png")
    if name == "idd":
        return {
            "train_list": count_lines(root / "train.txt"),
            "val_list": count_lines(root / "val.txt"),
            "test_list": count_lines(root / "test.txt"),
        }
    if name == "flir":
        return {subset: count_files(root / subset / "data", images) for subset in sorted(spec.get("subsets", {}))}
    if name == "llvip":
        return {
            "annotations": count_files(root / "Annotations", (".xml",)),
            "train_pairs": count_files(root / "infrared" / "train", images),
            "test_pairs": count_files(root / "infrared" / "test", images),
        }
    return {}


def verify_counts(name: str, root: Path, spec: dict, log: Logger, counters: Counters) -> dict:
    expected = spec.get("expected") or {}
    tolerance = float(spec.get("count_tolerance", 0.02))
    observed = observed_counts(name, root, spec)
    for key, want in sorted(expected.items()):
        have = observed.get(key)
        if have is None:
            continue
        if not want * (1 - tolerance) <= have <= want * (1 + tolerance):
            log.warn("count differs from the recorded figure", source=name, item=key, found=have, expected=want)
            counters.bump(f"{name}.count_mismatch")
        else:
            log.info("count verified", source=name, item=key, found=have, expected=want)
    return observed


# --------------------------------------------------------------------------- modes


def download_kaggle(name: str, spec: dict, dest: Path, log: Logger, args, counters: Counters) -> Path | None:
    """Download and unzip one mirror into <raw>/<name>. A `.complete` flag makes it resumable."""
    marker = list(spec.get("marker") or [])
    done_flag = dest / ".complete"
    existing = locate_root(dest, marker) if dest.exists() else None
    if existing is not None and done_flag.exists() and not args.force:
        log.info("already downloaded", source=name, root=str(existing))
        return existing
    if args.dry_run:
        log.info("dry-run: would download", source=name, ref=spec.get("kaggle_ref"), dest=str(dest))
        return existing
    if shutil.which("kaggle") is None:
        log.error("kaggle CLI not found", fix="pip install kaggle and place the API token under ~/.kaggle/")
        counters.bump(f"{name}.no_cli")
        return None
    dest.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "datasets", "download", "-d", spec["kaggle_ref"], "-p", str(dest), "--unzip"]
    log.info("downloading", source=name, command=" ".join(command))
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        log.error("download failed", source=name, exit=result.returncode)
        counters.bump(f"{name}.download_failed")
        return None
    root = locate_root(dest, marker)
    if root is None:
        log.error("download finished but the marker is missing", source=name, marker=",".join(marker))
        counters.bump(f"{name}.marker_missing")
        return None
    done_flag.write_text("complete\n", encoding="utf-8")
    return root


def record_licence(name: str, spec: dict, root: Path | None, log: Logger, dry_run: bool) -> None:
    """Write what is known about the licence. A mirror uploader's label is not a licence."""
    out = REPORT_DIR / "licences" / f"{name}.txt"
    if dry_run:
        return
    lines = [
        f"source: {name}",
        f"mirror: kaggle {spec.get('kaggle_ref', spec.get('repo_id', ''))}",
        f"original: {spec.get('mirror_of', 'n/a')}",
        f"third-party mirror: {spec.get('third_party_mirror', False)}",
        f"licence (sources.yaml): {spec.get('licence')}",
        f"redistribution (sources.yaml): {spec.get('redistribution')}",
        "The licence of the ORIGINAL release governs. See docs/DATASET_CARD.md.",
    ]
    if root is not None:
        for candidate in ("LICENSE", "LICENSE.txt", "license.txt", "README.md", "readme.txt"):
            path = root / candidate
            if path.is_file():
                lines += ["", f"--- {candidate}, verbatim ---", path.read_text(encoding="utf-8", errors="replace")]
                break
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("licence note recorded", source=name, path=str(out))


def parse_roots(values: list[str], log: Logger) -> dict[str, Path] | None:
    out: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            log.error("--root must be NAME=PATH", got=value)
            return None
        name, path = value.split("=", 1)
        out[name.strip()] = Path(path.strip())
    return out


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument("--only", default=None, help="one source only: idd, flir, llvip or kaist")
    ap.add_argument("--mode", choices=("download", "kaggle_input"), default="download")
    ap.add_argument("--input-base", default="/kaggle/input", help="where --mode kaggle_input searches")
    ap.add_argument("--root", action="append", default=[], help="NAME=PATH, an existing source root (repeatable)")
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    overrides = parse_roots(args.root, log)
    if overrides is None:
        return 2
    sources = load_sources()["sources"]
    unknown = sorted(set(overrides) - set(sources))
    if unknown:
        log.error("--root names an unknown source", names=",".join(unknown))
        return 2

    raw_root = Path(args.raw).resolve()
    processed = Path(args.processed).resolve()
    require_dirs(processed)

    state = load_state(processed, SCRIPT)
    roots: dict[str, str] = {}
    manifest: dict[str, dict] = {}
    failures = 0

    for name, spec in sources.items():
        if args.only and args.only != name:
            continue
        if not spec.get("enabled", True):
            log.info("source disabled, skipped", source=name, reason=" ".join(str(spec.get("disabled_reason", "")).split()))
            continue

        marker = list(spec.get("marker") or [])
        root: Path | None = None
        if name in overrides:
            origin = "--root"
            root = locate_root(overrides[name], marker) if marker else overrides[name]
            if root is None:
                log.error("--root has no marker", source=name, path=str(overrides[name]), marker=",".join(marker))
        elif args.mode == "kaggle_input":
            origin = "kaggle_input"
            root = locate_root(Path(args.input_base), marker, max_depth=7) if marker else None
            if root is None:
                log.error("marker not found under the input base", source=name, base=args.input_base, marker=",".join(marker))
        elif spec.get("kind") == "kaggle":
            origin = "download"
            root = download_kaggle(name, spec, raw_root / name, log, args, counters)
        else:
            origin = "none"
            log.error("no fetch path for this source kind", source=name, kind=spec.get("kind"))

        if root is None:
            if args.dry_run and origin == "download":
                continue
            failures += 1
            counters.bump(f"{name}.unresolved")
            continue

        root = root.resolve()
        log.info("source resolved", source=name, root=str(root), origin=origin, writable=os.access(root, os.W_OK))
        observed = verify_counts(name, root, spec, log, counters)
        record_licence(name, spec, root, log, args.dry_run)
        roots[name] = str(root)
        manifest[name] = {
            "root": str(root),
            "origin": origin,
            "kaggle_ref": spec.get("kaggle_ref"),
            "observed": observed,
            "expected": spec.get("expected"),
            "licence": spec.get("licence"),
            "redistribution": spec.get("redistribution"),
        }

    if not args.dry_run:
        recorded = read_json(roots_state_path(processed), default={}) or {}
        recorded.update(roots)
        write_json(roots_state_path(processed), recorded)
        state.update({k: {"root": v} for k, v in roots.items()})
        save_state(processed, SCRIPT, state)
        write_json(processed / "fetch_manifest.json", manifest)

    log.info("manifest")
    for name, entry in sorted(manifest.items()):
        print(f"    {name:<6} origin={entry['origin']:<13} root={entry['root']}", flush=True)
        print(f"           observed={entry['observed']}", flush=True)
    counters.report(log)

    if failures:
        log.error("one or more enabled sources could not be resolved", failures=failures)
        return 1
    log.info("done", sources=len(manifest))
    return 0


if __name__ == "__main__":
    run(main)
