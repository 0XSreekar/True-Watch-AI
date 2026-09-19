#!/usr/bin/env python3
"""Fetch IDD, LLVIP and KAIST into a git-ignored directory.

Resumable, count-verified, and it never re-downloads what is already present.
IDD is behind a registration and licence gate at IIIT Hyderabad: this script does
not scrape it. It prints the registration URL, verifies a manually placed archive
and extracts it. That refusal is deliberate.

Licence strings are READ AT FETCH TIME from the upstream dataset card and written
to reports/licences/. Nothing here paraphrases a licence from memory.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib import (  # noqa: E402
    REPORT_DIR,
    Counters,
    Logger,
    base_parser,
    count_images,
    load_sources,
    load_state,
    require_dirs,
    run,
    save_state,
    write_json,
)

SCRIPT = "00_fetch"


def fetch_huggingface(name: str, spec: dict, dest: Path, log: Logger, args, counters: Counters) -> bool:
    try:
        from huggingface_hub import list_repo_files, snapshot_download
    except ImportError:
        log.error("huggingface_hub is not installed", fix="pip install -r datasets/requirements.txt")
        return False

    repo_id = spec["repo_id"]
    repo_type = spec.get("repo_type", "dataset")

    try:
        files = list_repo_files(repo_id, repo_type=repo_type)
    except Exception as exc:  # network, auth, gating - all are failures worth naming
        log.error("listing failed", source=name, repo=repo_id, error=type(exc).__name__, detail=str(exc)[:200])
        counters.bump(f"{name}.list_failed")
        return False

    log.info("listed", source=name, repo=repo_id, files=len(files))

    expected = spec.get("expected_file_count")
    if expected:
        tolerance = float(spec.get("count_tolerance", 0.02))
        low, high = expected * (1 - tolerance), expected * (1 + tolerance)
        if not low <= len(files) <= high:
            log.warn(
                "upstream file count differs from the recorded figure",
                source=name,
                found=len(files),
                expected=expected,
            )
            counters.bump(f"{name}.count_mismatch")

    _record_licence(name, repo_id, repo_type, log, args)

    if args.dry_run:
        log.info("dry-run: would fetch", source=name, repo=repo_id, dest=str(dest))
        return True

    dest.mkdir(parents=True, exist_ok=True)
    patterns = spec.get("allow_patterns")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            local_dir=str(dest),
            allow_patterns=patterns,
            max_workers=4,
            resume_download=True,
        )
    except Exception as exc:
        log.error("download failed", source=name, error=type(exc).__name__, detail=str(exc)[:200])
        counters.bump(f"{name}.download_failed")
        return False

    present = sum(1 for p in dest.rglob("*") if p.is_file())
    log.info("fetched", source=name, files=present, dest=str(dest))
    counters.bump(f"{name}.files", present)
    return True


def _record_licence(name: str, repo_id: str, repo_type: str, log: Logger, args) -> None:
    """Write the upstream licence string verbatim. OQ-4 and OQ-5 in DATASET_SPEC.md."""
    out = REPORT_DIR / "licences" / f"{name}.txt"
    if args.dry_run:
        log.info("dry-run: would record licence", source=name, path=str(out))
        return
    text = f"source: {name}\nrepo: {repo_id}\n"
    try:
        from huggingface_hub import DatasetCard

        card = DatasetCard.load(repo_id)
        licence = getattr(card.data, "license", None)
        text += f"licence field: {licence}\n\n--- card text, verbatim ---\n{card.text}\n"
    except Exception as exc:
        text += (
            f"licence field: UNREAD ({type(exc).__name__})\n"
            "OPEN QUESTION OQ-5 remains open. Redistribution stays assumed FORBIDDEN.\n"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    log.info("licence recorded", source=name, path=str(out))


def fetch_manual(name: str, spec: dict, dest: Path, log: Logger, args, counters: Counters) -> bool:
    url = spec["registration_url"]
    archives = spec.get("expected_archives", [])

    log.info("this source is licence-gated and is never scraped", source=name, register_at=url)

    dest.mkdir(parents=True, exist_ok=True)
    found = [p for p in dest.iterdir() if p.is_file() and p.suffix in (".gz", ".tar", ".zip", ".tgz")]
    extracted_marker = dest / ".extracted"

    if extracted_marker.exists() and not args.force:
        images = count_images(dest)
        log.info("already extracted", source=name, images=images)
        counters.bump(f"{name}.images", images)
        return _verify_idd_count(name, spec, images, log, counters)

    if not found:
        log.error(
            "no archive found - download it manually, then re-run",
            source=name,
            register_at=url,
            place_in=str(dest),
            expected=",".join(archives) if archives else "the detection archive",
        )
        counters.bump(f"{name}.missing_archive")
        return False

    if args.dry_run:
        log.info("dry-run: would extract", source=name, archives=len(found))
        return True

    for archive in found:
        log.info("extracting", source=name, archive=archive.name)
        try:
            if archive.suffix == ".zip":
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(dest)
            else:
                with tarfile.open(archive) as tf:
                    tf.extractall(dest)
        except Exception as exc:
            log.error("extraction failed", archive=archive.name, error=type(exc).__name__)
            counters.bump(f"{name}.extract_failed")
            return False

    extracted_marker.write_text("extracted\n", encoding="utf-8")
    images = count_images(dest)
    counters.bump(f"{name}.images", images)
    log.info("extracted", source=name, images=images)
    return _verify_idd_count(name, spec, images, log, counters)


def _verify_idd_count(name: str, spec: dict, images: int, log: Logger, counters: Counters) -> bool:
    expected = spec.get("expected_image_count")
    if not expected:
        return True
    tolerance = float(spec.get("count_tolerance", 0.02))
    if not expected * (1 - tolerance) <= images <= expected * (1 + tolerance):
        log.warn("image count differs from the recorded figure", source=name, found=images, expected=expected)
        counters.bump(f"{name}.count_mismatch")
    return True


def main() -> int:
    ap = base_parser(__doc__)
    ap.add_argument(
        "--only",
        default=None,
        help="fetch one source only: kaist, llvip or idd",
    )
    args = ap.parse_args()
    log = Logger(SCRIPT)
    counters = Counters()

    sources = load_sources()["sources"]
    raw_root = Path(args.raw).resolve()
    processed = Path(args.processed).resolve()
    require_dirs(raw_root, REPORT_DIR)

    state = load_state(processed, SCRIPT)
    manifest: dict[str, dict] = {}
    failures = 0

    for name, spec in sources.items():
        if args.only and args.only != name:
            continue
        dest = raw_root / name
        log.info("source", source=name, kind=spec["kind"], dest=str(dest))

        if state.get(name, {}).get("complete") and not args.force:
            log.info("already complete, skipping", source=name, hint="--force to redo")
            manifest[name] = state[name]
            continue

        if spec["kind"] == "huggingface":
            ok = fetch_huggingface(name, spec, dest, log, args, counters)
        elif spec["kind"] == "manual_registration":
            ok = fetch_manual(name, spec, dest, log, args, counters)
        else:
            log.error("unknown source kind", source=name, kind=spec["kind"])
            ok = False

        if not ok:
            failures += 1
            continue

        entry = {
            "complete": not args.dry_run,
            "dest": str(dest),
            "files": sum(1 for p in dest.rglob("*") if p.is_file()) if dest.exists() else 0,
            "licence": spec.get("licence"),
            "redistribution": spec.get("redistribution"),
        }
        manifest[name] = entry
        if not args.dry_run:
            state[name] = entry

    if not args.dry_run:
        save_state(processed, SCRIPT, state)
        write_json(Path(args.processed).resolve() / "fetch_manifest.json", manifest)

    log.info("manifest")
    for name, entry in sorted(manifest.items()):
        print(f"    {name:<8} files={entry.get('files', 0):<8} dest={entry.get('dest')}", flush=True)

    counters.report(log)

    if failures:
        log.error("one or more sources failed", failures=failures)
        return 1
    log.info("done", sources=len(manifest))
    return 0


if __name__ == "__main__":
    run(main)
