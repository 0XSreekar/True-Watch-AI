"""Fetch a Kaggle kernel's output into a local directory, for a machine with no /kaggle/input mount (Colab).

The `kaggle kernels output` CLI already walks every result page by itself when no --page-token is
given, and already skips a file whose local copy is the same size or newer than the one on Kaggle
(see kaggle.api.kaggle_api_extended.KaggleApi.download_needed). What it does not do is let a caller
avoid even asking for a file it already knows, byte for byte, it has: the dataset build's output
carries truewatch_ds_shards/SHARDS.json, which records each shard's exact size (and its checksum,
verified again after unpacking by the notebook's own unpack_shards()). A Colab session that
reconnects after a disconnect can read that file locally, work out which shard tars are still
missing or short, and ask the CLI for only those by name -- instead of re-requesting files it would
just throw away.

Two pure, unit-tested pieces:
  * `parse_next_page_token` -- reads the "Next page token: <token>" line the CLI prints when a
    single-page request (one made with an explicit --page-token) was not the last page. Kept
    working even though the common case here never sees a non-empty token, because a caller that
    already holds a token logged by a previous `fetch_pages` call can resume from it.
  * `shard_needs_download` -- true unless a local file already has the exact size SHARDS.json
    records for it.

`fetch_pages` and `fetch_dataset_shards` / `fetch_prev_run_outputs` drive the real `kaggle` CLI as a
subprocess (dependency-injected via `run=`, so the tests never touch the network). Never call
`fetch_dataset_shards` in a test or CI job; it downloads real gigabytes.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Optional

NEXT_PAGE_RE = re.compile(r"next page token:\s*(\S+)", re.IGNORECASE)

# The build kernel's output also contains a `view/` mirror of every file and a `truewatch-*.log`;
# only the shard tars and SHARDS.json matter here.
SHARD_FILE_PATTERN = r"^truewatch_ds_shards/[^/]+$"

# The training kernel's output also contains a smoke/ scratch run and a full repo copy; only the
# restored run directories and the small results files matter here.
PREV_RUN_FILE_PATTERN = r"^(outputs/)?(runs/(day|ir)/|results/)"


def parse_next_page_token(cli_output: str) -> Optional[str]:
    """The token from a "Next page token: <token>" line the kaggle CLI prints, or None.

    The CLI prints this only for a request made with an explicit --page-token (a request with none
    walks every page itself and never prints one, having nothing left to resume from).
    """
    match = NEXT_PAGE_RE.search(cli_output)
    return match.group(1) if match else None


def shard_needs_download(local_path: Path, expected_bytes: int) -> bool:
    """False only when `local_path` exists with exactly the size SHARDS.json records for it."""
    return not (local_path.exists() and local_path.stat().st_size == expected_bytes)


def fetch_pages(
    kernel: str,
    dest: Path,
    file_pattern: Optional[str] = None,
    page_size: int = 200,
    start_token: Optional[str] = None,
    run: Callable[..., "subprocess.CompletedProcess[str]"] = subprocess.run,
    max_pages: int = 1000,
) -> str:
    """Download `kernel`'s output into `dest`, resuming from `start_token` when one is given.

    With no `start_token` the first call already walks every page (the CLI's own behaviour); the
    loop still checks for a printed token afterwards so a caller that resumes later from a token
    logged here only asks for the page(s) it has not seen. Returns the last token seen, or "" when
    the CLI reports none (nothing left to resume from).
    """
    dest.mkdir(parents=True, exist_ok=True)
    token = start_token
    last_token = ""
    for _ in range(max_pages):
        cmd = ["kaggle", "kernels", "output", kernel, "-p", str(dest), "--page-size", str(page_size)]
        if file_pattern:
            cmd += ["--file-pattern", file_pattern]
        if token:
            cmd += ["--page-token", token]
        result = run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
        token = parse_next_page_token(result.stdout)
        last_token = token or ""
        if not token:
            break
    return last_token


def fetch_dataset_shards(kernel: str, dest: Path, run: Callable = subprocess.run) -> Path:
    """Download truewatch_ds_shards/SHARDS.json, then only the tar shards missing or short.

    Returns the local path of SHARDS.json. Downloads real gigabytes for any shard not already
    present at its recorded size -- never call this outside a real Colab/fetch session.
    """
    fetch_pages(kernel, dest, file_pattern=r"^truewatch_ds_shards/SHARDS\.json$", run=run)
    index_path = dest / "truewatch_ds_shards" / "SHARDS.json"
    if not index_path.exists():
        raise RuntimeError(f"{kernel} output has no truewatch_ds_shards/SHARDS.json")
    index = json.loads(index_path.read_text())
    missing = [
        shard["name"]
        for shard in index["shards"]
        if shard_needs_download(dest / "truewatch_ds_shards" / shard["name"], shard["bytes"])
    ]
    if missing:
        pattern = r"^truewatch_ds_shards/(" + "|".join(re.escape(name) for name in missing) + r")$"
        fetch_pages(kernel, dest, file_pattern=pattern, run=run)
    else:
        print(f"all {len(index['shards'])} shards already present at {dest / 'truewatch_ds_shards'}")
    return index_path


def fetch_prev_run_outputs(kernel: str, dest: Path, run: Callable = subprocess.run) -> None:
    """Download only an earlier training kernel's runs/(day|ir)/ and results files.

    Skips its smoke/ scratch run and the repo copy the notebook also writes to its output.
    """
    fetch_pages(kernel, dest, file_pattern=PREV_RUN_FILE_PATTERN, run=run)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kernel", help="<owner>/<kernel-slug>, e.g. sreekar1206/truewatch-phase1-build")
    ap.add_argument("--dest", required=True, type=Path, help="directory to download into")
    ap.add_argument("--kind", choices=["shards", "prev-run", "raw"], default="raw")
    ap.add_argument("--file-pattern", default=None, help="only with --kind raw")
    ap.add_argument("--page-size", type=int, default=200)
    args = ap.parse_args(argv)

    if args.kind == "shards":
        fetch_dataset_shards(args.kernel, args.dest)
    elif args.kind == "prev-run":
        fetch_prev_run_outputs(args.kernel, args.dest)
    else:
        fetch_pages(args.kernel, args.dest, file_pattern=args.file_pattern, page_size=args.page_size)


if __name__ == "__main__":
    main()
