"""Tests for fetch_kaggle_outputs.py: the pure paging and shard-skip logic, no network.

fetch_dataset_shards / fetch_prev_run_outputs / fetch_pages are exercised here only with a stub
`run`; nothing calls the real `kaggle` CLI or downloads a byte.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import fetch_kaggle_outputs as F


# --------------------------------------------------------------------------------------------
# parse_next_page_token
# --------------------------------------------------------------------------------------------


def test_parse_next_page_token_reads_the_printed_line():
    assert F.parse_next_page_token("Next page token: abc123\n") == "abc123"


def test_parse_next_page_token_is_case_insensitive_and_ignores_surrounding_noise():
    out = "Output file downloaded to ./x/y.tar\nKernel log downloaded to ./k.log\nNEXT PAGE TOKEN: Zzz-9\n"
    assert F.parse_next_page_token(out) == "Zzz-9"


def test_parse_next_page_token_is_none_when_the_cli_paginated_internally():
    out = "Output file downloaded to ./a\nOutput file downloaded to ./b\n"
    assert F.parse_next_page_token(out) is None


def test_parse_next_page_token_is_none_on_empty_output():
    assert F.parse_next_page_token("") is None


# --------------------------------------------------------------------------------------------
# shard_needs_download
# --------------------------------------------------------------------------------------------


def test_shard_needs_download_true_when_missing(tmp_path):
    assert F.shard_needs_download(tmp_path / "shard-0.tar", 1000) is True


def test_shard_needs_download_true_when_size_differs(tmp_path):
    p = tmp_path / "shard-0.tar"
    p.write_bytes(b"x" * 500)
    assert F.shard_needs_download(p, 1000) is True


def test_shard_needs_download_false_when_size_matches_exactly(tmp_path):
    p = tmp_path / "shard-0.tar"
    p.write_bytes(b"x" * 1000)
    assert F.shard_needs_download(p, 1000) is False


# --------------------------------------------------------------------------------------------
# fetch_pages: argv construction and the resume loop, against a stub `run`
# --------------------------------------------------------------------------------------------


class FakeRun:
    """Stands in for subprocess.run: replays one canned stdout per call, records every argv."""

    def __init__(self, stdouts):
        self.stdouts = list(stdouts)
        self.calls = []

    def __call__(self, cmd, capture_output=True, text=True, check=True):
        self.calls.append(cmd)
        out = self.stdouts.pop(0)
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")


def test_fetch_pages_single_page_when_the_cli_reports_no_further_token(tmp_path):
    run = FakeRun(["Output file downloaded to ./x\n"])
    token = F.fetch_pages("owner/kernel", tmp_path, file_pattern=r"x", run=run)
    assert token == ""
    assert len(run.calls) == 1
    cmd = run.calls[0]
    assert cmd[:4] == ["kaggle", "kernels", "output", "owner/kernel"]
    assert "-p" in cmd and str(tmp_path) in cmd
    assert "--file-pattern" in cmd and "x" in cmd
    assert "--page-token" not in cmd


def test_fetch_pages_resumes_from_a_start_token_and_loops_until_no_token_left(tmp_path):
    run = FakeRun([
        "Output file downloaded to ./b\nNext page token: tok-2\n",
        "Output file downloaded to ./c\n",
    ])
    token = F.fetch_pages("owner/kernel", tmp_path, start_token="tok-1", run=run)
    assert token == ""
    assert len(run.calls) == 2
    assert "--page-token" in run.calls[0] and "tok-1" in run.calls[0]
    assert "--page-token" in run.calls[1] and "tok-2" in run.calls[1]


def test_fetch_pages_creates_the_destination_directory(tmp_path):
    dest = tmp_path / "nested" / "dest"
    run = FakeRun(["\n"])
    F.fetch_pages("owner/kernel", dest, run=run)
    assert dest.is_dir()


# --------------------------------------------------------------------------------------------
# fetch_dataset_shards: only the missing/short shards are asked for by name
# --------------------------------------------------------------------------------------------


def _write_shards_json(dest, shards):
    index_dir = dest / "truewatch_ds_shards"
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "SHARDS.json").write_text(json.dumps({"shards": shards, "files": 0}))


def test_fetch_dataset_shards_asks_only_for_missing_or_short_shards(tmp_path):
    calls = []

    def run(cmd, capture_output=True, text=True, check=True):
        calls.append(cmd)
        if "SHARDS" not in cmd[cmd.index("--file-pattern") + 1]:
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        _write_shards_json(tmp_path, [
            {"name": "shard-0.tar", "bytes": 10},
            {"name": "shard-1.tar", "bytes": 20},
            {"name": "shard-2.tar", "bytes": 30},
        ])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    # shard-0 already present at the right size; shard-1 present but short; shard-2 absent.
    shard_dir = tmp_path / "truewatch_ds_shards"
    shard_dir.mkdir(parents=True)
    (shard_dir / "shard-0.tar").write_bytes(b"x" * 10)
    (shard_dir / "shard-1.tar").write_bytes(b"x" * 5)

    F.fetch_dataset_shards("owner/kernel", tmp_path, run=run)

    assert len(calls) == 2  # the SHARDS.json call, then one call for the missing shards
    shard_call = calls[1]
    pattern = shard_call[shard_call.index("--file-pattern") + 1]
    import re as _re
    rx = _re.compile(pattern)
    assert rx.match("truewatch_ds_shards/shard-1.tar")
    assert rx.match("truewatch_ds_shards/shard-2.tar")
    assert not rx.match("truewatch_ds_shards/shard-0.tar")


def test_fetch_dataset_shards_makes_no_second_call_when_everything_is_present(tmp_path):
    calls = []

    def run(cmd, capture_output=True, text=True, check=True):
        calls.append(cmd)
        _write_shards_json(tmp_path, [{"name": "shard-0.tar", "bytes": 10}])
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    shard_dir = tmp_path / "truewatch_ds_shards"
    shard_dir.mkdir(parents=True)
    (shard_dir / "shard-0.tar").write_bytes(b"x" * 10)

    F.fetch_dataset_shards("owner/kernel", tmp_path, run=run)
    assert len(calls) == 1


def test_fetch_dataset_shards_raises_when_the_kernel_output_has_no_index(tmp_path):
    def run(cmd, capture_output=True, text=True, check=True):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    import pytest

    with pytest.raises(RuntimeError, match="SHARDS.json"):
        F.fetch_dataset_shards("owner/kernel", tmp_path, run=run)


# --------------------------------------------------------------------------------------------
# fetch_prev_run_outputs: the runs/(day|ir)/ and results file pattern, not smoke/ or the repo copy
# --------------------------------------------------------------------------------------------


def test_fetch_prev_run_outputs_pattern_matches_runs_and_results_not_smoke_or_repo():
    import re

    rx = re.compile(F.PREV_RUN_FILE_PATTERN)
    assert rx.match("runs/day/weights/best.pt")
    assert rx.match("outputs/runs/ir/train_log.csv")
    assert rx.match("results/eval_final.json")
    assert not rx.match("smoke/runs/day/weights/best.pt")
    assert not rx.match("truewatch/training/scripts/train.py")


def test_fetch_prev_run_outputs_calls_fetch_pages_with_that_pattern(tmp_path):
    run = FakeRun(["\n"])
    F.fetch_prev_run_outputs("owner/kernel", tmp_path, run=run)
    assert len(run.calls) == 1
    cmd = run.calls[0]
    assert "--file-pattern" in cmd
    assert cmd[cmd.index("--file-pattern") + 1] == F.PREV_RUN_FILE_PATTERN
