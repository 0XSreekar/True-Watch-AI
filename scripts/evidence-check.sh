#!/usr/bin/env bash
# Evidence chain smoke check used by CI (and runnable locally).
# Seeds a throwaway chain, requires verification to PASS, flips one byte in one sealed
# clip, and requires verification to FAIL. Run from anywhere; needs the edge Python deps.
# Override the interpreter with PYTHON=/path/to/python.
set -u
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
export EVIDENCE_DIR="$work/evidence"
export EVIDENCE_HQ_DIR="$work/evidence-hq"
PY="${PYTHON:-python}"
cd "$root/edge" || exit 2
range=(--from 2000-01-01 --to 2100-01-01)

"$PY" -m evidence.seed_demo || { echo "seed_demo failed"; exit 1; }

"$PY" -m evidence.verify "${range[@]}"
rc=$?
if [ "$rc" -ne 0 ]; then echo "FAIL: untouched chain did not verify (exit $rc)"; exit 1; fi

"$PY" -m evidence.tamper_demo
"$PY" -m evidence.verify "${range[@]}"
rc=$?
if [ "$rc" -eq 0 ]; then echo "FAIL: tampered chain still verified"; exit 1; fi
echo "OK: clean chain verified (exit 0); tampered chain rejected (exit $rc)"
