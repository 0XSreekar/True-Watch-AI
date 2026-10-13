#!/usr/bin/env bash
# Repository hygiene scan: fails (exit 1) when tooling attribution text appears in tracked files
# or in commit messages.
#
#   scripts/watermark-scan.sh                 scan every tracked file
#   scripts/watermark-scan.sh <range>         also scan commit messages in a git range, e.g. origin/main..HEAD
#                                             (CI passes the pushed range; an all-zero or unknown base scans only HEAD)
#
# Exit codes: 0 clean, 1 match found, 2 usage or environment error.
#
# WHY THE PATTERN IS BUILT FROM PARTS: this file is itself a tracked file and is scanned like every
# other one. Writing the banned words out literally would make the scan fail on itself, so each word
# is split with an empty string ("cl""aude" is the single word once the shell has parsed it) and the
# pieces are joined at run time. No literal banned word appears anywhere in this file.
set -u

words=(
  "cl""aude"
  "anthr""opic"
  "co-auth""ored-by"
  "generated"" with"
)
IFS='|'
pattern="${words[*]}"
unset IFS

root="$(git rev-parse --show-toplevel 2>/dev/null)" || { echo "watermark-scan: not inside a git repository" >&2; exit 2; }
cd "$root" || exit 2

status=0

# 1. Tracked files (plus new files not yet added, so a local run covers work in progress). Excluded: dependency and build output, and the lockfile (machine-generated text).
echo "watermark-scan: scanning tracked files"
hits="$(git ls-files -z --cached --others --exclude-standard \
  | grep -zvE '(^|/)(node_modules|dist|\.git)/|(^|/)package-lock\.json$' \
  | xargs -0 grep -InE -i -- "$pattern" 2>/dev/null)"
if [ -n "$hits" ]; then
  echo "FAIL: banned text found in tracked files:" >&2
  echo "$hits" >&2
  status=1
fi

# 2. Commit messages (subject and body) in the given range.
range="${1:-}"
if [ -n "$range" ]; then
  echo "watermark-scan: scanning commit messages in $range"
  if ! git rev-parse --verify --quiet "${range%%..*}^{commit}" >/dev/null && [[ "$range" == *..* ]]; then
    # Base is not a known commit (first push of a branch): fall back to the tip alone.
    range="HEAD -1"
  fi
  # shellcheck disable=SC2086
  msgs="$(git log --format='%H%n%B%n----' $range 2>/dev/null | grep -nE -i -- "$pattern")"
  if [ -n "$msgs" ]; then
    echo "FAIL: banned text found in commit messages:" >&2
    echo "$msgs" >&2
    status=1
  fi
fi

if [ "$status" -eq 0 ]; then echo "watermark-scan: clean"; fi
exit "$status"
