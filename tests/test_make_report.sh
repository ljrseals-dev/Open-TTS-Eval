#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SCRIPT="$HERE/../scripts/make_report.py"

check() {  # file, expected_substring, should_be_present(y/n)
  local out; out="$(python "$SCRIPT" "$1")"
  if [[ "$3" == "y" ]]; then
    echo "$out" | grep -qF -- "$2" || { echo "FAIL: '$2' missing in $1"; exit 1; }
  else
    echo "$out" | grep -qF -- "$2" && { echo "FAIL: '$2' unexpectedly present in $1"; exit 1; }
  fi
  echo "ok: $(basename "$1") — $3 '$2'"
}

check "$HERE/fixtures/full.json"     "## Reproducibility"        y
check "$HERE/fixtures/full.json"     "whisper 20231117"          y
check "$HERE/fixtures/full.json"     "tarepan/SpeechMOS:v1.2.0"  y
check "$HERE/fixtures/full.json"     "DRY RUN"                   n

check "$HERE/fixtures/partial.json"  "## Reproducibility"        n
check "$HERE/fixtures/partial.json"  "| WER |"                   n
check "$HERE/fixtures/partial.json"  "| UTMOS |"                 n
check "$HERE/fixtures/partial.json"  "| RTFx |"                  y

check "$HERE/fixtures/legacy.json"   "## Reproducibility"        y
check "$HERE/fixtures/legacy.json"   "whisper —"                y   # version unknown → em dash
check "$HERE/fixtures/legacy.json"   "DRY RUN"                   n

check "$HERE/fixtures/dry_run.json"  "2 samples were synthesized"  y
check "$HERE/fixtures/dry_run.json"  "**not** a valid leaderboard result"   y
check "$HERE/fixtures/dry_run.json"  "## Reproducibility"        y
check "$HERE/fixtures/dry_run.json"  "**Dry run:** yes"           y

echo "All report fixtures passed."
