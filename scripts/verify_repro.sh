#!/usr/bin/env bash
set -euo pipefail

SEED="${SEED:-20261021}"
NOW="${NOW:-2026-10-21T12:00:00}"
SNAPSHOT="${SNAPSHOT:-data/snapshots/lotto}"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/lotto-verify-repro.XXXXXX")"
trap 'rm -rf "$TMP_ROOT"' EXIT

run_cli() {
  local out_dir="$1"
  local log_file="$2"
  set +e
  uv run poc-lottery-randomness-auditor \
    --snapshot "$SNAPSHOT" \
    --out "$out_dir" \
    --seed "$SEED" \
    --now "$NOW" >"$log_file" 2>&1
  local status=$?
  set -e
  return "$status"
}

status_a=0
status_b=0
run_cli "$TMP_ROOT/a" "$TMP_ROOT/a.log" || status_a=$?
run_cli "$TMP_ROOT/b" "$TMP_ROOT/b.log" || status_b=$?

if [[ "$status_a" -ne 0 && "$status_b" -ne 0 ]] && \
   grep -q "WeasyPrint system libraries are unavailable" "$TMP_ROOT/a.log" && \
   grep -q "WeasyPrint system libraries are unavailable" "$TMP_ROOT/b.log"; then
  echo "PDF equality skipped: WeasyPrint system libraries are unavailable."
elif [[ "$status_a" -ne 0 || "$status_b" -ne 0 ]]; then
  echo "CLI failed; see logs under $TMP_ROOT" >&2
  cat "$TMP_ROOT/a.log" >&2
  cat "$TMP_ROOT/b.log" >&2
  exit 1
fi

[[ -f "$TMP_ROOT/a/report.html" && -f "$TMP_ROOT/b/report.html" ]] || {
  echo "report.html missing from one run" >&2
  exit 1
}
if ! cmp -s "$TMP_ROOT/a/report.html" "$TMP_ROOT/b/report.html"; then
  echo "HTML outputs differ" >&2
  diff -u "$TMP_ROOT/a/report.html" "$TMP_ROOT/b/report.html" | head -200 >&2 || true
  exit 1
fi

if [[ -f "$TMP_ROOT/a/report.pdf" && -f "$TMP_ROOT/b/report.pdf" ]]; then
  if ! cmp -s "$TMP_ROOT/a/report.pdf" "$TMP_ROOT/b/report.pdf"; then
    echo "PDF outputs differ" >&2
    exit 1
  fi
  echo "PDF outputs are byte-identical."
elif [[ ! -f "$TMP_ROOT/a/report.pdf" && ! -f "$TMP_ROOT/b/report.pdf" ]] && \
     grep -q "WeasyPrint system libraries are unavailable" "$TMP_ROOT/a.log" && \
     grep -q "WeasyPrint system libraries are unavailable" "$TMP_ROOT/b.log"; then
  echo "PDF files were not produced on this machine; HTML reproducibility verified."
else
  echo "report.pdf missing from one run" >&2
  exit 1
fi

echo "Reproducibility check passed for seed $SEED and now $NOW."
