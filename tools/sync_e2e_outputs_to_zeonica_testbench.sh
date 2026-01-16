#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")/.." rev-parse --show-toplevel)"

SRC_E2E="${ROOT}/test/e2e"
DST_TESTBENCH="${ROOT}/test/benchmark/Zeonica_Testbench"
DST_KERNEL="${DST_TESTBENCH}/kernel"

if [[ ! -d "${DST_TESTBENCH}" ]]; then
  echo "ERROR: Zeonica_Testbench submodule not found at: ${DST_TESTBENCH}" >&2
  echo "Hint: run: git submodule update --init --recursive" >&2
  exit 1
fi

mkdir -p "${DST_KERNEL}"

echo "[sync] SRC: ${SRC_E2E}"
echo "[sync] DST: ${DST_KERNEL}"

shopt -s nullglob
for d in "${SRC_E2E}"/*/; do
  k="$(basename "${d}")"

  # Ensure destination exists (user asked to sync all e2e outputs).
  if [[ ! -d "${DST_KERNEL}/${k}" ]]; then
    echo "[create] ${DST_KERNEL}/${k}"
    mkdir -p "${DST_KERNEL}/${k}"
  fi

  echo "[sync] kernel=${k}"

  # 1) Update canonical files expected by Zeonica_Testbench naming.
  [[ -f "${d}/tmp-generated-dfg.dot" ]] && cp -f "${d}/tmp-generated-dfg.dot" "${DST_KERNEL}/${k}/${k}-dfg.dot"
  [[ -f "${d}/tmp-generated-dfg.yaml" ]] && cp -f "${d}/tmp-generated-dfg.yaml" "${DST_KERNEL}/${k}/${k}-dfg.yaml"
  [[ -f "${d}/tmp-generated-instructions.asm" ]] && cp -f "${d}/tmp-generated-instructions.asm" "${DST_KERNEL}/${k}/${k}-instructions.asm"
  [[ -f "${d}/tmp-generated-instructions.yaml" ]] && cp -f "${d}/tmp-generated-instructions.yaml" "${DST_KERNEL}/${k}/${k}-instructions.yaml"

  # 2) Also archive full e2e outputs for debugging/repro.
  mkdir -p "${DST_KERNEL}/${k}/dataflow"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a "${d}/" "${DST_KERNEL}/${k}/dataflow/" --exclude ".git" --exclude ".gitignore"
  else
    # Fallback (no delete semantics): best-effort copy.
    cp -rf "${d}/." "${DST_KERNEL}/${k}/dataflow/"
  fi
done

echo "[done] synced e2e outputs into Zeonica_Testbench submodule"

