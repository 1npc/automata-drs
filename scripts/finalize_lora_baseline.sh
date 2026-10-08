#!/usr/bin/env bash
set -euo pipefail

source scripts/colossus_scratch.sh
automata_scratch_init

# Wait for one already-running full baseline, validate every expected row from
# the actual corpora, then create official scores and summaries.
# Usage: bash scripts/finalize_lora_baseline.sh MODEL_KEY WAIT_PID

if (( $# != 2 )); then
  echo "usage: $0 MODEL_KEY WAIT_PID" >&2
  exit 2
fi
model_key=$1
wait_pid=$2
if [[ ! "${wait_pid}" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "${wait_pid}" 2>/dev/null; then
  echo "WAIT_PID is not a live process: ${wait_pid}" >&2
  exit 2
fi

python_runtime=".server-venv/bin/python"
if [[ ! -x "${python_runtime}" ]]; then
  echo "missing ${python_runtime}" >&2
  exit 2
fi
run_root="runs/lora/${model_key}-r16-seed42"
if [[ ! -f "${run_root}/adapter/adapter_config.json" ]]; then
  echo "missing completed adapter under ${run_root}" >&2
  exit 2
fi

tail --pid="${wait_pid}" -f /dev/null

scorer_parent=$(automata_scratch_mktemp scorer)
cleanup() {
  if [[ -d "${scorer_parent}" ]]; then
    automata_scratch_cleanup "${scorer_parent}" scorer
  fi
}
trap cleanup EXIT INT TERM
scorer=$(bash scripts/checkout_reference_scorer.sh "${scorer_parent}/repo")

declare -A data_paths=(
  [dev]="data/en/dev/standard.sbn"
  [test]="data/en/test/standard.sbn"
  [tax]="data/en/test/tax_challenge.sbn"
)

for split in dev test tax; do
  prediction="${run_root}/${split}-u-greedy.jsonl"
  expected=$(
    env PYTHONPATH=src "${python_runtime}" - "${data_paths[$split]}" <<'PY'
import sys
from pathlib import Path
from automata_drs.causal_lm import iter_sbn_records

print(sum(1 for _ in iter_sbn_records(Path(sys.argv[1]))))
PY
  )
  if [[ ! -f "${prediction}" || ! -f "${prediction}.manifest.json" ]]; then
    echo "missing completed prediction or manifest: ${prediction}" >&2
    exit 3
  fi
  actual=$(wc -l < "${prediction}")
  if (( actual != expected )); then
    echo "incomplete ${split}: expected ${expected}, found ${actual}" >&2
    exit 3
  fi
  prefix=${prediction%.jsonl}
  for output in \
    "${prefix}.smatch.jsonl" \
    "${prefix}.smatch.jsonl.manifest.json" \
    "${prefix}.summary.md" \
    "${prefix}.summary.json" \
    "${prefix}.failures.json"; do
    if [[ -e "${output}" ]]; then
      echo "refusing to overwrite finalizer artifact: ${output}" >&2
      exit 3
    fi
  done

  nice -n 15 env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${python_runtime}" scripts/score_sbn_smatch.py \
      "${prediction}" --scorer "${scorer}" \
      --output "${prefix}.smatch.jsonl"
  nice -n 15 env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${python_runtime}" scripts/summarize_experiment.py \
      "${prediction}" --scores "${prefix}.smatch.jsonl" \
      > "${prefix}.summary.md"
  nice -n 15 env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${python_runtime}" scripts/summarize_experiment.py \
      "${prediction}" --scores "${prefix}.smatch.jsonl" --json \
      > "${prefix}.summary.json"
  nice -n 15 env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=src \
    "${python_runtime}" scripts/analyze_predictions.py \
      "${prediction}" --prediction-field raw_prediction \
      --label-field reference --configuration W --json \
      > "${prefix}.failures.json"
  echo "finalized ${split}: ${actual}/${expected} records"
done
