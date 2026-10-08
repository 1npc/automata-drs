#!/usr/bin/env bash
set -euo pipefail

source scripts/colossus_scratch.sh
automata_scratch_init

# Score only wholly missing split outputs for an already generated baseline.
# Complete split bundles are validated and skipped; partial bundles fail closed.

if (( $# != 1 )); then
  echo "usage: $0 MODEL_KEY" >&2
  exit 2
fi
model_key=$1
python_runtime=".server-venv/bin/python"
if [[ ! -x "${python_runtime}" ]]; then
  echo "missing ${python_runtime}" >&2
  exit 2
fi

if ! full_state=$(
  "${python_runtime}" scripts/check_lora_run_state.py \
    "${model_key}" --field full.state
); then
  "${python_runtime}" scripts/check_lora_run_state.py "${model_key}" || true
  echo "baseline outputs are not safely finalizable" >&2
  exit 2
fi
case "${full_state}" in
  complete)
    echo "validated complete scores: ${model_key}"
    exit 0
    ;;
  unscored|scoring_partial)
    ;;
  *)
    echo "generation is not complete: ${model_key} (${full_state})" >&2
    exit 2
    ;;
esac

scorer_parent=""
scorer=""
cleanup() {
  if [[ -n "${scorer_parent}" && -d "${scorer_parent}" ]]; then
    automata_scratch_cleanup "${scorer_parent}" scorer
  fi
}
trap cleanup EXIT INT TERM

ensure_scorer() {
  if [[ -z "${scorer_parent}" ]]; then
    scorer_parent=$(automata_scratch_mktemp scorer)
    scorer=$(bash scripts/checkout_reference_scorer.sh "${scorer_parent}/repo")
  fi
}

run_root="runs/lora/${model_key}-r16-seed42"
for split in dev test tax; do
  prediction="${run_root}/${split}-u-greedy.jsonl"
  prefix=${prediction%.jsonl}
  summary="${prefix}.summary.json"
  if [[ -f "${summary}" ]]; then
    echo "validated existing score bundle: ${split}"
    continue
  fi
  ensure_scorer
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
      > "${summary}"
  nice -n 15 env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=src \
    "${python_runtime}" scripts/analyze_predictions.py \
      "${prediction}" --prediction-field raw_prediction \
      --label-field reference --configuration W --json \
      > "${prefix}.failures.json"
done

final_state=$(
  "${python_runtime}" scripts/check_lora_run_state.py \
    "${model_key}" --field full.state
)
if [[ "${final_state}" != "complete" ]]; then
  echo "unexpected state after finalization: ${final_state}" >&2
  exit 2
fi
echo "validated complete baseline: ${model_key}"
