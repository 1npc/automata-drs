#!/usr/bin/env bash
set -euo pipefail

source scripts/colossus_scratch.sh
automata_scratch_init

# Add official SBN Smatch, paired summaries, and failure diagnostics to one
# completed LoRA experiment matrix. All outputs are exclusive and immutable.

if (( $# < 1 || $# > 2 )); then
  echo "usage: $0 PREDICTIONS.jsonl [POPULATION_SCOPE]" >&2
  exit 2
fi
prediction=$1
population_scope=${2:-fixed-without-u-outcome-selection}
python_runtime=".server-venv/bin/python"
if [[ ! -x "${python_runtime}" ]]; then
  echo "missing ${python_runtime}" >&2
  exit 2
fi
if [[ ! -f "${prediction}" || ! -f "${prediction}.manifest.json" ]]; then
  echo "missing completed prediction or generation manifest: ${prediction}" >&2
  exit 2
fi
case "${prediction}" in
  runs/lora/*.jsonl) ;;
  *)
    echo "prediction must be a JSONL directly under a runs/lora model directory" >&2
    exit 2
    ;;
esac

prefix=${prediction%.jsonl}
score_path="${prefix}.smatch.jsonl"
score_manifest="${score_path}.manifest.json"
if [[ -e "${score_path}" && ! -f "${score_manifest}" ]] \
  || [[ -e "${score_manifest}" && ! -f "${score_path}" ]]; then
  echo "score JSONL and manifest must either both exist or both be absent" >&2
  exit 2
fi

"${python_runtime}" - "${prediction}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

prediction = Path(sys.argv[1])
manifest = json.loads(Path(str(prediction) + ".manifest.json").read_text())
rows = sum(1 for line in prediction.open() if line.strip())
digest = hashlib.sha256(prediction.read_bytes()).hexdigest()
if manifest.get("status") != "complete":
    raise SystemExit("prediction manifest is not complete")
if manifest.get("rows") != rows:
    raise SystemExit("prediction manifest row count mismatch")
if manifest.get("output_sha256") != digest:
    raise SystemExit("prediction manifest output hash mismatch")
PY

scorer_parent=""
cleanup() {
  if [[ -n "${scorer_parent}" && -d "${scorer_parent}" ]]; then
    automata_scratch_cleanup "${scorer_parent}" scorer
  fi
}
trap cleanup EXIT INT TERM
if [[ ! -f "${score_path}" ]]; then
  scorer_parent=$(automata_scratch_mktemp scorer)
  scorer=$(bash scripts/checkout_reference_scorer.sh "${scorer_parent}/repo")
  nice -n 15 env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${python_runtime}" scripts/score_sbn_smatch.py \
      "${prediction}" --scorer "${scorer}" \
      --output "${score_path}"
fi
if [[ ! -f "${prefix}.summary.md" ]]; then
  nice -n 15 env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${python_runtime}" scripts/summarize_experiment.py \
      "${prediction}" --scores "${score_path}" \
      > "${prefix}.summary.md"
fi
if [[ ! -f "${prefix}.summary.json" ]]; then
  nice -n 15 env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${python_runtime}" scripts/summarize_experiment.py \
      "${prediction}" --scores "${score_path}" --json \
    > "${prefix}.summary.json"
fi
if [[ ! -f "${prefix}.paired-statistics.json" ]]; then
  nice -n 15 env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    "${python_runtime}" scripts/analyze_paired_statistics.py \
      "${prediction}" --scores "${score_path}" \
      --output "${prefix}.paired-statistics.json" \
      --population-scope "${population_scope}"
fi
if [[ ! -f "${prefix}.failures.json" ]]; then
  nice -n 15 env \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=src \
    "${python_runtime}" scripts/analyze_predictions.py \
      "${prediction}" --prediction-field raw_prediction \
      --label-field reference --configuration W --json \
      > "${prefix}.failures.json"
fi

"${python_runtime}" - "${prediction}" "${prefix}.smatch.jsonl" \
  "${prefix}.summary.json" "${prefix}.paired-statistics.json" \
  "${population_scope}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

prediction, scores, summary, paired = map(Path, sys.argv[1:5])
expected_population_scope = sys.argv[5]
manifest = json.loads(Path(str(scores) + ".manifest.json").read_text())
prediction_count = sum(1 for line in prediction.open() if line.strip())
score_count = sum(1 for line in scores.open() if line.strip())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if prediction_count != score_count or manifest.get("records") != prediction_count:
    raise SystemExit("score count does not match experiment matrix")
if manifest.get("input_sha256") != sha(prediction):
    raise SystemExit("score manifest input hash mismatch")
if manifest.get("output_sha256") != sha(scores):
    raise SystemExit("score manifest output hash mismatch")
report = json.loads(summary.read_text())
if report.get("records") != prediction_count:
    raise SystemExit("summary count does not match experiment matrix")
paired_report = json.loads(paired.read_text())
if paired_report.get("records") != prediction_count:
    raise SystemExit("paired-analysis count does not match experiment matrix")
if paired_report.get("population_scope") != expected_population_scope:
    raise SystemExit("paired-analysis population scope mismatch")
if paired_report.get("gate", {}).get("passed") is not True:
    raise SystemExit("paired-analysis gate did not pass")
source = paired_report.get("source", {})
if source.get("predictions_sha256") != sha(prediction):
    raise SystemExit("paired-analysis prediction hash mismatch")
if source.get("scores_sha256") != sha(scores):
    raise SystemExit("paired-analysis score hash mismatch")
print(f"finalized experiment matrix: {prediction_count} rows")
PY
