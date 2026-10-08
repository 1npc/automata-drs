#!/usr/bin/env bash
set -euo pipefail

source scripts/colossus_scratch.sh
automata_scratch_init
automata_scratch_require_private

# Queue the remaining immutable method suite behind one already-running
# publication job without overlapping model processes. This guard performs no
# GPU polling; each child stage applies the normal exclusive/shared claim gate.

poll_seconds=${AUTOMATA_DRS_JOB_POLL_SECONDS:-300}
after_pid=${AUTOMATA_DRS_AFTER_PID:-}
expected_fragment=${AUTOMATA_DRS_AFTER_COMMAND_FRAGMENT:-run_colossus_selected_lora_publication_matrix.sh dev}
if [[ ! "${poll_seconds}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AUTOMATA_DRS_JOB_POLL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ -n "${after_pid}" && ! "${after_pid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "AUTOMATA_DRS_AFTER_PID must be empty or a positive integer" >&2
  exit 2
fi

exec 8>runs/lora/.selected-method-suite-continuation.lock
if ! flock -n 8; then
  echo "another selected-host suite continuation is already queued" >&2
  exit 4
fi

if [[ -n "${after_pid}" ]] && kill -0 "${after_pid}" 2>/dev/null; then
  observed_command=$(ps -p "${after_pid}" -o args= 2>/dev/null || true)
  if [[ -z "${observed_command}" ]]; then
    start_ticks=""
  elif [[ "${observed_command}" != *"${expected_fragment}"* ]]; then
    echo "after-PID command does not match the expected publication job" >&2
    exit 4
  else
    start_ticks=$(awk '{print $22}' "/proc/${after_pid}/stat" 2>/dev/null || true)
  fi
  if [[ -n "${start_ticks}" ]]; then
    echo "waiting for publication PID ${after_pid}; poll=${poll_seconds}s"
    while kill -0 "${after_pid}" 2>/dev/null; do
      current_ticks=$(awk '{print $22}' "/proc/${after_pid}/stat" 2>/dev/null || true)
      [[ "${current_ticks}" == "${start_ticks}" ]] || break
      sleep "${poll_seconds}"
    done
  fi
fi

echo "starting the remaining selected-host method suite"
exec bash scripts/run_colossus_selected_lora_method_suite.sh
