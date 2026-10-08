#!/usr/bin/env bash
set -euo pipefail

source scripts/colossus_scratch.sh
automata_scratch_init

# Wait for one known original baseline queue, then invoke the fail-closed
# recovery runner. The caller must provide an explicit private scratch root.

if (( $# != 1 )); then
  echo "usage: $0 ORIGINAL_QUEUE_PID" >&2
  exit 2
fi
original_pid=$1
if [[ ! "${original_pid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ORIGINAL_QUEUE_PID must be a positive integer" >&2
  exit 2
fi
if [[ "${AUTOMATA_DRS_SCRATCH_ROOT}" == "/tmp" ]]; then
  echo "resume guard requires an explicit non-/tmp scratch root" >&2
  exit 2
fi
if ! kill -0 "${original_pid}" 2>/dev/null; then
  echo "original queue is not running: ${original_pid}" >&2
  exit 2
fi

echo "waiting for original recent-model queue: ${original_pid}"
tail --pid="${original_pid}" -f /dev/null

if pgrep -f 'bash scripts/run_colossus_recent_lora_baselines.sh' >/dev/null; then
  echo "another original recent-model queue is still active" >&2
  exit 4
fi

echo "original queue exited; starting fail-closed private-scratch recovery"
exec bash scripts/resume_colossus_recent_lora_baselines.sh
