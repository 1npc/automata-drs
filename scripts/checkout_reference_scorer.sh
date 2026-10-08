#!/usr/bin/env bash
set -euo pipefail

# Materialize the pinned official SBN Smatch implementation in a new directory.
# The caller owns and must clean the exact destination (normally a mktemp path).

if (( $# != 1 )); then
  echo "usage: $0 NEW_DESTINATION" >&2
  exit 2
fi
destination=$1
if [[ -e "${destination}" ]]; then
  echo "refusing existing destination: ${destination}" >&2
  exit 2
fi

tool_row=$(
  python3 - <<'PY'
import json

config = json.load(open("configs/evaluation_tools.json", encoding="utf-8"))
tool = config["sbn_smatch"]
print("\t".join((tool["repository"], tool["revision"], tool["relative_path"])))
PY
)
IFS=$'\t' read -r repository revision relative_path <<<"${tool_row}"

git clone --filter=blob:none --no-checkout "${repository}" "${destination}"
git -C "${destination}" fetch --depth 1 origin "${revision}"
git -C "${destination}" checkout --detach "${revision}"
resolved=$(git -C "${destination}" rev-parse HEAD)
if [[ "${resolved}" != "${revision}" ]]; then
  echo "scorer revision mismatch: ${resolved}" >&2
  exit 3
fi
scorer="${destination}/${relative_path}"
if [[ ! -f "${scorer}" ]]; then
  echo "pinned scorer missing: ${scorer}" >&2
  exit 3
fi
printf '%s\n' "${scorer}"
