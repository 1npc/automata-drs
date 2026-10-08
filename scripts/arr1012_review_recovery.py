"""Strict reuse of unaffected rows from a stopped, hash-locked predecessor.

The predecessor may be incomplete, but every complete line must be an exact
scheduled prefix. A final interrupted write is retained in the original file
and is not considered a completed row. All-masked rows are always regenerated.
"""
import copy
import hashlib
import json
from pathlib import Path

from automata_drs.causal_lm import validate_generation_resume_prefix

SCIENCE_FIELDS = (
    "model", "model_loader", "requested_revision", "resolved_revision", "adapter",
    "path", "start", "limit", "line_numbers", "max_input_tokens", "max_new_tokens",
    "num_beams", "batch_size", "configurations", "completion_budget_factor",
    "record_native_token_ids", "review_instrumentation", "fresh_proposal_denial_rate",
    "score_model_logprob", "constraint_lifting", "schedule", "source_sha256",
    "expected_examples", "expected_rows", "torch", "transformers", "peft",
)


def load_reusable_rows(path, manifest, expected_identities):
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != manifest["review_reuse_sha256"]:
        raise ValueError("predecessor hash changed")
    old = json.loads(Path(str(path) + ".manifest.json").read_text(encoding="utf-8"))
    mismatches = [key for key in SCIENCE_FIELDS if old.get(key) != manifest.get(key)]
    if mismatches:
        raise ValueError("predecessor science contract mismatch: " + ", ".join(mismatches))
    # A line is committed only with its newline. Never edit the source to repair it.
    lines = data.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b"\n"):
        lines = lines[:-1]
    rows = [json.loads(line) for line in lines]
    validate_generation_resume_prefix(rows, expected_identities, num_beams=1)
    reusable = {}
    for line_number, row in enumerate(rows, 1):
        stats = row.get("constraint") or {}
        generation = row["generation"]
        if stats.get("all_masked_rows", 0) or generation.get("decoder_failure") is not None:
            continue
        if stats.get("prefix_rows_unavailable", 0):
            raise ValueError("predecessor has an unavailable prefix")
        if "native_token_ids" not in generation:
            raise ValueError("predecessor lacks native token IDs")
        reused = copy.deepcopy(row)
        reused["review_reuse"] = {
            "path": str(path), "sha256": digest, "line": line_number,
            "original_line_sha256": hashlib.sha256(lines[line_number - 1]).hexdigest(),
            "reason": "no all-masked event; generation and costs unchanged",
        }
        reusable[(row["id"], row["configuration"])] = reused
    return reusable
