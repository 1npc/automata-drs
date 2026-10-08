"""Small GPU execution gate; no gold score or quality-based expansion rule."""
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
original = [json.loads(line) for line in (directory/"original68.jsonl").read_text().splitlines()]
instrumented = [json.loads(line) for line in (directory/"natural68.jsonl").read_text().splitlines()]
if [(row["id"],row["configuration"],row["raw_prediction"]) for row in original] != \
   [(row["id"],row["configuration"],row["raw_prediction"]) for row in instrumented]:
    raise RuntimeError("natural instrumentation parity requires diagnosis")
score_manifest = json.loads((directory/"smoke.scores.jsonl.manifest.json").read_text())
if not score_manifest["all_exact"]:
    raise RuntimeError("scoring preflight not exact")
counts = {"W320":4, "natural68":8, "natural78":8, "denial50":8, "denial100":4, "offline":32}
if (directory/"failfast.jsonl").exists():
    counts["failfast"] = 1
    failed = json.loads((directory/"failfast.jsonl").read_text())
    stats = failed["constraint"]
    gen = failed["generation"]
    if stats["all_masked_rows"] != 1 or gen["decoder_failure"]["reason"] != "all-masked":
        raise RuntimeError("fail-fast smoke did not stop on first empty support")
    if gen["native_token_ids"] != gen["decoder_failure"]["native_prefix_token_ids"]:
        raise RuntimeError("failure prefix was altered")
    if gen["decoder_steps"] != len(gen["native_token_ids"])+1 or gen["ended_with_eos"]:
        raise RuntimeError("failure incorrectly released or continued")
    failed_score = json.loads((directory/"failfast.scores.jsonl").read_text())
    if failed_score["f1"] != 0 or failed_score["raw_W_valid"] or not failed_score["exact"]:
        raise RuntimeError("decoder failure zero-score gate failed")
for name, expected in counts.items():
    rows = [json.loads(line) for line in (directory/(name+".jsonl")).read_text().splitlines()]
    if len(rows)!=expected:
        raise RuntimeError(f"incomplete smoke {name}")
    manifest = json.loads((directory/(name+".jsonl.manifest.json")).read_text())
    if manifest["status"] != "complete":
        raise RuntimeError("incomplete smoke manifest")
    for row in rows:
        if row["configuration"] in {"WWCD","WRollback"}:
            if not row["validity"]["raw_sbn"] or not row["validity"]["raw_clf_referee"]:
                raise RuntimeError("certified raw validity smoke failed")
            if row["configuration"] == "WWCD":
                stats = row["constraint"]
                if stats["all_masked_rows"] or stats["prefix_rows_unavailable"]:
                    raise RuntimeError("owned-prefix smoke failed")
                if stats["carried_tokens"]+stats["cached_tokens"]+stats["replanned_tokens"] != stats["rows"]:
                    raise RuntimeError("certificate accounting smoke failed")
        if "native_token_ids" not in row["generation"]:
            raise RuntimeError("native token recording failed")
with (directory/"smoke-gate.json").open("x", encoding="utf-8") as sink:
    json.dump({"status":"passed", "expected_counts":counts, "quality_not_used_for_gate":True,
               "natural_byte_parity":True,"smatchpp_exact":True,
               "failure_termination_checked": "failfast" in counts},sink,indent=2)
print("GPU execution smoke passed", flush=True)
