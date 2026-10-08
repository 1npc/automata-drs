"""Same-budget W rollback from preserved native tokens, never re-tokenized text."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from automata_drs import SBNPolicy, SubwordSBNLogitsProcessor, validate_sbn, check_sbn_state_wellformedness
from automata_drs.byt5 import ByT5ConstraintState
from automata_drs.causal_lm import extract_one_line_sbn
from automata_drs.certificates import verify_accepting_certificate


def rollback(ids, horizon, initial, propose, transition, is_accepting, fallback):
    """Latest exact-verified prefix; the token budget includes accepting EOS."""
    limited = tuple(ids[:horizon])
    states = [initial]
    for token in limited:
        nxt = transition(states[-1], token)
        if nxt is None:
            raise ValueError("W native trajectory has an invalid transition")
        states.append(nxt)
        if is_accepting(nxt):
            return limited[:len(states)-1], {"kept_tokens": len(states)-1, "dropped_tokens": 0,
                                            "completion_tokens": 0, "fresh_calls": 0, "fallback": False}
    calls = 0
    for kept in range(len(limited), 0, -1):
        remaining = horizon - kept
        if remaining <= 0:
            continue
        calls += 1
        witness = propose(states[kept], remaining)
        if witness is None:
            continue
        verified = verify_accepting_certificate(states[kept], witness, transition=transition,
                                                is_accepting=is_accepting, max_steps=remaining)
        if verified.accepted:
            return limited[:kept] + tuple(witness), {
                "kept_tokens": kept, "dropped_tokens": len(limited)-kept,
                "completion_tokens": len(witness), "fresh_calls": calls, "fallback": False}
    verified = verify_accepting_certificate(initial, fallback, transition=transition,
                                            is_accepting=is_accepting, max_steps=horizon)
    if not verified.accepted:
        raise ValueError("initial-certificate fallback failed verification")
    return tuple(fallback), {"kept_tokens": 0, "dropped_tokens": len(limited),
                            "completion_tokens": len(fallback), "fresh_calls": calls, "fallback": True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("predictions", type=Path)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--horizons", nargs="+", type=int, default=[68, 78, 256, 320])
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists() or Path(str(args.output)+".manifest.json").exists():
        p.error("refusing to overwrite offline evidence")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    policy = SBNPolicy.for_configuration("W")
    processor = SubwordSBNLogitsProcessor(tokenizer=tok, prompt_token_ids=[], policy=policy,
                                         max_new_tokens=max(args.horizons), bounded_completion=True)
    transition = processor.controller.transition
    initial = ByT5ConstraintState.initial(policy)
    source_digest = hashlib.sha256(args.predictions.read_bytes()).hexdigest()
    rows = [json.loads(line) for line in args.predictions.read_text().splitlines()]
    written = 0
    with args.output.open("x", encoding="utf-8") as sink:
        for row in rows:
            if row["configuration"] != "W":
                continue
            ids = row["generation"]["native_token_ids"]
            for horizon in args.horizons:
                initialization = processor.certificate_within(initial, horizon)
                if initialization.certificate is None:
                    raise RuntimeError("same-proposer initial certificate unavailable")
                fallback = initialization.certificate.token_ids
                started = time.perf_counter()
                repaired, info = rollback(ids, horizon, initial, processor._completion_token_witness_within,
                                          transition, lambda state: state.is_accepting, fallback)
                repair_ms = (time.perf_counter()-started)*1000
                for arm, out_ids, metadata in (("W", tuple(ids[:horizon]), None), ("WRollback", repaired, info)):
                    eos_positions = [i for i, token in enumerate(out_ids) if token in processor.vocabulary.eos_token_ids]
                    if eos_positions:
                        out_ids = out_ids[:eos_positions[0]+1]
                    raw = tok.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    structural = validate_sbn(raw)
                    valid = structural.valid and check_sbn_state_wellformedness(structural.state).valid
                    output = {k: row[k] for k in ("id", "input", "reference")}
                    output.update(configuration=arm, prediction=extract_one_line_sbn(raw), raw_prediction=raw,
                                  decode={"strategy": "offline-native-W", "num_beams": 1, "horizon": horizon},
                                  validity={"raw_sbn": structural.valid, "raw_clf_referee": valid},
                                  generation={"native_token_ids": list(out_ids), "output_tokens_including_eos": len(out_ids),
                                              "ended_with_eos": bool(eos_positions)},
                                  offline=metadata, initial_certificate_token_ids=list(fallback),
                                  timing={"posthoc_ms": repair_ms if arm == "WRollback" else 0,
                                          "source_W320_generation_ms": row["timing"]["generation_ms"],
                                          "bounded_generation_time_not_measured": True})
                    if arm == "WRollback" and (len(out_ids)>horizon or not valid or not eos_positions):
                        raise RuntimeError("rollback release gate failed")
                    sink.write(json.dumps(output, ensure_ascii=False)+"\n")
                    written += 1
    if hashlib.sha256(args.predictions.read_bytes()).hexdigest() != source_digest:
        raise RuntimeError("native source changed during derivation")
    with Path(str(args.output)+".manifest.json").open("x", encoding="utf-8") as sink:
        json.dump({"status": "complete", "rows": written, "source_sha256": source_digest,
                   "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                   "horizons": args.horizons, "includes_accepting_eos": True,
                   "native_tokens_only": True, "retokenization": False}, sink, indent=2)
    print(f"offline derivation complete: {written} rows", flush=True)


if __name__ == "__main__":
    main()
