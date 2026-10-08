"""Offline same-logit SBN baselines derived from a local-W token stream at horizon H.

The local W mask does not depend on H, so greedy W stopped at H-k is a prefix of
greedy W at H. Arms (no model call):
- FreeClose: W output + the WF/APD completion constructor, allowed to exceed H.
- Reserve-k: stop W at H-k tokens unless it already finished, then append the
  constructed completion; it counts as released only if completion + EOS fit in
  k tokens, otherwise the hard cap leaves an unfinished prefix.
Rows follow the generator schema so score_sbn_smatchpp.py scores them unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from automata_drs import SBNPolicy, SubwordSBNLogitsProcessor  # noqa: E402
from automata_drs.byt5 import ByT5ConstraintState  # noqa: E402
from automata_drs.causal_lm import extract_one_line_sbn  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="generator JSONL containing W rows")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--reserves", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--free-close-limit", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    proc = SubwordSBNLogitsProcessor(tokenizer=tok, prompt_token_ids=[], policy=SBNPolicy.for_configuration("W"),
                                     max_new_tokens=args.horizon, bounded_completion=True)
    eos = set(proc.vocabulary.eos_token_ids)
    H = args.horizon

    def replay(ids: list[int]) -> ByT5ConstraintState | None:
        state = ByT5ConstraintState.initial(SBNPolicy.for_configuration("W"))
        for token_id in ids:
            state = proc.controller.transition(state, token_id)
            if state is None:
                return None
        return state

    def surface(ids: list[int]) -> str:
        return tok.decode([i for i in ids if i not in eos], skip_special_tokens=True,
                          clean_up_tokenization_spaces=False)

    rows = [json.loads(line) for line in args.predictions.open() if line.strip()]
    w_rows = [r for r in rows if r["configuration"] == "W"]
    with args.output.open("x", encoding="utf-8") as sink:
        for row in w_rows:
            ids = tok(row["raw_prediction"], add_special_tokens=False)["input_ids"]
            finished = bool(row["generation"]["ended_with_eos"])
            outputs = {}
            state = replay(ids)
            if finished:
                outputs["FreeClose"] = (ids, {"overrun": 0, "fits": True})
            else:
                witness = None if state is None else proc._completion_token_witness_within(state, args.free_close_limit)
                full = ids + list(witness or ())
                outputs["FreeClose"] = (full, {"overrun": max(0, len(full) - H), "fits": witness is not None})
            for k in args.reserves:
                if finished and len(ids) + 1 <= H - k:
                    outputs[f"Reserve-{k}"] = (ids, {"truncated": False, "fits": True, "dropped": 0})
                    continue
                kept = ids[: H - k]
                kept_state = replay(kept)
                witness = None if kept_state is None else proc._completion_token_witness_within(kept_state, k)
                released = kept + list(witness) if witness is not None else kept
                outputs[f"Reserve-{k}"] = (released, {"truncated": True, "fits": witness is not None,
                                                      "dropped": max(0, len(ids) - len(kept))})
            for arm, (out_ids, info) in outputs.items():
                raw = surface(out_ids)
                sink.write(json.dumps({"id": row["id"], "input": row["input"], "reference": row["reference"],
                                       "prediction": extract_one_line_sbn(raw), "raw_prediction": raw,
                                       "configuration": arm, "decode": {"strategy": "offline-from-W", "horizon": H},
                                       "offline": info}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
