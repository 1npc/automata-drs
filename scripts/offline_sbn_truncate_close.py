"""L1 late arm: truncate an unconstrained SBN output at its longest W-viable prefix
and append the WF/APD completion constructor's suffix (within the same budget).

W-valid U outputs are released unchanged. Rows follow the generator schema.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from automata_drs import SBNPolicy, SubwordSBNLogitsProcessor, check_sbn_state_wellformedness, validate_sbn  # noqa: E402
from automata_drs.byt5 import ByT5ConstraintState  # noqa: E402
from automata_drs.causal_lm import extract_one_line_sbn  # noqa: E402


def w_valid(raw: str) -> bool:
    result = validate_sbn(raw)
    return result.valid and check_sbn_state_wellformedness(result.state).valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path, help="generator JSONL containing U rows")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    policy = SBNPolicy.for_configuration("W")
    proc = SubwordSBNLogitsProcessor(tokenizer=tok, prompt_token_ids=[], policy=policy,
                                     max_new_tokens=args.horizon, bounded_completion=True)
    eos = set(proc.vocabulary.eos_token_ids)

    rows = [json.loads(line) for line in args.predictions.open() if line.strip()]
    with args.output.open("x", encoding="utf-8") as sink:
        for row in (r for r in rows if r["configuration"] == "U"):
            raw = row["raw_prediction"]
            info = {"changed": False, "kept_tokens": None, "dropped_tokens": 0, "found": True}
            if not w_valid(raw):
                ids = tok(raw, add_special_tokens=False)["input_ids"]
                states = [ByT5ConstraintState.initial(policy)]
                for token_id in ids:
                    nxt = proc.controller.transition(states[-1], token_id)
                    if nxt is None:
                        break
                    states.append(nxt)
                released = None
                for kept in range(len(states) - 1, -1, -1):
                    witness = proc._completion_token_witness_within(states[kept], args.horizon - kept)
                    if witness is not None:
                        released = ids[:kept] + list(witness)
                        info = {"changed": True, "kept_tokens": kept, "dropped_tokens": len(ids) - kept,
                                "found": True}
                        break
                if released is None:
                    info = {"changed": False, "kept_tokens": None, "dropped_tokens": 0, "found": False}
                else:
                    raw = tok.decode([i for i in released if i not in eos], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
            sink.write(json.dumps({"id": row["id"], "input": row["input"], "reference": row["reference"],
                                   "prediction": extract_one_line_sbn(raw), "raw_prediction": raw,
                                   "configuration": "L1", "decode": {"strategy": "truncate-close-from-U"},
                                   "offline": info}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
