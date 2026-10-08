"""Check the greedy commutation: masked (W) greedy == U up to U's first dead token.

For each record, tokenize the unconstrained (U) output, replay it through the
W-policy subword automaton, and find the first token t* whose transition is
dead (EOS counts as dead if the state is not accepting). Greedy W decoding with
the same logits must then reproduce U's first t* tokens, and must equal U
exactly when U never dies. Localized post-hoc regeneration from t* is therefore
output-equivalent to online masking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from automata_drs import SBNPolicy  # noqa: E402
from automata_drs.byt5 import ByT5ConstraintState  # noqa: E402
from automata_drs.subword import ByteLevelVocabulary, SubwordTokenConstraintController  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("online", type=Path, help="dev-online.jsonl with U and W rows")
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    controller = SubwordTokenConstraintController(ByteLevelVocabulary.from_tokenizer(tok))
    policy = SBNPolicy.for_configuration("W")
    eos = tok.eos_token_id
    rows = [json.loads(line) for line in args.online.open(encoding="utf-8") if line.strip()]
    by = {(r["id"], r["configuration"]): r for r in rows}
    ids = sorted({r["id"] for r in rows if r["configuration"] == "U"})

    stats = {"records": len(ids), "u_never_dies": 0, "identical_when_u_never_dies": 0,
             "u_dies": 0, "w_shares_prefix_before_t_star": 0, "w_diverges_at_t_star": 0,
             "counterexamples": []}
    for i in ids:
        u, w = by[(i, "U")]["raw_prediction"], by[(i, "W")]["raw_prediction"]
        u_ids = tok(u, add_special_tokens=False)["input_ids"] + [eos]
        w_ids = tok(w, add_special_tokens=False)["input_ids"] + [eos]
        state = ByT5ConstraintState.initial(policy)
        t_star = None
        for position, token_id in enumerate(u_ids):
            state = controller.transition(state, token_id)
            if state is None:
                t_star = position
                break
        if t_star is None:
            stats["u_never_dies"] += 1
            if u_ids == w_ids:
                stats["identical_when_u_never_dies"] += 1
            elif len(stats["counterexamples"]) < 5:
                stats["counterexamples"].append({"id": i, "case": "U valid but W differs"})
            continue
        stats["u_dies"] += 1
        if w_ids[:t_star] == u_ids[:t_star]:
            stats["w_shares_prefix_before_t_star"] += 1
            if len(w_ids) <= t_star or w_ids[t_star] != u_ids[t_star]:
                stats["w_diverges_at_t_star"] += 1
        elif len(stats["counterexamples"]) < 5:
            first = next(k for k, (a, b) in enumerate(zip(w_ids, u_ids)) if a != b)
            stats["counterexamples"].append({"id": i, "case": "prefix differs", "t_star": t_star,
                                             "first_difference": first})
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
