"""Post-review instrumentation; original SBN/controller algorithms stay unchanged.

Forced UNKNOWN is a synthetic availability intervention, NOT measured natural
failure or a time limit. Initialization is exempt and uses the original exact
certificate verifier. Denial is a stable SHA256 key of input, state and budget,
shared by WF/APD; carry/cache never call the new proposer.
"""
from __future__ import annotations

import hashlib
import time
from collections import Counter

from automata_drs import SubwordSBNLogitsProcessor, WitnessCarryingSubwordSBNLogitsProcessor


class AllMaskedDecodingFailure(RuntimeError):
    """No permitted continuation; preserve the prefix before an arbitrary argmax."""

    def __init__(self, prefix_token_ids):
        self.prefix_token_ids = tuple(int(t) for t in prefix_token_ids)
        super().__init__("all-masked: no permitted next token")


def generate_with_failure_capture(model, encoded, input_tokens, **kwargs):
    """Greedy batch-one only. Other runtime errors must propagate unchanged."""
    try:
        output = model.generate(**encoded, **kwargs)
        return output[0, input_tokens:], None
    except AllMaskedDecodingFailure as error:
        prefix = list(error.prefix_token_ids)
        if prefix[:input_tokens] != encoded["input_ids"][0].tolist():
            raise RuntimeError("failed prefix does not belong to this input") from error
        suffix = encoded["input_ids"].new_tensor(prefix[input_tokens:])
        return suffix, {"reason": "all-masked", "position": len(suffix),
                        "policy": "terminate-at-first-all-masked-v1",
                        "native_prefix_token_ids": prefix[input_tokens:]}


def deliverable_raw_valid(row):
    """An unreleased failure prefix is not a delivered graph, even if parseable."""
    from automata_drs import validate_sbn, check_sbn_state_wellformedness
    if row.get("generation", {}).get("decoder_failure") is not None:
        return False
    parsed = validate_sbn(row["raw_prediction"])
    return parsed.valid and check_sbn_state_wellformedness(parsed.state).valid


def denial_key(identifier, state, remaining):
    # State repr includes frozensets. Use explicit dataclass normalization so
    # PYTHONHASHSEED or process ordering cannot change the intervention.
    from dataclasses import fields, is_dataclass
    import json

    def normalize(value):
        if is_dataclass(value):
            return {f.name: normalize(getattr(value, f.name)) for f in fields(value)}
        if isinstance(value, (set, frozenset)):
            return sorted((normalize(v) for v in value), key=lambda v: json.dumps(v, sort_keys=True))
        if isinstance(value, (tuple, list)):
            return [normalize(v) for v in value]
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        return value
    stable_id = str(identifier).rsplit(":", 1)[-1]
    payload = json.dumps(["arr1012-review-denial-seed42", stable_id, normalize(state), remaining], sort_keys=True)
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big") / 2**64


class _Operations:
    def __init__(self, *, review_identifier, review_denial_rate=0.0,
                 review_fail_fast=False, **kwargs):
        self.review_identifier = review_identifier
        self.review_denial_rate = review_denial_rate
        self.review_fail_fast = review_fail_fast
        self.operations = Counter()
        self.completion_requests = []
        self.decision_trace = []
        self.completion_histogram = Counter()
        self.initializing = True
        self._last_distance = None
        super().__init__(**kwargs)
        # Count actual primitive work without new transitions or cache lookups.
        compute = self.controller._compute_transition
        transition = self.controller.transition
        def counted_compute(state, token):
            self.operations["native_transition_evaluations_total"] += 1
            return compute(state, token)
        def counted_transition(state, token):
            self.operations["native_transition_requests_total"] += 1
            return transition(state, token)
        self.controller._compute_transition = counted_compute
        self.controller.transition = counted_transition
        if hasattr(self, "witnesses"):
            self.witnesses.transition = counted_transition
        initial = self.prefix_states.state_for(self.prefix_states.prompt_token_ids)
        initialized = self.certificate_within(initial, self.max_new_tokens)
        if initialized.certificate is None:
            raise RuntimeError("review initialization failed before availability intervention")
        self.initial_certificate = initialized.certificate
        if hasattr(self, "witnesses"):
            self.witnesses.certificates.bind((), initial, initialized.certificate)
        self.initializing = False

    def __call__(self, input_ids, scores):
        if self.review_fail_fast and int(input_ids.shape[0]) != 1:
            raise ValueError("review fail-fast is registered for greedy batch-one only")
        before = self.stats().all_masked_rows
        masked = super().__call__(input_ids, scores)
        if self.review_fail_fast and self.stats().all_masked_rows > before:
            # The parent has recorded the complete failed search and its cost.
            # Stop BEFORE Transformers argmax can turn all -inf into token zero.
            raise AllMaskedDecodingFailure(input_ids[0].tolist())
        return masked

    def _completion_token_witness_within(self, state, remaining):
        self.operations["initial_proposer_calls" if self.initializing else "fresh_proposer_calls"] += 1
        denied = (not self.initializing and self.review_denial_rate > 0 and
                  denial_key(self.review_identifier, state, remaining) < self.review_denial_rate)
        started = time.perf_counter()
        witness = None if denied else super()._completion_token_witness_within(state, remaining)
        elapsed = time.perf_counter() - started
        self.operations["proposer_seconds"] += elapsed
        self.operations["forced_unknown" if denied else "constructor_executions"] += 1
        self.operations["verified_witnesses" if witness is not None else "unknown_results"] += 1
        self.completion_histogram[(remaining, None if witness is None else len(witness),
                                   "initial" if self.initializing else ("forced" if denied else "natural"))] += 1
        if len(self.completion_requests) < 256:
            self.completion_requests.append({
            "initial": self.initializing, "remaining": remaining,
            "status": "forced-unknown" if denied else ("verified" if witness is not None else "natural-unknown"),
            "certified_length_upper_bound": None if witness is None else len(witness),
            "slack": None if witness is None else remaining - len(witness),
            })
        return witness

    def _successor_completion_distance(self, successor, remaining_after_token):
        result = super()._successor_completion_distance(successor, remaining_after_token)
        self._last_distance = result
        return result

    def _ranked_allowed(self, state, ranked_token_ids, *, generated, prefix_token_ids=None):
        ranked = list(ranked_token_ids)
        before = self.stats()
        accepted = super()._ranked_allowed(state, ranked, generated=generated, prefix_token_ids=prefix_token_ids)
        after = self.stats()
        distance = self._last_distance if self.bounded_completion else None
        kind = "local" if not self.bounded_completion else "fresh-or-distance-cache"
        if hasattr(self, "witnesses") and accepted:
            certificate = self.witnesses.certificates._entries.get(tuple(prefix_token_ids) + (accepted[0],))
            distance = None if certificate is None else certificate.distance
            kind = next((k for k in ("carried", "cached", "replanned")
                         if getattr(after, k + "_tokens") > getattr(before, k + "_tokens")), "unknown")
        self.decision_trace.append({
            "position": generated, "chosen_token_id": accepted[0] if accepted else None,
            "model_top1_token_id": ranked[0],
            "chosen_model_rank": ranked.index(accepted[0]) if accepted else None,
            "justification": kind, "certified_length_upper_bound": distance,
            "slack": None if distance is None else self.max_new_tokens - generated - 1 - distance,
        })
        return accepted

    def review_report(self):
        return {"denial_rate": self.review_denial_rate, "synthetic_intervention": self.review_denial_rate > 0,
                "all_masked_policy": "terminate-at-first-all-masked-v1" if self.review_fail_fast else "legacy-record-only",
                "initial_certificate_token_ids": list(self.initial_certificate.token_ids),
                "initial_certificate_verified": True, "operations": dict(self.operations),
                "completion_requests_first256": self.completion_requests,
                "completion_histogram": [{"remaining": r,"certified_length_upper_bound": length,
                                           "request_type": kind,"count": count}
                                          for (r,length,kind),count in self.completion_histogram.items()],
                "decision_trace": self.decision_trace}


class InstrumentedWF(_Operations, SubwordSBNLogitsProcessor):
    pass


class InstrumentedAPD(_Operations, WitnessCarryingSubwordSBNLogitsProcessor):
    pass
