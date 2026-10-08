"""Synthetic export guards; these fixtures are not measured paper results."""
import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_arr1012_review_paper_data import merge_reports


def reports():
    keys = {f"H{h}:deny0:{arm}" for h in (68, 78, 256, 320) for arm in ("W", "WRollback")}
    keys |= {f"H{h}:deny0:{arm}" for h in (68, 78) for arm in ("WF", "WWCD")}
    keys |= {f"H68:deny{r:g}:{arm}" for r in (.25, .5, .75) for arm in ("WF", "WWCD")}
    summary = {"n": 1132, "raw_valid": 1132, "macro_f1": .8, "accepting_eos": 1132,
               "all_masked": 0, "prefix_unavailable": 0, "operations": {}, "decoder_failures": 0,
               "certificate_paths": {"carried_tokens": 2, "cached_tokens": 3, "replanned_tokens": 4, "rows": 9},
               "output_length_mean": 20, "online_generation_seconds": 40, "posthoc_seconds": 0}
    errors = ["independent semantic checker disagreement requires diagnosis"]
    frozen = {"status": "failed", "errors": errors, "independent_false_accepts": 1,
              "summaries": {k: copy.deepcopy(summary) for k in keys}, "comparisons": {}}
    mechanism = {"status": "complete_mechanism_evidence", "frozen_scientific_gate": "failed",
                 "frozen_gate_errors": errors, "summaries": {k: {"n": 1132} for k in keys}}
    checked = {"n": 1132, "delivered_native_valid": 1132,
               "fallback_referee_compatible_delivered_valid": 1131,
               "native_macro_points": 80, "compatibility_zeroed_macro_points": 79.9}
    audit = {"status": "complete_corrected_audit_not_release", "frozen_scientific_gate": "failed",
             "frozen_gate_errors": errors, "existing_corrected_false_accepts": 0, "new_false_accept_observations": 18,
             "bootstrap": {"resamples": 20000, "unit": "input", "seed": 20261001},
             "summaries": {k: copy.deepcopy(checked) for k in keys}, "comparisons": {}}
    pairs = [(h, 0, arm) for h in (68, 78) for arm in ("W", "WF", "WRollback")]
    pairs += [(68, r, "WF") for r in (.25, .5, .75)]
    contribution = {"mean_points": 0, "ci95_points": [0, 0]}
    policy = {"n": 1132, "counts": {"common_valid": 1132, "rescue": 0, "lost": 0, "neither_valid": 0},
              **{name: copy.deepcopy(contribution) for name in ("delta", "common_valid", "rescue", "lost")}}
    for h, r, arm in pairs:
        k = f"H{h}:deny{r:g}:APD-minus-{arm}"
        frozen["comparisons"][k] = {"paired_n": 1132, "byte_identical": 1132, "delta": {"mean": 0}}
        audit["comparisons"][k] = {"native": copy.deepcopy(policy), "compatibility_zeroed": copy.deepcopy(policy)}
        audit["comparisons"][k]["compatibility_zeroed"]["counts"].update(common_valid=1131, neither_valid=1)
    return frozen, mechanism, audit


class PaperDataTest(unittest.TestCase):
    def test_real_audit_schema_checks_both_policy_denominators(self):
        f, m, a = reports()
        self.assertTrue(all("n" not in pair for pair in a["comparisons"].values()))
        result = merge_reports(f, m, a)
        self.assertTrue(all(pair["n"] == 1132 for pair in result["comparisons"].values()))
        for policy in ("native", "compatibility_zeroed"):
            f, m, a = reports()
            next(iter(a["comparisons"].values()))[policy]["n"] = 1131
            with self.assertRaises(ValueError):
                merge_reports(f, m, a)

    def test_complete_export_preserves_failed_gate_and_separate_metrics(self):
        result = merge_reports(*reports())
        self.assertEqual(len(result["cells"]), 18)
        self.assertEqual(len(result["comparisons"]), 9)
        self.assertEqual(result["frozen_scientific_gate"], "failed")
        self.assertFalse(result["corrected_independent_agreement_passed"])
        cell = result["cells"]["H68:deny0:WWCD"]
        self.assertEqual((cell["native_delivered_valid"], cell["fallback_referee_compatible_delivered_valid"]), (1132, 1131))
        self.assertEqual((cell["native_macro_f1_points"], cell["compatibility_zeroed_macro_f1_points"]), (80, 79.9))

    def test_missing_condition_partial_report_or_changed_gate_is_rejected(self):
        for change in ("condition", "partial", "gate"):
            f, m, a = reports()
            if change == "condition":
                del a["summaries"]["H68:deny0:WWCD"]
            elif change == "partial":
                m["status"] = "partial"
            else:
                a["frozen_scientific_gate"] = "passed"
            with self.assertRaises(ValueError):
                merge_reports(f, m, a)

    def test_native_score_and_independent_zero_policy_cannot_be_changed(self):
        for field, value in (("native_macro_points", 81), ("compatibility_zeroed_macro_points", 81),
                             ("delivered_native_valid", 1131), ("native_macro_points", float("nan"))):
            f, m, a = reports()
            a["summaries"]["H68:deny0:WWCD"][field] = value
            with self.assertRaises(ValueError):
                merge_reports(f, m, a)

    def test_all_paired_comparisons_and_full_strata_are_required(self):
        for change in ("missing", "strata", "identity", "interval", "cohort"):
            f, m, a = reports()
            key = next(iter(a["comparisons"]))
            pair = a["comparisons"][key]
            if change == "missing":
                del a["comparisons"][key]
            elif change == "strata":
                pair["native"]["counts"]["common_valid"] = 1131
            elif change == "identity":
                pair["native"]["rescue"]["mean_points"] = 1
            elif change == "cohort":
                pair["native"]["counts"].update(common_valid=1131, rescue=1)
            else:
                pair["native"]["delta"]["ci95_points"] = [1, 0]
            with self.assertRaises(ValueError):
                merge_reports(f, m, a)

    def test_apd_eos_and_accounting_invariants_are_required(self):
        for field, value in (("accepting_eos", 1131), ("all_masked", 1), ("prefix_unavailable", 1),
                             ("certificate_paths", {"carried_tokens": 2, "cached_tokens": 3, "replanned_tokens": 4, "rows": 10})):
            f, m, a = reports()
            f["summaries"]["H68:deny0:WWCD"][field] = value
            with self.assertRaises(ValueError):
                merge_reports(f, m, a)


if __name__ == "__main__":
    unittest.main()
