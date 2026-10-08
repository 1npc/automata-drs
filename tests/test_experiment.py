from __future__ import annotations

import unittest

from automata_drs.experiment import summarize_experiment, summary_to_markdown


def row(
    identifier: str,
    configuration: str,
    *,
    prediction: str,
    sbn: bool,
    referee: bool | None,
    generation_ms: float,
    logprob: float,
):
    return {
        "id": identifier,
        "reference": "gold",
        "prediction": prediction,
        "configuration": configuration,
        "decode": {"strategy": "greedy", "num_beams": 1},
        "validity": {"sbn": sbn, "clf_referee": referee},
        "timing": {"generation_ms": generation_ms, "constraint_ms": 1.0},
        "generation": {"hit_max_new_tokens": False},
        "score": {
            "raw_model_logprob": logprob,
            "raw_model_mean_logprob": logprob / 2,
        },
        "constraint": None,
    }


class ExperimentSummaryTest(unittest.TestCase):
    def test_aggregate_and_paired_changes(self):
        records = [
            row("1", "U", prediction="bad", sbn=False, referee=None, generation_ms=10, logprob=-2),
            row("2", "U", prediction="same", sbn=True, referee=True, generation_ms=12, logprob=-3),
            row("1", "B", prediction="fixed", sbn=True, referee=True, generation_ms=13, logprob=-5),
            row("2", "B", prediction="same", sbn=True, referee=True, generation_ms=14, logprob=-3),
        ]
        summary = summarize_experiment(records)
        self.assertEqual(4, summary["records"])
        self.assertEqual(2, summary["unique_examples"])
        b_group = next(group for group in summary["groups"] if group["configuration"] == "B")
        self.assertEqual(2, b_group["sbn_valid"])
        self.assertEqual(2, b_group["raw_sbn_valid"])
        self.assertEqual(2, b_group["referee_valid"])
        pair = summary["paired_vs_u"][0]
        self.assertEqual(1, pair["prediction_changed"])
        self.assertEqual(1, pair["raw_sbn_fixed"])
        self.assertEqual(1, pair["sbn_fixed"])
        self.assertEqual(1, pair["wellformed_fixed"])
        self.assertEqual(-1.5, pair["raw_model_logprob_delta_mean"])
        self.assertEqual(-0.75, pair["raw_model_mean_logprob_delta_mean"])
        self.assertIn("Paired changes versus U", summary_to_markdown(summary))
        self.assertIn("Δ raw mean log-p", summary_to_markdown(summary))

    def test_duplicates_are_rejected(self):
        record = row("1", "U", prediction="x", sbn=True, referee=True, generation_ms=1, logprob=-1)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summarize_experiment([record, dict(record)])

    def test_completion_witness_and_slack_are_aggregated(self):
        record = row(
            "1",
            "BF",
            prediction="x",
            sbn=True,
            referee=True,
            generation_ms=1,
            logprob=-1,
        )
        record["constraint"] = {
            "bounded_rows": 2,
            "completion_certificates_tested": 3,
            "completion_certificates_accepted": 2,
            "certified_limit_hits": 2,
            "completion_witness_tokens_total": 10,
            "completion_slack_total": 14,
            "completion_slack_min": 3,
            "all_masked_rows": 1,
        }
        group = summarize_experiment([record])["groups"][0]
        self.assertEqual(5.0, group["completion_witness_tokens_mean"])
        self.assertEqual(7.0, group["completion_slack_mean"])
        self.assertEqual(3, group["completion_slack_min"])
        self.assertEqual(1, group["all_masked_rows"])

    def test_raw_and_postprocessed_validity_are_not_conflated(self):
        record = row(
            "1",
            "U",
            prediction="male.n.02",
            sbn=True,
            referee=True,
            generation_ms=1,
            logprob=-1,
        )
        record["validity"]["raw_sbn"] = False
        record["generation"]["canonical_tokenization"] = True
        group = summarize_experiment([record])["groups"][0]
        self.assertEqual(0, group["raw_sbn_valid"])
        self.assertEqual(1, group["sbn_valid"])
        self.assertEqual(1, group["referee_valid"])
        self.assertEqual(0, group["raw_referee_valid"])
        self.assertEqual(0, group["raw_referee_scored"])
        self.assertEqual(1, group["canonical_tokenization"])
        self.assertEqual(1.0, group["canonical_tokenization_rate"])
        markdown = summary_to_markdown(summarize_experiment([record]))
        self.assertIn("Raw SBN valid", markdown)

    def test_postw_beam_selection_diagnostics_are_aggregated(self):
        records = []
        for identifier, fallback, valid_candidates in (
            ("1", False, 3),
            ("2", True, 0),
        ):
            record = row(
                identifier,
                "PostW",
                prediction="x",
                sbn=True,
                referee=True,
                generation_ms=1,
                logprob=-1,
            )
            record["decode"] = {"strategy": "beam4", "num_beams": 4}
            record["beam_selection"] = {
                "method": "highest-sequence-score-w-valid-u-beam",
                "fallback_to_u_top1": fallback,
                "w_valid_candidate_count": valid_candidates,
            }
            records.append(record)

        group = summarize_experiment(records)["groups"][0]
        self.assertEqual(2, group["postw_selection_records"])
        self.assertEqual(1, group["postw_fallbacks"])
        self.assertEqual(1.5, group["postw_w_valid_candidates_mean"])


if __name__ == "__main__":
    unittest.main()
