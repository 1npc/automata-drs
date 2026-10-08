import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from arr1012_review_mechanism_analysis import (
    AUDIT_ONLY_ERROR, common_prefix, distribution, online_measurements,
    rollback_measurements, validate_aggregate,
)


def online_row():
    return {
        "configuration": "WWCD", "decode": {"strategy":"greedy"},
        "generation": {"native_token_ids":[10,20]},
        "constraint": {"carried_tokens":1,"cached_tokens":0,"replanned_tokens":1,
                       "rows":2,"all_masked_rows":0,"prefix_rows_unavailable":0},
        "review_operations": {
            "initial_certificate_verified": True, "initial_certificate_token_ids":[10,20],
            "operations": {"initial_proposer_calls":1,"fresh_proposer_calls":2,
                           "unknown_results":1,"verified_witnesses":2},
            "completion_histogram":[
                {"remaining":2,"certified_length_upper_bound":2,"request_type":"initial","count":1},
                {"remaining":1,"certified_length_upper_bound":1,"request_type":"natural","count":1},
                {"remaining":1,"certified_length_upper_bound":None,"request_type":"forced","count":1},
            ],
            "decision_trace":[
                {"position":0,"chosen_token_id":10,"model_top1_token_id":10,
                 "chosen_model_rank":0,"justification":"replanned",
                 "certified_length_upper_bound":1,"slack":0},
                {"position":1,"chosen_token_id":20,"model_top1_token_id":30,
                 "chosen_model_rank":1,"justification":"carried",
                 "certified_length_upper_bound":0,"slack":0},
            ],
        },
    }


class MechanismAnalysisTest(unittest.TestCase):
    def test_incomplete_or_failed_non_audit_aggregate_cannot_be_released(self):
        keys={f"H{h}:deny0:{arm}" for h in (68,78,256,320) for arm in ("W","WRollback")}
        keys|={f"H{h}:deny0:{arm}" for h in (68,78) for arm in ("WF","WWCD")}
        keys|={f"H68:deny{r:g}:{arm}" for r in (.25,.5,.75) for arm in ("WF","WWCD")}
        report={"status":"failed","errors":[AUDIT_ONLY_ERROR],
                "independent_false_accepts":12,"summaries":{k:{"n":1132} for k in keys}}
        validate_aggregate(report)
        self.assertEqual(report["status"],"failed")
        for field,value in (("errors",["missing primary paired cell"]),
                            ("status","passed"),("independent_false_accepts",0)):
            broken=copy.deepcopy(report);broken[field]=value
            with self.assertRaises(ValueError):
                validate_aggregate(broken)
        broken=copy.deepcopy(report)
        broken["summaries"]["H68:deny0:WF"]["n"]=1131
        with self.assertRaises(ValueError):
            validate_aggregate(broken)
        broken=copy.deepcopy(report)
        broken["summaries"]["H68:deny0:bogus"]=broken["summaries"].pop("H68:deny0:WF")
        with self.assertRaises(ValueError):
            validate_aggregate(broken)

    def test_weighted_quantiles_include_multiplicity(self):
        result = distribution({0:1,10:3})
        self.assertEqual(result["n"],4)
        self.assertEqual(result["mean"],7.5)
        self.assertEqual(result["quantiles"]["p25"],7.5)
        self.assertEqual(result["quantiles"]["p50"],10)
        self.assertEqual(distribution({})["n"],0)
        for bad in ({0:0},{0:-1},{0:True},{float("nan"):1}):
            with self.assertRaises(ValueError):
                distribution(bad)

    def test_native_decisions_and_owned_completion_are_validated(self):
        row=online_row()
        result=online_measurements(row,2)
        self.assertEqual(result["decisions"]["carried:changed"],1)
        self.assertEqual(result["calls"]["forced"],1)
        self.assertEqual(result["lengths"],{2:1,1:1})
        self.assertEqual(row["generation"]["native_token_ids"],[10,20])

    def test_false_bounds_or_missing_native_token_are_rejected(self):
        row=online_row()
        row["review_operations"]["decision_trace"][0]["slack"]=1
        with self.assertRaises(ValueError):
            online_measurements(row,2)
        row=online_row()
        row["generation"]["native_token_ids"]=[10,30]
        with self.assertRaises(ValueError):
            online_measurements(row,2)

    def test_missing_accounting_or_false_forced_witness_is_rejected(self):
        row=online_row()
        row["review_operations"]["operations"]["unknown_results"]=0
        with self.assertRaises(ValueError):
            online_measurements(row,2)
        row=online_row()
        row["review_operations"]["completion_histogram"][2]["certified_length_upper_bound"]=1
        with self.assertRaises(ValueError):
            online_measurements(row,2)

    def test_failed_wf_decision_is_not_an_emitted_token(self):
        row=online_row()
        row["configuration"]="WF"
        row["generation"]["native_token_ids"]=[10]
        row["generation"]["decoder_failure"]={"reason":"all-masked"}
        row["review_operations"]["decision_trace"][1]["chosen_token_id"]=None
        result=online_measurements(row,2)
        self.assertEqual(result["decisions"]["failed"],1)
        self.assertEqual(result["decisions"]["chosen"],1)

    def test_rollback_retained_prefix_and_fallback_are_exact(self):
        row={"offline":{"kept_tokens":1,"dropped_tokens":1,"completion_tokens":1,
                        "fresh_calls":1,"fallback":False},
             "generation":{"native_token_ids":[10,30],"ended_with_eos":True},
             "initial_certificate_token_ids":[40,30]}
        result=rollback_measurements(row,[10,20,30],2)
        self.assertEqual(result["kept_fraction"],.5)
        changed=copy.deepcopy(row)
        changed["generation"]["native_token_ids"][0]=40
        with self.assertRaises(ValueError):
            rollback_measurements(changed,[10,20,30],2)
        row["offline"].update(kept_tokens=0,dropped_tokens=2,completion_tokens=2,fallback=True)
        row["generation"]["native_token_ids"]=[40,30]
        self.assertEqual(rollback_measurements(row,[10,20,30],2)["fallback"],1)
        self.assertEqual(common_prefix([10,20],[10,30]),1)


if __name__ == "__main__":
    unittest.main()
