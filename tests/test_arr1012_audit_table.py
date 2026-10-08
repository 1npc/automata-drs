import copy
import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "paper/arr1012-drs-placement/scripts/verify_independent_audit_table.py"
SPEC = importlib.util.spec_from_file_location("verify_independent_audit_table", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def report():
    counts = {}
    for arm in MODULE.ROW_LABELS.values():
        n = 2264 if arm == "gold" else 1132
        counts[arm] = {"total": n, "automaton_accept": n,
                       "structural_accept": n, "official_signature_accept": n - 1}
    counts["WWCD"].update(structural_accept=1131, false_accept_structural=1)
    return {"counts": counts, "false_accept_structural": 1}


def table(rows):
    lines = [r"\label{tab:independent-audit}"]
    for label, values in rows.items():
        lines.append(label + " & " + " & ".join(f"{v:,}" for v in values) + r" \\")
    return "\n".join(lines) + r"\end{table}"


class AuditTableTest(unittest.TestCase):
    def test_completed_failed_agreement_is_not_changed(self):
        data = report()
        rows = MODULE.audit_rows(data)
        MODULE.verify_tex_table(table(rows), rows)
        self.assertEqual(data["false_accept_structural"], 1)
        self.assertEqual(rows["APD"][2], 1131)

    def test_wrong_number_missing_or_duplicate_arm_is_rejected(self):
        rows = MODULE.audit_rows(report())
        original = table(rows)
        for changed in (original.replace("1,131", "1,132"),
                        original.replace("APD &", "Other &"),
                        original.replace(r"\end{table}", "APD & 1,132 & 1,132 & 1,131 & 1,131 \\\n" + r"\end{table}")):
            with self.assertRaises(ValueError):
                MODULE.verify_tex_table(changed, rows)

    def test_partial_cohort_or_false_accept_waiver_is_rejected(self):
        for field, value in (("total", 1131), ("structural_accept", True)):
            data = copy.deepcopy(report())
            data["counts"]["WWCD"][field] = value
            with self.assertRaises(ValueError):
                MODULE.audit_rows(data)
        data = report()
        data["false_accept_structural"] = 0
        with self.assertRaises(ValueError):
            MODULE.audit_rows(data)


if __name__ == "__main__":
    unittest.main()
