from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_amr_gold_token_lengths.py"


class AMRGoldTokenLengthsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("amr_gold_lengths", SCRIPT)
        assert spec is not None and spec.loader is not None
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def test_nearest_rank_percentiles(self):
        values = [1, 2, 3, 4, 5]
        self.assertEqual(3, self.module.percentile(values, 0.50))
        self.assertEqual(5, self.module.percentile(values, 0.99))

    def test_horizon_summary(self):
        summary = self.module.horizon_summary([100, 200, 300, 400], [192, 288])
        self.assertEqual({"fit": 1, "over": 3, "over_rate": 0.75}, summary["192"])
        self.assertEqual({"fit": 2, "over": 2, "over_rate": 0.5}, summary["288"])


if __name__ == "__main__":
    unittest.main()
