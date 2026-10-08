from __future__ import annotations

import unittest
from pathlib import Path

from automata_drs import (
    ByT5ConstraintState,
    ByT5TokenConstraintController,
    ByT5Vocabulary,
)
from automata_drs.sbn import (
    SBNPolicy,
    check_sbn_state_wellformedness,
    validate_sbn,
    validate_sbn_lexemes,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class GoldCorpusAcceptanceTest(unittest.TestCase):
    relative_paths = [
        "data/en/train/gold.sbn",
        "data/en/dev/standard.sbn",
        "data/en/test/standard.sbn",
    ]

    def test_all_strict_gold_sbn_is_accepted(self):
        failures: list[str] = []
        checked = 0
        for relative_path in self.relative_paths:
            path = REPOSITORY_ROOT / relative_path
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    checked += 1
                    sbn = line.rstrip("\n").split("\t", 1)[1]
                    result = validate_sbn(sbn)
                    if not result.valid:
                        failures.append(
                            f"{relative_path}:{line_number}: {result.reason}: {sbn}"
                        )
                        if len(failures) == 20:
                            break
            if failures:
                break
        self.assertEqual([], failures)
        self.assertGreater(checked, 10_000)

    def test_all_strict_gold_is_accepted_by_byt5_adapter(self):
        vocabulary = ByT5Vocabulary()
        controller = ByT5TokenConstraintController(vocabulary)
        failures: list[str] = []
        checked = 0
        for relative_path in self.relative_paths:
            path = REPOSITORY_ROOT / relative_path
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    sbn = line.rstrip("\n").split("\t", 1)[1]
                    state = ByT5ConstraintState.initial()
                    for byte_offset, byte in enumerate(sbn.encode("utf-8")):
                        state = controller.transition(
                            state, vocabulary.token_for_byte(byte)
                        )  # type: ignore[assignment]
                        if state is None:
                            failures.append(
                                f"{relative_path}:{line_number}: byte {byte_offset}: {sbn}"
                            )
                            break
                    if state is not None and controller.transition(
                        state, vocabulary.eos_token_id
                    ) is None:
                        failures.append(
                            f"{relative_path}:{line_number}: EOS rejected: {sbn}"
                        )
                    checked += 1
                    if len(failures) == 20:
                        break
            if failures:
                break
        self.assertEqual([], failures)
        self.assertGreater(checked, 10_000)

    def test_referee_aligned_semantic_audit_has_one_documented_exception(self):
        failures: list[tuple[str, int, str | None]] = []
        checked = 0
        online_policy = SBNPolicy.semantic_referee()
        for relative_path in self.relative_paths:
            path = REPOSITORY_ROOT / relative_path
            with path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    sbn = line.rstrip("\n").split("\t", 1)[1]
                    parsed = validate_sbn_lexemes(sbn)
                    self.assertTrue(parsed.valid, parsed.reason)
                    result = check_sbn_state_wellformedness(parsed.state)
                    online = validate_sbn_lexemes(sbn, online_policy)
                    self.assertEqual(result.valid, online.valid)
                    if not result.valid:
                        failures.append((relative_path, line_number, result.family))
                    checked += 1
        self.assertEqual(
            [("data/en/train/gold.sbn", 8021, "subordination-loop")],
            failures,
        )
        self.assertEqual(11_321, checked)


if __name__ == "__main__":
    unittest.main()
