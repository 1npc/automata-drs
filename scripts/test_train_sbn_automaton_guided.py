#!/usr/bin/env python3
"""Local tests for scripts/train_sbn_automaton_guided.py.

Run with ``.venv/bin/python scripts/test_train_sbn_automaton_guided.py`` (plain
unittest) or with pytest if installed.  The real-tokenizer test looks for a
Qwen3.5 tokenizer in ``$AUTOMATA_TEST_TOKENIZER`` (falls back to the path used
while developing) and is skipped if it is absent.
"""

from __future__ import annotations

import importlib.util
import os
import random
import sys
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

_spec = importlib.util.spec_from_file_location(
    "train_sbn_automaton_guided", REPOSITORY_ROOT / "scripts" / "train_sbn_automaton_guided.py"
)
guided = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(guided)

DEFAULT_TOKENIZER = "Qwen/Qwen3.5-0.8B-Base"
TOKENIZER_PATH = os.environ.get("AUTOMATA_TEST_TOKENIZER", DEFAULT_TOKENIZER)
GOLD_PATH = REPOSITORY_ROOT / "data" / "en" / "train" / "gold.sbn"


def torch_or_skip():
    try:
        import torch
    except ImportError as error:  # pragma: no cover
        raise unittest.SkipTest("torch unavailable") from error
    return torch


def random_problem(torch, n=37, vocab=211, k=8, seed=0, dead_prob=0.4):
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn(n, vocab, generator=generator) * 3.0
    gold = torch.randint(0, vocab, (n,), generator=generator)
    topk = torch.topk(rows, k, dim=-1).indices
    dead = torch.rand(n, k, generator=generator) < dead_prob
    return rows, gold, topk, dead


class LossHelperTest(unittest.TestCase):
    def test_none_equals_cross_entropy_exactly(self):
        torch = torch_or_skip()
        import torch.nn.functional as F

        rows, gold, topk, dead = random_problem(torch)
        result = guided.automaton_guided_loss(rows, gold, topk, dead, mode="none")
        expected = F.cross_entropy(rows.float(), gold, reduction="sum")
        self.assertTrue(torch.equal(result["loss_sum"], expected))
        # bf16 logits: same upcast as HF ForCausalLMLoss
        rows_bf16 = rows.to(torch.bfloat16)
        result = guided.automaton_guided_loss(rows_bf16, gold, topk, dead, mode="none")
        self.assertTrue(torch.equal(result["loss_sum"], F.cross_entropy(rows_bf16.float(), gold, reduction="sum")))

    def test_none_matches_hf_causal_lm_loss_normalisation(self):
        torch = torch_or_skip()
        try:
            from transformers.loss.loss_utils import ForCausalLMLoss
        except Exception as error:  # pragma: no cover
            raise unittest.SkipTest(f"transformers loss utils unavailable: {error}")
        generator = torch.Generator().manual_seed(3)
        B, T, V = 3, 9, 50
        logits = torch.randn(B, T, V, generator=generator)
        labels = torch.full((B, T), -100)
        labels[0, 3:8] = torch.randint(0, V, (5,), generator=generator)
        labels[1, 2:5] = torch.randint(0, V, (3,), generator=generator)
        labels[2, 5:9] = torch.randint(0, V, (4,), generator=generator)
        selection = guided.target_rows(logits, labels)
        n = selection["gold"].numel()
        topk = torch.topk(selection["rows"], 4, dim=-1).indices
        result = guided.automaton_guided_loss(
            selection["rows"], selection["gold"], topk, torch.zeros_like(topk, dtype=torch.bool), mode="none"
        )
        for num_items in (None, torch.tensor(n + 7)):
            hf = ForCausalLMLoss(logits, labels, V, num_items_in_batch=num_items)
            ours = result["loss_sum"] / guided.normaliser(num_items, n, use_global=True)
            self.assertTrue(torch.allclose(ours, hf, rtol=0, atol=1e-6), (float(ours), float(hf)))

    def test_mask_never_masks_gold_and_equals_ce_without_dead(self):
        torch = torch_or_skip()
        import torch.nn.functional as F

        rows, gold, topk, dead = random_problem(torch, seed=1)
        ce = F.cross_entropy(rows, gold, reduction="sum")
        none_dead = torch.zeros_like(dead)
        result = guided.automaton_guided_loss(rows, gold, topk, none_dead, mode="mask")
        self.assertTrue(torch.equal(result["loss_sum"], ce))
        # Force every top-k candidate dead, and put the gold token in the top-k.
        topk_with_gold = topk.clone()
        topk_with_gold[:, 0] = gold
        # keep indices unique per row
        for i in range(topk_with_gold.shape[0]):
            others = [int(t) for t in topk[i] if int(t) != int(gold[i])][: topk.shape[1] - 1]
            topk_with_gold[i, 1:] = torch.tensor(others)
        all_dead = torch.ones_like(dead)
        result = guided.automaton_guided_loss(rows, gold, topk_with_gold, all_dead, mode="mask")
        self.assertEqual(result["gold_in_dead"], rows.shape[0])
        self.assertTrue(torch.isfinite(result["loss_sum"]))
        # reference: -inf on the non-gold top-k only
        reference = rows.clone()
        reference.scatter_(1, topk_with_gold[:, 1:], float("-inf"))
        self.assertTrue(torch.allclose(result["loss_sum"], F.cross_entropy(reference, gold, reduction="sum")))
        # mask loss = CE + log(1 - P_dead) <= CE
        partial = guided.automaton_guided_loss(rows, gold, topk, dead, mode="mask")
        p = torch.softmax(rows, -1).gather(1, topk)
        effective = dead & (topk != gold[:, None])
        closed_form = ce + torch.log1p(-(p * effective).sum(-1)).sum()
        self.assertTrue(torch.allclose(partial["loss_sum"], closed_form, rtol=1e-5, atol=1e-4))
        self.assertLessEqual(float(partial["loss_sum"]), float(ce) + 1e-5)

    def test_mask_gradient_matches_out_of_place_reference(self):
        torch = torch_or_skip()
        import torch.nn.functional as F

        base, gold, topk, dead = random_problem(torch, seed=4)
        for dtype in (torch.bfloat16, torch.float32):
            leaf = base.clone().to(dtype).requires_grad_(True)
            rows = leaf * 1  # non-leaf, like logits[:, :-1][mask]
            result = guided.automaton_guided_loss(rows, gold, topk, dead, mode="mask")
            result["loss_sum"].backward()
            effective = dead & (topk != gold[:, None])
            ref_leaf = base.clone().to(dtype).requires_grad_(True)
            ref_rows = ref_leaf.float()
            penalty = torch.zeros_like(ref_rows).scatter(
                1, topk, torch.where(effective, float("-inf"), 0.0)
            )
            F.cross_entropy(ref_rows + penalty, gold, reduction="sum").backward()
            self.assertTrue(torch.isfinite(leaf.grad.float()).all())
            self.assertTrue(torch.allclose(leaf.grad.float(), ref_leaf.grad.float(), atol=1e-6))

    def test_ul_bounded_and_equals_ce_without_dead(self):
        torch = torch_or_skip()
        import torch.nn.functional as F

        rows, gold, topk, dead = random_problem(torch, seed=2)
        rows.requires_grad_(True)
        ce = F.cross_entropy(rows, gold, reduction="sum")
        result = guided.automaton_guided_loss(rows, gold, topk, torch.zeros_like(dead), mode="ul", ul_weight=1.0)
        self.assertEqual(float(result["ul_sum"]), 0.0)
        self.assertTrue(torch.allclose(result["loss_sum"], ce, rtol=1e-6, atol=1e-4))
        result = guided.automaton_guided_loss(rows, gold, topk, dead, mode="ul", ul_weight=1.0)
        self.assertTrue(torch.isfinite(result["loss_sum"]))
        self.assertGreaterEqual(float(result["loss_sum"]), float(ce) - 1e-4)
        self.assertGreater(float(result["ul_sum"]), 0.0)
        result["loss_sum"].backward()
        self.assertTrue(torch.isfinite(rows.grad).all())
        # near-certain dead mass: the clamp keeps the term bounded by -log(1e-6)
        extreme = rows.detach().clone()
        extreme.scatter_(1, topk[:, :1], 80.0)
        gold2 = (topk[:, 0] + 1) % rows.shape[1]
        gold2 = torch.where(gold2 == topk[:, 0], gold2 + 1, gold2)
        all_dead = torch.zeros_like(dead)
        all_dead[:, 0] = True
        result = guided.automaton_guided_loss(extreme, gold2, topk, all_dead, mode="ul")
        self.assertTrue(torch.isfinite(result["loss_sum"]))
        per_row = result["ul_sum"] / rows.shape[0]
        self.assertLessEqual(float(per_row), -torch.log(torch.tensor(1e-6)).item() + 1e-3)


class AlignmentTest(unittest.TestCase):
    def test_shift_alignment_hand_made(self):
        torch = torch_or_skip()
        B, T, V = 2, 6, 10
        labels = torch.tensor(
            [[-100, -100, 3, 5, 7, -100], [-100, 4, 2, -100, -100, -100]]
        )
        logits = torch.zeros(B, T, V)
        for b in range(B):
            for j in range(1, T):
                if labels[b, j] != -100:
                    logits[b, j - 1, labels[b, j]] = 10.0 + j
        selection = guided.target_rows(logits, labels)
        self.assertEqual(selection["gold"].tolist(), [3, 5, 7, 4, 2])
        self.assertEqual(selection["rows"].argmax(-1).tolist(), [3, 5, 7, 4, 2])
        self.assertEqual(selection["targets"], [[3, 5, 7], [4, 2]])
        self.assertEqual(selection["positions"], [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)])
        # rows come from column j-1: row for label at (0, 2) is logits[0, 1]
        self.assertTrue(torch.equal(selection["rows"][0], logits[0, 1]))
        self.assertTrue(torch.equal(selection["rows"][3], logits[1, 0]))

        class PrefixOracle:
            """Records which gold prefix each candidate is checked against."""

            def __init__(self):
                self.calls = []

            def gold_state_ids(self, target):
                return tuple(("prefix", tuple(target[:i])) for i in range(len(target))), None

            def dead_rows(self, state_ids, candidates, gold):
                self.calls.append((list(state_ids), list(gold)))
                return [[False] * len(row) for row in candidates]

        oracle = PrefixOracle()
        guided.dead_candidates(oracle, selection, topk=3)
        state_ids, gold = oracle.calls[0]
        self.assertEqual(gold, [3, 5, 7, 4, 2])
        self.assertEqual(
            [s[1] for s in state_ids], [(), (3,), (3, 5), (), (4,)]
        )


class RealTokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not Path(TOKENIZER_PATH).is_dir():
            raise unittest.SkipTest(f"tokenizer not found at {TOKENIZER_PATH}")
        from transformers import AutoTokenizer

        cls.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        from automata_drs.causal_lm import encode_supervised_record, iter_sbn_records

        cls.examples = [
            encode_supervised_record(cls.tokenizer, record, max_length=1024)
            for record in iter_sbn_records(GOLD_PATH, limit=20)
        ]
        cls.oracle = guided.GoldPrefixAutomaton(cls.tokenizer, policy="W")

    def test_gold_tokens_never_dead(self):
        oracle = self.oracle
        self.assertEqual(self.tokenizer.eos_token_id, 248044)
        rng = random.Random(0)
        vocab = len(self.tokenizer)
        total_dead = 0
        positions = 0
        for example in self.examples:
            target = example["input_ids"][example["prompt_tokens"]:]
            self.assertEqual(target[-1], 248044)
            state_ids, rejected = oracle.gold_state_ids(target)
            self.assertIsNone(rejected, "gold prefix rejected by the W automaton")
            self.assertTrue(all(sid is not None for sid in state_ids))
            # each gold token is legal from its own gold-prefix state
            for sid, token in zip(state_ids, target):
                self.assertFalse(oracle.is_dead(sid, token))
            # candidate rows: gold + neighbouring gold tokens + random ASCII-ish ids
            candidates = []
            for i, token in enumerate(target):
                row = [token, target[(i + 1) % len(target)], target[i - 1]]
                row += [rng.randrange(vocab) for _ in range(5)]
                candidates.append(row)
            flags = oracle.dead_rows(state_ids, candidates, target)
            for row_flags in flags:
                self.assertFalse(row_flags[0])
                total_dead += sum(row_flags)
            positions += len(target)
        self.assertEqual(oracle.gold_dead_hits, 0)
        self.assertEqual(oracle.gold_rejections, 0)
        self.assertGreater(total_dead, positions)  # automaton actually rejects things
        print(
            f"\n[real-tokenizer] examples={len(self.examples)} target_positions={positions} "
            f"dead_candidates={total_dead} oracle={oracle.info()}"
        )

    def test_end_to_end_batch_all_modes(self):
        torch = torch_or_skip()
        from automata_drs.causal_lm import pad_supervised_batch

        items = []
        for example in self.examples[:4]:
            items.append({
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
                "labels": example["labels"],
            })
        padded = pad_supervised_batch(items, pad_token_id=248044)
        labels = torch.tensor(padded["labels"])
        B, T = labels.shape
        V = 248320  # model head is wider than the tokenizer; extra ids must be dead
        generator = torch.Generator().manual_seed(0)
        logits = torch.randn(B, T, V, generator=generator).to(torch.bfloat16)
        # make the gold token likely so it appears in the top-k
        for b in range(B):
            for j in range(1, T):
                if labels[b, j] != -100:
                    logits[b, j - 1, labels[b, j]] += 6.0
        selection = guided.target_rows(logits, labels)
        topk_ids, dead, seconds = guided.dead_candidates(self.oracle, selection, topk=32)
        gold = selection["gold"]
        in_topk = (topk_ids == gold[:, None])
        self.assertFalse((dead & in_topk).any().item())
        self.assertGreater(dead.float().mean().item(), 0.5)
        losses = {}
        for mode in guided.MODES:
            result = guided.automaton_guided_loss(selection["rows"], gold, topk_ids, dead, mode=mode)
            self.assertTrue(torch.isfinite(result["loss_sum"]))
            losses[mode] = float(result["loss_sum"]) / result["tokens"]
        self.assertLessEqual(losses["mask"], losses["none"] + 1e-4)
        self.assertGreaterEqual(losses["ul"], losses["none"] - 1e-4)
        print(
            f"\n[end-to-end] rows={gold.numel()} dead_frac={dead.float().mean():.3f} "
            f"automaton_s={seconds:.3f} per-token losses={losses}"
        )


class CliAndPatchTest(unittest.TestCase):
    def test_split_arguments(self):
        ours, rest = guided.split_arguments([
            "--task", "sbn", "--automaton-mode", "ul", "--checkpoint-steps", "8,16", "32",
            "--output", "x", "--ul-weight", "0.5",
        ])
        self.assertEqual(ours.automaton_mode, "ul")
        self.assertEqual(ours.checkpoint_steps, [8, 16, 32])
        self.assertEqual(ours.ul_weight, 0.5)
        self.assertEqual(rest, ["--task", "sbn", "--output", "x"])
        self.assertEqual(str(guided.output_argument(rest)), "x")

    # These run in a fresh interpreter: the bug they guard against (transformers 5.x
    # replacing sys.modules["transformers"] on the first ``Trainer`` access via
    # processing_utils -> direct_transformers_import) only reproduces if nothing
    # in the process has imported processing_utils yet.

    def _run_fresh(self, code, *args):
        import subprocess

        try:
            import transformers  # noqa: F401
        except Exception as error:  # pragma: no cover
            raise unittest.SkipTest(f"transformers unavailable locally: {error}")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPOSITORY_ROOT / "src"), str(REPOSITORY_ROOT / "scripts"), env.get("PYTHONPATH", "")]
        )
        return subprocess.run(
            [sys.executable, "-c", code, *args], capture_output=True, text=True, env=env, timeout=600
        )

    def test_patch_is_seen_by_function_local_import_real_transformers(self):
        code = (
            "import sys\n"
            "import train_sbn_automaton_guided as g\n"
            "import transformers\n"
            "bound = transformers\n"
            "config, _ = g.split_arguments([])\n"
            "sub = g.install_trainer_patch(transformers, config, None)\n"
            "def f():\n"
            "    import transformers\n"
            "    from transformers import Trainer\n"
            "    return Trainer\n"
            "seen = f()\n"
            "print('SWAPPED', sys.modules['transformers'] is not bound)\n"
            "print('SEEN', seen.__module__, seen.__qualname__)\n"
            "g.verify_trainer_patch(sub)\n"
            "assert seen is sub, (seen, sub)\n"
            "import transformers.trainer as tt\n"
            "assert issubclass(sub, tt.Trainer)\n"
            "print('PATCH_OK')\n"
        )
        proc = self._run_fresh(code)
        print(f"\n[patch subprocess] {proc.stdout.strip()!r}")
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("PATCH_OK", proc.stdout)
        self.assertIn("AutomatonGuidedTrainer", proc.stdout)

    def test_main_patch_visible_to_upstream_and_fails_if_trainer_never_built(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / "fake_upstream.py"
            stub.write_text(
                "def main():\n"
                "    import transformers\n"
                "    from transformers import Trainer\n"
                "    print('UPSTREAM SEES:', Trainer.__module__, Trainer.__qualname__, flush=True)\n"
                "    return 0\n",
                encoding="utf-8",
            )
            code = (
                "import sys, train_sbn_automaton_guided as g\n"
                "sys.exit(g.main(sys.argv[1:]))\n"
            )
            proc = self._run_fresh(
                code, "--upstream-trainer", str(stub), "--automaton-mode", "ul", "--output", str(Path(tmp) / "out")
            )
        print(f"\n[main subprocess] rc={proc.returncode} stdout={proc.stdout.strip()!r}")
        self.assertIn("UPSTREAM SEES:", proc.stdout, proc.stderr)
        self.assertIn("AutomatonGuidedTrainer", proc.stdout)
        # the stub never constructs a Trainer -> main must report failure, not success
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)
        self.assertIn("never instantiated", proc.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
