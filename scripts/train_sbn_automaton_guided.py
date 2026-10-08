#!/usr/bin/env python3
"""LoRA fine-tuning of a causal-LM SBN parser with executable-automaton guidance.

This is a thin wrapper around ``scripts/train_causal_lm_lora.py``.  The upstream
``main()`` is imported by file path and run unchanged, except that
``transformers.Trainer`` is replaced by :class:`AutomatonGuidedTrainer` before
``main()`` performs its function-local ``from transformers import Trainer``.

Training objectives (``--automaton-mode``):

``none``
    The upstream Hugging Face causal-LM loss, obtained by delegating to
    ``Trainer.compute_loss``.  This is the baseline and is bit-for-bit the
    upstream objective.  Automaton diagnostics (dead top-k mass) are computed
    under ``no_grad`` on a sample of micro-batches and timed separately.

``mask``
    Approximate masked maximum likelihood (GraFT-style).  For every supervised
    position the gold prefix is replayed through the full W SBN automaton; of the
    model's top-k next-token candidates, those whose transition from the gold
    prefix state is illegal ("dead") get logit ``-inf`` before the softmax.
    Only illegal tokens *inside the top-k* are masked; illegal tokens outside
    the top-k keep their mass, so this approximates the exact masked MLE whose
    normaliser ranges over the legal vocabulary only.

``ul``
    Bounded aggregate unlikelihood:
    ``CE + ul_weight * mean_t -log(clamp(1 - sum_{c dead in top-k} p_t(c), 1e-6))``
    with ``p_t`` the float32 softmax at target position ``t``.

CE normalisation follows Hugging Face: the sum over supervised tokens divided by
``num_items_in_batch`` (the token count of the whole gradient-accumulation
batch) when the Trainer supplies it and the model accepts loss kwargs,
otherwise the mean over the micro-batch's supervised tokens.  The UL term uses
the same denominator, so it is a per-target-position mean.

Extra outputs under ``--output``: ``checkpoints/step-<n>/`` adapter snapshots
(each with ``snapshot.json`` holding wall-clock seconds and training loss so
far), ``automaton_stats.jsonl`` (one line per logging step) and
``automaton_manifest.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

SCRIPT_PATH = Path(__file__).resolve()
UPSTREAM_CANDIDATES = (
    Path("/workspace/automata-drs/scripts/train_causal_lm_lora.py"),
    SCRIPT_PATH.parent / "train_causal_lm_lora.py",
)
MODES = ("none", "mask", "ul")
UL_FLOOR = 1e-6
IGNORE_INDEX = -100


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_step_list(values: Sequence[str] | None) -> list[int]:
    steps: set[int] = set()
    for value in values or ():
        for piece in str(value).split(","):
            piece = piece.strip()
            if not piece:
                continue
            step = int(piece)
            if step <= 0:
                raise ValueError("checkpoint steps must be positive")
            steps.add(step)
    return sorted(steps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--automaton-mode", choices=MODES, default="none")
    parser.add_argument("--automaton-topk", type=int, default=32)
    parser.add_argument("--ul-weight", type=float, default=1.0)
    parser.add_argument("--automaton-policy", default="W")
    parser.add_argument(
        "--checkpoint-steps",
        nargs="+",
        default=None,
        help="optimizer steps at which to snapshot the adapter (space/comma separated)",
    )
    parser.add_argument(
        "--baseline-diagnostics-every",
        type=int,
        default=10,
        help="mode none only: compute dead-mass diagnostics on every n-th training "
        "micro-batch (0 disables); never changes the loss",
    )
    parser.add_argument(
        "--automaton-cache-entries",
        type=int,
        default=5_000_000,
        help="bound on the (gold-prefix state, token) -> dead memo; cleared when full",
    )
    parser.add_argument(
        "--upstream-trainer",
        type=Path,
        default=None,
        help="path to train_causal_lm_lora.py (default: /workspace/automata-drs "
        "copy if present, else the sibling script)",
    )
    return parser


def split_arguments(argv: Sequence[str]) -> tuple[argparse.Namespace, list[str]]:
    ours, rest = build_parser().parse_known_args(list(argv))
    if ours.automaton_topk <= 0:
        raise SystemExit("--automaton-topk must be positive")
    if ours.ul_weight < 0:
        raise SystemExit("--ul-weight must be non-negative")
    if ours.baseline_diagnostics_every < 0:
        raise SystemExit("--baseline-diagnostics-every must be >= 0")
    ours.checkpoint_steps = parse_step_list(ours.checkpoint_steps)
    return ours, rest


def upstream_path(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    env = os.environ.get("AUTOMATA_UPSTREAM_TRAINER")
    if env:
        return Path(env).resolve()
    for candidate in UPSTREAM_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise SystemExit("cannot locate train_causal_lm_lora.py; pass --upstream-trainer")


def load_upstream(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("train_causal_lm_lora_upstream", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import upstream trainer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # inserts <repo>/src into sys.path
    return module


def output_argument(argv: Sequence[str]) -> Path | None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", type=Path)
    known, _ = parser.parse_known_args(list(argv))
    return known.output


# --------------------------------------------------------------------------
# Gold-prefix automaton oracle
# --------------------------------------------------------------------------


class GoldPrefixAutomaton:
    """Replay gold SBN targets through the subword W controller.

    ``gold_state_ids(target)`` returns, for every target token, an interned id of
    the automaton state *before* that token (``None`` once the gold prefix has
    been rejected, which should never happen for Referee-valid gold).
    ``dead_rows`` answers whether candidate tokens are illegal from those
    states, memoised on ``(state id, token id)``.
    """

    def __init__(self, tokenizer: Any, *, policy: str = "W", cache_entries: int = 5_000_000) -> None:
        from automata_drs import SBNPolicy
        from automata_drs.byt5 import ByT5ConstraintState
        from automata_drs.subword import (
            ByteLevelVocabulary,
            SubwordTokenConstraintController,
        )

        self.vocabulary = ByteLevelVocabulary.from_tokenizer(tokenizer)
        # Our own int-keyed memo replaces the controller's (state, token) LRU.
        self.controller = SubwordTokenConstraintController(self.vocabulary, max_cache_entries=0)
        self.initial = ByT5ConstraintState.initial(SBNPolicy.for_configuration(policy))
        self.policy = policy
        self.cache_entries = int(cache_entries)
        self._state_ids: dict[Any, int] = {}
        self._states: list[Any] = []
        self._prefix: dict[tuple[int, ...], tuple[tuple[int | None, ...], int | None]] = {}
        self._dead: dict[int, bool] = {}
        self.gold_rejections = 0  # distinct targets whose gold prefix was rejected
        self.gold_dead_hits = 0  # gold token listed among dead candidates (must stay 0)
        self.dead_cache_clears = 0
        self.transitions_computed = 0

    def _intern(self, state: Any) -> int:
        sid = self._state_ids.get(state)
        if sid is None:
            sid = len(self._states)
            self._state_ids[state] = sid
            self._states.append(state)
        return sid

    def gold_state_ids(self, target: Sequence[int]) -> tuple[tuple[int | None, ...], int | None]:
        """Return (state id before each token, index of first rejected gold token)."""
        key = tuple(int(t) for t in target)
        cached = self._prefix.get(key)
        if cached is not None:
            return cached
        ids: list[int | None] = []
        state = self.initial
        rejected_at: int | None = None
        for index, token in enumerate(key):
            if state is None:
                ids.append(None)
                continue
            ids.append(self._intern(state))
            successor = self.controller.transition(state, token)
            if successor is None and rejected_at is None:
                rejected_at = index
                self.gold_rejections += 1
                print(
                    f"[automaton] WARNING gold token rejected at target index {index} "
                    f"(token {token}); later positions are left unconstrained",
                    file=sys.stderr,
                    flush=True,
                )
            state = successor
        result = (tuple(ids), rejected_at)
        self._prefix[key] = result
        return result

    def is_dead(self, state_id: int, token: int) -> bool:
        key = (state_id << 20) | int(token)
        hit = self._dead.get(key)
        if hit is not None:
            return hit
        if self.cache_entries and len(self._dead) >= self.cache_entries:
            self._dead.clear()
            self.dead_cache_clears += 1
        dead = self.controller.transition(self._states[state_id], int(token)) is None
        self.transitions_computed += 1
        self._dead[key] = dead
        return dead

    def dead_rows(
        self,
        state_ids: Sequence[int | None],
        candidates: Sequence[Sequence[int]],
        gold: Sequence[int],
    ) -> list[list[bool]]:
        rows: list[list[bool]] = []
        for sid, row, gold_token in zip(state_ids, candidates, gold):
            if sid is None:
                rows.append([False] * len(row))
                continue
            flags = []
            for token in row:
                dead = self.is_dead(sid, token)
                if dead and int(token) == int(gold_token):
                    self.gold_dead_hits += 1
                    dead = False
                flags.append(dead)
            rows.append(flags)
        return rows

    def info(self) -> dict[str, int]:
        return {
            "interned_states": len(self._states),
            "cached_targets": len(self._prefix),
            "dead_memo_entries": len(self._dead),
            "dead_memo_clears": self.dead_cache_clears,
            "transitions_computed": self.transitions_computed,
            "gold_rejections": self.gold_rejections,
            "gold_dead_hits": self.gold_dead_hits,
        }


_ORACLES: dict[tuple[Any, ...], GoldPrefixAutomaton] = {}


def get_oracle(tokenizer: Any, *, policy: str = "W", cache_entries: int = 5_000_000) -> GoldPrefixAutomaton:
    key = (getattr(tokenizer, "name_or_path", None), len(tokenizer), policy)
    oracle = _ORACLES.get(key)
    if oracle is None:
        oracle = GoldPrefixAutomaton(tokenizer, policy=policy, cache_entries=cache_entries)
        _ORACLES[key] = oracle
    return oracle


# --------------------------------------------------------------------------
# Loss helpers (pure torch; unit-tested without a model)
# --------------------------------------------------------------------------


def target_rows(logits: Any, labels: Any) -> dict[str, Any]:
    """Select the logits rows that predict supervised tokens.

    ``logits[b, j-1]`` predicts ``labels[b, j]``; rows are returned in
    row-major (b, j) order together with the per-example gold targets and, for
    each row, its index inside the example's full supervised target.
    """
    shift_labels = labels[:, 1:]
    mask = shift_labels != IGNORE_INDEX
    rows = logits[:, :-1][mask]
    gold = shift_labels[mask]
    labels_cpu = labels.detach().cpu().tolist()
    mask_cpu = mask.detach().cpu().tolist()
    targets: list[list[int]] = []
    positions: list[tuple[int, int]] = []  # (example, index in full target)
    for b, (label_row, mask_row) in enumerate(zip(labels_cpu, mask_cpu)):
        full = [int(t) for t in label_row if t != IGNORE_INDEX]
        targets.append(full)
        supervised = sum(1 for m in mask_row if m)
        offset = len(full) - supervised  # 1 only if the target started at column 0
        positions.extend((b, offset + i) for i in range(supervised))
    return {"rows": rows, "gold": gold, "targets": targets, "positions": positions}


def automaton_guided_loss(
    rows: Any,
    gold: Any,
    topk_ids: Any,
    dead: Any,
    *,
    mode: str,
    ul_weight: float = 1.0,
) -> dict[str, Any]:
    """Return summed losses over target rows plus detached diagnostics.

    rows: [N, V] logits of target rows; gold: [N]; topk_ids: [N, k];
    dead: [N, k] bool (True = illegal candidate).  The gold token is never
    treated as dead.  ``loss_sum`` is the objective before normalisation.
    """
    import torch
    import torch.nn.functional as F

    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    dead = dead.to(device=rows.device, dtype=torch.bool)
    topk_ids = topk_ids.to(device=rows.device, dtype=torch.long)
    gold = gold.to(device=rows.device, dtype=torch.long)
    gold_in_dead = dead & (topk_ids == gold[:, None])
    dead = dead & ~gold_in_dead

    rows32 = rows.float()
    with torch.no_grad():
        lse_plain = torch.logsumexp(rows32, dim=-1)
        top_logits = rows32.gather(1, topk_ids)
        dead_prob = torch.where(dead, torch.exp(top_logits - lse_plain[:, None]), torch.zeros_like(top_logits))
        dead_mass = dead_prob.sum(-1)
        ce_plain = lse_plain - rows32.gather(1, gold[:, None]).squeeze(1)

    ul_sum = rows32.new_zeros(())
    if mode == "none":
        loss_sum = F.cross_entropy(rows32, gold, reduction="sum")
    elif mode == "mask":
        # In-place write of -inf at dead (row, token) cells only.  rows32 is a
        # fresh float32 copy (its producer's backward does not need it), so
        # the in-place index_put_ is autograd-safe and avoids a second [N, V]
        # buffer; the -inf cells simply receive zero gradient.
        masked = rows32.clone() if rows32 is rows else rows32
        dead_rows_idx, dead_cols = dead.nonzero(as_tuple=True)
        if dead_rows_idx.numel():
            masked.index_put_(
                (dead_rows_idx, topk_ids[dead_rows_idx, dead_cols]),
                torch.tensor(float("-inf"), dtype=masked.dtype, device=masked.device),
            )
        loss_sum = F.cross_entropy(masked, gold, reduction="sum")
    else:
        lse = torch.logsumexp(rows32, dim=-1)
        ce = lse - rows32.gather(1, gold[:, None]).squeeze(1)
        probs = torch.exp(rows32.gather(1, topk_ids) - lse[:, None])
        p_dead = (probs * dead.to(probs.dtype)).sum(-1)
        ul = -torch.log(torch.clamp(1.0 - p_dead, min=UL_FLOOR))
        ul_sum = ul.sum()
        loss_sum = ce.sum() + float(ul_weight) * ul_sum
    return {
        "loss_sum": loss_sum,
        "ul_sum": ul_sum.detach(),
        "ce_plain_sum": ce_plain.sum(),
        "dead_mass": dead_mass,
        "dead_any": dead.any(-1),
        "gold_in_dead": int(gold_in_dead.sum().item()),
        "tokens": int(gold.numel()),
    }


def dead_candidates(
    oracle: GoldPrefixAutomaton,
    selection: dict[str, Any],
    topk: int,
) -> tuple[Any, Any, float]:
    """Top-k candidates (GPU) -> dead mask via the gold-prefix automaton (CPU)."""
    import torch

    rows = selection["rows"]
    k = min(int(topk), rows.shape[-1])
    with torch.no_grad():
        topk_ids = torch.topk(rows.detach(), k, dim=-1).indices
    # .cpu() synchronises with the forward pass; keep that wait out of the
    # automaton timer so automaton seconds measure CPU automaton work only.
    candidate_lists = topk_ids.cpu().tolist()
    gold_list = selection["gold"].detach().cpu().tolist()
    started = time.perf_counter()
    state_cache: dict[int, tuple[int | None, ...]] = {}
    state_ids: list[int | None] = []
    for b, index in selection["positions"]:
        if b not in state_cache:
            state_cache[b], _ = oracle.gold_state_ids(selection["targets"][b])
        state_ids.append(state_cache[b][index])
    flags = oracle.dead_rows(state_ids, candidate_lists, gold_list)
    dead = torch.tensor(flags, dtype=torch.bool).reshape(len(flags), k)
    return topk_ids, dead, time.perf_counter() - started


def normaliser(num_items_in_batch: Any, tokens: int, use_global: bool) -> Any:
    if use_global and num_items_in_batch is not None:
        return num_items_in_batch
    return max(int(tokens), 1)


# --------------------------------------------------------------------------
# Trainer subclass and snapshot callback
# --------------------------------------------------------------------------


class RunningStats:
    def __init__(self) -> None:
        self.reset_window()
        self.total_automaton_seconds = 0.0
        self.total_diag_seconds = 0.0
        self.objective_sum = 0.0  # sum of reported micro-batch losses
        self.ce_sum = 0.0  # same normalisation, plain CE component
        self.micro_batches = 0

    def reset_window(self) -> None:
        self.w_positions = 0
        self.w_dead_positions = 0
        self.w_dead_mass = 0.0
        self.w_ce = 0.0
        self.w_ul = 0.0
        self.w_seconds = 0.0
        self.w_batches = 0
        self.w_gold_in_dead = 0


def make_trainer_class(base: type, config: argparse.Namespace, output_dir: Path | None) -> type:
    from transformers import TrainerCallback

    class SnapshotCallback(TrainerCallback):
        def __init__(self, trainer: Any) -> None:
            self.trainer = trainer
            self.pending = set(config.checkpoint_steps)
            self.started: float | None = None
            self.save_seconds = 0.0
            self.saved: list[dict[str, Any]] = []

        def on_train_begin(self, args, state, control, **kwargs):  # noqa: D401
            self.started = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            step = int(state.global_step)
            if step not in self.pending or output_dir is None:
                return control
            self.pending.discard(step)
            trainer = self.trainer
            wall = time.time() - (self.started or time.time())
            stats: RunningStats = trainer.automaton_stats
            logged = [
                {"step": e.get("step"), "loss": e.get("loss")}
                for e in state.log_history
                if "loss" in e
            ]
            eval_seconds = sum(float(e.get("eval_runtime", 0.0)) for e in state.log_history)
            target = output_dir / "checkpoints" / f"step-{step}"
            target.mkdir(parents=True, exist_ok=True)
            save_start = time.time()
            trainer.model.save_pretrained(target, safe_serialization=True)
            if trainer.processing_class is not None:
                trainer.processing_class.save_pretrained(target)
            save_elapsed = time.time() - save_start
            record = {
                "status": "complete",
                "step": step,
                "epoch": state.epoch,
                "automaton_mode": config.automaton_mode,
                "wall_seconds_since_train_begin": wall,
                "prior_snapshot_save_seconds": self.save_seconds,
                "eval_seconds_so_far": eval_seconds,
                "train_wall_seconds_net": wall - self.save_seconds - eval_seconds,
                "automaton_seconds_cumulative": stats.total_automaton_seconds,
                "baseline_diagnostic_seconds_cumulative": stats.total_diag_seconds,
                "micro_batches": stats.micro_batches,
                "objective_loss_mean_per_step": stats.objective_sum / max(step, 1),
                "ce_loss_mean_per_step": stats.ce_sum / max(step, 1),
                "last_logged_loss": logged[-1]["loss"] if logged else None,
                "logged_losses": logged,
                "this_snapshot_save_seconds": save_elapsed,
                "oracle": trainer.automaton_oracle.info() if trainer.automaton_oracle else None,
            }
            self.save_seconds += save_elapsed
            (target / "snapshot.json").write_text(json.dumps(record, indent=2, default=str) + "\n", encoding="utf-8")
            self.saved.append({"step": step, "path": str(target), "wall_seconds": wall})
            print(f"[automaton] snapshot step={step} wall={wall:.1f}s -> {target}", flush=True)
            return control

    class AutomatonGuidedTrainer(base):
        automaton_config = config

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.automaton_stats = RunningStats()
            self.automaton_oracle: GoldPrefixAutomaton | None = None
            needs_oracle = config.automaton_mode != "none" or config.baseline_diagnostics_every > 0
            if needs_oracle:
                started = time.perf_counter()
                self.automaton_oracle = get_oracle(
                    self.processing_class,
                    policy=config.automaton_policy,
                    cache_entries=config.automaton_cache_entries,
                )
                self.automaton_stats.total_automaton_seconds += time.perf_counter() - started
            self.snapshot_callback = SnapshotCallback(self)
            self.add_callback(self.snapshot_callback)
            self._stats_path = None if output_dir is None else output_dir / "automaton_stats.jsonl"
            AutomatonGuidedTrainer.last_instance = self
            if output_dir is not None:
                manifest = {
                    "status": "running",
                    "automaton_mode": config.automaton_mode,
                    "automaton_topk": config.automaton_topk,
                    "ul_weight": config.ul_weight,
                    "ul_floor": UL_FLOOR,
                    "automaton_policy": config.automaton_policy,
                    "checkpoint_steps": config.checkpoint_steps,
                    "baseline_diagnostics_every": config.baseline_diagnostics_every,
                    "model_accepts_loss_kwargs": getattr(self, "model_accepts_loss_kwargs", None),
                    "script": str(SCRIPT_PATH),
                }
                (output_dir / "automaton_manifest.json").write_text(
                    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
                )

        def _use_global_normaliser(self) -> bool:
            return bool(getattr(self, "model_accepts_loss_kwargs", False))

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            import torch

            stats = self.automaton_stats
            if not model.training or "labels" not in inputs:
                return super().compute_loss(
                    model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
                )
            stats.micro_batches += 1
            mode = config.automaton_mode
            if mode == "none":
                every = config.baseline_diagnostics_every
                diagnose = every > 0 and (stats.micro_batches - 1) % every == 0
                loss, outputs = super().compute_loss(
                    model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
                )
                stats.objective_sum += float(loss.detach().float().item())
                stats.ce_sum += float(loss.detach().float().item())
                if diagnose:
                    started = time.perf_counter()
                    with torch.no_grad():
                        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
                        selection = target_rows(logits.detach(), inputs["labels"])
                        topk_ids, dead, _ = dead_candidates(self.automaton_oracle, selection, config.automaton_topk)
                        result = automaton_guided_loss(
                            selection["rows"], selection["gold"], topk_ids, dead, mode="none"
                        )
                    elapsed = time.perf_counter() - started
                    stats.total_diag_seconds += elapsed
                    self._accumulate(result, elapsed)
                return (loss, outputs) if return_outputs else loss

            labels = inputs["labels"]
            model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
            outputs = model(**model_inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
            selection = target_rows(logits, labels)
            topk_ids, dead, automaton_seconds = dead_candidates(
                self.automaton_oracle, selection, config.automaton_topk
            )
            result = automaton_guided_loss(
                selection["rows"],
                selection["gold"],
                topk_ids,
                dead,
                mode=mode,
                ul_weight=config.ul_weight,
            )
            use_global = self._use_global_normaliser()
            denominator = normaliser(num_items_in_batch, result["tokens"], use_global)
            loss = result["loss_sum"] / denominator
            if (
                use_global
                and num_items_in_batch is not None
                and getattr(self.args, "average_tokens_across_devices", False)
            ):
                loss = loss * self.accelerator.num_processes
            stats.total_automaton_seconds += automaton_seconds
            stats.objective_sum += float(loss.detach().float().item())
            ce_norm = result["ce_plain_sum"] / denominator
            stats.ce_sum += float(torch.as_tensor(ce_norm).float().item())
            self._accumulate(result, automaton_seconds)
            return (loss, outputs) if return_outputs else loss

        def _accumulate(self, result: dict[str, Any], seconds: float) -> None:
            stats = self.automaton_stats
            stats.w_positions += result["tokens"]
            stats.w_dead_positions += int(result["dead_any"].sum().item())
            stats.w_dead_mass += float(result["dead_mass"].sum().item())
            stats.w_ce += float(result["ce_plain_sum"].item())
            stats.w_ul += float(result["ul_sum"].item())
            stats.w_seconds += seconds
            stats.w_batches += 1
            stats.w_gold_in_dead += int(result["gold_in_dead"])

        def log(self, logs: dict[str, float], *args: Any, **kwargs: Any) -> None:
            stats = getattr(self, "automaton_stats", None)
            if stats is not None and "loss" in logs:
                positions = max(stats.w_positions, 1)
                extra = {
                    "automaton_mode": config.automaton_mode,
                    "auto_measured_batches": stats.w_batches,
                    "auto_positions": stats.w_positions,
                    "auto_dead_position_frac": stats.w_dead_positions / positions,
                    "auto_dead_mass_mean": stats.w_dead_mass / positions,
                    "auto_ce_token_mean": stats.w_ce / positions,
                    "auto_ul_token_mean": stats.w_ul / positions,
                    "auto_seconds": stats.w_seconds,
                    "auto_seconds_cumulative": stats.total_automaton_seconds,
                    "auto_diag_seconds_cumulative": stats.total_diag_seconds,
                    "auto_gold_in_dead": stats.w_gold_in_dead,
                }
                if self.automaton_oracle is not None:
                    extra.update({f"oracle_{k}": v for k, v in self.automaton_oracle.info().items()})
                if self._stats_path is not None:
                    line = {
                        "step": int(self.state.global_step),
                        "epoch": self.state.epoch,
                        "loss": logs.get("loss"),
                        "time": time.time(),
                        **extra,
                    }
                    with self._stats_path.open("a", encoding="utf-8") as sink:
                        sink.write(json.dumps(line, default=str) + "\n")
                logs = {**logs, **{k: v for k, v in extra.items() if isinstance(v, (int, float))}}
                stats.reset_window()
            return super().log(logs, *args, **kwargs)

    AutomatonGuidedTrainer.last_instance = None
    return AutomatonGuidedTrainer


def install_trainer_patch(transformers_module: Any, config: argparse.Namespace, output_dir: Path | None) -> type:
    """Replace ``transformers.Trainer`` so upstream ``from transformers import Trainer`` sees the subclass.

    In transformers 5.x the first access to ``transformers.Trainer`` imports
    ``processing_utils``, which calls ``direct_transformers_import()``; that
    re-executes ``transformers/__init__.py`` and installs a *new* ``_LazyModule``
    in ``sys.modules["transformers"]``.  The module object we were handed is
    then stale, so the subclass must be set on the live ``sys.modules`` entry
    (and, harmlessly, on the stale one too).  Callers should verify the patch
    with :func:`verify_trainer_patch`.
    """
    base = transformers_module.Trainer  # may swap sys.modules["transformers"]
    subclass = make_trainer_class(base, config, output_dir)
    live = sys.modules.get("transformers")
    targets = {id(module): module for module in (transformers_module, live) if module is not None}
    for module in targets.values():
        setattr(module, "Trainer", subclass)
    return subclass


def verify_trainer_patch(trainer_class: type) -> None:
    """Fail loudly unless a fresh function-local ``from transformers import Trainer`` sees the subclass."""
    from transformers import Trainer as visible  # noqa: N813

    if visible is not trainer_class:
        raise RuntimeError(
            "Trainer patch not visible to upstream: 'from transformers import Trainer' returned "
            f"{visible.__module__}.{visible.__qualname__}, expected {trainer_class.__qualname__}"
        )


def finalize_manifest(output_dir: Path, trainer_class: type, status: str) -> None:
    path = output_dir / "automaton_manifest.json"
    if not path.is_file():
        return
    manifest = json.loads(path.read_text(encoding="utf-8"))
    trainer = getattr(trainer_class, "last_instance", None)
    manifest["status"] = status
    if trainer is not None:
        stats: RunningStats = trainer.automaton_stats
        manifest["automaton_seconds_total"] = stats.total_automaton_seconds
        manifest["baseline_diagnostic_seconds_total"] = stats.total_diag_seconds
        manifest["micro_batches"] = stats.micro_batches
        manifest["snapshots"] = trainer.snapshot_callback.saved
        manifest["snapshot_steps_not_reached"] = sorted(trainer.snapshot_callback.pending)
        manifest["oracle"] = trainer.automaton_oracle.info() if trainer.automaton_oracle else None
    path.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    config, rest = split_arguments(argv)
    upstream = load_upstream(upstream_path(config.upstream_trainer))
    output_dir = output_argument(rest)

    import transformers

    trainer_class = install_trainer_patch(transformers, config, output_dir)
    verify_trainer_patch(trainer_class)
    sys.argv = [sys.argv[0], *rest]
    status = "failed"
    try:
        code = upstream.main()
        if code in (None, 0) and getattr(trainer_class, "last_instance", None) is None:
            print(
                "[automaton] ERROR: upstream main() finished but AutomatonGuidedTrainer was never "
                "instantiated (Trainer patch bypassed); marking run failed",
                file=sys.stderr,
                flush=True,
            )
            code = 1
        code = 0 if code is None else int(code)
        status = "complete" if code == 0 else "failed"
        return code
    finally:
        if output_dir is not None:
            finalize_manifest(output_dir, trainer_class, status)


if __name__ == "__main__":
    raise SystemExit(main())
