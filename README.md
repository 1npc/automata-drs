# automata-drs

`automata-drs` studies hard, prefix-safe constrained decoding for linearized
graph generation. The first target is the Parallel Meaning Bank's Sequence Box
Notation (SBN), used to represent Discourse Representation Structures (DRSs).

This repository contains code only: the automaton and decoding library
(`src/`), experiment and analysis scripts (`scripts/`), tests (`tests/`), and
public experiment configurations (`configs/`). Corpora, model weights,
experiment outputs, and cluster launch scripts are not included.

The implementation keeps its constraint core model-independent while also
providing a real ByT5 integration. Its immutable incremental SBN state:

- consumes arbitrary text fragments rather than assuming word-level model
  tokens;
- checks lexical and SBN sequencing constraints while a fragment is consumed;
- validates relative concept references, including unresolved forward
  references;
- rejects cycles in the complete typed SBN graph (box membership, roles,
  propositions, and box edges), with a cycle-tolerant compatibility policy for
  reproducing the reference parser's warning-and-continue behavior;
- tracks future box obligations introduced by `Proposition`;
- retains a lossless typed graph projection and translates accepted SBNs to
  PMB-style CLF clauses;
- checks box subordination, inferred accessibility, loop freedom, binding, and
  box connectedness with an implementation cross-validated against the
  upstream PMB Referee;
- exposes whether EOS is legal at the current prefix.

A generic `TokenConstraintController` turns exact token-to-surface fragments
into cached allowed-token sets. It depends only on the exported
`IncrementalConstraintState` protocol, not on SBN, so a new graph language can
reuse masking and caching by providing an immutable/hashable state that
implements `advance_text`, `finish`, `is_extendable`, and `is_accepting`. A
representation-neutral ByteLevel-BPE
bridge additionally preserves partial UTF-8 scalars across token boundaries,
reconstructs states after beam reordering, and applies a score-ranked logits
mask to any such state. The first concrete model integration targets
[`XiaoZhang98/byT5-DRS`](https://huggingface.co/XiaoZhang98/byT5-DRS): it maps
the checkpoint's byte IDs exactly, buffers incomplete UTF-8 characters, masks
sentinel tokens, and exposes a Hugging Face-compatible logits processor that
remains correct when beam rows are reordered.

The first cross-representation migration is also executable: a conservative
`PENMANConstraintState` recognizes common AMR serializations with a parenthesis
stack, variable declarations, forward/backward references, reentrancies, and
quoted constants. It deliberately allows cyclic AMR graphs instead of copying
SBN's DAG rule. See the [PENMAN/AMR migration note](docs/cross-representation.md)
for the implemented boundary and corpus-coverage gate.

A third contract adapter, `CanonicalTripleConstraintState`, implements a small
deterministically serialized KG language rather than another SBN-shaped
grammar. It combines finite `NODE`/`EDGE` record control with a node register,
online endpoint checks, duplicate and canonical-order guards, and weak
connectivity at EOS. Its tests drive both surface-fragment and exact-byte token
controllers, including rejection of irreparable endpoint/order prefixes. This
is executable interface evidence, not a KG task-quality result. A separate
record-oriented batch oracle shares no incremental parser control flow; bounded
enumeration over 446 canonical connected graphs with one to three nodes finds
zero disagreement, while registered structural corruptions are rejected by
both paths.

For fail-closed application integration, use `release_sbn_surface`,
`release_penman_surface`, or `release_canonical_triple_surface` instead of
consuming a row's `raw_prediction` directly. They return a `CertifiedSurface`
carrying the exact policy-relative guarantee, or raise
`CertifiedReleaseError`; they never return an unchecked surface. PENMAN and
canonical triples instantiate the shared `FinalSurfaceReleasePolicy` gate for
observed EOS, no truncation, nonempty masks, fresh membership, and registered
validity evidence, while keeping representation semantics separate. Canonical
KG release requires both the incremental state and its independent batch
oracle. This does not certify task-semantic correctness. Population reports
must still state attempts, releases, and refusals separately.

The requirement-by-requirement [completion audit](docs/completion-audit.md)
separates implemented evidence from experiments that are still pending.
The [adversarial paper audit](docs/paper-review-audit.md) records reviewer risks
and prevents unsupported implementation claims from entering the paper story.
The claim-controlled [manuscript scaffold](docs/manuscript-scaffold.md) turns
that ledger into a reverse-outlined Introduction draft while leaving all
full-suite and AMR task effects as explicit `RESULT PENDING` placeholders.
The companion [Method draft](docs/manuscript-method.md) organizes the layered
state, exact token lifting, finite-horizon certificate, score-ranked mask,
release boundary, and PENMAN transfer by motivation, forward process, and
technical guarantee.
The [Experiments draft](docs/manuscript-experiments.md) freezes the datasets,
host-selection record, paired comparisons, release/coverage semantics,
uncertainty protocol, and result slots without promoting partial runs to final
evidence.
The [Related Work draft](docs/manuscript-related-work.md) places grammar,
tokenizer, semantic-completion, probability, systems, and DRS precedents around
the deliberately narrow constructive-certificate claim.
The [Abstract scaffold](docs/manuscript-abstract.md) uses result-locked slots so
that the central guarantee can be stated without turning pilots or pending AMR
work into publication claims.
The [Conclusion scaffold](docs/manuscript-conclusion.md) preserves the same
result lock while separating practical fail-closed impact from semantic,
availability, representation, and systems limitations.
The generated [unified manuscript](docs/manuscript.md) assembles the six
canonical section drafts into one reviewer-readable paper while removing
author-only outlines and preserving every `RESULT PENDING` slot. Regenerate it
with `python3 scripts/assemble_manuscript.py`; CI-style checks can use
`python3 scripts/assemble_manuscript.py --check`.
`scripts/check_manuscript_result_lock.py` keeps those result slots locked until
the DRS full-scope and AMR exact-chain collectors supply complete, hash-bound
evidence and `scripts/build_manuscript_result_package.py` binds both suites into
the exact paper tables; logs, partial JSONL files, individual summaries, and
suite-only manual transcription cannot authorize a manuscript number.
For this active server migration, the completed schema-v6 DRS suite is kept
immutable; after all model jobs exit, `scripts/promote_colossus_lora_suite_v3.sh`
recollects schema-v7 v3 with one CPU thread, records both v2 hashes, and freezes
the promotion gate required by the AMR full runner.
After both canonical server suites and their manuscript result package pass
the result lock,
`scripts/pull_colossus_evidence.sh` provides a default-dry-run, size-capped,
non-overwriting pull of JSON/Markdown evidence into a fresh isolated snapshot
without fetching predictions, checkpoints, raw data, or logs. Applied
snapshots receive a deterministic `snapshot-manifest.json`; recheck it with
`scripts/verify_evidence_snapshot.py` to reproduce both suite Markdown files,
the manuscript-result Markdown, and verify every transferred compact input
against its collector-recorded SHA-256 digest.

Run the tests from the repository root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Audit an SBN/TSV corpus:

```bash
python3 scripts/audit_sbn.py data/en/test/standard.sbn
```

Audit the incremental PENMAN state against a versioned corpus with the
independent `penman` parser. On Colossus, the pinned GrAPES runner uses one CPU
thread at `nice=15`, keeps the checkout in private local scratch, verifies its
SHA-256, and removes the checkout when finished:

```bash
AUTOMATA_DRS_SCRATCH_ROOT=/path/to/scratch/automata-drs \
  bash scripts/run_colossus_penman_grapes_audit.sh \
  runs/penman/grapes-reproduction.json
```

After a recent-model host is frozen, replay the same raw corpus through that
model's pinned tokenizer and the representation-neutral PENMAN controller:

```bash
AUTOMATA_DRS_SCRATCH_ROOT=/path/to/scratch/automata-drs \
  bash scripts/run_colossus_penman_token_lifting.sh selected
```

Run constrained generation after installing the optional model runtime (the
large checkpoint is downloaded only by the generation command):

```bash
python3 -m pip install -e '.[hf]'
python3 scripts/byt5_constrained_generate.py "I am a student."
```

Run the reproducible U/B/W × greedy/beam-4 matrix on a TSV corpus. The runner
refuses to overwrite an existing JSONL unless `--resume` is supplied and writes
a companion manifest with the checkpoint revision, environment, decoding
parameters, per-example validity, timing, cache statistics, and truncation
flags:

```bash
python3 scripts/run_byt5_experiment.py data/en/dev/standard.sbn \
  --output runs/byt5-dev.jsonl --limit 100
```

The default primary matrix is U/B/W. Add `--configurations U S R G B W` for
the full executable ablation ladder. `BF` and `WF` add a finite generation-
budget completion certificate to B and W respectively:

```bash
python3 scripts/run_byt5_experiment.py data/en/test/tax_challenge.sbn \
  --output runs/byt5-tax-budget.jsonl \
  --configurations U BF WF --max-new-tokens 256
```

BF/WF default to enforcing the certificate for the full decoding horizon. An
explicit smaller `--bounded-completion-window` is a faster late-repair
ablation, not a global guarantee: a prefix may already be impossible to finish
when the shorter window activates.

Summarize aggregate results and paired changes against U:

```bash
python3 scripts/summarize_experiment.py runs/byt5-dev.jsonl \
  > runs/byt5-dev-summary.md
```

The runner teacher-forces every selected prediction by default, recording its
raw LM log-probability separately from constrained generation time. This makes
it possible to quantify whether a validity repair required choosing a much
lower-probability path rather than reporting validity alone.

For official PMB5-style SBN Smatch, install the lightweight scoring extra and
point the adapter at the reference `sbn_smatch.py`. Scores are written to a
sidecar instead of mutating the original run. Its companion manifest records
the immutable input/output hashes plus the scorer file hash and Git revision.
`configs/evaluation_tools.json` pins the reference implementation used by the
reported experiments; `scripts/checkout_reference_scorer.sh NEW_DESTINATION`
checks out that exact revision when reproducing it:

```bash
python3 -m pip install -e '.[score]'
python3 scripts/score_sbn_smatch.py runs/byt5-dev.jsonl \
  --scorer /path/to/parsing_smatch/sbn/sbn_smatch.py \
  --output runs/byt5-dev-smatch.jsonl
python3 scripts/summarize_experiment.py runs/byt5-dev.jsonl \
  --scores runs/byt5-dev-smatch.jsonl \
  > runs/byt5-dev-summary.md
```

Measure pure constraint-mask overhead without torch or model weights:

```bash
python3 scripts/benchmark_byt5.py data/en/dev/standard.sbn --limit 100
```

Classify the earliest hard-constraint event in existing JSONL predictions:

```bash
python3 scripts/analyze_predictions.py data/en/test/generated_predictions.jsonl
```

Replay already valid U outputs through the full finite-budget policy. This is
the regression gate for completion-planner false rejections:

```bash
python3 scripts/audit_completion_paths.py runs/byt5-dev.jsonl \
  --source-configuration U --strategy greedy --policy B \
  --max-new-tokens 256 \
  --output runs/byt5-dev-completion-audit.json
```

This is an empirical path-coverage gate for the constructive certificate
planner, not a proof that its search is complete. A positive certificate is
always an automaton-checked accepting suffix; the audit detects valid known
paths that the current short-witness planner would nevertheless remove.
The same gate can replay strict gold paths that fit the experiment budget:

```bash
python3 scripts/audit_completion_paths.py data/en/dev/standard.sbn \
  --input-format sbn-tsv --policy W --max-new-tokens 256 \
  --output runs/dev-gold-w-completion-audit.json
```

Audit semantic well-formedness and optionally cross-check the exact upstream
Referee implementation:

```bash
python3 scripts/audit_wellformedness.py \
  data/en/train/gold.sbn data/en/dev/standard.sbn data/en/test/standard.sbn \
  --referee /path/to/DRS_parsing/evaluation/clf_referee.py
```

The semantic oracle is also available as the opt-in online W policy via
`SBNPolicy.semantic_referee()` and the scripts' `--semantic-referee` switch.
Its corpus audit finds one Referee-confirmed subordination loop in the
11,321-record structural gold set; this is kept as a documented semantic-data
exception rather than weakening the rule. The staged baselines, data gates,
metrics, and run schema are fixed in the
[experiment plan](docs/experiment-plan.md).
The reviewer-facing thesis, claim–evidence map, paper outline, and falsification
tests are consolidated in the [research brief](docs/research-brief.md).
The durable requirements and corrected interpretation extracted from the five
shared design conversations are recorded in the
[conversation synthesis](docs/conversation-synthesis.md).
The BF/WF guarantee and its incompleteness boundary are stated precisely in the
[finite-budget formalization](docs/formalization.md).
The [related-work map](docs/related-work-map.md) records the closest primary
sources, claims ruled out by prior work, and the novelty decision gate.
The pinned-checkpoint pilot, including negative results, finite-budget repairs,
Smatch, likelihood cost, and runtime limitations, is recorded in the
[current results](docs/results.md).
The role of the older byte-level checkpoint, the stronger-model candidates,
artifact gaps, quota implications, and tokenizer integration gate are recorded
in the [model matrix](docs/model-matrix.md).
The pinned recent checkpoints, shared LoRA recipe, passed smoke gates, and
serial resource contract are recorded in the
[sub-10B LoRA baseline plan](docs/lora-baselines.md).
Resource-capped GPU reproduction on the shared Colossus host is documented in
the [server runbook](docs/colossus-runbook.md); it contains no login secrets.
