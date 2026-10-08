"""Late (post-decoding) uses of the SBN automaton for text-to-SBN parsing.

Two same-state late arms, both with unconstrained generation followed by the
same W validity check (SBN transition state + Referee-aligned well-formedness)
that the online W/WF/APD decoders use:

- L4 beam-filter: beam search with N returned sequences; release the first
  W-valid candidate in beam order, else the top beam (Liu, LREC-COLING 2024).
- L3 sample-filter: K ancestral samples (temperature 1, no top-k/top-p);
  release the W-valid sample with the highest model log-probability, else the
  highest-log-probability sample.

Rows follow generate_causal_lm_baseline.py so score_sbn_smatchpp.py can score
them unchanged; every candidate is kept for audit.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs import check_sbn_state_wellformedness, validate_sbn  # noqa: E402
from automata_drs.causal_lm import (  # noqa: E402
    build_sbn_prompt,
    extract_one_line_sbn,
    iter_sbn_records,
)
from automata_drs.generation_manifest import sha256_file  # noqa: E402
from automata_drs.model_loading import MODEL_LOADERS, load_generation_model  # noqa: E402


def w_valid(raw: str) -> tuple[bool, str | None]:
    structural = validate_sbn(raw)
    if not structural.valid:
        return False, structural.reason
    semantic = check_sbn_state_wellformedness(structural.state)
    return semantic.valid, semantic.reason


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--model-loader", choices=MODEL_LOADERS, default="causal-lm")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=320)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    manifest_path = Path(str(args.output) + ".manifest.json")
    if manifest_path.exists():
        parser.error(f"output already complete: {args.output}")
    done_ids: set[str] = set()
    if args.output.exists():
        # resume: keep only records whose two arms (L4 then L3) were both written
        kept, pending = [], {}
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            pending.setdefault(row["id"], []).append(line)
            if len(pending[row["id"]]) == 2:
                kept.extend(pending.pop(row["id"]))
                done_ids.add(row["id"])
        args.output.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
        print(f"resuming: {len(done_ids)} complete records kept", flush=True)

    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoTokenizer

    load_kwargs = {"revision": args.revision, "dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    if args.cache_dir:
        load_kwargs["cache_dir"] = str(args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = load_generation_model(transformers, args.model, loader=args.model_loader, load_kwargs=load_kwargs)
    model = PeftModel.from_pretrained(base, args.adapter).to("cuda").eval()
    eos = tokenizer.eos_token_id
    records = list(iter_sbn_records(args.path, limit=args.limit))

    def decode(suffix: list[int]) -> tuple[str, int]:
        if eos in suffix:
            suffix = suffix[: suffix.index(eos) + 1]
        return tokenizer.decode(suffix, skip_special_tokens=True, clean_up_tokenization_spaces=False), len(suffix)

    def candidate(suffix: list[int], score: float) -> dict:
        raw, n_tokens = decode(suffix)
        valid, reason = w_valid(raw)
        return {"raw_prediction": raw, "w_valid": valid, "w_reason": reason,
                "score": score, "tokens_including_eos": n_tokens}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    with args.output.open("a", encoding="utf-8") as sink:
        for index, record in enumerate(records):
            if record.identifier in done_ids:
                continue
            encoded = tokenizer([build_sbn_prompt(record.source)], return_tensors="pt",
                                truncation=True, max_length=args.max_input_tokens).to("cuda")
            width = int(encoded["input_ids"].shape[-1])
            common = dict(max_new_tokens=args.max_new_tokens, eos_token_id=eos,
                          pad_token_id=tokenizer.pad_token_id, return_dict_in_generate=True,
                          output_scores=True)
            with torch.inference_mode():
                t0 = time.perf_counter()
                beam = model.generate(**encoded, do_sample=False, num_beams=args.beams,
                                      num_return_sequences=args.beams, length_penalty=1.0, **common)
                t_beam = time.perf_counter() - t0
                torch.manual_seed(args.seed + index)
                t0 = time.perf_counter()
                sample = model.generate(**encoded, do_sample=True, temperature=1.0, top_k=0,
                                        top_p=1.0, num_return_sequences=args.samples, **common)
                t_sample = time.perf_counter() - t0
                # accumulate chosen-token log-probs step by step (stacking all steps over the
                # 248k vocabulary with compute_transition_scores can exceed the memory cap)
                chosen = sample.sequences[:, width:]
                transition = torch.zeros(chosen.shape, dtype=torch.float32, device=chosen.device)
                for step, step_scores in enumerate(sample.scores):
                    logp = torch.log_softmax(step_scores.float(), dim=-1)
                    transition[:, step] = logp.gather(1, chosen[:, step : step + 1]).squeeze(1)
                    del logp
            beams = [candidate(beam.sequences[j, width:].tolist(), float(beam.sequences_scores[j]))
                     for j in range(args.beams)]
            samples = []
            for j in range(args.samples):
                suffix = sample.sequences[j, width:].tolist()
                length = suffix.index(eos) + 1 if eos in suffix else len(suffix)
                logprob = float(transition[j, :length].sum())
                samples.append(candidate(suffix, logprob))

            valid_beams = [c for c in beams if c["w_valid"]]
            l4 = valid_beams[0] if valid_beams else beams[0]
            valid_samples = [c for c in samples if c["w_valid"]]
            pool = valid_samples or samples
            l3 = max(pool, key=lambda c: c["score"])
            for arm, chosen, pool_rows, seconds, decode_info in (
                ("L4", l4, beams, t_beam, {"strategy": "beam-filter", "num_beams": args.beams}),
                ("L3", l3, samples, t_sample, {"strategy": "sample-filter", "samples": args.samples,
                                               "temperature": 1.0}),
            ):
                raw = chosen["raw_prediction"]
                sink.write(json.dumps({
                    "id": record.identifier,
                    "input": record.source,
                    "reference": record.target,
                    "prediction": extract_one_line_sbn(raw),
                    "raw_prediction": raw,
                    "configuration": arm,
                    "decode": decode_info,
                    "validity": {"raw_w": chosen["w_valid"], "raw_w_reason": chosen["w_reason"]},
                    "selection": {"any_valid": any(c["w_valid"] for c in pool_rows),
                                  "valid_candidates": sum(c["w_valid"] for c in pool_rows),
                                  "generated_tokens": sum(c["tokens_including_eos"] for c in pool_rows),
                                  "seconds": seconds},
                    "candidates": pool_rows,
                }, ensure_ascii=False) + "\n")
            sink.flush()
            if (index + 1) % 50 == 0:
                print(f"{index + 1}/{len(records)} {time.perf_counter() - started_all:.0f}s", flush=True)

    manifest = {"status": "complete", "path": str(args.path), "source_sha256": sha256_file(args.path),
                "records": len(records), "rows": 2 * len(records), "adapter": str(args.adapter),
                "model": args.model, "revision": args.revision, "beams": args.beams,
                "samples": args.samples, "seed": args.seed, "max_new_tokens": args.max_new_tokens,
                "physical_gpu_index": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "wall_seconds": time.perf_counter() - started_all,
                "output_sha256": sha256_file(args.output)}
    Path(str(args.output) + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
