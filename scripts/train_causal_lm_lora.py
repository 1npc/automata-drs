#!/usr/bin/env python3
"""Fine-tune a decoder-only language model for registered SBN or AMR tasks."""

from __future__ import annotations

import argparse
from array import array
import inspect
from itertools import islice
import json
from hashlib import sha256
import math
import os
import platform
import shutil
import sys
import time
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from automata_drs.causal_lm import (  # noqa: E402
    encode_supervised_record,
    iter_sbn_records,
    pad_supervised_batch,
)
from automata_drs.amr_causal_lm import encode_amr_supervised_record  # noqa: E402
from automata_drs.amr_data import (  # noqa: E402
    iter_amr_records,
    select_amr_records_by_ordinal,
)
from automata_drs.model_loading import (  # noqa: E402
    MODEL_LOADERS,
    disable_generation_cache,
    load_generation_model,
    lora_target_modules,
    target_module_manifest,
)


class EncodedDataset:
    """Store token IDs once in compact arrays; build masks only per batch.

    AMR has over fifty-five thousand records. Retaining Python-int copies of
    input IDs, attention masks, and labels for the full corpus would multiply
    host RAM before the model is even loaded.
    """

    def __init__(self) -> None:
        self.examples: list[tuple[array, int]] = []

    def append(self, example: dict[str, Any]) -> None:
        input_ids = array("I", (int(value) for value in example["input_ids"]))
        prompt_tokens = int(example["prompt_tokens"])
        if not 0 <= prompt_tokens <= len(input_ids):
            raise ValueError("invalid prompt token boundary")
        self.examples.append((input_ids, prompt_tokens))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        compact_ids, prompt_tokens = self.examples[index]
        input_ids = list(compact_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": [-100] * prompt_tokens + input_ids[prompt_tokens:],
            "prompt_tokens": prompt_tokens,
            "target_tokens": len(input_ids) - prompt_tokens,
        }


class TorchCollator:
    def __init__(self, torch: Any, pad_token_id: int) -> None:
        self.torch = torch
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        padded = pad_supervised_batch(
            examples, pad_token_id=self.pad_token_id, pad_to_multiple_of=8
        )
        return {
            key: self.torch.tensor(value, dtype=self.torch.long)
            for key, value in padded.items()
        }


def encode_corpus(
    tokenizer: Any,
    path: Path,
    *,
    task: str,
    limit: int,
    max_length: int,
    record_numbers: list[int] | None = None,
) -> tuple[EncodedDataset, dict[str, int]]:
    if record_numbers is not None and task != "amr":
        raise ValueError("record_numbers are supported only for AMR")
    if record_numbers is not None and limit != -1:
        raise ValueError("record_numbers cannot be combined with limit")
    dataset = EncodedDataset()
    stats = {
        "records": 0,
        "max_sequence_tokens": 0,
        "max_prompt_tokens": 0,
        "max_target_tokens": 0,
    }

    def add(example: dict[str, Any]) -> None:
        dataset.append(example)
        stats["records"] += 1
        stats["max_sequence_tokens"] = max(
            stats["max_sequence_tokens"], len(example["input_ids"])
        )
        stats["max_prompt_tokens"] = max(
            stats["max_prompt_tokens"], int(example["prompt_tokens"])
        )
        stats["max_target_tokens"] = max(
            stats["max_target_tokens"], int(example["target_tokens"])
        )

    if task == "sbn":
        for record in iter_sbn_records(path, limit=limit):
            add(encode_supervised_record(tokenizer, record, max_length=max_length))
    elif task == "amr":
        with path.open(encoding="utf-8") as source:
            records = iter_amr_records(source)
            if record_numbers is not None:
                records = iter(select_amr_records_by_ordinal(records, record_numbers))
            elif limit > 0:
                records = islice(records, limit)
            for record in records:
                add(encode_amr_supervised_record(
                    tokenizer,
                    record,
                    corpus_path=path,
                    max_length=max_length,
                ))
    else:
        raise ValueError(f"unsupported task: {task}")
    if not dataset:
        raise ValueError(f"no records found in {path}")
    return dataset, stats


def training_argument_name() -> str:
    from transformers import TrainingArguments

    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in parameters:
        return "eval_strategy"
    return "evaluation_strategy"


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("sbn", "amr"), default="sbn")
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--model-loader", choices=MODEL_LOADERS, default="causal-lm"
    )
    parser.add_argument("--train", type=Path)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--train-limit", type=int, default=-1)
    parser.add_argument("--eval-limit", type=int, default=-1)
    parser.add_argument(
        "--train-record-numbers",
        type=int,
        nargs="+",
        help="exact paired-block AMR record ordinals for worst-case smoke training",
    )
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=2)
    args = parser.parse_args()

    if args.train is None:
        args.train = Path(
            "data/en/train/gold.sbn" if args.task == "sbn" else "data/amr/train.amr"
        )
    if args.eval is None:
        args.eval = Path(
            "data/en/dev/standard.sbn"
            if args.task == "sbn"
            else "data/amr/validation.amr"
        )

    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"non-empty output directory already exists: {args.output}")
    if args.batch_size <= 0 or args.gradient_accumulation <= 0:
        parser.error("batch size and gradient accumulation must be positive")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        parser.error("warmup ratio must be in [0, 1]")
    if args.train_limit == 0 or args.eval_limit == 0:
        parser.error("corpus limits must be -1 or positive")
    if args.train_record_numbers is not None:
        if args.task != "amr":
            parser.error("train-record-numbers are supported only for --task amr")
        if args.train_limit != -1:
            parser.error("train-record-numbers cannot be combined with train-limit")
        if any(value <= 0 for value in args.train_record_numbers):
            parser.error("train-record-numbers must be positive")
        if len(set(args.train_record_numbers)) != len(args.train_record_numbers):
            parser.error("train-record-numbers must not contain duplicates")
    args.output.mkdir(parents=True, exist_ok=True)

    try:
        import accelerate
        import peft
        import torch
        import transformers
        from peft import LoraConfig, TaskType, get_peft_model
        from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed
    except ImportError as error:
        parser.error("install the LoRA runtime with: pip install -e '.[lora]'")
        raise AssertionError("unreachable") from error

    torch.set_num_threads(args.torch_threads)
    set_seed(args.seed)
    load_kwargs: dict[str, Any] = {"revision": args.revision}
    if args.cache_dir:
        load_kwargs["cache_dir"] = str(args.cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    if tokenizer.eos_token_id is None:
        parser.error("tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset, train_stats = encode_corpus(
        tokenizer,
        args.train,
        task=args.task,
        limit=args.train_limit,
        max_length=args.max_length,
        record_numbers=args.train_record_numbers,
    )
    eval_dataset, eval_stats = encode_corpus(
        tokenizer,
        args.eval,
        task=args.task,
        limit=args.eval_limit,
        max_length=args.max_length,
    )

    if not torch.cuda.is_available():
        parser.error("this resource-capped recipe requires CUDA")
    if not torch.cuda.is_bf16_supported():
        parser.error("selected GPU does not support bfloat16")

    model = load_generation_model(
        transformers,
        args.model,
        loader=args.model_loader,
        load_kwargs={
            **load_kwargs,
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        },
    )
    disable_generation_cache(model)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    target_modules = lora_target_modules(
        model,
        loader=args.model_loader,
        linear_type=torch.nn.Linear,
    )
    target_manifest = target_module_manifest(target_modules)
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    updates_per_epoch = math.ceil(
        len(train_dataset) / (args.batch_size * args.gradient_accumulation)
    )
    planned_steps = (
        args.max_steps
        if args.max_steps > 0
        else math.ceil(updates_per_epoch * args.epochs)
    )
    warmup_steps = max(1, math.ceil(planned_steps * args.warmup_ratio))
    common_arguments: dict[str, Any] = {
        "output_dir": str(args.output / "trainer"),
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup_steps,
        "weight_decay": 0.0,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "save_only_model": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": [],
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "seed": args.seed,
        "data_seed": args.seed,
    }
    common_arguments[training_argument_name()] = "epoch"
    training_args = TrainingArguments(**common_arguments)

    manifest = {
        "status": "starting",
        "task": args.task,
        "model": args.model,
        "model_loader": args.model_loader,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "train_path": str(args.train),
        "train_source_sha256": sha256_file(args.train),
        "train_record_numbers": args.train_record_numbers,
        "eval_path": str(args.eval),
        "eval_source_sha256": sha256_file(args.eval),
        "train": train_stats,
        "eval": eval_stats,
        "recipe": {
            "max_length": args.max_length,
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "gradient_accumulation": args.gradient_accumulation,
            "effective_batch_size": args.batch_size * args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "planned_optimizer_steps": planned_steps,
            "warmup_steps": warmup_steps,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": target_manifest,
            "precision": "bf16",
            "seed": args.seed,
            "torch_threads": args.torch_threads,
            "dataloader_num_workers": 0,
            "dataset_storage": "compact-uint32-input-ids; masks-and-labels-materialized-per-item",
        },
        "parameters": {
            "total": total_parameters,
            "trainable": trainable_parameters,
            "trainable_fraction": trainable_parameters / total_parameters,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "accelerate": accelerate.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0),
        },
    }
    write_json(args.output / "train_manifest.json", manifest)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=TorchCollator(torch, tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    train_result = trainer.train()
    evaluation = trainer.evaluate()
    torch.cuda.synchronize()
    cuda_memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
    }
    elapsed = time.time() - started

    adapter_path = args.output / "adapter"
    best_model_checkpoint = trainer.state.best_model_checkpoint
    model.save_pretrained(adapter_path, safe_serialization=True)
    tokenizer.save_pretrained(adapter_path)
    # Checkpoints were created by this run solely to restore the best epoch.
    # Keep the final adapter and metrics, not a duplicate adapter snapshot.
    trainer_output = args.output / "trainer"
    removed_checkpoints: list[str] = []
    for checkpoint in trainer_output.glob("checkpoint-*"):
        if checkpoint.is_dir() and checkpoint.parent == trainer_output:
            removed_checkpoints.append(str(checkpoint))
            shutil.rmtree(checkpoint)
    metrics = {
        "train": train_result.metrics,
        "eval": evaluation,
        "wall_seconds": elapsed,
        "best_model_checkpoint": best_model_checkpoint,
        "cuda_memory": cuda_memory,
    }
    write_json(args.output / "train_metrics.json", metrics)
    manifest["status"] = "complete"
    manifest["wall_seconds"] = elapsed
    manifest["adapter_path"] = str(adapter_path)
    manifest["best_model_checkpoint"] = best_model_checkpoint
    manifest["removed_duplicate_checkpoints"] = removed_checkpoints
    manifest["cuda_memory"] = cuda_memory
    write_json(args.output / "train_manifest.json", manifest)
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
