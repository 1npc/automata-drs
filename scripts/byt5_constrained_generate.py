#!/usr/bin/env python3
"""Generate an SBN with XiaoZhang98/byT5-DRS under hard constraints.

Install the optional runtime first with ``pip install -e '.[hf]'``. The model
checkpoint is intentionally downloaded only when this script is invoked.
"""

from __future__ import annotations

import argparse

from automata_drs import ByT5SBNLogitsProcessor, SBNPolicy, validate_sbn


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="English sentence to parse into flat SBN")
    parser.add_argument("--model", default="XiaoZhang98/byT5-DRS")
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--semantic-referee",
        action="store_true",
        help="enable the full W scope/binding policy",
    )
    args = parser.parse_args()

    try:
        import torch
        from transformers import (
            AutoModelForSeq2SeqLM,
            AutoTokenizer,
            LogitsProcessorList,
        )
    except ImportError as error:
        parser.error("install the Hugging Face runtime with: pip install -e '.[hf]'")
        raise AssertionError("unreachable") from error

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    expected = {
        "vocab_size": 384,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "decoder_start_token_id": 0,
    }
    actual = {
        "vocab_size": model.config.vocab_size,
        "pad_token_id": model.config.pad_token_id,
        "eos_token_id": model.config.eos_token_id,
        "decoder_start_token_id": model.config.decoder_start_token_id,
    }
    if actual != expected or tokenizer.__class__.__name__ != "ByT5Tokenizer":
        raise ValueError(
            "the exact byte adapter does not match this checkpoint: "
            f"tokenizer={tokenizer.__class__.__name__}, config={actual}"
        )

    inputs = tokenizer(
        args.text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)
    policy = (
        SBNPolicy.semantic_referee() if args.semantic_referee else SBNPolicy()
    )
    constraints = ByT5SBNLogitsProcessor(
        policy=policy,
        decoder_start_token_id=model.config.decoder_start_token_id
    )
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            logits_processor=LogitsProcessorList([constraints]),
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )

    sbn = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    validation = validate_sbn(sbn, policy)
    print(sbn)
    print(f"valid={validation.valid}")
    if not validation.valid:
        print(f"reason={validation.reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
