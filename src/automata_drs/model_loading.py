"""Fail-closed model loading and LoRA scoping for causal SBN baselines."""

from __future__ import annotations

from typing import Any


MODEL_LOADERS = ("causal-lm", "multimodal-text-only")
QWEN35_LANGUAGE_PREFIX = "model.language_model."


def load_generation_model(
    transformers: Any,
    model_name: str,
    *,
    loader: str,
    load_kwargs: dict[str, Any],
) -> Any:
    """Load an explicitly selected generation interface.

    Multimodal checkpoints are kept as their official conditional-generation
    wrapper so Hugging Face can map the published weight prefixes exactly. The
    SBN caller supplies text tokens only, and LoRA targeting is separately
    restricted to the language backbone.
    """

    if loader == "causal-lm":
        factory = transformers.AutoModelForCausalLM
    elif loader == "multimodal-text-only":
        factory = getattr(transformers, "AutoModelForMultimodalLM", None)
        if factory is None:
            raise RuntimeError(
                "the installed Transformers version has no "
                "AutoModelForMultimodalLM loader"
            )
    else:
        raise ValueError(
            f"unsupported model loader {loader!r}; expected one of {MODEL_LOADERS}"
        )
    model = factory.from_pretrained(model_name, **load_kwargs)
    if not callable(getattr(model, "forward", None)):
        raise TypeError("loaded generation model has no forward method")
    if not callable(getattr(model, "generate", None)):
        raise TypeError("loaded generation model has no generate method")
    return model


def disable_generation_cache(model: Any) -> None:
    """Disable KV caching in both outer and nested text configurations."""

    configurations = [getattr(model, "config", None)]
    outer = configurations[0]
    if outer is not None:
        configurations.append(getattr(outer, "text_config", None))
    for configuration in configurations:
        if configuration is not None and hasattr(configuration, "use_cache"):
            configuration.use_cache = False


def lora_target_modules(
    model: Any,
    *,
    loader: str,
    linear_type: type[Any],
) -> str | list[str]:
    """Return all linear layers, excluding an unused multimodal vision tower."""

    if loader == "causal-lm":
        return "all-linear"
    if loader != "multimodal-text-only":
        raise ValueError(
            f"unsupported model loader {loader!r}; expected one of {MODEL_LOADERS}"
        )
    targets = sorted(
        name
        for name, module in model.named_modules()
        if name.startswith(QWEN35_LANGUAGE_PREFIX)
        and isinstance(module, linear_type)
    )
    if not targets:
        raise RuntimeError(
            "multimodal text-only LoRA found no linear modules below "
            f"{QWEN35_LANGUAGE_PREFIX!r}"
        )
    if any("visual" in name or "vision" in name for name in targets):
        raise RuntimeError("vision module leaked into text-only LoRA targets")
    return targets


def target_module_manifest(targets: str | list[str]) -> dict[str, Any]:
    """Summarize the adapter scope without duplicating a long module list."""

    if isinstance(targets, str):
        return {"scope": targets, "count": None, "prefix": None}
    return {
        "scope": "language-linear",
        "count": len(targets),
        "prefix": QWEN35_LANGUAGE_PREFIX,
    }


__all__ = [
    "MODEL_LOADERS",
    "QWEN35_LANGUAGE_PREFIX",
    "disable_generation_cache",
    "load_generation_model",
    "lora_target_modules",
    "target_module_manifest",
]
