from __future__ import annotations

import unittest

from automata_drs.model_loading import (
    QWEN35_LANGUAGE_PREFIX,
    disable_generation_cache,
    lora_target_modules,
    target_module_manifest,
)


class Linear:
    pass


class Other:
    pass


class NamedModel:
    def named_modules(self):
        return iter(
            [
                ("model.language_model.layers.0.q_proj", Linear()),
                ("model.language_model.layers.0.norm", Other()),
                ("model.visual.blocks.0.q_proj", Linear()),
                ("lm_head", Linear()),
                ("model.language_model.layers.0.down_proj", Linear()),
            ]
        )


class Config:
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.text_config = None


class ModelWithConfig:
    def __init__(self):
        self.config = Config()
        self.config.text_config = Config()


class ModelLoadingTest(unittest.TestCase):
    def test_multimodal_lora_targets_language_linear_modules_only(self):
        targets = lora_target_modules(
            NamedModel(),
            loader="multimodal-text-only",
            linear_type=Linear,
        )
        self.assertEqual(
            [
                "model.language_model.layers.0.down_proj",
                "model.language_model.layers.0.q_proj",
            ],
            targets,
        )
        self.assertTrue(
            all(name.startswith(QWEN35_LANGUAGE_PREFIX) for name in targets)
        )
        self.assertEqual(
            {
                "scope": "language-linear",
                "count": 2,
                "prefix": QWEN35_LANGUAGE_PREFIX,
            },
            target_module_manifest(targets),
        )

    def test_causal_loader_preserves_all_linear_scope(self):
        self.assertEqual(
            "all-linear",
            lora_target_modules(
                NamedModel(), loader="causal-lm", linear_type=Linear
            ),
        )

    def test_multimodal_scope_fails_when_language_prefix_is_absent(self):
        class VisionOnly:
            def named_modules(self):
                return iter([("model.visual.q_proj", Linear())])

        with self.assertRaisesRegex(RuntimeError, "found no linear modules"):
            lora_target_modules(
                VisionOnly(),
                loader="multimodal-text-only",
                linear_type=Linear,
            )

    def test_cache_is_disabled_on_outer_and_text_configs(self):
        model = ModelWithConfig()
        disable_generation_cache(model)
        self.assertFalse(model.config.use_cache)
        self.assertFalse(model.config.text_config.use_cache)


if __name__ == "__main__":
    unittest.main()
