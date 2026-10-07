# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Checkpoint detection selects the model and the checkpoint key names it loads from."""

from argparse import Namespace

import pytest

from megatron.core.inference.text_generation_server.dynamic_text_gen_server import (
    vlm_dynamic_inference,
)


@pytest.mark.parametrize("provider", ["nemotron-moe-vlm", "nemotron-moe-mistral-vit"])
def test_mimo_text_detection_selects_text_backbone(monkeypatch, provider):
    """Model detection selects the text backbone without constructing a VLM."""
    args = Namespace(model_provider="gpt")
    saved = Namespace(model_provider=provider, mimo_llm_tp=1)
    monkeypatch.setattr(vlm_dynamic_inference, "load_args_from_checkpoint", lambda _: (args, saved))
    # A language model without a vision encoder.
    monkeypatch.setattr(
        vlm_dynamic_inference,
        "_checkpoint_tensor_keys",
        lambda args, checkpoint_dir=None: ["language_model.module.module.output_layer.weight"],
    )
    assert not vlm_dynamic_inference._detect_vlm_from_checkpoint(args)
    assert args.model_provider == "hybrid"
    assert args.checkpoint_model_prefix == "language_model.module.module."


@pytest.mark.parametrize("vision", [False, True], ids=["text", "vision"])
def test_checkpoint_without_saved_args_uses_cli_vision_args(monkeypatch, vision):
    """Without saved args (e.g. Megatron-Bridge), --vision-model-type selects the VLM."""
    args = Namespace(model_provider="gpt", vision_model_type="pixtral-vit-large")
    monkeypatch.setattr(vlm_dynamic_inference, "load_args_from_checkpoint", lambda _: args)
    passed = {"vision_model_type"} if vision else set()

    assert vlm_dynamic_inference._detect_vlm_from_checkpoint(args, passed) is vision
    if vision:
        # LLaVAModel's own key names, each loaded from itself.
        assert args.mimo_checkpoint_prefix_map == {
            "language_model.": "language_model.",
            "vision_model.": "vision_model.",
            "vision_projection.": "vision_projection.",
        }
        assert args.img_h == 1540  # from the encoder registry
    else:
        assert not hasattr(args, "mimo_checkpoint_prefix_map")
