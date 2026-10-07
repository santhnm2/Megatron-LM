# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Detokenized output drops the trailing tokens that ended generation."""

from types import SimpleNamespace

from megatron.core.inference.utils import detokenize_tokens


def _tokenizer(generation_config=None):
    return SimpleNamespace(
        eod=2,
        generation_config=generation_config,
        detokenize=lambda tokens: " ".join(str(token) for token in tokens),
    )


def test_strips_every_generation_config_eos():
    """A chat model can stop on `<|im_end|>` (11) as well as `</s>` (2)."""
    tokenizer = _tokenizer({"eos_token_id": [2, 11]})
    assert detokenize_tokens(tokenizer, [5, 6, 11]) == "5 6"
    assert detokenize_tokens(tokenizer, [5, 6, 2]) == "5 6"
    assert detokenize_tokens(tokenizer, [11]) == ""


def test_keeps_eos_inside_the_text_and_when_asked():
    tokenizer = _tokenizer({"eos_token_id": [2, 11]})
    assert detokenize_tokens(tokenizer, [5, 11, 6]) == "5 11 6"
    assert detokenize_tokens(tokenizer, [5, 6, 11], remove_EOD=False) == "5 6 11"


def test_without_generation_config_only_eod_is_stripped():
    tokenizer = _tokenizer()
    assert detokenize_tokens(tokenizer, [5, 11, 2]) == "5 11"
