# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Tests for the SSM chunk alignment quantum threaded through the inference config.

A mixer's `chunk_size` is not always the chunk length its inference kernels run
at: the forked Gated Delta Product prefill kernels chunk at a fixed 64 whatever
`chunk_size` says. Scheduling decisions that must land on a chunk boundary read
`ssm_chunk_alignment`, which is the LCM over the model's mixers, so this is the
seam where a wrong answer turns into silently unrecordable state boundaries.
"""

import types

import pytest
import torch

from megatron.core.inference.config import MambaInferenceStateConfig
from megatron.core.ssm.gated_delta_product import GatedDeltaProductMixer
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.ssm.ops.gdp.metadata import CHUNK_SIZE as GDP_CHUNK_SIZE


def _model(mixers):
    """A stand-in model exposing only what `from_model` reads."""
    from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

    decoder = types.SimpleNamespace(
        layer_type_list=[Symbols.MAMBA] * len(mixers),
        layers=[types.SimpleNamespace(mixer=mixer) for mixer in mixers],
        mamba_state_shapes_per_request=lambda: ((16, 4), (2, 8, 16)),
    )
    return types.SimpleNamespace(
        decoder=decoder,
        config=types.SimpleNamespace(params_dtype=torch.bfloat16, batch_invariant_mode=False),
    )


def _mamba_mixer(chunk_size=128):
    return types.SimpleNamespace(chunk_size=chunk_size, ssm_inference_chunk_size=chunk_size)


def _gdp_mixer(chunk_size=128):
    return types.SimpleNamespace(
        chunk_size=chunk_size, ssm_inference_chunk_size=GDP_CHUNK_SIZE, num_householder=2
    )


@pytest.mark.internal
def test_mixer_classes_expose_the_inference_chunk_size():
    """Both mixers answer the same question, so `from_model` can ask uniformly."""
    assert isinstance(MambaMixer.ssm_inference_chunk_size, property)
    assert isinstance(GatedDeltaProductMixer.ssm_inference_chunk_size, property)
    # GDP's answer is a constant, so it can be read without an instance.
    assert GatedDeltaProductMixer.ssm_inference_chunk_size.fget(None) == GDP_CHUNK_SIZE
    assert GDP_CHUNK_SIZE == 64


@pytest.mark.internal
def test_mamba_only_model_aligns_to_the_mamba_chunk_size():
    config = MambaInferenceStateConfig.from_model(_model([_mamba_mixer(128)]))
    assert config.mamba_chunk_size == 128
    assert config.ssm_chunk_alignment == 128
    assert config.gdp_num_householder == 0


@pytest.mark.internal
def test_gdp_only_model_aligns_to_the_gdp_kernel_chunk_size():
    """The training-path `chunk_size` of 128 must not be mistaken for the real one."""
    config = MambaInferenceStateConfig.from_model(_model([_gdp_mixer(chunk_size=128)]))
    assert config.ssm_chunk_alignment == GDP_CHUNK_SIZE
    assert config.gdp_num_householder == 2
    # mamba_chunk_size keeps its old meaning -- it sizes the Mamba2 chunk
    # metadata buffers, which a GDP-only model never reads.
    assert config.mamba_chunk_size == 128


@pytest.mark.internal
def test_mixed_model_aligns_to_the_lcm():
    """A boundary is only clean if it is clean for every mixer in the model."""
    config = MambaInferenceStateConfig.from_model(_model([_mamba_mixer(128), _gdp_mixer()]))
    assert config.ssm_chunk_alignment == 128  # lcm(128, 64)
    assert config.gdp_num_householder == 2


@pytest.mark.internal
def test_alignment_defaults_to_the_mamba_chunk_size():
    """Hand-built configs that predate the field keep their old behaviour."""
    config = MambaInferenceStateConfig(
        layer_type_list=["M"],
        conv_states_shape=(16, 4),
        ssm_states_shape=(2, 8, 16),
        conv_states_dtype=torch.bfloat16,
        ssm_states_dtype=torch.bfloat16,
        mamba_chunk_size=64,
    )
    assert config.ssm_chunk_alignment == 64
