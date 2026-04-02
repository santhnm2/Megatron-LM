# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
import math

import pytest
import torch

from megatron.core.inference.config import (
    InferenceConfig,
    MambaInferenceStateConfig,
    compute_kv_block_size_bytes,
    compute_mamba_memory_per_request,
    measure_peak_activation_memory,
)
from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _make_tiny_gpt_model():
    """Build a tiny GPT model on GPU for tests that need a real model."""
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec

    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=2,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=128,
        max_sequence_length=512,
    )
    model.cuda().eval()
    return model, config


def _make_tiny_mamba_model():
    """Build a tiny hybrid Mamba model on GPU for tests that need a real model."""
    from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
    from megatron.core.models.mamba.mamba_model import MambaModel
    from megatron.core.transformer.enums import AttnBackend

    model_parallel_cuda_manual_seed(123)
    config = TransformerConfig(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        use_cpu_initialization=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_backend=AttnBackend.local,
    )
    model = MambaModel(
        config=config,
        mamba_stack_spec=mamba_stack_spec,
        vocab_size=128,
        max_sequence_length=512,
        hybrid_layer_pattern="M*",  # alternating Mamba/Attention
    )
    model.cuda().eval()
    return model, config


class TestComputeHelpers:
    """Tests for the shared memory computation helpers."""

    def test_kv_block_size_mha(self):
        result = compute_kv_block_size_bytes(
            kv_dtype_size=2,
            num_attention_layers=32,
            block_size_tokens=256,
            cache_mla_latent=False,
            heads_per_partition=8,
            head_size=128,
        )
        assert result == 2 * 2 * 32 * 256 * 8 * 128

    def test_kv_block_size_mla(self):
        result = compute_kv_block_size_bytes(
            kv_dtype_size=2,
            num_attention_layers=32,
            block_size_tokens=64,
            cache_mla_latent=True,
            kv_reduced_dim=576,
        )
        assert result == 2 * 32 * 64 * 576

    def test_mamba_memory_no_layers(self):
        config = MambaInferenceStateConfig(
            layer_type_list=[Symbols.ATTENTION],
            conv_states_shape=(128, 4),
            ssm_states_shape=(128, 64),
            conv_states_dtype=torch.bfloat16,
            ssm_states_dtype=torch.bfloat16,
        )
        assert compute_mamba_memory_per_request(config, num_mamba_layers=0) == 0

    def test_mamba_memory_basic(self):
        config = MambaInferenceStateConfig(
            layer_type_list=[Symbols.MAMBA],
            conv_states_shape=(128, 4),
            ssm_states_shape=(128, 64),
            conv_states_dtype=torch.bfloat16,
            ssm_states_dtype=torch.bfloat16,
        )
        conv_bytes = 128 * 4 * 2  # bf16
        ssm_bytes = 128 * 64 * 2
        assert compute_mamba_memory_per_request(config, num_mamba_layers=4) == (
            (conv_bytes + ssm_bytes) * 4
        )

    def test_mamba_memory_with_speculative(self):
        config = MambaInferenceStateConfig(
            layer_type_list=[Symbols.MAMBA],
            conv_states_shape=(128, 4),
            ssm_states_shape=(128, 64),
            conv_states_dtype=torch.bfloat16,
            ssm_states_dtype=torch.bfloat16,
        )
        base = compute_mamba_memory_per_request(config, num_mamba_layers=4)
        with_spec = compute_mamba_memory_per_request(
            config, num_mamba_layers=4, num_speculative_tokens=3,
        )
        assert with_spec > base
        # Speculative adds (spec+1) * per_layer * num_layers.
        conv_bytes = 128 * 4 * 2
        ssm_bytes = 128 * 64 * 2
        expected = base + (conv_bytes + ssm_bytes) * 4 * 4  # (3+1) * per_layer * 4 layers
        assert with_spec == expected


class TestInferenceConfig:
    def test_mutual_exclusivity_with_transformer_config(self):
        """
        Ensure mutual exclusivity between fields in `InferenceConfig` and
        `TransformerConfig`.
        """
        dynamic_inference_config_fields = set(dataclasses.fields(InferenceConfig))
        transformer_config_fields = set(dataclasses.fields(TransformerConfig))
        assert len(dynamic_inference_config_fields.intersection(transformer_config_fields)) == 0

    def test_requires_buffer_size_without_model_config(self):
        """buffer_size_gb must be set when model_config is not provided."""
        with pytest.raises(ValueError, match="buffer_size_gb must be set"):
            InferenceConfig()

    def test_explicit_buffer_size_without_model_config(self):
        """Legacy path: explicit buffer_size_gb works without model_config."""
        config = InferenceConfig(buffer_size_gb=20.5)
        assert config.buffer_size_gb == 20.5

    def test_prefix_caching_routing_alpha_validation(self):
        with pytest.raises(ValueError, match="prefix_caching_routing_alpha"):
            InferenceConfig(buffer_size_gb=10.0, prefix_caching_routing_alpha=1.5)


class TestAutoConfigurePureTransformer:
    """Tests for InferenceConfig auto-configuration with standard Transformer models."""

    @staticmethod
    def _make_model_config(**overrides):
        defaults = dict(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
        )
        defaults.update(overrides)
        return TransformerConfig(**defaults)

    def test_basic(self):
        model_config = self._make_model_config()
        config = InferenceConfig(
            model_config=model_config,
            max_sequence_length=4096,
            gpu_memory_budget_gb=20.0,
        )
        assert config.max_sequence_length == 4096
        assert config.block_size_tokens == 256
        assert config.mamba_memory_ratio is None
        assert config.max_requests > 0
        assert config.max_requests % 4 == 0
        assert config.buffer_size_gb == pytest.approx(20.0)

    def test_max_requests_matches_manual_calculation(self):
        model_config = self._make_model_config()
        budget_gb = 20.0

        config = InferenceConfig(
            model_config=model_config,
            max_sequence_length=4096,
            gpu_memory_budget_gb=budget_gb,
        )

        # Reproduce the expected calculation.
        kv_dtype_size = torch.bfloat16.itemsize  # 2
        num_kv_heads = 32
        tp_size = 1
        pp_size = 1
        num_attn_layers = 32 // pp_size
        head_size = 4096 // 32  # 128
        heads_per_partition = num_kv_heads // tp_size
        block_size_bytes = (
            kv_dtype_size * 2 * num_attn_layers * 256 * heads_per_partition * head_size
        )
        budget_bytes = int(budget_gb * 1024**3)
        block_count = max(2, budget_bytes // block_size_bytes)
        expected_max = block_count - 1
        expected_max = (expected_max // tp_size) * tp_size
        expected_max = (expected_max // 4) * 4
        expected_max = max(4, expected_max)

        assert config.max_requests == expected_max

    def test_gqa_model(self):
        """GQA with fewer KV heads should yield more requests."""
        model_config_mha = self._make_model_config()
        model_config_gqa = self._make_model_config(num_query_groups=8)

        config_mha = InferenceConfig(
            model_config=model_config_mha,
            max_sequence_length=4096,
            gpu_memory_budget_gb=20.0,
        )
        config_gqa = InferenceConfig(
            model_config=model_config_gqa,
            max_sequence_length=4096,
            gpu_memory_budget_gb=20.0,
        )
        assert config_gqa.max_requests >= config_mha.max_requests


class TestAutoConfigureHybridMamba:
    """Tests for auto-configuration with hybrid Mamba models."""

    @staticmethod
    def _make_mamba_config():
        return MambaInferenceStateConfig(
            layer_type_list=[Symbols.MAMBA, Symbols.MAMBA, Symbols.ATTENTION, Symbols.MAMBA, Symbols.MAMBA, Symbols.ATTENTION] * 4,
            conv_states_shape=(128, 4),
            ssm_states_shape=(128, 64),
            conv_states_dtype=torch.bfloat16,
            ssm_states_dtype=torch.bfloat16,
        )

    @staticmethod
    def _make_model_config():
        return TransformerConfig(
            num_layers=24,
            hidden_size=2048,
            num_attention_heads=16,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
        )

    def test_mamba_memory_ratio_auto_computed(self):
        model_config = self._make_model_config()
        mamba_config = self._make_mamba_config()

        config = InferenceConfig(
            model_config=model_config,
            max_sequence_length=4096,
            gpu_memory_budget_gb=20.0,
            mamba_inference_state_config=mamba_config,
        )

        assert config.mamba_memory_ratio is not None
        assert 0 < config.mamba_memory_ratio < 1
        assert config.mamba_inference_state_config is mamba_config
        assert config.max_requests > 0
        assert config.max_requests % 4 == 0

    def test_mamba_ratio_is_natural_proportion(self):
        """The ratio should be mamba_per_request / total_per_request."""
        model_config = self._make_model_config()
        mamba_config = self._make_mamba_config()

        config = InferenceConfig(
            model_config=model_config,
            max_sequence_length=4096,
            gpu_memory_budget_gb=20.0,
            mamba_inference_state_config=mamba_config,
        )

        # Compute expected ratio manually.
        conv_bytes = math.prod((128, 4)) * 2  # bf16
        ssm_bytes = math.prod((128, 64)) * 2
        # 16 mamba layers (4 per group * 4 groups), per PP stage: 16 / 1 = 16
        num_mamba_layers = 16
        mamba_per_req = (conv_bytes + ssm_bytes) * num_mamba_layers

        kv_dtype_size = 2
        num_attn_layers = 8  # 2 per group * 4 groups
        heads_per_part = 16
        head_size = 2048 // 16
        block_size_bytes = kv_dtype_size * 2 * num_attn_layers * 256 * heads_per_part * head_size
        blocks_per_req = math.ceil(4096 / 256)
        kv_per_req = blocks_per_req * block_size_bytes

        expected_ratio = mamba_per_req / (kv_per_req + mamba_per_req)
        assert config.mamba_memory_ratio == pytest.approx(expected_ratio, rel=1e-6)

    def test_speculative_tokens_increase_mamba_memory(self):
        model_config = self._make_model_config()
        mamba_config = self._make_mamba_config()

        config_no_spec = InferenceConfig(
            model_config=model_config,
            max_sequence_length=4096,
            gpu_memory_budget_gb=20.0,
            mamba_inference_state_config=mamba_config,
        )
        config_spec = InferenceConfig(
            model_config=model_config,
            max_sequence_length=4096,
            gpu_memory_budget_gb=20.0,
            mamba_inference_state_config=mamba_config,
            num_speculative_tokens=3,
        )

        # With speculative tokens, mamba uses more memory per request,
        # so the ratio should be higher and max_requests should be lower.
        assert config_spec.mamba_memory_ratio > config_no_spec.mamba_memory_ratio
        assert config_spec.max_requests <= config_no_spec.max_requests

    def test_prefix_caching_mamba_gb_reduces_budget(self):
        """prefix_caching_mamba_gb should be reserved from the KV cache budget."""
        model_config = self._make_model_config()
        mamba_config = self._make_mamba_config()

        config_no_cache = InferenceConfig(
            model_config=model_config,
            max_sequence_length=4096,
            gpu_memory_budget_gb=20.0,
            mamba_inference_state_config=mamba_config,
        )
        config_with_cache = InferenceConfig(
            model_config=model_config,
            max_sequence_length=4096,
            gpu_memory_budget_gb=20.0,
            mamba_inference_state_config=mamba_config,
            prefix_caching_mamba_gb=2.0,
            enable_prefix_caching=True,
        )

        # Reserving 2 GB for Mamba prefix caching leaves less for the KV buffer.
        assert config_with_cache.max_requests < config_no_cache.max_requests
        assert config_with_cache.prefix_caching_mamba_gb == 2.0


class TestAutoConfigureWithMaxRequests:
    """Tests for explicit max_requests validation."""

    @staticmethod
    def _make_model_config():
        return TransformerConfig(
            num_layers=32,
            hidden_size=4096,
            num_attention_heads=32,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
        )

    def test_explicit_max_requests_with_sufficient_budget(self):
        """Explicit max_requests is accepted when the budget is sufficient."""
        model_config = self._make_model_config()
        config = InferenceConfig(
            model_config=model_config,
            max_sequence_length=4096,
            max_requests=128,
            gpu_memory_budget_gb=500.0,
        )
        assert config.max_requests == 128

    def test_raises_when_budget_too_small(self):
        """Explicit max_requests that exceeds the budget raises ValueError."""
        model_config = self._make_model_config()
        with pytest.raises(ValueError, match="requires.*GB but buffer_size_gb"):
            InferenceConfig(
                model_config=model_config,
                max_sequence_length=4096,
                max_requests=1024,
                gpu_memory_budget_gb=1.0,
            )


class TestAutoConfigureEdgeCases:
    """Edge case and pass-through tests."""

    @staticmethod
    def _make_model_config():
        return TransformerConfig(
            num_layers=4,
            hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=True,
            params_dtype=torch.bfloat16,
        )

    def test_auto_configure_from_gpu_memory(self):
        """Without explicit budget, buffer_size_gb is derived from real GPU memory."""
        model_config = self._make_model_config()
        config = InferenceConfig(
            model_config=model_config, max_sequence_length=2048,
        )
        assert config.buffer_size_gb > 0
        assert config.max_requests > 0

    def test_kwargs_passthrough(self):
        model_config = self._make_model_config()
        config = InferenceConfig(
            model_config=model_config,
            max_sequence_length=2048,
            gpu_memory_budget_gb=10.0,
            enable_chunked_prefill=True,
            num_cuda_graphs=8,
            enable_prefix_caching=True,
        )
        assert config.enable_chunked_prefill is True
        assert config.num_cuda_graphs == 8
        assert config.enable_prefix_caching is True

    def test_explicit_max_tokens(self):
        model_config = self._make_model_config()
        config = InferenceConfig(
            model_config=model_config,
            max_sequence_length=2048,
            gpu_memory_budget_gb=10.0,
            max_tokens=8192,
        )
        assert config.max_tokens == 8192


@pytest.mark.internal
class TestModelAutoConfiguration:
    """End-to-end auto-configuration tests parametrized over GPT and Mamba models."""

    @classmethod
    def setup_class(cls):
        Utils.initialize_model_parallel()

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    @staticmethod
    def _build(model_type):
        """Return (model, model_config, mamba_inference_state_config_or_None)."""
        if model_type == "gpt":
            model, config = _make_tiny_gpt_model()
            return model, config, None
        model, config = _make_tiny_mamba_model()
        mamba_config = MambaInferenceStateConfig.from_model(model)
        return model, config, mamba_config

    @pytest.mark.parametrize("model_type", ["gpt", "mamba"])
    def test_measure_activation_memory(self, model_type):
        model, _, _ = self._build(model_type)
        result = measure_peak_activation_memory(model, max_tokens=256, tp_size=1)
        assert result > 0

    @pytest.mark.parametrize("model_type", ["gpt", "mamba"])
    def test_measure_with_tp_rounding(self, model_type):
        """Non-aligned token count is rounded up internally."""
        model, _, _ = self._build(model_type)
        result = measure_peak_activation_memory(model, max_tokens=1003, tp_size=8)
        assert result > 0

    @pytest.mark.parametrize("model_type", ["gpt", "mamba"])
    def test_auto_configure_from_real_model(self, model_type):
        """Auto-configure derives buffer_size_gb and max_requests from the model
        and GPU memory.  For Mamba, mamba_memory_ratio is also derived."""
        model, model_config, mamba_config = self._build(model_type)

        config = InferenceConfig(
            model_config=model_config,
            max_sequence_length=512,
            model=model,
            mamba_inference_state_config=mamba_config,
        )

        assert config.buffer_size_gb > 0
        assert config.max_requests > 0
        assert config.max_requests % 4 == 0
        if mamba_config is not None:
            assert 0 < config.mamba_memory_ratio < 1
        else:
            assert config.mamba_memory_ratio is None

    @pytest.mark.parametrize("model_type", ["gpt", "mamba"])
    def test_auto_configure_from_gpu_fraction(self, model_type):
        """Without model, gpu_memory_fraction path produces a valid config."""
        _, model_config, mamba_config = self._build(model_type)

        config = InferenceConfig(
            model_config=model_config,
            max_sequence_length=512,
            gpu_memory_fraction=0.90,
            mamba_inference_state_config=mamba_config,
        )

        assert config.buffer_size_gb > 0
        assert config.max_requests > 0

    @pytest.mark.parametrize("model_type", ["gpt", "mamba"])
    def test_explicit_budget_overrides_model(self, model_type):
        """gpu_memory_budget_gb takes precedence over model measurement."""
        model, model_config, mamba_config = self._build(model_type)

        config = InferenceConfig(
            model_config=model_config,
            max_sequence_length=512,
            model=model,
            mamba_inference_state_config=mamba_config,
            gpu_memory_budget_gb=10.0,
        )

        assert config.buffer_size_gb == pytest.approx(10.0)

    @pytest.mark.parametrize("model_type", ["gpt", "mamba"])
    def test_explicit_max_requests_validated(self, model_type):
        """Explicit max_requests succeeds with sufficient budget, fails otherwise."""
        _, model_config, mamba_config = self._build(model_type)

        config = InferenceConfig(
            model_config=model_config,
            max_sequence_length=512,
            mamba_inference_state_config=mamba_config,
            max_requests=64,
            gpu_memory_budget_gb=100.0,
        )
        assert config.max_requests == 64

        with pytest.raises(ValueError, match="requires.*GB but buffer_size_gb"):
            InferenceConfig(
                model_config=model_config,
                max_sequence_length=512,
                mamba_inference_state_config=mamba_config,
                max_requests=100000,
                gpu_memory_budget_gb=0.001,
            )
