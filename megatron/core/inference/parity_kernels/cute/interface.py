# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# Modified for Megatron: BF16 SM10x forward/combine only; no autograd API.
# Native forward dispatch and kernel arithmetic are retained.

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32
from quack.compile_utils import make_fake_tensor as fake_tensor
from torch._guards import active_fake_mode

from megatron.core.inference.parity_kernels.cute import fa_logging, utils
from megatron.core.inference.parity_kernels.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    get_block_sparse_broadcast_pattern,
    normalize_block_sparse_config,
    to_cute_block_sparse_tensors,
)
from megatron.core.inference.parity_kernels.cute.cache_utils import get_jit_cache
from megatron.core.inference.parity_kernels.cute.cute_dsl_utils import (
    get_aux_tensor_metadata,
    get_broadcast_dims,
    to_cute_aux_tensor,
    to_cute_tensor,
)
from megatron.core.inference.parity_kernels.cute.flash_fwd_combine import (
    FlashAttentionForwardCombine,
)
from megatron.core.inference.parity_kernels.cute.flash_fwd_sm100 import (
    DescaleTensors,
    FlashAttentionForwardSm100,
)

if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from megatron.core.inference.parity_kernels.cute import cute_dsl_ptxas

    cute_dsl_ptxas.patch()


def is_fake_mode() -> bool:
    """Preserve the upstream fake-tensor dispatch without its test utilities."""
    return active_fake_mode() is not None


def _parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_80', 'sm_90a', '80', '100') to int (e.g. 80, 90, 100)."""
    import re

    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch():
    """Read the device architecture or explicit upstream architecture override."""
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + int(minor)


def _validate_head_dims(
    head_dim: int, head_dim_v: int, compute_capability: int, alignment: int
) -> None:
    """Validate the retained standard Blackwell attention specialization."""
    assert compute_capability == 10
    assert all(
        8 <= dim <= 128 and dim % alignment == 0 for dim in (head_dim, head_dim_v)
    ), f"Head dimensions must be between 8 and 128 and divisible by {alignment}."


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    if not is_fake_mode():
        assert t.is_cuda, f"{name} must be on CUDA"


torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}


def num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, max_splits):
    # If num_n_blocks is too small, use 1 split. For example, we never split for hdim = 128 and seqlen_k = 512.
    if num_n_blocks <= 4:
        return 1
    # Avoid ZeroDivisionError when batch_size or seqlen_q is 0. The empty-Q
    # early-exit in _flash_attn_fwd handles correctness for those shapes; this
    # guard just keeps the heuristic safe if called in other contexts.
    if total_mblocks == 0:
        return 1

    # NOTE: We should revisit this heuristic after persistence is supported for split KV.
    # Sometimes, it's ideal to over-schedule splits for better efficiency.
    return min(num_SMs // total_mblocks, max_splits, num_n_blocks)


def _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod=None):
    """Resolve causal/local/window settings into canonical form.

    Returns (causal, local, window_size_left, window_size_right).
    """
    if mask_mod is not None:
        return False, False, window_size_left, window_size_right
    if causal:
        window_size_right = 0
    if (
        window_size_left is not None
        and window_size_right is not None
        and window_size_left + window_size_right < 0
    ):
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


def _flash_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    dynamic_causal: Optional[torch.Tensor] = None,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    tile_mn: Optional[Tuple[int, int]] = None,
    mma_pv_is_rs: Optional[bool] = None,
    intra_wg_overlap: Optional[bool] = None,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _arch: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
    compile_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
            The returned LSE supports taking gradient.
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
            FP8 (e4m3fn) dtype is selected automatically when `output_scale` is set.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
        output_scale: 0-d FP32 GPU tensor. Presence opts into the static per-tensor
            FP8 (e4m3fn) fused-quant output: the kernel writes FP8 with
            dequant = out_fp8 * output_scale. SM100/SM110 only.
        compile_only: If True, compile the selected kernel and return without
            launching it.
    """
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("Local parity attention requires BF16 inputs")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("Local parity attention is inference-only")
    if qv is not None or q.shape[-1] > 128 or v.shape[-1] > 128 or compile_only:
        raise ValueError("Local parity attention requires head dimensions <=128 without MLA")
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    q_descale, k_descale, v_descale = [
        maybe_contiguous(t) for t in (q_descale, k_descale, v_descale)
    ]
    num_head, head_dim = q.shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q.shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q.shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = k.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = k.shape[-3]
    num_head_kv = k.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (
            batch_size + 1,
        ), "cu_seqlens_k must have shape (batch_size + 1,)"

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (
            batch_size + 1,
        ), "cu_seqlens_q must have shape (batch_size + 1,)"
    assert seqused_q is None or seqused_q.shape == (
        batch_size,
    ), "seqused_q must have shape (batch_size,)"
    assert seqused_k is None or seqused_k.shape == (
        batch_size,
    ), "seqused_k must have shape (batch_size,)"
    assert q.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
    ], "inputs must be float16, bfloat16, fp8 e4m3fn, or fp8 e5m2"
    assert q.dtype == k.dtype == v.dtype, "inputs must have the same dtype"
    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert (
                t.dtype == torch.int32
            ), "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            assert (
                t.stride(0) == 1
            ), "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    if not is_fake_mode():
        assert all(
            t is None or t.is_cuda
            for t in (
                q,
                k,
                v,
                q_descale,
                k_descale,
                v_descale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                learnable_sink,
                output_scale,
            )
        ), "inputs must be on CUDA device"
    arch = _get_device_arch() if _arch is None else _arch
    if arch // 10 != 10:
        raise ValueError("Local parity attention requires Blackwell SM10x")
    if dynamic_causal is not None:
        assert arch // 10 == 9, "dynamic_causal is only supported on SM90 (Hopper)."
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    alignment = 16 // q.element_size()
    _validate_head_dims(head_dim, head_dim_v, arch // 10, alignment)
    if softmax_scale is None:
        softmax_scale = (
            1.0 / math.sqrt(head_dim) if qv is None else 1.0 / math.sqrt(head_dim + head_dim_v)
        )
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    is_fp8 = q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    if is_fp8 and (q.requires_grad or k.requires_grad or v.requires_grad):
        raise NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only).")
    if output_scale is not None:
        assert output_scale.dtype == torch.float32, "output_scale must be float32"
        assert output_scale.numel() == 1, "output_scale must be a scalar (numel == 1) tensor"
        assert output_scale.device == q.device, "output_scale must be on the same device as q"
        assert block_sparse_tensors is None, "fused FP8 output + block sparsity not supported yet"
        assert page_table is None, "fused FP8 output + paged KV not supported yet"
        out_torch_dtype = torch.float8_e4m3fn
        # Derived output quant key for use as tag in compile_key.
        # The tag values are inspired by vLLM. More support will be added for keys:
        # kFp8Dynamic128Sym, kFp8Dynamic64Sym, kNvfp4Dynamic
        output_quant_key = "kFp8StaticTensorSym"
        output_scale = output_scale.reshape(1)
    else:
        out_torch_dtype = torch.bfloat16 if is_fp8 else q.dtype
        output_quant_key = None
    device = q.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)
    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    requires_grad = q.requires_grad or k.requires_grad or v.requires_grad

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )
    else:
        _validate_tensor(
            out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v), out_torch_dtype, device
        )

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    if seqlen_k == 0 or total_q == 0:
        out.zero_()
        if lse is not None:
            lse.fill_(float("-inf"))
        return out, lse

    if is_fp8:
        for t, name in (
            (q_descale, "q_descale"),
            (k_descale, "k_descale"),
            (v_descale, "v_descale"),
        ):
            if t is not None:
                _validate_tensor(t, name, (batch_size, num_head_kv), torch.float32, device)
    else:
        assert (
            q_descale is None and k_descale is None and v_descale is None
        ), "q_descale/k_descale/v_descale are only supported for FP8 inputs"

    dtype = torch2cute_dtype_map[q.dtype]
    if is_fp8:
        assert (
            arch // 10 == 10
        ), "FP8 is only supported on SM100 (compute capability 10.x) for FA4 CuTe."
    use_block_sparsity = block_sparse_tensors is not None

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right, mask_mod
    )

    requested_use_clc_scheduler = utils._get_use_clc_scheduler_default()
    requested_disable_2cta = utils._get_disable_2cta_default(is_fwd=True)

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    fwd_cfg = FwdConfig(128, 128, True, True)  # default
    if tile_mn is not None:
        fwd_cfg = FwdConfig(tile_mn[0], tile_mn[1], fwd_cfg.mma_pv_is_rs, fwd_cfg.intra_wg_overlap)
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    if mma_pv_is_rs is None:
        mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
    if intra_wg_overlap is None:
        intra_wg_overlap = fwd_cfg.intra_wg_overlap

    # TODO: fix GQA + SplitKV + non-varlen
    if pack_gqa and num_splits != 1 and cu_seqlens_q is None:
        pack_gqa = False

    if pack_gqa and qv is not None and 128 % qhead_per_kvhead != 0:
        pack_gqa = False

    if max_seqlen_q is None:
        max_seqlen_q = seqlen_q if cu_seqlens_q is None else total_q
    if max_seqlen_k is None:
        max_seqlen_k = seqlen_k
    if cu_seqlens_k is None and seqused_k is None:
        min_seqlen_k = seqlen_k
    seqlen_q_packgqa = max_seqlen_q * qhead_per_kvhead
    q_stage = 2 if seqlen_q_packgqa > tile_m else 1

    m_block_size_effective = q_stage * tile_m
    if local:
        win_right = max_seqlen_k if window_size_right is None else window_size_right
        win_left = max_seqlen_k if window_size_left is None else window_size_left
        seqlen_k_loaded = max(0, min(max_seqlen_k, win_right + win_left + 1 + tile_m))
    else:
        seqlen_k_loaded = max_seqlen_k
    num_m_blocks = (seqlen_q_packgqa + m_block_size_effective - 1) // m_block_size_effective
    total_mblocks = batch_size * num_head_kv * num_m_blocks
    num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
    num_SMs = (
        132 if is_fake_mode() else torch.cuda.get_device_properties(device).multi_processor_count
    )
    if num_splits < 1:
        num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)

    # SplitKV uses float32 partial output, which doubles the O buffer size
    # in shared memory, causing OOM for diff-headdim (192, 128)
    if arch // 10 in [10, 11] and head_dim != head_dim_v and num_splits > 1:
        if num_n_blocks >= 64 and head_dim_v != 512:
            tile_n = 64
            num_n_blocks = (seqlen_k_loaded + tile_n - 1) // tile_n
            num_splits = num_splits_heuristic(total_mblocks, num_SMs, num_n_blocks, 128)
        else:
            num_splits = 1

    is_split_kv = num_splits > 1
    if is_split_kv:
        out_partial = torch.empty(
            num_splits,
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=torch.float32,
            device=device,
        )
        lse_partial = torch.empty(num_splits, *lse_shape, dtype=torch.float32, device=device)

    use_2cta_instrs = (
        arch // 10 in [10, 11]
        and not requested_disable_2cta
        and not causal
        and not local
        and not is_split_kv
        and cu_seqlens_q is None
        and seqused_q is None
        and not use_block_sparsity
        and page_size in [None, 128]
        and int(math.ceil(head_dim / 16) * 16) in [128, 192]
        and int(math.ceil(head_dim_v / 16) * 16) == 128
        and seqlen_q_packgqa > 2 * tile_m
        and (tile_m % qhead_per_kvhead == 0 or not pack_gqa)
    )

    # hd=256 2CTA forward uses dedicated kernel (Blackwell family)
    use_dedicated_hd256_kernel = arch // 10 in [10, 11] and head_dim == 256 and head_dim_v == 256
    use_2cta_instrs = use_2cta_instrs or use_dedicated_hd256_kernel

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)

    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    is_varlen = (
        cu_seqlens_q is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or seqused_k is not None
    )

    # CLC regressed for varlen MHA and dense noncausal. Imbalanced varlen shapes
    # keep more K/V blocks in flight and hurt L2; dense noncausal mostly just
    # pays work-stealing overhead.
    is_varlen_mha = is_varlen and qhead_per_kvhead == 1
    is_dense_noncausal = not is_varlen and not causal and not local
    use_clc_scheduler = requested_use_clc_scheduler and not is_varlen_mha and not is_dense_noncausal

    if use_block_sparsity:
        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)
        head_dim_idx = 0 if block_sparse_tensors.mask_block_cnt.ndim == 2 else 1
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[head_dim_idx] != 1:
            pack_gqa = False
        if cu_seqlens_q is not None:
            assert (
                block_sparse_tensors.cu_total_m_blocks is not None
            ), "Varlen block sparsity requires block_sparse_tensors.cu_total_m_blocks."

    # See get_broadcast_dims for why this is needed in compile key
    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = None
    if block_sparse_tensors is not None:
        normalized_block_sparse_tensors, block_sparse_broadcast_pattern, q_subtile_factor = (
            normalize_block_sparse_config(
                block_sparse_tensors,
                batch_size=batch_size,
                num_head=num_head,
                seqlen_q=seqlen_q,
                seqlen_k=seqlen_k,
                block_size=(tile_m, tile_n),
                q_stage=q_stage,
            )
        )
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None

    assert gather_kv_indices is None, 'gather_kv_indices is only supported with qv'
    gather_kv_length = None
    sparse_kv = None
    disable_sparse_kv_bitmask = None

    compile_key = (
        dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        q_descale is not None,
        k_descale is not None,
        v_descale is not None,
        block_sparse_tensors is None or block_sparse_tensors.cu_total_m_blocks is None,
        block_sparse_tensors is None or block_sparse_tensors.cu_block_idx_offsets is None,
        tile_m,
        tile_n,
        q_stage,
        num_threads,
        is_split_kv,
        pack_gqa,
        arch,
        page_size not in [None, tile_n],  # paged KV non-TMA
        use_2cta_instrs,
        q_subtile_factor,
        mma_pv_is_rs,
        intra_wg_overlap,
        use_clc_scheduler,
        qv is not None,
        gather_kv_length,
        sparse_kv,
        disable_sparse_kv_bitmask,
        fa_logging.get_fa_log_level(),
        output_quant_key,
    )
    if compile_key not in _flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
            output_scale_tensor,  # 1d scalar tensor
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0) if t is not None else None
            for t in (
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                learnable_sink,
                output_scale,
            )
        ]
        dynamic_causal_tensor = (
            to_cute_tensor(dynamic_causal, assumed_align=4, leading_dim=0)
            if dynamic_causal is not None
            else None
        )
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out if not is_split_kv else out_partial)
        ]
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        elif lse is not None:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)
        else:
            lse_tensor = None

        q_descale_tensor = (
            to_cute_tensor(q_descale, assumed_align=4, leading_dim=1)
            if q_descale is not None
            else None
        )
        k_descale_tensor = (
            to_cute_tensor(k_descale, assumed_align=4, leading_dim=1)
            if k_descale is not None
            else None
        )
        v_descale_tensor = (
            to_cute_tensor(v_descale, assumed_align=4, leading_dim=1)
            if v_descale is not None
            else None
        )
        descale_tensors_tensor = (
            DescaleTensors(
                q_descale=q_descale_tensor, k_descale=k_descale_tensor, v_descale=v_descale_tensor
            )
            if q_descale_tensor is not None
            or k_descale_tensor is not None
            or v_descale_tensor is not None
            else None
        )

        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        cute_aux_tensors = None
        aux_tensor_metadata = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        qv_tensor = to_cute_tensor(qv) if qv is not None else None
        gather_kv_indices_tensor = (
            to_cute_tensor(gather_kv_indices) if gather_kv_indices is not None else None
        )

        if output_quant_key is not None:
            assert qv is None, 'fused FP8 output + MLA (qv) not supported yet'
            assert (
                not use_dedicated_hd256_kernel
            ), 'fused FP8 output + head_dim=256 kernel not supported yet'
        fa_fwd = FlashAttentionForwardSm100(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=causal,
            is_local=local,
            is_split_kv=is_split_kv,
            pack_gqa=pack_gqa,
            m_block_size=tile_m,
            n_block_size=tile_n,
            q_stage=q_stage,
            is_persistent=not causal
            and (not local)
            and (cu_seqlens_q is None)
            and (seqused_q is None)
            and (not is_split_kv),
            score_mod=score_mod,
            mask_mod=mask_mod,
            has_aux_tensors=aux_tensors is not None,
            paged_kv_non_tma=page_size not in [None, tile_n],
            is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
            q_subtile_factor=q_subtile_factor,
            use_2cta_instrs=use_2cta_instrs,
            use_clc_scheduler=use_clc_scheduler,
            output_quant_key=output_quant_key if not is_split_kv else None,
        )
        # TODO: check @can_implement
        compile_args = [
            fa_fwd,
            q_tensor,
            k_tensor,
            v_tensor,
            o_tensor,
            lse_tensor,
            softmax_scale,
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            dynamic_causal_tensor,
            page_table_tensor,
            window_size_left,
            window_size_right,
            learnable_sink_tensor,
        ]
        compile_args.append(descale_tensors_tensor)
        compile_args.extend([sparse_tensors, cute_aux_tensors])
        compile_args.append(output_scale_tensor)
        compile_args.append(current_stream)
        _flash_attn_fwd.compile_cache[compile_key] = cute.compile(
            *compile_args, options='--enable-tvm-ffi'
        )

    if compile_only:
        return out, lse

    if not is_fake_mode():
        q_call, k_call, v_call = q.detach(), k.detach(), v.detach()
        qv_call = qv.detach() if qv is not None else None
        if is_fp8:
            # need uint8 workaround until we pin torch >= 2.11.0 where fp8 export is supported
            q_call = q_call.view(torch.uint8)
            k_call = k_call.view(torch.uint8)
            v_call = v_call.view(torch.uint8)
            if qv_call is not None:
                qv_call = qv_call.view(torch.uint8)
        out_call = out.detach()
        if out_call.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            out_call = out_call.view(torch.uint8)
        descale_tensors = (
            DescaleTensors(q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
            if q_descale is not None or k_descale is not None or v_descale is not None
            else None
        )
        call_args = [
            q_call,
            k_call,
            v_call,
            out_call if not is_split_kv else out_partial,
            lse_partial if is_split_kv else lse,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            dynamic_causal,
            page_table,
            window_size_left,
            window_size_right,
            learnable_sink,
        ]
        call_args.append(descale_tensors)
        call_args.extend(
            [
                (
                    (
                        normalized_block_sparse_tensors.mask_block_cnt,
                        normalized_block_sparse_tensors.mask_block_idx,
                        normalized_block_sparse_tensors.full_block_cnt,
                        normalized_block_sparse_tensors.full_block_idx,
                        normalized_block_sparse_tensors.cu_total_m_blocks,
                        normalized_block_sparse_tensors.cu_block_idx_offsets,
                        normalized_block_sparse_tensors.dq_write_order,
                        normalized_block_sparse_tensors.dq_write_order_full,
                    )
                    if normalized_block_sparse_tensors is not None
                    else None
                ),
                aux_tensors,
            ]
        )
        call_args.append(output_scale)
        _flash_attn_fwd.compile_cache[compile_key](*call_args)
    if is_split_kv:
        _flash_attn_fwd_combine(
            out_partial,
            lse_partial.transpose(-1, -2),
            out,
            lse.transpose(-1, -2) if lse is not None else None,
            cu_seqlens_q,
            seqused_q,
            output_scale=output_scale,
        )
    return out, lse


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")


def _compile_fwd_combine(
    dtype,
    dtype_partial,
    head_dim,
    tile_m,
    k_block_size,
    log_max_splits,
    has_cu_seqlens,
    has_seqused,
    has_lse,
    has_varlen_batch_idx,
    output_quant_key,
):
    """Compile fwd combine kernel using cute fake tensors (no real GPU tensors needed)."""
    sym = cute.sym_int
    div = 128 // dtype_partial.width  # 16-byte alignment in elements

    fa_combine = FlashAttentionForwardCombine(
        dtype=dtype,
        dtype_partial=dtype_partial,
        head_dim=head_dim,
        tile_m=tile_m,
        k_block_size=k_block_size,
        log_max_splits=log_max_splits,
        output_quant_key=output_quant_key,
    )
    if not fa_combine.can_implement(
        dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits, num_threads=256
    ):
        raise RuntimeError(
            "FlashAttention combine kernel cannot be implemented with given parameters"
        )

    if has_cu_seqlens:
        # Varlen: (num_splits, total_q, nheads, headdim)
        num_splits, total_q, nheads = sym(), sym(), sym()
        mO_partial = fake_tensor(
            dtype_partial, (num_splits, total_q, nheads, head_dim), divisibility=div
        )
        mLSE_partial = fake_tensor(
            Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=1
        )
        mO = fake_tensor(dtype, (total_q, nheads, head_dim), divisibility=div)
        mLSE = (
            fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=0)
            if has_lse
            else None
        )
    else:
        # Batched: (num_splits, batch, seqlen, nheads, headdim)
        num_splits, batch, seqlen, nheads = sym(), sym(), sym(), sym()
        mO_partial = fake_tensor(
            dtype_partial, (num_splits, batch, seqlen, nheads, head_dim), divisibility=div
        )
        mLSE_partial = fake_tensor(
            Float32, (num_splits, batch, seqlen, nheads), divisibility=1, leading_dim=2
        )
        mO = fake_tensor(dtype, (batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE = (
            fake_tensor(Float32, (batch, seqlen, nheads), divisibility=1, leading_dim=1)
            if has_lse
            else None
        )
        batch = mO_partial.shape[1]

    batch_for_1d = batch if not has_cu_seqlens else sym()
    batchp1 = sym()
    mCuSeqlens = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cu_seqlens else None
    mSeqused = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_seqused else None
    mNumSplitsDynamic = None  # Not parametrized in compile_key
    mVarlenBatchIdx = (
        fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_varlen_batch_idx else None
    )
    mSemaphore = None  # Not parametrized in compile_key
    output_scale = (
        fake_tensor(Float32, (1,), divisibility=1) if output_quant_key is not None else None
    )

    return cute.compile(
        fa_combine,
        mO_partial,
        mLSE_partial,
        mO,
        mLSE,
        mCuSeqlens,
        mSeqused,
        mNumSplitsDynamic,
        mVarlenBatchIdx,
        mSemaphore,
        output_scale,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
    output_scale: Optional[torch.Tensor] = None,
) -> None:
    """Forward combine kernel for split attention computation.

    Combines partial outputs and log-sum-exp values from multiple splits
    of attention computation into final outputs.

    Args:
        out_partial: Partial outputs tensor (num_splits, batch, seqlen, nheads, headdim) or
                                            (num_splits, total_q, nheads, headdim) if there's cu_seqlens
        lse_partial: Partial LSE tensor (num_splits, batch, seqlen, nheads) or
                                       (num_splits, total_q, nheads) if there's cu_seqlens
        out: Output tensor (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim) if there's cu_seqlens
        lse: Output LSE tensor (batch, seqlen, nheads) or (total_q, nheads) if there's cu_seqlens.
        cu_seqlens: Cumulative sequence lengths for variable length sequences
        seqused: Used sequence lengths for each batch
        num_splits_dynamic_ptr: Dynamic number of splits per batch
        semaphore_to_reset: Semaphore for synchronization
        k_block_size: Block size for head dimension

    Returns:
        None
    """
    assert out_partial.dtype in [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ], "out_partial must be fp16, bf16, or fp32"

    if output_scale is not None:
        # Derived output quant key for use in compile_key and const_expr.
        output_quant_key = "kFp8StaticTensorSym"
    else:
        output_quant_key = None

    if not is_fake_mode():
        assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    # Determine if this is variable length based on dimensions
    is_varlen = out_partial.dim() == 4
    # Validate optional tensors
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            if not is_fake_mode():
                assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"
    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    # If hdim is 96 or 192, it's faster to round them to 128 or 256 respectively
    # so that kBlockM is smaller and we have more parallelism.
    k_block_size = 64 if head_dim <= 64 else 128
    # We want kBlockM to be as small as possible to maximize parallelism.
    # E.g., if hdim is 64, we want kBlockM to be 16 so that we can use 256 threads, each reading 4 elements (floats).
    tile_m = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if tile_m == 8:
        # If kBlockM == 8 then the minimum number of splits is 32.
        # TODO: we can deal w this by using 128 threads instead
        log_max_splits = max(log_max_splits, 5)

    # Create combine kernel configuration
    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]
    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
        varlen_batch_idx is not None,
        output_quant_key,
    )
    if compile_key not in _flash_attn_fwd_combine.compile_cache:
        _flash_attn_fwd_combine.compile_cache[compile_key] = _compile_fwd_combine(*compile_key)
    if not is_fake_mode():
        _flash_attn_fwd_combine.compile_cache[compile_key](
            out_partial,
            lse_partial,
            out,
            lse,
            cu_seqlens,
            seqused,
            num_splits_dynamic_ptr,
            varlen_batch_idx,
            semaphore_to_reset,
            output_scale,
        )


_flash_attn_fwd_combine.compile_cache = get_jit_cache("fwd_combine")
