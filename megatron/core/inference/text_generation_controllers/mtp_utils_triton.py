# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:
    from unittest.mock import MagicMock

    from megatron.core.utils import null_decorator

    triton = MagicMock()
    triton.jit = null_decorator
    tl = MagicMock()
    HAVE_TRITON = False


# ---------------------------------------------------------------------------
# Kernel 1: KV-cache rewind for speculative decoding
# ---------------------------------------------------------------------------
@triton.jit
def _rewind_kv_cache_kernel(
    # Per-request input (read-only)
    ACCEPTED_COUNTS_PTR,
    PREFILL_STATUS_PTR,
    # Per-request state (read-write, updated in-place)
    LAST_KV_BLOCK_OFFSET_PTR,
    KV_LENGTH_OFFSETS_PTR,
    KV_BLOCK_COUNTS_PTR,
    LAST_KV_BLOCK_ID_PTR,
    # 2-D table [N, max_blocks] (read-write)
    KV_BLOCK_IDS_PTR,
    # Per-request outputs
    BLOCKS_TO_RELEASE_PTR,
    REMOVE_MASK_PTR,
    # Strides / limits
    kv_block_ids_stride,
    max_blocks_minus_1,
    num_active_requests,
    # Compile-time constants
    NUM_SPEC_TOKENS: tl.constexpr,
    BLOCK_SIZE_TOKENS: tl.constexpr,
):
    """Rewind KV-cache bookkeeping for one request after speculative verification.

    Grid: may be padded beyond active requests for CUDA-graph compatibility.
    Each program handles exactly one request.  Programs with
    `pid >= num_active_requests` are padding and produce safe no-op outputs.
    """
    pid = tl.program_id(0)

    # Padding programs: write safe defaults and skip all state mutation.
    if pid >= num_active_requests:
        tl.store(BLOCKS_TO_RELEASE_PTR + pid, 0)
        tl.store(REMOVE_MASK_PTR + pid, False)
        return

    # --- Load per-request scalars ---
    accepted = tl.load(ACCEPTED_COUNTS_PTR + pid)
    prefill = tl.load(PREFILL_STATUS_PTR + pid)
    last_offset = tl.load(LAST_KV_BLOCK_OFFSET_PTR + pid)
    kv_length = tl.load(KV_LENGTH_OFFSETS_PTR + pid)
    block_count = tl.load(KV_BLOCK_COUNTS_PTR + pid)
    last_block_id = tl.load(LAST_KV_BLOCK_ID_PTR + pid)

    # --- Compute rewind (zero for prefill requests) ---
    num_to_rewind = tl.where(prefill == 1, 0, NUM_SPEC_TOKENS - accepted)
    diff = last_offset - num_to_rewind
    remove = diff < 0

    # Python-style modulo: ((diff % M) + M) % M  to handle negative diff
    new_offset = ((diff % BLOCK_SIZE_TOKENS) + BLOCK_SIZE_TOKENS) % BLOCK_SIZE_TOKENS
    tl.store(LAST_KV_BLOCK_OFFSET_PTR + pid, new_offset)
    tl.store(KV_LENGTH_OFFSETS_PTR + pid, kv_length - num_to_rewind)

    # Save current last block id (will be released by caller if remove is True)
    tl.store(BLOCKS_TO_RELEASE_PTR + pid, last_block_id)

    # Decrement block count when a block boundary was crossed
    new_block_count = tl.where(remove, block_count - 1, block_count)
    tl.store(KV_BLOCK_COUNTS_PTR + pid, new_block_count)

    # Gather previous block id from the 2-D table
    kv_row_base = pid.to(tl.int64) * kv_block_ids_stride
    prev_idx = tl.maximum(new_block_count - 1, 0)
    prev_block_id = tl.load(KV_BLOCK_IDS_PTR + kv_row_base + prev_idx)

    # Conditionally update last block id
    tl.store(LAST_KV_BLOCK_ID_PTR + pid, tl.where(remove, prev_block_id, last_block_id))

    # Clear released block entry via scatter
    scatter_idx = tl.minimum(new_block_count, max_blocks_minus_1)
    current_val = tl.load(KV_BLOCK_IDS_PTR + kv_row_base + scatter_idx)
    tl.store(KV_BLOCK_IDS_PTR + kv_row_base + scatter_idx, tl.where(remove, -1, current_val))

    # Output remove mask for the caller (to release blocks outside this kernel)
    tl.store(REMOVE_MASK_PTR + pid, remove)


def rewind_kv_cache(
    accepted_counts,
    prefill_status,
    last_kv_block_offset,
    kv_length_offsets,
    kv_block_counts,
    last_kv_block_id,
    kv_block_ids,
    num_speculative_tokens,
    block_size_tokens,
    num_active_requests=None,
):
    """Launch the KV-cache rewind Triton kernel.

    Args:
        num_active_requests: Number of real (non-padding) requests. When the
            grid is padded beyond this count, the kernel skips padding
            programs so stale data in padding slots cannot corrupt
            bookkeeping.  Defaults to `accepted_counts.shape[0]` (no
            padding).

    Returns:
        (blocks_to_release, remove_mask) — same semantics as the original
        torch.compile'd `_rewind_kv_cache` (KV-cache portion only; Mamba
        state updates are handled separately by the caller).
    """
    N = accepted_counts.shape[0]
    if num_active_requests is None:
        num_active_requests = N
    if N == 0:
        return (
            torch.empty(0, device=accepted_counts.device, dtype=last_kv_block_id.dtype),
            torch.empty(0, device=accepted_counts.device, dtype=torch.bool),
        )

    blocks_to_release = torch.empty_like(last_kv_block_id)
    remove_mask = torch.empty(N, device=accepted_counts.device, dtype=torch.bool)

    _rewind_kv_cache_kernel[(N,)](
        accepted_counts,
        prefill_status,
        last_kv_block_offset,
        kv_length_offsets,
        kv_block_counts,
        last_kv_block_id,
        kv_block_ids,
        blocks_to_release,
        remove_mask,
        kv_block_ids_stride=kv_block_ids.stride(0),
        max_blocks_minus_1=kv_block_ids.shape[1] - 1,
        num_active_requests=num_active_requests,
        NUM_SPEC_TOKENS=num_speculative_tokens,
        BLOCK_SIZE_TOKENS=block_size_tokens,
    )
    return blocks_to_release, remove_mask


# ---------------------------------------------------------------------------
# Kernel 2: Verify speculative tokens
# ---------------------------------------------------------------------------
@triton.jit
def _verify_speculative_tokens_kernel(
    INPUT_TOKENS_PTR,
    OUTPUT_TOKENS_PTR,
    # Outputs
    ACCEPTED_MASK_PTR,
    LAST_ONE_INDICES_PTR,
    # Runtime scalars
    num_decode_requests,
    decode_len,
    # Compile-time constants
    STRIDE: tl.constexpr,  # num_speculative_tokens + 1
    BLOCK_SIZE: tl.constexpr,  # next_power_of_2(STRIDE)
):
    """Verify speculative tokens for one request.

    Grid: (active_request_count,)
    Programs 0..num_decode_requests-1 handle decode requests.
    Programs num_decode_requests..end handle prefill requests.
    """
    pid = tl.program_id(0)

    if pid < num_decode_requests:
        base = pid * STRIDE
        offsets = tl.arange(0, BLOCK_SIZE)
        valid = offsets < STRIDE

        input_toks = tl.load(INPUT_TOKENS_PTR + base + offsets, mask=valid, other=0)

        # Build shifted output: shifted[i] = output[i-1].
        # Position 0 uses a dummy load (always accepted regardless).
        safe_shifted = tl.where(offsets > 0, offsets - 1, 0)
        shifted_output = tl.load(OUTPUT_TOKENS_PTR + base + safe_shifted, mask=valid, other=0)

        # First token is always accepted; rest must match shifted output.
        match = tl.where(offsets == 0, 1, (input_toks == shifted_output).to(tl.int32))
        match = tl.where(valid, match, 0)

        # Consecutive acceptance via cumulative-sum trick:
        # accepted[i] iff cumsum(match)[i] == i + 1
        cumsum = tl.cumsum(match, axis=0)
        accepted = (cumsum == (offsets + 1)) & valid

        tl.store(ACCEPTED_MASK_PTR + base + offsets, accepted, mask=valid)

        accepted_count = tl.sum(accepted.to(tl.int32))
        tl.store(LAST_ONE_INDICES_PTR + pid, (base + accepted_count - 1).to(tl.int64))
    else:
        # Prefill request — single token, always accepted
        prefill_idx = decode_len + (pid - num_decode_requests)
        tl.store(ACCEPTED_MASK_PTR + prefill_idx, 1)
        tl.store(LAST_ONE_INDICES_PTR + pid, prefill_idx.to(tl.int64))


def verify_speculative_tokens(
    input_tokens, output_tokens, num_decode_requests, num_prefill_requests, num_speculative_tokens
):
    """Launch the speculative-token verification Triton kernel.

    Returns:
        (last_one_indices, accepted_tokens_mask, input_tokens)
        matching the original `_verify_speculative_tokens` signature.
    """
    if input_tokens.ndim == 2:
        input_tokens = input_tokens.squeeze(0)

    device = input_tokens.device
    active_request_count = num_decode_requests + num_prefill_requests
    stride = num_speculative_tokens + 1
    decode_len = num_decode_requests * stride

    accepted_tokens_mask = torch.zeros_like(input_tokens, dtype=torch.bool)
    last_one_indices = torch.full((active_request_count,), -1, device=device, dtype=torch.long)

    if active_request_count > 0:
        block_size = triton.next_power_of_2(stride)
        _verify_speculative_tokens_kernel[(active_request_count,)](
            input_tokens,
            output_tokens,
            accepted_tokens_mask,
            last_one_indices,
            num_decode_requests=num_decode_requests,
            decode_len=decode_len,
            STRIDE=stride,
            BLOCK_SIZE=block_size,
        )

    return last_one_indices, accepted_tokens_mask, input_tokens


# ---------------------------------------------------------------------------
# Kernel 3: Prepare speculative tokens for next forward pass
# ---------------------------------------------------------------------------
@triton.jit
def _prepare_next_forward_pass_kernel(
    OUTPUT_TOKENS_PTR,
    REQUIRED_LOGIT_INDICES_PTR,
    LAST_ONE_INDICES_PTR,
    INPUT_TOKENS_PTR,
    ACCEPTED_MASK_PTR,
    # Outputs
    SAMPLED_TOKENS_OUT_PTR,
    LAST_ACCEPTED_SEQ_OUT_PTR,
    ACCEPTED_TOKENS_OUT_PTR,
    ACCEPTED_COUNTS_OUT_PTR,
    # Strides
    accepted_tokens_out_stride,
    # Runtime scalars
    num_decode_requests,
    # Compile-time constants
    STRIDE: tl.constexpr,  # num_speculative_tokens + 1
    NUM_SPEC_TOKENS: tl.constexpr,
    SPEC_BLOCK_SIZE: tl.constexpr,  # next_power_of_2(NUM_SPEC_TOKENS)
):
    """Gather final tokens and extract accepted speculative tokens per request.

    Grid: (active_request_count,)
    """
    pid = tl.program_id(0)

    # --- Gather final sampled token and sequence index for every request ---
    idx = tl.load(LAST_ONE_INDICES_PTR + pid)
    tl.store(SAMPLED_TOKENS_OUT_PTR + pid, tl.load(OUTPUT_TOKENS_PTR + idx))
    tl.store(LAST_ACCEPTED_SEQ_OUT_PTR + pid, tl.load(REQUIRED_LOGIT_INDICES_PTR + idx))

    # --- For decode requests: extract accepted tokens and count ---
    if pid < num_decode_requests:
        base = pid * STRIDE
        spec_offsets = tl.arange(0, SPEC_BLOCK_SIZE)
        spec_valid = spec_offsets < NUM_SPEC_TOKENS
        token_positions = base + 1 + spec_offsets  # skip first (base) token

        tokens = tl.load(INPUT_TOKENS_PTR + token_positions, mask=spec_valid, other=0)
        mask_val = tl.load(ACCEPTED_MASK_PTR + token_positions, mask=spec_valid, other=0)
        accepted = mask_val != 0

        result = tl.where(accepted & spec_valid, tokens, -1)

        out_base = pid.to(tl.int64) * accepted_tokens_out_stride
        tl.store(ACCEPTED_TOKENS_OUT_PTR + out_base + spec_offsets, result, mask=spec_valid)

        count = tl.sum((accepted & spec_valid).to(tl.int64))
        tl.store(ACCEPTED_COUNTS_OUT_PTR + pid, count)
    else:
        spec_offsets = tl.arange(0, SPEC_BLOCK_SIZE)
        spec_valid = spec_offsets < NUM_SPEC_TOKENS
        out_base = pid.to(tl.int64) * accepted_tokens_out_stride
        tl.store(ACCEPTED_TOKENS_OUT_PTR + out_base + spec_offsets, -1, mask=spec_valid)
        tl.store(ACCEPTED_COUNTS_OUT_PTR + pid, 0)


def prepare_next_forward_pass(
    num_decode_requests,
    output_tokens,
    required_logit_indices,
    last_one_indices,
    accepted_tokens_mask,
    input_tokens,
    sampled_tokens_buf,
    last_accepted_seq_buf,
    accepted_tokens_per_request,
    accepted_token_counts,
    num_speculative_tokens,
):
    """Launch the prepare-next-forward-pass Triton kernel.

    Writes results into the pre-allocated buffers provided by the caller.
    """
    active_request_count = last_one_indices.shape[0]
    if active_request_count == 0:
        return

    stride = num_speculative_tokens + 1
    spec_block_size = triton.next_power_of_2(num_speculative_tokens)

    _prepare_next_forward_pass_kernel[(active_request_count,)](
        output_tokens,
        required_logit_indices,
        last_one_indices,
        input_tokens,
        accepted_tokens_mask,
        sampled_tokens_buf,
        last_accepted_seq_buf,
        accepted_tokens_per_request,
        accepted_token_counts,
        accepted_tokens_out_stride=accepted_tokens_per_request.stride(0),
        num_decode_requests=num_decode_requests,
        STRIDE=stride,
        NUM_SPEC_TOKENS=num_speculative_tokens,
        SPEC_BLOCK_SIZE=spec_block_size,
    )


# ---------------------------------------------------------------------------
# Kernel 3b: Verify + prepare-next in one launch, reading tokens through the row maps
# ---------------------------------------------------------------------------
@triton.jit
def _verify_and_prepare_next_kernel(
    # Sampled tokens (any integer width), one per required row.
    OUTPUT_TOKENS_PTR,
    # Required row -> packed input row (`speculative_required_logit_indices`).
    REQUIRED_ROWS_PTR,
    # Optional packed input row -> original input-storage row (async survivor compaction).
    TOKEN_ROW_PTR,
    # Input-token storage, indexed by (mapped) input row.
    INPUT_IDS_PTR,
    # Optional position ids, indexed like INPUT_IDS_PTR.
    POS_IDS_PTR,
    # Outputs
    SAMPLED_TOKENS_OUT_PTR,
    LAST_ACCEPTED_SEQ_OUT_PTR,
    ACCEPTED_TOKENS_OUT_PTR,
    ACCEPTED_COUNTS_OUT_PTR,
    BASE_POSITION_OUT_PTR,
    # Strides
    accepted_tokens_out_stride,
    # Runtime scalars
    num_decode_requests,
    decode_len,
    # Compile-time constants
    STRIDE: tl.constexpr,  # num_speculative_tokens + 1
    NUM_SPEC_TOKENS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # next_power_of_2(STRIDE)
    HAS_TOKEN_ROW: tl.constexpr,
    HAS_POS_IDS: tl.constexpr,
):
    """Verify one request's drafts and write its next-step state.

    Grid: (active_request_count,). Fuses `_verify_speculative_tokens_kernel` and
    `_prepare_next_forward_pass_kernel`: both work per request on that request's own rows, so the
    accepted mask and last-accepted index stay in registers. The input tokens are read through
    the row maps rather than from a pre-gathered copy, and the next draft's base position
    (`pos_ids[last accepted row] + 1`) is written alongside.
    """
    pid = tl.program_id(0)
    spec_offsets = tl.arange(0, BLOCK_SIZE)
    spec_valid = spec_offsets < NUM_SPEC_TOKENS
    out_base = pid.to(tl.int64) * accepted_tokens_out_stride

    if pid < num_decode_requests:
        base = pid * STRIDE
        offsets = tl.arange(0, BLOCK_SIZE)
        valid = offsets < STRIDE

        rows = tl.load(REQUIRED_ROWS_PTR + base + offsets, mask=valid, other=0).to(tl.int64)
        if HAS_TOKEN_ROW:
            rows = tl.load(TOKEN_ROW_PTR + rows, mask=valid, other=0).to(tl.int64)
        input_toks = tl.load(INPUT_IDS_PTR + rows, mask=valid, other=0).to(tl.int64)

        # shifted[i] = output[i-1]; position 0 is always accepted.
        safe_shifted = tl.where(offsets > 0, offsets - 1, 0)
        shifted_output = tl.load(OUTPUT_TOKENS_PTR + base + safe_shifted, mask=valid, other=0)
        match = tl.where(offsets == 0, 1, (input_toks == shifted_output.to(tl.int64)).to(tl.int32))
        match = tl.where(valid, match, 0)
        # accepted[i] iff every row up to i matched.
        cumsum = tl.cumsum(match, axis=0)
        accepted = (cumsum == (offsets + 1)) & valid
        accepted_count = tl.sum(accepted.to(tl.int32))
        last = base + accepted_count - 1

        # Accepted drafts: rows 1..NUM_SPEC of this request, -1 where rejected.
        is_spec = (offsets >= 1) & valid
        draft_tokens = tl.where(accepted & is_spec, input_toks, -1)
        tl.store(ACCEPTED_TOKENS_OUT_PTR + out_base + offsets - 1, draft_tokens, mask=is_spec)
        tl.store(ACCEPTED_COUNTS_OUT_PTR + pid, (accepted_count - 1).to(tl.int64))
    else:
        last = decode_len + (pid - num_decode_requests)
        tl.store(ACCEPTED_TOKENS_OUT_PTR + out_base + spec_offsets, -1, mask=spec_valid)
        tl.store(ACCEPTED_COUNTS_OUT_PTR + pid, 0)

    # Final sampled token and the input row of the last accepted token.
    tl.store(SAMPLED_TOKENS_OUT_PTR + pid, tl.load(OUTPUT_TOKENS_PTR + last).to(tl.int64))
    last_row = tl.load(REQUIRED_ROWS_PTR + last).to(tl.int64)
    if HAS_TOKEN_ROW:
        last_row = tl.load(TOKEN_ROW_PTR + last_row).to(tl.int64)
    tl.store(LAST_ACCEPTED_SEQ_OUT_PTR + pid, last_row)
    if HAS_POS_IDS:
        tl.store(BASE_POSITION_OUT_PTR + pid, tl.load(POS_IDS_PTR + last_row).to(tl.int64) + 1)


def verify_and_prepare_next_forward_pass(
    output_tokens,
    required_rows,
    input_ids,
    num_decode_requests,
    num_prefill_requests,
    num_speculative_tokens,
    sampled_tokens_buf,
    last_accepted_seq_buf,
    accepted_tokens_per_request,
    accepted_token_counts,
    token_row_indices=None,
    pos_ids=None,
    base_position_buf=None,
):
    """Verify drafts and prepare the next forward pass in one launch.

    Equivalent to::

        rows = required_rows if token_row_indices is None else token_row_indices[required_rows]
        last, mask, inp = verify_speculative_tokens(input_ids[rows], output_tokens, ...)
        prepare_next_forward_pass(..., output_tokens, rows, last, mask, inp, <buffers>)
        base_position_buf[:active] = pos_ids[last_accepted_seq_buf[:active]] + 1

    without materializing `rows`, `inp`, the accepted mask or the last-accepted indices.

    Args:
        output_tokens: Sampled ids (any integer dtype), one per required row.
        required_rows: Packed input row per required row (int32 or int64).
        input_ids: 1-D input-token storage.
        num_decode_requests, num_prefill_requests: Request counts.
        num_speculative_tokens: Drafts per decode request.
        sampled_tokens_buf, last_accepted_seq_buf, accepted_tokens_per_request,
            accepted_token_counts: Output buffers, as for `prepare_next_forward_pass`.
        token_row_indices: Optional packed -> storage row map.
        pos_ids: Optional position ids indexed like `input_ids`; requires `base_position_buf`.
        base_position_buf: Receives `pos_ids[last accepted row] + 1` per request.
    """
    active_request_count = num_decode_requests + num_prefill_requests
    if active_request_count == 0:
        return
    if input_ids.ndim == 2:
        input_ids = input_ids.squeeze(0)
    stride = num_speculative_tokens + 1
    has_pos = pos_ids is not None
    assert not has_pos or base_position_buf is not None
    has_token_row = token_row_indices is not None
    _verify_and_prepare_next_kernel[(active_request_count,)](
        output_tokens,
        required_rows,
        token_row_indices if has_token_row else required_rows,
        input_ids,
        pos_ids if has_pos else input_ids,
        sampled_tokens_buf,
        last_accepted_seq_buf,
        accepted_tokens_per_request,
        accepted_token_counts,
        base_position_buf if has_pos else sampled_tokens_buf,
        accepted_tokens_out_stride=accepted_tokens_per_request.stride(0),
        num_decode_requests=num_decode_requests,
        decode_len=num_decode_requests * stride,
        STRIDE=stride,
        NUM_SPEC_TOKENS=num_speculative_tokens,
        BLOCK_SIZE=triton.next_power_of_2(stride),
        HAS_TOKEN_ROW=has_token_row,
        HAS_POS_IDS=has_pos,
    )


# ---------------------------------------------------------------------------
# Kernel 4: Mamba state selective copy (eliminates temporary allocations)
# ---------------------------------------------------------------------------
@triton.jit
def _mamba_state_selective_copy_kernel(
    # Source: intermediate states  [L, M, S+1, *state_shape]
    SRC_PTR,
    # Destination: current states  [L, M, *state_shape]
    DST_PTR,
    # Per-request index arrays
    PREFILL_STATUS_PTR,  # [N] 0=decode, 1=prefill
    STATE_IDX_PTR,  # [N] maps request → mamba state slot
    ACCEPTED_PTR,  # [N] accepted token index per request
    # Strides (in elements)
    src_stride_layer,
    src_stride_slot,
    src_stride_spec,
    dst_stride_layer,
    dst_stride_slot,
    # Data size
    STATE_SIZE,
    # Compile-time
    BLOCK_SIZE: tl.constexpr,
):
    """Copy intermediate Mamba state to current state for decode requests.

    Grid: (N, L, num_chunks)
      - dim 0: active request index
      - dim 1: mamba layer index
      - dim 2: chunk of the flattened state vector

    No-op for prefill requests.
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1)
    pid_chunk = tl.program_id(2)

    # Skip prefill requests immediately.
    prefill = tl.load(PREFILL_STATUS_PTR + pid_req)
    if prefill == 1:
        return

    state_idx = tl.load(STATE_IDX_PTR + pid_req).to(tl.int64)
    accepted = tl.load(ACCEPTED_PTR + pid_req).to(tl.int64)

    chunk_start = pid_chunk * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    elem_offsets = chunk_start + offsets
    mask = elem_offsets < STATE_SIZE

    src_base = (
        pid_layer.to(tl.int64) * src_stride_layer
        + state_idx * src_stride_slot
        + accepted * src_stride_spec
    )
    dst_base = pid_layer.to(tl.int64) * dst_stride_layer + state_idx * dst_stride_slot

    data = tl.load(SRC_PTR + src_base + elem_offsets, mask=mask)
    tl.store(DST_PTR + dst_base + elem_offsets, data, mask=mask)


def mamba_state_selective_copy(
    intermediate_states, current_states, prefill_status, state_idx, accepted_counts, num_layers
):
    """Copy accepted intermediate Mamba states to current states in-place.

    For each decode request, copies
    `intermediate[layer, slot, accepted_count, ...]` →
    `current[layer, slot, ...]` for every Mamba layer.

    Args:
        intermediate_states: `(L, M, S+1, *state_shape)` — intermediate buffer.
        current_states: `(L, M, *state_shape)` — current state buffer (updated in-place).
        prefill_status: `(N,)` int tensor — 0 for decode, 1 for prefill.
        state_idx: `(N,)` int tensor — mamba state slot index per request.
        accepted_counts: `(N,)` int tensor — accepted token index per request.
        num_layers: number of Mamba layers (first dim of the state tensors).
    """
    N = prefill_status.shape[0]
    if N == 0:
        return

    # The state vector to copy per (layer, request) is the product of all
    # trailing dimensions after the speculative-token axis.
    # intermediate shape: (L, M, S+1, *state_shape) → state_size = prod(state_shape)
    state_size = math.prod(intermediate_states.shape[3:])

    BLOCK_SIZE = 1024
    num_chunks = triton.cdiv(state_size, BLOCK_SIZE)
    grid = (N, num_layers, num_chunks)

    _mamba_state_selective_copy_kernel[grid](
        intermediate_states,
        current_states,
        prefill_status,
        state_idx,
        accepted_counts,
        src_stride_layer=intermediate_states.stride(0),
        src_stride_slot=intermediate_states.stride(1),
        src_stride_spec=intermediate_states.stride(2),
        dst_stride_layer=current_states.stride(0),
        dst_stride_slot=current_states.stride(1),
        STATE_SIZE=state_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# Kernel 5: Per-depth attention metadata for the MTP draft loop
# ---------------------------------------------------------------------------
@triton.jit
def _mtp_draft_step_metadata_kernel(
    # Draft-loop state (read-only)
    OFFSETS_PTR,
    SRC_BLOCK_TABLE_PTR,
    # Per-token KV write maps
    TOKEN_TO_BLOCK_IDX_PTR,
    TOKEN_TO_LOCAL_POS_PTR,
    TOKEN_TO_REQUEST_IDX_PTR,
    TOKEN_TO_POS_IN_REQUEST_PTR,
    TOKEN_TO_POS_IDS_PTR,
    # Per-request length staging
    STAGED_QUERY_LENGTHS_PTR,
    STAGED_KV_LENGTHS_PTR,
    # Per-request MHA read metadata
    MHA_QUERY_LENGTHS_PTR,
    MHA_CU_QUERY_PTR,
    MHA_KV_LENGTHS_PTR,
    MHA_CU_KV_PTR,
    MHA_BLOCK_TABLE_PTR,
    # Strides / limits
    src_block_table_stride,
    mha_block_table_stride,
    num_block_columns,
    active_request_count,
    padded_count,
    dummy_block_idx,
    # Compile-time constants
    BLOCK_SIZE_TOKENS: tl.constexpr,
    BLOCK_R: tl.constexpr,  # next_power_of_2(padded_count)
    BLOCK_C: tl.constexpr,  # block-table columns copied per iteration
):
    """Write one draft depth's KV write maps and MHA read metadata.

    Grid: (1,). One program covers every request slot, since the cumulative lengths span
    requests. Each of the first `active_request_count` slots writes one token at its MTP
    position P (from OFFSETS) and attends over P + 1 keys. Slots in
    [active_request_count, padded_count) are padding: their tokens go to the dummy block and
    their requests get zero lengths and a -1 block table.
    """
    rows = tl.arange(0, BLOCK_R)
    real = rows < active_request_count
    padded = rows < padded_count

    # --- Per-token KV write maps (one token per request) ---
    positions = tl.load(OFFSETS_PTR + rows, mask=real, other=0)
    block_col = positions // BLOCK_SIZE_TOKENS
    destinations = tl.load(
        SRC_BLOCK_TABLE_PTR + rows.to(tl.int64) * src_block_table_stride + block_col,
        mask=real,
        other=0,
    )
    block_idx = tl.where(real, destinations.to(tl.int64), dummy_block_idx)
    tl.store(TOKEN_TO_BLOCK_IDX_PTR + rows, block_idx, mask=padded)
    local_pos = tl.where(real, positions % BLOCK_SIZE_TOKENS, 0)
    tl.store(TOKEN_TO_LOCAL_POS_PTR + rows, local_pos, mask=padded)
    tl.store(TOKEN_TO_REQUEST_IDX_PTR + rows, rows, mask=real)
    tl.store(TOKEN_TO_POS_IN_REQUEST_PTR + rows, positions, mask=real)
    tl.store(TOKEN_TO_POS_IDS_PTR + rows, positions.to(tl.int64), mask=real)

    # --- Per-request lengths: query_length = 1, kv_length = P + 1 (write-then-attend) ---
    query_lengths = tl.where(real, 1, 0)
    kv_lengths = tl.where(real, positions + 1, 0)
    tl.store(STAGED_QUERY_LENGTHS_PTR + rows, query_lengths, mask=real)
    tl.store(STAGED_KV_LENGTHS_PTR + rows, kv_lengths, mask=real)
    tl.store(MHA_QUERY_LENGTHS_PTR + rows, query_lengths, mask=padded)
    tl.store(MHA_KV_LENGTHS_PTR + rows, kv_lengths, mask=padded)

    # Cumulative lengths: entry 0 is 0, entry r + 1 sums slots 0..r. Padding slots add zero,
    # so they repeat the last real total.
    tl.store(MHA_CU_QUERY_PTR + rows, tl.zeros_like(rows), mask=rows == 0)
    tl.store(MHA_CU_KV_PTR + rows, tl.zeros_like(rows), mask=rows == 0)
    tl.store(MHA_CU_QUERY_PTR + 1 + rows, tl.cumsum(query_lengths, axis=0), mask=padded)
    tl.store(MHA_CU_KV_PTR + 1 + rows, tl.cumsum(kv_lengths, axis=0), mask=padded)

    # --- Block table: real rows copy the draft loop's table, padding rows get -1 ---
    real_2d = real[:, None]
    padded_2d = padded[:, None]
    src_rows = rows.to(tl.int64)[:, None] * src_block_table_stride
    dst_rows = rows.to(tl.int64)[:, None] * mha_block_table_stride
    for col_start in range(0, num_block_columns, BLOCK_C):
        cols = (col_start + tl.arange(0, BLOCK_C))[None, :]
        col_mask = cols < num_block_columns
        values = tl.load(SRC_BLOCK_TABLE_PTR + src_rows + cols, mask=real_2d & col_mask, other=0)
        values = tl.where(real_2d, values, -1)
        tl.store(
            MHA_BLOCK_TABLE_PTR + dst_rows + cols,
            values.to(MHA_BLOCK_TABLE_PTR.dtype.element_ty),
            mask=padded_2d & col_mask,
        )


def mtp_draft_step_metadata(
    gpu_view,
    offsets,
    block_table,
    staged_query_lengths,
    staged_kv_lengths,
    active_request_count,
    padded_count,
    block_size_tokens,
    dummy_block_idx,
):
    """Launch the per-depth MTP draft metadata kernel.

    Writes exactly what `MTPMetadata.stage_decode_lengths`, `write_token_maps` and
    `write_mha_metadata` write for one draft depth, in a single launch and without host
    syncs.

    Args:
        gpu_view: `ContextGPUView` whose token maps and `mha_*` buffers are written.
        offsets: `(>=active_request_count,)` int32 MTP write position per request. Must be
            non-negative.
        block_table: `(>=active_request_count, num_block_columns)` draft-loop block table.
        staged_query_lengths, staged_kv_lengths: `(>=active_request_count,)` int32 staging
            buffers, written with the per-request query and KV lengths.
        active_request_count: Requests that draft this depth.
        padded_count: Request (and token) slots the forward launches over, including padding.
        block_size_tokens: KV block size, in tokens.
        dummy_block_idx: Scratch block that padding tokens write to.
    """
    if padded_count == 0:
        return
    assert active_request_count <= padded_count
    num_block_columns = gpu_view.mha_block_table.shape[1]
    assert block_table.shape[1] == num_block_columns

    _mtp_draft_step_metadata_kernel[(1,)](
        offsets,
        block_table,
        gpu_view.token_to_block_idx,
        gpu_view.token_to_local_position_within_kv_block,
        gpu_view.token_to_request_idx,
        gpu_view.token_to_position_in_request,
        gpu_view.token_to_pos_ids,
        staged_query_lengths,
        staged_kv_lengths,
        gpu_view.mha_query_lengths,
        gpu_view.mha_cu_query_seq_lengths,
        gpu_view.mha_kv_seq_lengths,
        gpu_view.mha_cu_kv_seq_lengths,
        gpu_view.mha_block_table,
        src_block_table_stride=block_table.stride(0),
        mha_block_table_stride=gpu_view.mha_block_table.stride(0),
        num_block_columns=num_block_columns,
        active_request_count=active_request_count,
        padded_count=padded_count,
        dummy_block_idx=dummy_block_idx,
        BLOCK_SIZE_TOKENS=block_size_tokens,
        BLOCK_R=triton.next_power_of_2(padded_count),
        BLOCK_C=min(triton.next_power_of_2(num_block_columns), 64),
    )


# ---------------------------------------------------------------------------
# Kernel 6: Attention metadata for the CUDA-graphed MTP commit pass
# ---------------------------------------------------------------------------
@dataclass
class MTPCommitGather:
    """Inputs gathered by the commit-metadata launch (see `_mtp_commit_step_metadata_kernel`).

    Args:
        main_hidden (Tensor): [T, 1, H] main-forward hidden states (`num_slots + 1` rows per
            decode request), last dim contiguous.
        input_ids (Tensor): [>=T] 1-D input-token storage.
        last_accepted_rows (Tensor): [>=active] main-forward row of each request's last
            accepted token.
        commit_hidden_out (Tensor): [>=padded * num_slots, 1, H] contiguous, main_hidden dtype.
        commit_tokens_out (Tensor): [>=padded * num_slots] contiguous, input_ids dtype.
        draft_hidden_out (Tensor): [>=padded, 1, H] contiguous, main_hidden dtype.
    """

    main_hidden: torch.Tensor
    input_ids: torch.Tensor
    last_accepted_rows: torch.Tensor
    commit_hidden_out: torch.Tensor
    commit_tokens_out: torch.Tensor
    draft_hidden_out: torch.Tensor


@triton.jit
def _mtp_commit_step_metadata_kernel(
    # Per-request input (read-only)
    ACCEPTED_COUNTS_PTR,
    BASE_POSITIONS_PTR,
    SRC_BLOCK_TABLE_PTR,
    # Per-token KV write maps
    TOKEN_TO_BLOCK_IDX_PTR,
    TOKEN_TO_LOCAL_POS_PTR,
    TOKEN_TO_REQUEST_IDX_PTR,
    TOKEN_TO_POS_IN_REQUEST_PTR,
    TOKEN_TO_POS_IDS_PTR,
    # Per-request MHA read metadata
    MHA_QUERY_LENGTHS_PTR,
    MHA_CU_QUERY_PTR,
    MHA_KV_LENGTHS_PTR,
    MHA_CU_KV_PTR,
    MHA_BLOCK_TABLE_PTR,
    # Strides / limits
    src_block_table_stride,
    mha_block_table_stride,
    num_block_columns,
    active_request_count,
    padded_count,
    dummy_block_idx,
    # Optional commit-pass / draft-input gathers (GATHER only)
    MAIN_HIDDEN_PTR,
    INPUT_IDS_PTR,
    LAST_ACCEPTED_ROWS_PTR,
    COMMIT_HIDDEN_OUT_PTR,
    COMMIT_TOKENS_OUT_PTR,
    DRAFT_HIDDEN_OUT_PTR,
    main_hidden_stride,
    max_main_hidden_row,
    max_input_row,
    hidden_size,
    # Compile-time constants
    SLOTS: tl.constexpr,  # num_speculative_tokens: commit rows per request
    BLOCK_SIZE_TOKENS: tl.constexpr,
    BLOCK_R: tl.constexpr,  # next_power_of_2(padded_count)
    BLOCK_S: tl.constexpr,  # next_power_of_2(SLOTS)
    BLOCK_C: tl.constexpr,  # block-table columns copied per iteration
    GATHER: tl.constexpr = False,
    BLOCK_H: tl.constexpr = 1024,  # hidden elements copied per iteration
):
    """Write the fixed-shape commit-pass metadata for a decode-only step.

    With GATHER, the same launch also builds the commit pass's inputs and the first draft
    depth's hidden input (each request has `SLOTS + 1` main-forward rows):
      * tokens:  row r*SLOTS+j = input_ids[min(r*(SLOTS+1)+j+1, max_input_row)], 0 past the
        real requests;
      * hidden:  row r*SLOTS+j = main_hidden[min(r*(SLOTS+1)+j, max_main_hidden_row)], 0 past
        the real requests;
      * draft hidden: row r = main_hidden[last_accepted_rows[r]], 0 for padding requests.
    Program 0 writes the metadata and tokens; programs 1.. copy one hidden row each.

    Grid: (1,), or (1 + padded_count * (SLOTS + 1),) with GATHER. Request r owns token rows
    r*SLOTS .. r*SLOTS+SLOTS-1. Request r accepted a_r
    drafts, so its refreshed run starts at MTP position `start_r = base_r - 1 - a_r`; slot j
    writes position start_r + j when j < a_r and the dummy block otherwise. Every real request
    attends with SLOTS query rows over `start_r + SLOTS` keys, so causal (bottom-right aligned)
    masking lets slot j see keys up to start_r + j. Requests at or past active_request_count are
    padding: dummy-block writes, zero lengths, and a -1 block table.
    """
    if GATHER:
        pid = tl.program_id(0)
        if pid > 0:
            # Hidden-row copies: commit rows first, then the draft rows.
            out_row = pid - 1
            num_commit_rows = padded_count * SLOTS
            if out_row < num_commit_rows:
                is_real = out_row < active_request_count * SLOTS
                req = out_row // SLOTS
                src_row = tl.minimum(req * (SLOTS + 1) + out_row % SLOTS, max_main_hidden_row).to(
                    tl.int64
                )
                dst = COMMIT_HIDDEN_OUT_PTR + out_row.to(tl.int64) * hidden_size
            else:
                req = out_row - num_commit_rows
                is_real = req < active_request_count
                src_row = tl.load(LAST_ACCEPTED_ROWS_PTR + req, mask=is_real, other=0).to(tl.int64)
                dst = DRAFT_HIDDEN_OUT_PTR + req.to(tl.int64) * hidden_size
            src = MAIN_HIDDEN_PTR + src_row * main_hidden_stride
            for h_start in range(0, hidden_size, BLOCK_H):
                h = h_start + tl.arange(0, BLOCK_H)
                h_mask = h < hidden_size
                values = tl.load(src + h, mask=h_mask & is_real, other=0.0)
                tl.store(dst + h, values, mask=h_mask)
            return

    rows = tl.arange(0, BLOCK_R)
    real = rows < active_request_count
    padded = rows < padded_count

    if GATHER:
        # Commit tokens: the input token one main-forward row after each commit hidden row.
        g_slots = tl.arange(0, BLOCK_S)
        g_tokens = rows[:, None] * SLOTS + g_slots[None, :]
        g_in_range = padded[:, None] & (g_slots < SLOTS)[None, :]
        g_real = real[:, None] & (g_slots < SLOTS)[None, :]
        g_src = tl.minimum(rows[:, None] * (SLOTS + 1) + g_slots[None, :] + 1, max_input_row)
        g_values = tl.load(INPUT_IDS_PTR + g_src.to(tl.int64), mask=g_real, other=0)
        tl.store(COMMIT_TOKENS_OUT_PTR + g_tokens, g_values, mask=g_in_range)

    accepted = tl.load(ACCEPTED_COUNTS_PTR + rows, mask=real, other=0).to(tl.int32)
    base = tl.load(BASE_POSITIONS_PTR + rows, mask=real, other=1).to(tl.int32)
    starts = base - 1 - accepted

    # --- Per-token KV write maps ---
    slots = tl.arange(0, BLOCK_S)
    slot_in_range = (slots < SLOTS)[None, :]
    tokens = rows[:, None] * SLOTS + slots[None, :]
    token_mask = padded[:, None] & slot_in_range
    positions = starts[:, None] + slots[None, :]
    writes = real[:, None] & (slots[None, :] < accepted[:, None]) & slot_in_range
    destinations = tl.load(
        SRC_BLOCK_TABLE_PTR
        + rows.to(tl.int64)[:, None] * src_block_table_stride
        + positions // BLOCK_SIZE_TOKENS,
        mask=writes,
        other=0,
    )
    block_idx = tl.where(writes, destinations.to(tl.int64), dummy_block_idx)
    local_pos = tl.where(writes, positions % BLOCK_SIZE_TOKENS, 0)
    request_positions = tl.where(real[:, None], positions, 0)
    tl.store(TOKEN_TO_BLOCK_IDX_PTR + tokens, block_idx, mask=token_mask)
    tl.store(TOKEN_TO_LOCAL_POS_PTR + tokens, local_pos, mask=token_mask)
    tl.store(
        TOKEN_TO_REQUEST_IDX_PTR + tokens, rows[:, None] + tl.zeros_like(tokens), mask=token_mask
    )
    tl.store(TOKEN_TO_POS_IN_REQUEST_PTR + tokens, request_positions, mask=token_mask)
    tl.store(TOKEN_TO_POS_IDS_PTR + tokens, request_positions.to(tl.int64), mask=token_mask)

    # --- Per-request lengths and their cumulative sums ---
    query_lengths = tl.where(real, SLOTS, 0)
    kv_lengths = tl.where(real, starts + SLOTS, 0)
    tl.store(MHA_QUERY_LENGTHS_PTR + rows, query_lengths, mask=padded)
    tl.store(MHA_KV_LENGTHS_PTR + rows, kv_lengths, mask=padded)
    tl.store(MHA_CU_QUERY_PTR + rows, tl.zeros_like(rows), mask=rows == 0)
    tl.store(MHA_CU_KV_PTR + rows, tl.zeros_like(rows), mask=rows == 0)
    tl.store(MHA_CU_QUERY_PTR + 1 + rows, tl.cumsum(query_lengths, axis=0), mask=padded)
    tl.store(MHA_CU_KV_PTR + 1 + rows, tl.cumsum(kv_lengths, axis=0), mask=padded)

    # --- Block table: real rows copy the source table, padding rows get -1 ---
    real_2d = real[:, None]
    padded_2d = padded[:, None]
    src_rows = rows.to(tl.int64)[:, None] * src_block_table_stride
    dst_rows = rows.to(tl.int64)[:, None] * mha_block_table_stride
    for col_start in range(0, num_block_columns, BLOCK_C):
        cols = (col_start + tl.arange(0, BLOCK_C))[None, :]
        col_mask = cols < num_block_columns
        values = tl.load(SRC_BLOCK_TABLE_PTR + src_rows + cols, mask=real_2d & col_mask, other=0)
        values = tl.where(real_2d, values, -1)
        tl.store(
            MHA_BLOCK_TABLE_PTR + dst_rows + cols,
            values.to(MHA_BLOCK_TABLE_PTR.dtype.element_ty),
            mask=padded_2d & col_mask,
        )


def mtp_commit_step_metadata(
    gpu_view,
    accepted_counts,
    base_positions,
    block_table,
    active_request_count,
    padded_count,
    num_slots,
    block_size_tokens,
    dummy_block_idx,
    gather=None,
):
    """Launch the graphed commit-pass metadata kernel.

    Lays the commit pass out as `num_slots` token rows per request (so its shape depends only on
    `padded_count`) and writes the token maps and MHA metadata for it, without host syncs.

    Args:
        gpu_view: `ContextGPUView` whose token maps and `mha_*` buffers are written.
        accepted_counts: `(>=active_request_count,)` drafts each request accepted this step.
        base_positions: `(>=active_request_count,)` position of each request's next draft
            (one past its last committed token).
        block_table: `(>=active_request_count, num_block_columns)` block ids per request.
        active_request_count: Real (decode) requests.
        padded_count: Request slots the forward launches over, including padding.
        num_slots: Commit rows per request (`num_speculative_tokens`).
        block_size_tokens: KV block size, in tokens.
        dummy_block_idx: Scratch block that unused and padding rows write to.
        gather (Optional[MTPCommitGather]): Also build the commit pass's hidden / token inputs
            and the first draft depth's hidden input in the same launch.
    """
    if padded_count == 0:
        return
    assert active_request_count <= padded_count
    num_block_columns = gpu_view.mha_block_table.shape[1]
    assert block_table.shape[1] == num_block_columns

    if gather is not None:
        main_hidden = gather.main_hidden
        hidden_size = main_hidden.shape[-1]
        assert main_hidden.stride(-1) == 1 and main_hidden[0].numel() == hidden_size
        assert gather.commit_hidden_out.is_contiguous()
        assert gather.draft_hidden_out.is_contiguous()
        assert gather.commit_hidden_out.numel() >= padded_count * num_slots * hidden_size
        assert gather.draft_hidden_out.numel() >= padded_count * hidden_size
        assert gather.commit_tokens_out.numel() >= padded_count * num_slots
        grid = (1 + padded_count * (num_slots + 1),)
        gather_args = dict(
            MAIN_HIDDEN_PTR=main_hidden,
            INPUT_IDS_PTR=gather.input_ids,
            LAST_ACCEPTED_ROWS_PTR=gather.last_accepted_rows,
            COMMIT_HIDDEN_OUT_PTR=gather.commit_hidden_out,
            COMMIT_TOKENS_OUT_PTR=gather.commit_tokens_out,
            DRAFT_HIDDEN_OUT_PTR=gather.draft_hidden_out,
            main_hidden_stride=main_hidden.stride(0),
            max_main_hidden_row=main_hidden.shape[0] - 1,
            max_input_row=gather.input_ids.shape[0] - 1,
            hidden_size=hidden_size,
            GATHER=True,
            BLOCK_H=min(triton.next_power_of_2(hidden_size), 1024),
        )
    else:
        grid = (1,)
        gather_args = dict(
            MAIN_HIDDEN_PTR=accepted_counts,
            INPUT_IDS_PTR=accepted_counts,
            LAST_ACCEPTED_ROWS_PTR=accepted_counts,
            COMMIT_HIDDEN_OUT_PTR=accepted_counts,
            COMMIT_TOKENS_OUT_PTR=accepted_counts,
            DRAFT_HIDDEN_OUT_PTR=accepted_counts,
            main_hidden_stride=0,
            max_main_hidden_row=0,
            max_input_row=0,
            hidden_size=0,
        )

    _mtp_commit_step_metadata_kernel[grid](
        accepted_counts,
        base_positions,
        block_table,
        gpu_view.token_to_block_idx,
        gpu_view.token_to_local_position_within_kv_block,
        gpu_view.token_to_request_idx,
        gpu_view.token_to_position_in_request,
        gpu_view.token_to_pos_ids,
        gpu_view.mha_query_lengths,
        gpu_view.mha_cu_query_seq_lengths,
        gpu_view.mha_kv_seq_lengths,
        gpu_view.mha_cu_kv_seq_lengths,
        gpu_view.mha_block_table,
        src_block_table_stride=block_table.stride(0),
        mha_block_table_stride=gpu_view.mha_block_table.stride(0),
        num_block_columns=num_block_columns,
        active_request_count=active_request_count,
        padded_count=padded_count,
        dummy_block_idx=dummy_block_idx,
        **gather_args,
        SLOTS=num_slots,
        BLOCK_SIZE_TOKENS=block_size_tokens,
        BLOCK_R=triton.next_power_of_2(padded_count),
        BLOCK_S=triton.next_power_of_2(num_slots),
        BLOCK_C=min(triton.next_power_of_2(num_block_columns), 64),
    )
