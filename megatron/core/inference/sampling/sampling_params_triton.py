# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-row sampling parameters for the FlashInfer sampler, built in one launch.

The FlashInfer sampler needs, per logits row, the request's temperature (clamped away from zero)
and its top-k / top-p with the "disabled" sentinels replaced by the values that make the filter a
no-op (`top_k -> vocab_size`, `top_p -> 1.0`). Done with eager torch ops this is a gather per
parameter through a token -> request map (itself an arange / repeat_interleave / cat under
speculative decoding), a clamp, and a compare / compare / or / masked_fill per filter: about a
dozen tiny launches on the critical path between the forward pass and the sampler.

`gather_sampling_params` writes all three columns in a single Triton launch. The row -> request
map is either explicit, or the speculative-decoding layout, which is static given the counts:
the first `num_decode * stride` rows are decode requests, `stride` rows each, and the remaining
rows are one row per prefill request. Non-speculative sampling is the `stride == 1` case.

The outputs are bit-identical to the torch ops they replace: the work is a gather, a max and two
selects, with no arithmetic rounding.
"""

from typing import Optional

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


@triton.jit
def _gather_sampling_params_kernel(
    # Per-request inputs, indexed by active request.
    TEMPERATURE_PTR,
    TOP_K_PTR,
    TOP_P_PTR,
    # Optional explicit row -> request map.
    ROW_TO_REQUEST_PTR,
    # Per-row outputs.
    TEMPERATURE_OUT_PTR,
    TOP_K_OUT_PTR,
    TOP_P_OUT_PTR,
    # Runtime scalars.
    num_rows,
    num_decode_rows,
    num_decode_requests,
    vocab_size,
    min_temperature,
    # Compile-time constants.
    STRIDE: tl.constexpr,
    HAS_MAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Grid: (cdiv(num_rows, BLOCK),). Each program fills BLOCK rows."""
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = rows < num_rows
    if HAS_MAP:
        requests = tl.load(ROW_TO_REQUEST_PTR + rows, mask=valid, other=0)
    else:
        # Decode rows: `STRIDE` consecutive rows per request. Prefill rows: one row each.
        requests = tl.where(
            rows < num_decode_rows, rows // STRIDE, num_decode_requests + (rows - num_decode_rows)
        )
    temperature = tl.load(TEMPERATURE_PTR + requests, mask=valid, other=1.0)
    top_k = tl.load(TOP_K_PTR + requests, mask=valid, other=0)
    top_p = tl.load(TOP_P_PTR + requests, mask=valid, other=0.0)

    # `temperature.clamp(min=MIN_SAMPLING_TEMPERATURE)`.
    temperature = tl.maximum(temperature, min_temperature)
    # `is_no_op_top_k`: top_k <= 0 keeps every token.
    top_k = tl.where(top_k <= 0, vocab_size, top_k)
    # `is_no_op_top_p`: top_p <= 0 (sentinel) or >= 1 (full mass) keeps every token.
    top_p = tl.where((top_p <= 0.0) | (top_p >= 1.0), 1.0, top_p)

    tl.store(TEMPERATURE_OUT_PTR + rows, temperature, mask=valid)
    tl.store(TOP_K_OUT_PTR + rows, top_k.to(TOP_K_OUT_PTR.dtype.element_ty), mask=valid)
    tl.store(TOP_P_OUT_PTR + rows, top_p, mask=valid)


def gather_sampling_params(
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    temperature_out: torch.Tensor,
    top_k_out: torch.Tensor,
    top_p_out: torch.Tensor,
    num_rows: int,
    *,
    vocab_size: int,
    min_temperature: float,
    num_decode_requests: Optional[int] = None,
    rows_per_decode_request: int = 1,
    row_to_request: Optional[torch.Tensor] = None,
) -> None:
    """Write the per-row sampler parameters for `num_rows` logits rows.

    Equivalent to (with `idx` the row -> request map)::

        temperature_out[:n] = temperature[idx].clamp(min=min_temperature)
        top_k_out[:n] = top_k[idx].masked_fill(top_k[idx] <= 0, vocab_size)
        top_p_out[:n] = top_p[idx].masked_fill((top_p[idx] <= 0) | (top_p[idx] >= 1), 1.0)

    Args:
        temperature, top_k, top_p: Per-active-request parameters (float32, int32, float32).
        temperature_out, top_k_out, top_p_out: Destinations, at least `num_rows` long.
        num_rows: Logits rows to fill.
        vocab_size: Value that disables top-k.
        min_temperature: Lower clamp for the temperature.
        num_decode_requests: Decode requests in the speculative layout. When None, every row is
            treated as a decode row (`rows_per_decode_request` rows per request).
        rows_per_decode_request: Rows per decode request (`1 + num_speculative_tokens`).
        row_to_request: Explicit row -> request map; overrides the static layout.
    """
    if num_rows <= 0:
        return
    stride = rows_per_decode_request
    if num_decode_requests is None:
        num_decode_requests = triton.cdiv(num_rows, stride)
    num_decode_rows = min(num_decode_requests * stride, num_rows)
    has_map = row_to_request is not None
    block = min(triton.next_power_of_2(num_rows), 1024)
    _gather_sampling_params_kernel[(triton.cdiv(num_rows, block),)](
        temperature,
        top_k,
        top_p,
        row_to_request if has_map else temperature,
        temperature_out,
        top_k_out,
        top_p_out,
        num_rows,
        num_decode_rows,
        num_decode_requests,
        vocab_size,
        min_temperature,
        STRIDE=stride,
        HAS_MAP=has_map,
        BLOCK=block,
    )
