# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Lazy-initialized symmetric memory manager for inference.

Provides a registry of SymmetricMemoryBuffer instances keyed by a
user-supplied identifier (e.g. "tp", "ep").  Buffers are created on first
access so that callers never need to worry about initialization ordering
relative to the inference context.
"""

from __future__ import annotations

import operator
from functools import reduce
from typing import Optional

import torch

try:
    import torch.distributed._symmetric_memory as symm_mem

    HAVE_TORCH_SYMM_MEM = True
except ImportError:
    HAVE_TORCH_SYMM_MEM = False

try:
    import triton  # pylint: disable=unused-import

    HAVE_TRITON = True
except ImportError:
    HAVE_TRITON = False

# 128 max CTA blocks per kernel launch (matches MAX_NUM_BLOCKS in collectives).
_SIGNAL_PAD_MAX_BLOCKS = 128


class _SymmetricMemoryHandleWrapper:
    """Proxies a _SymmetricMemory handle but overrides signal_pad_ptrs_dev.

    PyTorch 2.11 changed how signal pads are allocated via the implicit mempool,
    which can break the fixed-offset barrier arithmetic in our Triton kernels.
    This wrapper substitutes a Megatron-managed signal pad carved from the tail
    of the symmetric buffer itself, giving us a stable layout across versions.
    """

    def __init__(self, inner_handle, custom_signal_pad_ptrs_dev):
        self._inner = inner_handle
        self._custom_signal_pad_ptrs_dev = custom_signal_pad_ptrs_dev

    @property
    def signal_pad_ptrs_dev(self):
        return self._custom_signal_pad_ptrs_dev

    def __getattr__(self, name):
        return getattr(self._inner, name)


class SymmetricMemoryBuffer:
    """
     symmetric memory buffer used in inference.
    This buffer is used by mcore-inference's low-latency
    NVLS all-gather and reduce-scatter collectives.
    """

    def __init__(self, size_in_mb, process_group):
        self.init_failure_reason: Optional[str] = None
        if not HAVE_TORCH_SYMM_MEM:
            self.init_failure_reason = "torch.distributed._symmetric_memory not importable"
            self.symm_buffer = None
            self.symm_mem_hdl = None
        elif not HAVE_TRITON:
            self.init_failure_reason = "triton not installed"
            self.symm_buffer = None
            self.symm_mem_hdl = None
        else:
            numel = int(size_in_mb * 1024 * 1024)  # size in bytes
            try:
                symm_mem.enable_symm_mem_for_group(process_group.group_name)
                self.symm_buffer = symm_mem.empty(numel, dtype=torch.uint8, device='cuda')
                self.symm_mem_hdl = symm_mem.rendezvous(self.symm_buffer, process_group)
                self._init_custom_signal_pad()
            except RuntimeError as e:
                self.init_failure_reason = f"{type(e).__name__}: {e}"
                self.symm_buffer = None
                self.symm_mem_hdl = None

    def _init_custom_signal_pad(self):
        """Carve a signal pad region from the tail of the symmetric buffer.

        The region is sized for _SIGNAL_PAD_MAX_BLOCKS * world_size int32 slots
        (matching the barrier kernel's offset math: block_id * world_size + rank).
        We build a signal_pad_ptrs_dev tensor from buffer_ptrs_dev + byte_offset
        so every rank's signal pad is P2P-accessible with a known, stable layout.
        """
        hdl = self.symm_mem_hdl
        world_size = hdl.world_size
        pad_numel = _SIGNAL_PAD_MAX_BLOCKS * world_size  # int32 slots
        pad_bytes = pad_numel * 4  # sizeof(int32)

        # Reserve from the end of the buffer; reduce usable capacity accordingly.
        buf_bytes = self.symm_buffer.numel()
        pad_offset = buf_bytes - pad_bytes
        assert pad_offset > 0, (
            f"Symmetric buffer too small for signal pad: "
            f"need {pad_bytes} bytes, buffer is {buf_bytes} bytes"
        )

        # Zero-initialize the signal pad region (barrier uses CAS 0→1 / 1→0).
        self.symm_buffer[pad_offset:].zero_()

        # Shrink the usable buffer so data allocations can't overlap the pad.
        self.symm_buffer = self.symm_buffer[:pad_offset]

        # Build custom signal_pad_ptrs_dev: one P2P pointer per rank, each
        # pointing to that rank's signal pad at (buffer_base + pad_offset).
        buffer_ptrs = hdl.buffer_ptrs_dev  # [world_size] uint64, P2P base per rank
        self._custom_signal_pad_ptrs_dev = buffer_ptrs + pad_offset

    def _can_allocate(self, numel, dtype) -> bool:
        """
        Returns whether enough symmetric memory is available
        for the given tensor shape and dtype.
        """
        if self.symm_mem_hdl is None:
            return False
        size_of_dtype = torch.tensor([], dtype=dtype).element_size()
        required_len = numel * size_of_dtype
        return required_len <= self.symm_buffer.numel()

    def _allocate(self, numel, dtype) -> torch.Tensor:
        """
        Allocates a sub-tensor from the self.symm_buffer for the given numel and dtype"""
        required_bytes = numel * torch.tensor([], dtype=dtype).element_size()
        return self.symm_buffer[0:required_bytes].view(dtype).view(numel)

    def _wrapped_handle(self):
        """Return the handle with our custom signal pad, or the raw handle as fallback."""
        if hasattr(self, '_custom_signal_pad_ptrs_dev'):
            return _SymmetricMemoryHandleWrapper(
                self.symm_mem_hdl, self._custom_signal_pad_ptrs_dev
            )
        return self.symm_mem_hdl

    def maybe_get_tensors(self, tensor_specs, alignment=16):
        """
        Pack multiple tensors contiguously in the symmetric buffer with alignment.

        Each tensor's starting offset is aligned to `alignment` bytes (default 16
        for 128-bit multimem access).

        Args:
            tensor_specs: list of (numel, dtype) tuples.
            alignment: byte alignment for each tensor's start offset (default 16).

        Returns:
            {"handle": None, "tensors": None} if unavailable or insufficient space.
            {"handle": symm_mem_hdl, "tensors": [(raw_byte_view, byte_offset), ...]}
            on success, where raw_byte_view is a uint8 slice of the buffer.
        """
        _NONE_RESULT = {"handle": None, "tensors": None}
        if self.symm_mem_hdl is None:
            return _NONE_RESULT

        # Compute aligned byte sizes and running offsets
        slices = []
        current_offset = 0
        for numel, dtype in tensor_specs:
            nbytes = numel * torch.tensor([], dtype=dtype).element_size()
            aligned_nbytes = ((nbytes + alignment - 1) // alignment) * alignment
            slices.append((current_offset, nbytes))
            current_offset += aligned_nbytes

        if not self._can_allocate(current_offset, torch.uint8):
            return _NONE_RESULT

        tensors = []
        for offset, nbytes in slices:
            tensors.append((self.symm_buffer[offset : offset + nbytes], offset))

        return {"handle": self._wrapped_handle(), "tensors": tensors}

    def maybe_get_tensor(self, tensor_shape, dtype):
        """
        Returns (potentially) a sub-tensor from the self.symm_buffer for the given shape.
        If enough symmetric memory is not available, returns None.
        """
        if self.symm_mem_hdl is None:
            return {"tensor": None, "handle": None}
        numel = reduce(operator.mul, tensor_shape, 1)
        if not self._can_allocate(numel, dtype):
            return {"tensor": None, "handle": None}
        return {
            "tensor": self._allocate(numel, dtype).view(*tensor_shape),
            "handle": self._wrapped_handle(),
        }


class SymmetricMemoryManager:
    """Registry of lazily-initialized symmetric memory buffers.

    Usage::

        buf = SymmetricMemoryManager.get_buffer("tp", process_group=tp_group)
        result = buf.maybe_get_tensor(shape, dtype)
    """

    _buffers: dict[str, SymmetricMemoryBuffer] = {}
    _default_size_mb: int = 512

    @classmethod
    def get_buffer(
        cls,
        key: str,
        process_group: Optional[torch.distributed.ProcessGroup] = None,
        size_mb: Optional[int] = None,
    ) -> SymmetricMemoryBuffer:
        """Return the buffer for *key*, creating it on first call.

        Args:
            key: Unique identifier (e.g. "tp", "ep").
            process_group: Required on the first call for a given key.
                Subsequent calls may omit it.
            size_mb: Buffer size in MiB (default 256).
        """
        if key not in cls._buffers:
            assert (
                process_group is not None
            ), f"SymmetricMemoryManager: process_group is required on first access for key='{key}'"
            cls._buffers[key] = SymmetricMemoryBuffer(
                size_in_mb=size_mb or cls._default_size_mb, process_group=process_group
            )
        return cls._buffers[key]

    @classmethod
    def destroy(cls, key: Optional[str] = None) -> None:
        """Destroy one or all buffers.

        Args:
            key: If provided, destroy only that buffer. Otherwise destroy all.
        """
        if key is not None:
            cls._buffers.pop(key, None)
        else:
            cls._buffers.clear()

    @classmethod
    def is_initialized(cls, key: str) -> bool:
        """Check whether a buffer has been created for *key*."""
        return key in cls._buffers
