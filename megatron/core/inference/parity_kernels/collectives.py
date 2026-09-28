# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for Megatron's explicit process groups and the audited CUDA profile.
"""Match custom / symmetric-memory / NCCL all-reduce dispatch and rounding."""

import ctypes
import logging
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist

from .collective_sizes import CUSTOM_ALL_REDUCE_MAX_SIZES
from .ops import load_ops
from .symm_mem import SymmMemCommunicator

logger = logging.getLogger(__name__)


class _UniqueId(ctypes.Structure):
    _fields_ = [('internal', ctypes.c_byte * 128)]


class PyNcclCommunicator:
    """A separate NCCL communicator preserves the reference's collective path."""

    def __init__(self, group, device):
        self.device = device
        self.disabled = False
        candidates = [
            line.split()[-1]
            for line in Path('/proc/self/maps').read_text().splitlines()
            if '/libnccl.so' in line
        ]
        path = os.environ.get('MEGATRON_PARITY_NCCL_SO_PATH') or (
            candidates[0] if candidates else 'libnccl.so.2'
        )
        self.lib = ctypes.CDLL(path)
        functions = {
            'ncclGetUniqueId': [ctypes.POINTER(_UniqueId)],
            'ncclCommInitRank': [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_int,
                _UniqueId,
                ctypes.c_int,
            ],
            'ncclAllReduce': [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_void_p,
            ],
            'ncclCommAbort': [ctypes.c_void_p],
        }
        for name, args in functions.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = ctypes.c_int
        rank, size = dist.get_rank(group), dist.get_world_size(group)
        uid = _UniqueId()
        if rank == 0:
            self.check(self.lib.ncclGetUniqueId(ctypes.byref(uid)))
        encoded = torch.tensor(list(bytes(uid)), dtype=torch.uint8)
        dist.broadcast(encoded, src=dist.get_process_group_ranks(group)[0], group=group)
        ctypes.memmove(ctypes.byref(uid), bytes(encoded.tolist()), 128)
        self.comm = ctypes.c_void_p()
        self.check(self.lib.ncclCommInitRank(ctypes.byref(self.comm), size, uid, rank))
        self.all_reduce(torch.zeros(1, device=device))
        torch.cuda.current_stream(device).synchronize()

    @staticmethod
    def check(result):
        if result:
            raise RuntimeError(f'Parity NCCL call failed with status {result}')

    def all_reduce(self, tensor):
        assert tensor.device == self.device and tensor.is_contiguous()
        dtype = {torch.float32: 7, torch.float16: 6, torch.bfloat16: 9}[tensor.dtype]
        output = torch.empty_like(tensor)
        self.check(
            self.lib.ncclAllReduce(
                tensor.data_ptr(),
                output.data_ptr(),
                tensor.numel(),
                dtype,
                0,
                self.comm,
                torch.cuda.current_stream(self.device).cuda_stream,
            )
        )
        return output

    def close(self):
        """Let graph destruction proceed while NCCL abort releases its resources."""
        if not self.disabled:
            self.disabled = True

            def abort():
                with torch.cuda.device(self.device):
                    self.lib.ncclCommAbort(self.comm)

            thread = threading.Thread(target=abort, daemon=True)
            thread.start()
            thread.join(timeout=5)


class CustomAllreduce:
    """IPC buffers stay alive for the lifetime of all captured decode graphs."""

    def __init__(self, group, device, symm_mem_enabled):
        load_ops()
        self.ops = torch.ops.mcore_parity_ar
        self.group = group
        self.device = device
        self.rank, self.world_size = dist.get_rank(group), dist.get_world_size(group)
        self._IS_CAPTURING = False
        self.disabled = False
        self._ptr = 0
        if self.world_size not in (2, 4, 6, 8):
            raise ValueError('Parity custom all-reduce supports 2, 4, 6 or 8 GPUs per node')
        hosts = [None] * self.world_size
        dist.all_gather_object(hosts, os.uname().nodename, group=group)
        if len(set(hosts)) != 1:
            raise ValueError('Parity collective profile requires a single NVLink node')
        devices = [None] * self.world_size
        dist.all_gather_object(devices, device.index, group=group)
        if not all(
            i == device.index or torch.cuda.can_device_access_peer(device.index, i) for i in devices
        ):
            raise ValueError('Parity collective profile requires peer access between TP GPUs')
        self.fully_connected = True
        self.max_size = 8 * 1024 * 1024
        capability = '.'.join(map(str, torch.cuda.get_device_capability(device)))
        if symm_mem_enabled and capability in CUSTOM_ALL_REDUCE_MAX_SIZES:
            self.max_size = min(
                self.max_size, CUSTOM_ALL_REDUCE_MAX_SIZES[capability][self.world_size]
            )
        self.meta_ptrs = self.create_shared_buffer(self.ops.meta_size() + self.max_size)
        self.buffer_ptrs = self.create_shared_buffer(self.max_size)
        self.rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
        self._ptr = self.ops.init_custom_ar(self.meta_ptrs, self.rank_data, self.rank, True)
        self.ops.register_buffer(self._ptr, self.buffer_ptrs)

    def create_shared_buffer(self, size):
        pointer, handle = self.ops.allocate_shared_buffer_and_handle(size)
        handles = [None] * self.world_size
        dist.all_gather_object(handles, handle, group=self.group)
        return [
            pointer if i == self.rank else self.ops.open_mem_handle(h)
            for i, h in enumerate(handles)
        ]

    def should_custom_ar(self, tensor):
        size = tensor.numel() * tensor.element_size()
        return tensor.is_contiguous() and size % 16 == 0 and size < self.max_size

    @contextmanager
    def capture(self):
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            handle, offsets = self.ops.get_graph_buffer_ipc_meta(self._ptr)
            data = [[None, None] for _ in range(self.world_size)]
            data[self.rank] = [handle, offsets]
            for i, rank in enumerate(sorted(dist.get_process_group_ranks(self.group))):
                dist.broadcast_object_list(data[i], src=rank, group=self.group, device='cpu')
            self.ops.register_graph_buffers(self._ptr, [d[0] for d in data], [d[1] for d in data])

    def custom_all_reduce(self, tensor):
        output = torch.empty_like(tensor)
        if self._IS_CAPTURING and not torch.cuda.is_current_stream_capturing():
            return output
        registered = self._IS_CAPTURING
        self.ops.all_reduce(
            self._ptr,
            tensor,
            output,
            0 if registered else self.buffer_ptrs[self.rank],
            0 if registered else self.max_size,
        )
        return output

    def close(self):
        """Release graph registration metadata and this rank's IPC allocations."""
        if self._ptr:
            self.ops.dispose(self._ptr)
            self._ptr = 0
            self.ops.free_shared_buffer(self.meta_ptrs[self.rank])
            self.ops.free_shared_buffer(self.buffer_ptrs[self.rank])


class CudaCommunicator:
    """Reference native policy: custom kernel, symmetric memory, then NCCL."""

    def __init__(self, cpu_group, device):
        self.size = dist.get_world_size(cpu_group)
        self.ca_comm = None
        if self.size == 1:
            return
        self.pynccl_comm = PyNcclCommunicator(cpu_group, device)
        self.symm_mem_comm = SymmMemCommunicator(cpu_group, device)
        self.ca_comm = CustomAllreduce(cpu_group, device, not self.symm_mem_comm.disabled)

    def all_reduce(self, tensor):
        if self.size == 1:
            return tensor
        if self.ca_comm.should_custom_ar(tensor):
            return self.ca_comm.custom_all_reduce(tensor)
        if self.symm_mem_comm.should_use_symm_mem(tensor):
            return self.symm_mem_comm.all_reduce(tensor)
        return self.pynccl_comm.all_reduce(tensor)

    def __del__(self):
        if sys.is_finalizing():
            return
        if getattr(self, 'ca_comm', None) is not None:
            self.ca_comm.close()
        if getattr(self, 'pynccl_comm', None) is not None:
            self.pynccl_comm.close()
