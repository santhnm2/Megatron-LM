# Local numerical parity primitives

This package implements the audited BF16 Nemotron-H execution policy without
importing the `vllm` Python package or loading its extension libraries.

Sources are pinned to vLLM `v0.25.1` and its FlashAttention source revision
`2c839c33742309ec41e620bf837495ec9926c56e`. The source hashes and original paths
are recorded in `UPSTREAM.json`. The Apache 2.0 and BSD licenses are included
as `LICENSE.vllm` and `LICENSE.flash-attention`. Copyright notices are retained.

## Components

- `moe.py` and `moe_gemm.py`: sigmoid/bias routing, BF16 Triton expert GEMMs,
  separate ReLU and square, and ordered expert summation. The reference's
  native GB300 fallback policy is retained. No device/shape tuning tables are
  bundled: the pinned reference has no GB300 table. Other device profiles must
  supply the reference's complete BF16 table directory through
  `MEGATRON_PARITY_MOE_CONFIG_DIR` to preserve its table-or-fallback choice.
- `collectives.py`, `symm_mem.py`, and `collective_sizes.py`: reference dispatch
  thresholds for custom IPC all-reduce, PyTorch symmetric memory, and a
  separate NCCL communicator, using explicit caller-owned process groups.
- `csrc/`: only the CUDA translation units and headers for routing, alignment,
  expert summation, RMSNorm and custom all-reduce. The bindings register local
  `mcore_parity` / `mcore_parity_ar` namespaces. Internal C++ namespaces are
  renamed to avoid collisions with an independently loaded audit reference.
- `cute/`: only the BF16 Blackwell SM10x forward/combine entry point and its
  transitive local dependencies. Autograd/backward, benchmark/testing utilities,
  unused architecture kernels, MLA and head-dimension-256 kernels are omitted.
  Supported head dimensions are 8–128, aligned as required by the kernel. The
  retained kernel/helper arithmetic and native split/tile/scheduler choices are
  unchanged. Unsupported attention profiles raise an explicit error.

## Build and dependencies

The local CUDA extension builds on first use through PyTorch's extension cache,
before graph capture. An offline installation needs the CUDA toolkit (including
`nvcc`), a C++20 compiler, Ninja and PyTorch 2.11 or newer. No code is downloaded
at runtime. `TORCH_EXTENSIONS_DIR`, `TORCH_CUDA_ARCH_LIST` and `MAX_JOBS` are the
standard PyTorch controls for the build cache, target architectures and jobs.

The `inference-parity` optional extra declares the standalone kernel dependencies.
The audited CUDA 13 versions are also listed in `requirements/inference-parity.txt`
for provisioning an existing nemoRL policy-worker environment. Install these
inside that environment before launching Ray. Keep its CUDA-enabled PyTorch
2.11.0 build and NCCL installation; the audit used Triton 3.6.0, CUTLASS DSL
4.5.2 and QuACK 0.4.1. See `docs/inference/vllm_numerical_parity.md` for setup.

The `dev` extra pins cuDNN's CUTLASS DSL dependency to 4.5.0, so uv treats `dev`
and `inference-parity` as mutually exclusive extras. Use the dedicated inference
worker environment for this profile. The existing default development pins are
preserved. `uv.lock` resolves both profiles independently.

The supported numerical profile remains native autotuning, BF16, full attention,
ReLU-squared MoE, and one NVLink node with TP-sharded experts (EP1/ETP=TP).
Alternative vLLM backend/environment profiles are not selected implicitly.
Optional local overrides use `MEGATRON_PARITY_MOE_CONFIG_DIR` and
`MEGATRON_PARITY_NCCL_SO_PATH`; equivalent choices must be supplied to a
reference comparison.

Local port validation (GB300, 2026-09-26): job 4022228 passed 19 MoE shapes,
7 attention shapes, 12 norm checks, 12 MoE graph replays, and 26 collective
checks on each of four ranks against the pinned reference. Local-only MoE,
attention and collective execution ran before exposing the reference package.
Job 4022390 then repeated all seven integrated scenarios: all 92 full-logit
events, 2,116 Mamba and 552 KV state comparisons, returned tokens/logprobs and
184 observer controls pass byte-for-byte under matched autotuning choices.

The reduced package contains 52 files (previously 217). The 23 CuTe files
are exactly the transitive local import closure of the forward entry point,
including its optional PTXAS hook; the 18 C++/CUDA files are required translation
units and headers. Shared helpers remain intact to preserve upstream arithmetic.
After pruning, job 4023291 repeated all primitive/reference comparisons above
and passed all four graph-replay test families on each of four ranks with vLLM
imports blocked. The seven full-model scenarios were run before this pruning.
The same job built a wheel with exactly the 52 retained package files, checked
their bytes against the frozen source, and verified no vLLM dependency.

`tests/unit_tests/determinism/kernels/test_local_parity_kernels.py` provides
portable eager/graph replay checks with changing inputs and blocked vLLM imports.
It requires the audited CUDA/NVLink profile and pytest; run it directly with
four-rank torchrun to preserve production NCCL settings. This is an inference
package; it does not expose the upstream training/autograd API.
