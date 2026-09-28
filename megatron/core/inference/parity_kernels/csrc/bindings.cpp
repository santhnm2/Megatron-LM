// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Modified: expose only the primitives used by Megatron numerical parity.
#include "libtorch_stable/ops.h"
#include "libtorch_stable/moe/moe_ops.h"
#include <torch/csrc/stable/library.h>

STABLE_TORCH_LIBRARY_FRAGMENT(mcore_parity, m) {
m.def(
      "rms_norm(Tensor! result, Tensor input, Tensor? weight, float epsilon) "
      "-> "
      "()");
m.def(
      "fused_add_rms_norm(Tensor! input, Tensor! residual, Tensor? weight, "
      "float epsilon) -> ()");
m.def("moe_sum(Tensor input, Tensor! output) -> ()");
m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad,"
      "                     Tensor? maybe_expert_map) -> ()");
m.def(
      "grouped_topk(Tensor scores, int n_group, int "
      "topk_group, int topk, bool renormalize, float "
      "routed_scaling_factor, Tensor bias, int scoring_func) -> (Tensor, "
      "Tensor)");
}
STABLE_TORCH_LIBRARY_IMPL(mcore_parity, CUDA, m) {
m.impl("rms_norm", TORCH_BOX(&rms_norm));
m.impl("fused_add_rms_norm", TORCH_BOX(&fused_add_rms_norm));
m.impl("moe_sum", TORCH_BOX(&moe_sum));
m.impl("moe_align_block_size", TORCH_BOX(&moe_align_block_size));
m.impl("grouped_topk", TORCH_BOX(&grouped_topk));
}
STABLE_TORCH_LIBRARY_FRAGMENT(mcore_parity_ar, custom_ar) {
  custom_ar.def(
      "init_custom_ar(int[] ipc_tensors, Tensor rank_data, "
      "int rank, bool fully_connected) -> int");
  custom_ar.def(
      "all_reduce(int fa, Tensor inp, Tensor! out, int reg_buffer, "
      "int reg_buffer_sz_bytes) -> ()");
  custom_ar.def("dispose(int fa) -> ()");
  custom_ar.def("meta_size() -> int");
  custom_ar.def("register_buffer(int fa, int[] ipc_tensors) -> ()");
  custom_ar.def("get_graph_buffer_ipc_meta(int fa) -> (int[], int[])");
  custom_ar.def(
      "register_graph_buffers(int fa, int[][] handles, int[][] offsets) -> ()");
  custom_ar.def("allocate_shared_buffer_and_handle(int size) -> (int, Tensor)");
  custom_ar.def("open_mem_handle(Tensor mem_handle) -> int");
  custom_ar.def("free_shared_buffer(int ptr) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(mcore_parity_ar, CUDA, custom_ar) {
  custom_ar.impl("init_custom_ar", TORCH_BOX(&init_custom_ar));
  custom_ar.impl("all_reduce", TORCH_BOX(&all_reduce));
}

STABLE_TORCH_LIBRARY_IMPL(mcore_parity_ar, CPU, custom_ar) {
  custom_ar.impl("open_mem_handle", TORCH_BOX(&open_mem_handle));
}

STABLE_TORCH_LIBRARY_IMPL(mcore_parity_ar, CompositeExplicitAutograd, custom_ar) {
  custom_ar.impl("dispose", TORCH_BOX(&dispose));
  custom_ar.impl("meta_size", TORCH_BOX(&meta_size));
  custom_ar.impl("register_buffer", TORCH_BOX(&register_buffer));
  custom_ar.impl("get_graph_buffer_ipc_meta",
                 TORCH_BOX(&get_graph_buffer_ipc_meta));
  custom_ar.impl("register_graph_buffers", TORCH_BOX(&register_graph_buffers));
  custom_ar.impl("allocate_shared_buffer_and_handle",
                 TORCH_BOX(&allocate_shared_buffer_and_handle));
  custom_ar.impl("free_shared_buffer", TORCH_BOX(&free_shared_buffer));
}
