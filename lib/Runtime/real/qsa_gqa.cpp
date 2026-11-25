/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Native GQA specialization: one runtime call, EP stream and per-node state.
#include "../hipdnn_ep_runtime.h"
#include "../op_state.h"
#include "../runtime_state_internal.h"
#include "hip_qsa_gqa.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <map>

namespace {
constexpr int Chunk = 32;
struct Buffer {
  void *ptr = nullptr;
  size_t size = 0;
  ~Buffer() {
    if (ptr)
      hipFree(ptr);
  }
  bool reserve(size_t bytes) {
    if (bytes <= size)
      return true;
    void *next = nullptr;
    if (hipMalloc(&next, bytes) != hipSuccess)
      return false;
    if (ptr)
      hipFree(ptr);
    ptr = next;
    size = bytes;
    return true;
  }
};
struct Gemm {
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
  hipblasLtMatmulAlgo_t algo{};
  size_t workspace = 0;
  ~Gemm() {
    if (desc)
      hipblasLtMatmulDescDestroy(desc);
    if (a)
      hipblasLtMatrixLayoutDestroy(a);
    if (b)
      hipblasLtMatrixLayoutDestroy(b);
    if (c)
      hipblasLtMatrixLayoutDestroy(c);
  }
  bool init(hipblasLtHandle_t handle, int S) {
    hipblasOperation_t trans = HIPBLAS_OP_T;
    if (hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) ||
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                        &trans, sizeof(trans)) ||
        hipblasLtMatrixLayoutCreate(&a, HIP_R_16BF, 4096, 640, 4096) ||
        hipblasLtMatrixLayoutCreate(&b, HIP_R_16BF, 4096, S, 4096) ||
        hipblasLtMatrixLayoutCreate(&c, HIP_R_16BF, 640, S, 640))
      return false;
    hipblasLtMatmulPreference_t pref = nullptr;
    if (hipblasLtMatmulPreferenceCreate(&pref))
      return false;
    size_t limit = 32u << 20;
    hipblasLtMatmulPreferenceSetAttribute(
        pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &limit, sizeof(limit));
    hipblasLtMatmulHeuristicResult_t result{};
    int count = 0;
    auto status = hipblasLtMatmulAlgoGetHeuristic(handle, desc, a, b, c, c,
                                                  pref, 1, &result, &count);
    hipblasLtMatmulPreferenceDestroy(pref);
    if (status || count != 1 || result.state != HIPBLAS_STATUS_SUCCESS)
      return false;
    algo = result.algo;
    workspace = result.workspaceSize;
    return true;
  }
};
struct QsaState : OpStateT<QsaState> {
  Buffer hidden, projected, features, pool, tail, scores, selected, partial,
      stats, blas;
  std::map<int, std::unique_ptr<Gemm>> gemms;
  int seen = 0, capacity = 0;
  void *keys = nullptr, *values = nullptr;
  bool reserve(int S, int C) {
    int P = (C + 3) / 4;
    return hidden.reserve(size_t(S) * 4096 * 2) &&
           projected.reserve(size_t(S) * 640 * 2) &&
           features.reserve(size_t(S) * 512 * 2) &&
           pool.reserve(size_t(P) * 128 * 2) && tail.reserve(4 * 128 * 2) &&
           scores.reserve(size_t(Chunk) * P * 4) &&
           selected.reserve(Chunk * 128 * 4) &&
           partial.reserve(Chunk * 32 * 8 * 128 * 4) &&
           stats.reserve(Chunk * 32 * 8 * 2 * 4);
  }
};
int report_failure(RuntimeState *state, const char *message) {
  fprintf(stderr, "QsaGroupQueryAttention: %s\n", message);
  // Runtime call return values alone are not propagated by every lowering.
  // Use the existing EP error flag so inference_compute fails visibly.
  if (state && state->device_error_flag)
    hipMemsetAsync(state->device_error_flag, 0xff, sizeof(int), state->stream);
  return -1;
}
} // namespace

extern "C" int8_t hipdnn_ep_op_state_construct_qsa_gqa(RuntimeState *state,
                                                       int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, QsaState::create().release());
  return 0;
}

extern "C" int wrap_qsa_group_query_attention(
    RuntimeState *state, int slot, void *query, void *key, void *value,
    void *past_key, void *past_value, void *seqlens_k, void *total_seq_len,
    void *hidden, void *projection, void *q_norm, void *k_norm, void *cos,
    void *sin, void *output, void *present_key, void *present_value, int64_t B,
    int64_t S64, int64_t C64) {
  auto failure = [state](const char *message) {
    return report_failure(state, message);
  };
  if (!state || B != 1 || S64 < 1 || C64 < S64 || C64 > 65536 || !query ||
      !key || !value || !past_key || !past_value || !seqlens_k ||
      !total_seq_len || !hidden || !projection || !q_norm || !k_norm || !cos ||
      !sin || !output || !present_key || !present_value ||
      !state->device_error_flag)
    return failure("invalid geometry or null input");
  auto *qs = QsaState::get_op_state(state, slot);
  if (!qs)
    return failure("missing per-node state");
  auto stream = static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  int S = int(S64), C = int(C64), P = (C + 3) / 4;
  // Reuse GQA's existing per-Compute readback. Layers preceding QSA normally
  // populate it, so this adds no host synchronization to the original graph.
  const char *gate = std::getenv("HIPDNN_EP_GQA_CACHE_SEQLENS");
  bool cache = !gate || std::strcmp(gate, "0") != 0;
  int32_t length = -2;
  if (cache && state->seqlens_k_cached_valid &&
      state->seqlens_k_cached_ptr == seqlens_k)
    length = state->seqlens_k_cached_val;
  else {
    if (hipMemcpyAsync(&length, seqlens_k, sizeof(length),
                       hipMemcpyDeviceToHost, stream) ||
        hipStreamSynchronize(stream))
      return failure("sequence length read failed");
    if (cache) {
      state->seqlens_k_cached_valid = true;
      state->seqlens_k_cached_ptr = seqlens_k;
      state->seqlens_k_cached_val = length;
    }
  }
  int64_t past64 = length == -1 ? 0 : int64_t(length) + 1 - S;
  if (past64 < 0 || past64 > C - S || (past64 && S != 1))
    return failure("requires full prefill followed by one-token decode");
  int past = int(past64);
  if (past && (qs->seen != past || qs->capacity != C || qs->keys != past_key ||
               qs->values != past_value || present_key != past_key ||
               present_value != past_value))
    return failure("decode requires matching in-place cache and prior prefill "
                   "on this node");
  if (!past) {
    int device = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&device) || hipGetDeviceProperties(&prop, device) ||
        prop.warpSize != 32 || std::strncmp(prop.gcnArchName, "gfx1151", 7))
      return failure("currently qualified for gfx1151 wave32 only");
  }
  if (!qs->reserve(S, C))
    return failure("workspace allocation failed");
  if (S == 1) {
    if (hip_qsa_project_decode(stream, hidden, projection, qs->projected.ptr))
      return failure("selector projection launch failed");
  } else {
    if (hip_qsa_cast_hidden(stream, hidden, qs->hidden.ptr, S))
      return failure("BF16 input cast failed");
    auto handle = static_cast<hipblasLtHandle_t>(
        hipdnn_ep_state_get_hipblas_handle(state));
    if (!handle)
      return failure("BF16 selector prefill requires hipBLASLt");
    auto &entry = qs->gemms[S];
    if (!entry) {
      auto next = std::make_unique<Gemm>();
      if (!next->init(handle, S))
        return failure("BF16 selector GEMM unavailable");
      entry = std::move(next);
    }
    if (!qs->blas.reserve(entry->workspace))
      return failure("GEMM workspace allocation failed");
    float alpha = 1, beta = 0;
    if (hipblasLtMatmul(handle, entry->desc, &alpha, projection, entry->a,
                        qs->hidden.ptr, entry->b, &beta, qs->projected.ptr,
                        entry->c, qs->projected.ptr, entry->c, &entry->algo,
                        qs->blas.ptr, entry->workspace, stream))
      return failure("BF16 selector GEMM failed");
  }
  if (hip_qsa_features(stream, qs->projected.ptr, q_norm, k_norm, cos, sin,
                       qs->features.ptr, qs->pool.ptr, qs->tail.ptr, key, value,
                       present_key, present_value, 1, S, past, C, P, 1e-6f))
    return failure("feature/cache launch failed");
  for (int first = 0; first < S; first += Chunk) {
    int chunk = S - first < Chunk ? S - first : Chunk;
    if (hip_qsa_select(stream, qs->features.ptr, qs->pool.ptr, qs->scores.ptr,
                       qs->selected.ptr, 1, S, past, first, chunk, P) ||
        hip_qsa_attend(stream, query, present_key, present_value,
                       qs->selected.ptr, qs->partial.ptr, qs->stats.ptr, output,
                       1, S, past, first, chunk, C))
      return failure("selection/attention launch failed");
  }
  qs->seen = past + S;
  qs->capacity = C;
  qs->keys = present_key;
  qs->values = present_value;
  return 0;
}
