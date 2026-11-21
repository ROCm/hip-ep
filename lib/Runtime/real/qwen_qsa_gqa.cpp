/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Qwen specialization inside the native GQA runtime: one EP stream and call.
#include "../hipdnn_ep_runtime.h"
#include "../op_state.h"
#include "../runtime_state_internal.h"
#include "hip_qwen_qsa.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <map>
#include <vector>

extern "C" OpState *hipdnn_ep_qwen_qsa_create_dense_state();
extern "C" int hipdnn_ep_qwen_qsa_dense_attention(RuntimeState *, OpState *,
                                                  void *, void *, void *,
                                                  void *, void *, void *,
                                                  void *, void *, void *,
                                                  int64_t, int64_t, int64_t);

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
  bool init(hipblasLtHandle_t handle, int rows) {
    hipblasOperation_t trans = HIPBLAS_OP_T;
    if (hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) ||
        hipblasLtMatmulDescSetAttribute(desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                        &trans, sizeof(trans)) ||
        hipblasLtMatrixLayoutCreate(&a, HIP_R_16BF, 5120, 640, 5120) ||
        hipblasLtMatrixLayoutCreate(&b, HIP_R_16BF, 5120, rows, 5120) ||
        hipblasLtMatrixLayoutCreate(&c, HIP_R_16BF, 640, rows, 640))
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
struct QwenQsaState : OpStateT<QwenQsaState> {
  Buffer hidden, projected, features, pool, tail, scores, selected, partial,
      stats, blas;
  std::map<int, std::unique_ptr<Gemm>> gemms;
  OpState *dense = hipdnn_ep_qwen_qsa_create_dense_state();
  int seen = 0, capacity = 0, batch = 0;
  void *keys = nullptr, *values = nullptr;
  ~QwenQsaState() {
    if (dense)
      dense->deletor(dense);
  }
  bool reserve(int B, int S, int C) {
    size_t rows = size_t(B) * S, chunks = size_t(B) * (S < Chunk ? S : Chunk);
    int P = (C + 3) / 4;
    int splits = S == 1 ? 16 : 1;
    return (S == 1 || hidden.reserve(rows * 5120 * 2)) &&
           projected.reserve(rows * 640 * 2) &&
           features.reserve(rows * 512 * 2) &&
           pool.reserve(size_t(B) * P * 128 * 2) &&
           tail.reserve(size_t(B) * 4 * 128 * 2) &&
           scores.reserve(chunks * P * 4) &&
           selected.reserve(chunks * 513 * 4) &&
           partial.reserve(chunks * 24 * splits * 256 * 4) &&
           stats.reserve(chunks * 24 * splits * 2 * 4);
  }
};
int report_failure(RuntimeState *state, const char *message) {
  fprintf(stderr, "QwenQsaGroupQueryAttention: %s\n", message);
  if (state && state->device_error_flag)
    hipMemsetAsync(state->device_error_flag, 0xff, sizeof(int), state->stream);
  return -1;
}
} // namespace

extern "C" int8_t hipdnn_ep_op_state_construct_qwen_qsa_gqa(RuntimeState *state,
                                                            int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, QwenQsaState::create().release());
  return 0;
}

extern "C" int wrap_qwen_qsa_group_query_attention(
    RuntimeState *state, int slot, void *query, void *key, void *value,
    void *past_key, void *past_value, void *seqlens_k, void *total_seq_len,
    void *hidden, void *projection, void *q_norm, void *token_k_norm,
    void *pool_k_norm, void *cos, void *sin, void *positions, void *output,
    void *present_key, void *present_value, int64_t B64, int64_t S64,
    int64_t C64, int64_t D64) {
  auto failure = [state](const char *message) {
    return report_failure(state, message);
  };
  if (!state || B64 < 1 || B64 > 8 || S64 < 1 || C64 < S64 || C64 > 262144 ||
      D64 != 256 || !query || !key || !value || !past_key || !past_value ||
      !seqlens_k || !total_seq_len || !hidden || !projection || !q_norm ||
      !token_k_norm || !pool_k_norm || !cos || !sin || !positions || !output ||
      !present_key || !present_value || !state->device_error_flag)
    return failure("invalid geometry or null input");
  auto *qs = QwenQsaState::get_op_state(state, slot);
  if (!qs || !qs->dense)
    return failure("missing per-node state");
  auto stream = static_cast<hipStream_t>(hipdnn_ep_state_get_stream(state));
  int B = int(B64), S = int(S64), C = int(C64), P = (C + 3) / 4;
  const char *gate = std::getenv("HIPDNN_EP_GQA_CACHE_SEQLENS");
  bool cache = B == 1 && (!gate || std::strcmp(gate, "0") != 0);
  int32_t length = -2;
  if (cache && state->seqlens_k_cached_valid &&
      state->seqlens_k_cached_ptr == seqlens_k)
    length = state->seqlens_k_cached_val;
  else {
    std::vector<int32_t> lengths(B);
    if (hipMemcpyAsync(lengths.data(), seqlens_k, B * sizeof(int32_t),
                       hipMemcpyDeviceToHost, stream) ||
        hipStreamSynchronize(stream))
      return failure("sequence length read failed");
    length = lengths[0];
    for (int value : lengths)
      if (value != length)
        return failure("requires uniform unpadded batch lengths");
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
  if (past && (qs->seen != past || qs->capacity != C || qs->batch != B ||
               qs->keys != past_key || qs->values != past_value ||
               present_key != past_key || present_value != past_value))
    return failure("decode requires matching in-place cache and prior prefill "
                   "on this node");
  if (!past) {
    int device = 0;
    hipDeviceProp_t prop{};
    if (hipGetDevice(&device) || hipGetDeviceProperties(&prop, device) ||
        prop.warpSize != 32 || std::strncmp(prop.gcnArchName, "gfx1151", 7))
      return failure("currently implemented for gfx1151 wave32 only");
  }
  if (!qs->reserve(B, S, C))
    return failure("workspace allocation failed");
  if (S == 1) {
    if (hip_qwen_qsa_project_decode(stream, hidden, projection, positions,
                                    qs->projected.ptr, state->device_error_flag,
                                    B, past))
      return failure("selector projection launch failed");
  } else {
    if (hip_qwen_qsa_cast_hidden(stream, hidden, positions, qs->hidden.ptr,
                                 state->device_error_flag, B, S, past))
      return failure("BF16 input cast failed");
    auto handle = static_cast<hipblasLtHandle_t>(
        hipdnn_ep_state_get_hipblas_handle(state));
    if (!handle)
      return failure("BF16 selector prefill requires hipBLASLt");
    auto &entry = qs->gemms[B * S];
    if (!entry) {
      auto next = std::make_unique<Gemm>();
      if (!next->init(handle, B * S))
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
  if (hip_qwen_qsa_features(stream, qs->projected.ptr, q_norm, token_k_norm,
                            pool_k_norm, cos, sin, qs->features.ptr,
                            qs->pool.ptr, qs->tail.ptr, B, S, past, P))
    return failure("selector feature launch failed");
  if (past + S <= 2052) {
    if (hipdnn_ep_qwen_qsa_dense_attention(
            state, qs->dense, query, key, value, past_key, past_value,
            seqlens_k, output, present_key, present_value, B, S, C))
      return failure("full-support native attention failed");
  } else {
    if (hip_qwen_qsa_pack_kv(stream, key, value, present_key, present_value, B,
                             S, past, C))
      return failure("native KV append failed");
    for (int first = 0; first < S; first += Chunk) {
      int count = S - first < Chunk ? S - first : Chunk;
      if (hip_qwen_qsa_select(stream, qs->features.ptr, qs->pool.ptr,
                              qs->scores.ptr, qs->selected.ptr, B, S, past,
                              first, count, P) ||
          hip_qwen_qsa_attend(stream, query, present_key, present_value,
                              qs->selected.ptr, qs->partial.ptr, qs->stats.ptr,
                              output, B, S, past, first, count, C))
        return failure("selection/attention launch failed");
    }
  }
  qs->seen = past + S;
  qs->capacity = C;
  qs->batch = B;
  qs->keys = present_key;
  qs->values = present_value;
  return 0;
}
