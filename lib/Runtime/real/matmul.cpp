/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "dense_gemm.h"

namespace {
struct MatmulState : OpStateT<MatmulState> {};
} // namespace

extern "C" int8_t hipdnn_ep_op_state_construct_matmul(RuntimeState *state,
                                                      int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, MatmulState::create().release());
  return 0;
}

// Historical symbol/ABI retained for tight, flat batches. elem_size is an
// element-kind marker: 2=f16, 4=f32, 8=f64, -2=bf16.
int wrap_matmul_typed(RuntimeState *state, int op_state_slot, const void *A,
                      const void *B, void *output, int64_t M, int64_t N,
                      int64_t K, int64_t batch_count, int64_t elem_size,
                      int64_t b_batch_stride, int64_t transA, int64_t transB,
                      int64_t outputCode) {
  OP_PROFILE(
      "matmul",
      [&] {
        char text[64];
        snprintf(text, sizeof(text), "m=%lld,n=%lld,k=%lld", (long long)M,
                 (long long)N, (long long)K);
        return std::string(text);
      },
      state);
  (void)op_state_slot;
  using namespace hipdnn_ep::dense_gemm;
  const int code = typeCode(elem_size);
  int64_t aStride, bMatrix, yStride, aElements, bLast, bElements, yElements;
  size_t aBytes, bBytes, yBytes;
  if (!state || outputCode < 0 || outputCode > 3 ||
      (outputCode != code && !((code == 0 || code == 3) && outputCode == 1)) ||
      !geometry(M, N, K, transA, transB, code) || batch_count < 0 ||
      b_batch_stride < 0 || !multiply(M, K, aStride) ||
      !multiply(K, N, bMatrix) || !multiply(M, N, yStride) ||
      !multiply(aStride, batch_count, aElements) ||
      !multiply(b_batch_stride, batch_count ? batch_count - 1 : 0, bLast) ||
      !add(bLast, batch_count ? bMatrix : 0, bElements) ||
      !multiply(yStride, batch_count, yElements) ||
      !byteSize(aElements, elementBytes(code), aBytes) ||
      !byteSize(bElements, elementBytes(code), bBytes) ||
      !byteSize(yElements, elementBytes(static_cast<int>(outputCode)), yBytes))
    return -1;
  if (!batch_count || !M || !N)
    return 0;
  if ((K && (!A || !B)) || !output)
    return -1;
  void *stream = hipdnn_ep_state_get_stream(state);
  OP_ROCTX_MARK("hip_ep.op.matmul.run.custom_gemm");
  size_t workspaceBytes;
  if (!workspaceSize(M, N, K, transA, transB, code,
                     static_cast<int>(outputCode), batch_count, workspaceBytes))
    return -1;
  if (workspaceBytes && hipdnn_ep_state_ensure_workspace(state, workspaceBytes))
    return -1;
  return launchBatched(stream, A, B, nullptr, output, M, N, K, 1.0f, 0.0f,
                       transA, transB, code, static_cast<int>(outputCode),
                       batch_count, aStride, b_batch_stride, 0, yStride,
                       hipdnn_ep_state_get_workspace(state), workspaceBytes);
}

int wrap_matmul_broadcast_typed(
    RuntimeState *state, int op_state_slot, const void *A, const void *B,
    void *output, int64_t M, int64_t N, int64_t K, int64_t elem_size,
    int64_t transA, int64_t transB, int64_t a_batch_rank,
    const int64_t *a_batch_dims, int64_t b_batch_rank,
    const int64_t *b_batch_dims, int64_t outputCode) {
  OP_PROFILE(
      "matmul",
      [&] {
        char text[64];
        snprintf(text, sizeof(text), "m=%lld,n=%lld,k=%lld", (long long)M,
                 (long long)N, (long long)K);
        return std::string(text);
      },
      state);
  (void)op_state_slot;
  using namespace hipdnn_ep::dense_gemm;
  const int code = typeCode(elem_size);
  int64_t aMatrix, bMatrix, yMatrix, yElements;
  size_t aBytes, bBytes, yBytes;
  BatchLayout layout;
  if (!state || outputCode < 0 || outputCode > 3 ||
      (outputCode != code && !((code == 0 || code == 3) && outputCode == 1)) ||
      !geometry(M, N, K, transA, transB, code) || a_batch_rank < 0 ||
      a_batch_rank > INT_MAX || b_batch_rank < 0 || b_batch_rank > INT_MAX ||
      !multiply(M, K, aMatrix) || !multiply(K, N, bMatrix) ||
      !multiply(M, N, yMatrix) ||
      !layout.initialize(static_cast<int>(a_batch_rank), a_batch_dims,
                         static_cast<int>(b_batch_rank), b_batch_dims, aMatrix,
                         bMatrix) ||
      !multiply(yMatrix, layout.count, yElements) ||
      !byteSize(layout.aElements, elementBytes(code), aBytes) ||
      !byteSize(layout.bElements, elementBytes(code), bBytes) ||
      !byteSize(yElements, elementBytes(static_cast<int>(outputCode)),
                yBytes)) {
    fprintf(stderr, "MatMul: invalid or unsupported batch broadcast\n");
    return -1;
  }
  if (!layout.count || !M || !N)
    return 0;
  if ((K && (!A || !B)) || !output)
    return -1;
  void *stream = hipdnn_ep_state_get_stream(state);
  OP_ROCTX_MARK("hip_ep.op.matmul.run.custom_gemm");
  const size_t bytes = elementBytes(code);
  const size_t outputBytes = elementBytes(static_cast<int>(outputCode));
  int64_t aStride, bStride;
  const bool uniform =
      layout.count <= 65535 && layout.uniformStrides(aStride, bStride);
  size_t workspaceBytes;
  if (!workspaceSize(M, N, K, transA, transB, code,
                     static_cast<int>(outputCode), uniform ? layout.count : 1,
                     workspaceBytes))
    return -1;
  if (workspaceBytes && hipdnn_ep_state_ensure_workspace(state, workspaceBytes))
    return -1;
  void *workspace = hipdnn_ep_state_get_workspace(state);
  if (uniform)
    return launchBatched(stream, A, B, nullptr, output, M, N, K, 1.0f, 0.0f,
                         transA, transB, code, static_cast<int>(outputCode),
                         layout.count, aStride, bStride, 0, yMatrix, workspace,
                         workspaceBytes);
  for (int64_t batch = 0; batch < layout.count; ++batch) {
    int64_t aOffset, bOffset;
    layout.offsets(batch, aOffset, bOffset);
    size_t aByteOffset, bByteOffset;
    if (!byteSize(aOffset, bytes, aByteOffset) ||
        !byteSize(bOffset, bytes, bByteOffset))
      return -1;
    int rc = launchBatched(
        stream, A ? static_cast<const char *>(A) + aByteOffset : nullptr,
        B ? static_cast<const char *>(B) + bByteOffset : nullptr, nullptr,
        static_cast<char *>(output) + batch * yMatrix * outputBytes, M, N, K,
        1.0f, 0.0f, transA, transB, code, static_cast<int>(outputCode), 1, 0, 0,
        0, yMatrix, workspace, workspaceBytes);
    if (rc)
      return rc;
  }
  return 0;
}

int wrap_hipblasLtMatmul(RuntimeState *state, int op_state_slot, const void *A,
                         const void *B, void *output, int64_t M, int64_t N,
                         int64_t K, int64_t batch_count, int64_t elem_size,
                         int64_t b_batch_stride, int64_t transA,
                         int64_t transB) {
  return wrap_matmul_typed(state, op_state_slot, A, B, output, M, N, K,
                           batch_count, elem_size, b_batch_stride, transA,
                           transB, hipdnn_ep::dense_gemm::typeCode(elem_size));
}

int wrap_matmul_broadcast(RuntimeState *state, int op_state_slot, const void *A,
                          const void *B, void *output, int64_t M, int64_t N,
                          int64_t K, int64_t elem_size, int64_t transA,
                          int64_t transB, int64_t a_batch_rank,
                          const int64_t *a_batch_dims, int64_t b_batch_rank,
                          const int64_t *b_batch_dims) {
  return wrap_matmul_broadcast_typed(
      state, op_state_slot, A, B, output, M, N, K, elem_size, transA, transB,
      a_batch_rank, a_batch_dims, b_batch_rank, b_batch_dims,
      hipdnn_ep::dense_gemm::typeCode(elem_size));
}

int wrap_matmul_i32(RuntimeState *state, const void *A, const void *B,
                    void *output, int64_t M, int64_t N, int64_t K,
                    int64_t batch_count, int64_t b_batch_stride) {
  if (!state || !A || !B || !output) {
    fprintf(stderr, "wrap_matmul_i32: null argument\n");
    return -1;
  }
  void *stream = hipdnn_ep_state_get_stream(state);
  int rc = hip_matmul_i32(stream, A, B, output, M, N, K, batch_count,
                          b_batch_stride);
  if (rc != 0)
    fprintf(stderr, "wrap_matmul_i32: hip_matmul_i32 failed (%d)\n", rc);
  return rc;
}
