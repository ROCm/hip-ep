/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "../hipdnn_ep_runtime.h"
#include "../op_profile.h"
#include "../op_state.h"
#include "dense_gemm.h"

namespace {
struct GemmState : OpStateT<GemmState> {};
} // namespace

extern "C" int8_t hipdnn_ep_op_state_construct_gemm(RuntimeState *state,
                                                    int32_t slot) {
  hipdnn_ep_op_state_set(state, slot, GemmState::create().release());
  return 0;
}

int wrap_gemm_typed(RuntimeState *state, int op_state_slot, const void *A,
                    const void *B, const void *C, void *output, int64_t M,
                    int64_t N, int64_t K, float alpha, float beta,
                    int64_t transA, int64_t transB, int64_t typeCode,
                    int64_t cDim0, int64_t cDim1, int64_t outputCode) {
  OP_PROFILE(
      "gemm",
      [&] {
        char text[64];
        snprintf(text, sizeof(text), "m=%lld,n=%lld,k=%lld", (long long)M,
                 (long long)N, (long long)K);
        return std::string(text);
      },
      state);
  (void)op_state_slot;
  if (!state || typeCode < 0 || typeCode > 3 || outputCode < 0 ||
      outputCode > 3)
    return -1;
  using namespace hipdnn_ep::dense_gemm;
  size_t gemmBytes, savedBytes = 0, scratchOffset = 0;
  if (!workspaceSize(M, N, K, transA, transB, static_cast<int>(typeCode),
                     static_cast<int>(outputCode), 1, gemmBytes))
    return -1;
  const bool saveC = C && C == output && beta != 0.0f && M && N;
  if (saveC) {
    int64_t count;
    if ((cDim0 != 1 && cDim0 != M) || (cDim1 != 1 && cDim1 != N) ||
        !multiply(cDim0, cDim1, count) ||
        !byteSize(count, elementBytes(static_cast<int>(outputCode)),
                  savedBytes) ||
        savedBytes > SIZE_MAX - 63)
      return -1;
    scratchOffset = (savedBytes + 63) & ~size_t(63);
  }
  if (gemmBytes > SIZE_MAX - scratchOffset)
    return -1;
  const size_t totalBytes = scratchOffset + gemmBytes;
  if (totalBytes && hipdnn_ep_state_ensure_workspace(state, totalBytes))
    return -1;
  char *workspace = static_cast<char *>(hipdnn_ep_state_get_workspace(state));
  void *stream = hipdnn_ep_state_get_stream(state);
  if (saveC) {
    // Keep beta*C immutable during online config selection and across
    // broadcast output tiles. The session workspace uses the transient arena
    // and is reused safely by operations submitted on this same stream.
    hipError_t status =
        hipMemcpyAsync(workspace, C, savedBytes, hipMemcpyDeviceToDevice,
                       static_cast<hipStream_t>(stream));
    if (status != hipSuccess)
      return static_cast<int>(status);
    C = workspace;
  }
  return launchBatched(stream, A, B, C, output, M, N, K, alpha, beta, transA,
                       transB, static_cast<int>(typeCode),
                       static_cast<int>(outputCode), 1, 0, 0, 0, M * N,
                       gemmBytes ? workspace + scratchOffset : nullptr,
                       gemmBytes, cDim0, cDim1);
}

int wrap_gemm(RuntimeState *state, int op_state_slot, const void *A,
              const void *B, const void *C, void *output, int64_t M, int64_t N,
              int64_t K, float alpha, float beta, int64_t transA,
              int64_t transB, int64_t typeCode, int64_t cDim0, int64_t cDim1) {
  return wrap_gemm_typed(state, op_state_slot, A, B, C, output, M, N, K, alpha,
                         beta, transA, transB, typeCode, cDim0, cDim1,
                         typeCode);
}
