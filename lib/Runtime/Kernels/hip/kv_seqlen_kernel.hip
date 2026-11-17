/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Valid-row counts to the GQA seqlens_k form on the device, so an attention
// over a nonpad_kv_seqlen-bounded buffer needs no host sync per step.

#include <hip/hip_runtime.h>
#include <cstdint>

#include "hip_custom_kernels.h"

namespace {

__global__ void lengths_to_seqlens_k_kernel(const int64_t *__restrict__ lengths,
                                            int32_t *__restrict__ seqlens_k,
                                            int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n)
    seqlens_k[i] = static_cast<int32_t>(lengths[i] - 1);
}

} // namespace

extern "C" int hip_lengths_to_seqlens_k(void *stream, const int64_t *lengths,
                                        int32_t *seqlens_k, int64_t n) {
  if (n <= 0)
    return 0;
  hipStream_t s = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  const int block = 64;
  hipLaunchKernelGGL(lengths_to_seqlens_k_kernel,
                     dim3(static_cast<unsigned>((n + block - 1) / block)),
                     dim3(block), 0, s, lengths, seqlens_k, n);
  return hipGetLastError() == hipSuccess ? 0 : -1;
}
