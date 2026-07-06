/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Memory-bandwidth microbenchmark (STREAM-style) to CALIBRATE the roofline peak
// instead of assuming a spec number. Measures:
//   - copy  BW  = read+write (2 x bytes / time), the classic STREAM-copy metric
//   - read  BW  = read-only grid-stride reduction (1 x bytes / time)
// on a large device buffer. Used once at EP init when HIPDNN_EP_BW_PROBE=1.
//===----------------------------------------------------------------------===//

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

__global__ void bw_copy_kernel(const float4 *__restrict__ in,
                               float4 *__restrict__ out, int64_t n4) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (; i < n4; i += stride)
    out[i] = in[i];
}

__global__ void bw_read_kernel(const float4 *__restrict__ in, int64_t n4,
                               float *__restrict__ sink) {
  int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  int64_t stride = (int64_t)gridDim.x * blockDim.x;
  float acc = 0.0f;
  for (; i < n4; i += stride) {
    float4 v = in[i];
    acc += v.x + v.y + v.z + v.w;
  }
  // Prevent the read loop from being optimized away.
  if (acc == -1.0f && sink)
    sink[0] = acc;
}

extern "C" int hip_bw_probe(void *stream, double *copy_gbps, double *read_gbps) {
  hipStream_t s = static_cast<hipStream_t>(stream);
  const int64_t bytes = 256LL * 1024 * 1024; // 256 MB
  const int64_t n4 = bytes / (int64_t)sizeof(float4);
  const int iters = 30;
  const int warmup = 5;

  float4 *in = nullptr, *out = nullptr;
  float *sink = nullptr;
  if (hipMalloc(&in, bytes) != hipSuccess ||
      hipMalloc(&out, bytes) != hipSuccess ||
      hipMalloc(&sink, sizeof(float)) != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_bw_probe: hipMalloc failed\n");
    if (in) hipFree(in);
    if (out) hipFree(out);
    if (sink) hipFree(sink);
    return -1;
  }
  hipMemsetAsync(in, 1, bytes, s);

  const int block = 256;
  int grid = (int)((n4 + block - 1) / block);
  if (grid > 65535) grid = 65535;

  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);

  auto timeKernel = [&](bool copy) -> double {
    for (int i = 0; i < warmup; ++i) {
      if (copy)
        hipLaunchKernelGGL(bw_copy_kernel, dim3(grid), dim3(block), 0, s, in,
                           out, n4);
      else
        hipLaunchKernelGGL(bw_read_kernel, dim3(grid), dim3(block), 0, s, in, n4,
                           sink);
    }
    hipStreamSynchronize(s);
    hipEventRecord(e0, s);
    for (int i = 0; i < iters; ++i) {
      if (copy)
        hipLaunchKernelGGL(bw_copy_kernel, dim3(grid), dim3(block), 0, s, in,
                           out, n4);
      else
        hipLaunchKernelGGL(bw_read_kernel, dim3(grid), dim3(block), 0, s, in, n4,
                           sink);
    }
    hipEventRecord(e1, s);
    hipEventSynchronize(e1);
    float ms = 0.0f;
    hipEventElapsedTime(&ms, e0, e1);
    return ms / iters; // ms per iteration
  };

  double copyMs = timeKernel(true);
  double readMs = timeKernel(false);
  if (copy_gbps)
    *copy_gbps = copyMs > 0 ? (2.0 * bytes / 1e9) / (copyMs / 1000.0) : 0.0;
  if (read_gbps)
    *read_gbps = readMs > 0 ? ((double)bytes / 1e9) / (readMs / 1000.0) : 0.0;

  hipEventDestroy(e0);
  hipEventDestroy(e1);
  hipFree(in);
  hipFree(out);
  hipFree(sink);
  return 0;
}
