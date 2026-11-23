/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "roctx.h"
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

namespace {
void Check(hipError_t status, const char *call) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "%s failed: %s\n", call, hipGetErrorString(status));
    std::exit(1);
  }
}

__global__ void VectorKernel(float *values, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    values[index] = values[index] * 1.01f + 0.25f;
}

__global__ void MatrixKernel(float *values, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const float value = values[index];
    values[index] = value * value + 1.0f;
  }
}

__global__ void ActivationKernel(float *values, int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count)
    values[index] = values[index] > 0.0f ? values[index] : 0.0f;
}

int ParseRounds(int argc, char **argv) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--rounds") {
      return (std::max)(1, std::atoi(argv[i + 1]));
    }
  }
  return 200;
}
} // namespace

int main(int argc, char **argv) {
  constexpr int kElements = 4096;
  const int rounds = ParseRounds(argc, argv);
  float *values = nullptr;
  Check(hipSetDevice(0), "hipSetDevice");
  Check(hipMalloc(&values, kElements * sizeof(float)), "hipMalloc");
  Check(hipMemset(values, 0, kElements * sizeof(float)), "hipMemset");

  const dim3 block(256);
  const dim3 grid((kElements + block.x - 1) / block.x);

  roctxMark("startup");
  VectorKernel<<<grid, block>>>(values, kElements);

  roctxRangePush("iteration");
  roctxRangePush("vector_ops");
  for (int i = 0; i < rounds; ++i)
    VectorKernel<<<grid, block>>>(values, kElements);
  Check(hipDeviceSynchronize(), "hipDeviceSynchronize(vector_ops)");
  roctxRangePop();

  roctxRangePush("matrix_ops");
  for (int i = 0; i < rounds; ++i)
    MatrixKernel<<<grid, block>>>(values, kElements);
  Check(hipDeviceSynchronize(), "hipDeviceSynchronize(matrix_ops)");
  roctxRangePop();
  roctxRangePop();

  const roctx_range_id_t cross_thread =
      roctxRangeStart("cross_thread_activation");
  for (int i = 0; i < rounds; ++i)
    ActivationKernel<<<grid, block>>>(values, kElements);
  Check(hipDeviceSynchronize(), "hipDeviceSynchronize(activation)");
  std::thread stop_thread([cross_thread] { roctxRangeStop(cross_thread); });
  stop_thread.join();

  roctxMark("complete");
  Check(hipFree(values), "hipFree");
  std::printf("ROCTx workload complete: rounds=%d dispatches=%d\n", rounds,
              1 + 3 * rounds);
  return 0;
}
