/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "udna_custom_kernels.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Generic gather kernel for axis=0 with a single scalar index.
// T is the element type used for copying (uint16_t, uint32_t, int64_t).
// Indices are always i64.
template <typename T>
__global__ void gather_axis0_kernel(
    const T* __restrict__ data,
    const int64_t* __restrict__ indices,
    T* __restrict__ output,
    int64_t slice_size,
    int64_t data_axis_size) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < slice_size) {
    int64_t index_val = indices[0];
    if (index_val < 0) index_val += data_axis_size;
    if (index_val < 0 || index_val >= data_axis_size) {
      output[idx] = T(0);
      return;
    }
    int64_t src_offset = index_val * slice_size + idx;
    output[idx] = data[src_offset];
  }
}

extern "C" int udna_gather(
    void* stream,
    const void* data,
    const void* indices,
    void* output,
    int64_t axis,
    int64_t data_num_elements,
    int64_t output_num_elements,
    int element_size_bytes) {
  if (output_num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (output_num_elements + kBlockSize - 1) / kBlockSize);

  if (axis != 0) {
    fprintf(stderr,
            "[custom_kernels] udna_gather: only axis=0 supported, got %lld\n",
            (long long)axis);
    return -1;
  }

  int64_t slice_size = output_num_elements;
  int64_t data_axis_size = data_num_elements / slice_size;

  fprintf(stderr,
          "[custom_kernels] udna_gather: axis=0, elem_size=%d, "
          "data_num=%lld, output_num=%lld, data_axis_size=%lld, "
          "grid=%d, block=%d\n",
          element_size_bytes,
          (long long)data_num_elements, (long long)output_num_elements,
          (long long)data_axis_size, grid_size, kBlockSize);

  switch (element_size_bytes) {
    case 2:
      hipLaunchKernelGGL(gather_axis0_kernel<uint16_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const uint16_t*>(data),
                         static_cast<const int64_t*>(indices),
                         static_cast<uint16_t*>(output),
                         slice_size, data_axis_size);
      break;
    case 4:
      hipLaunchKernelGGL(gather_axis0_kernel<uint32_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const uint32_t*>(data),
                         static_cast<const int64_t*>(indices),
                         static_cast<uint32_t*>(output),
                         slice_size, data_axis_size);
      break;
    case 8:
      hipLaunchKernelGGL(gather_axis0_kernel<int64_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int64_t*>(data),
                         static_cast<const int64_t*>(indices),
                         static_cast<int64_t*>(output),
                         slice_size, data_axis_size);
      break;
    default:
      fprintf(stderr,
              "[custom_kernels] udna_gather: unsupported element_size=%d\n",
              element_size_bytes);
      return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] udna_gather launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
