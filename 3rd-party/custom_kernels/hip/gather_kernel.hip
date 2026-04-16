/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Generic gather kernel for axis=0 with arbitrary number of indices.
// Each thread copies one element of the output tensor.
// Thread tid maps to: index_idx = tid / slice_size, offset = tid % slice_size.
// output[tid] = data[indices[index_idx] * slice_size + offset]
template <typename T>
__global__ void gather_axis0_kernel(
    const T* __restrict__ data,
    const int64_t* __restrict__ indices,
    T* __restrict__ output,
    int64_t slice_size,
    int64_t data_axis_size,
    int64_t total_elements) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < total_elements) {
    int64_t index_idx = tid / slice_size;
    int64_t offset = tid % slice_size;
    int64_t index_val = indices[index_idx];
    if (index_val < 0) index_val += data_axis_size;
    if (index_val < 0 || index_val >= data_axis_size) {
      output[tid] = T(0);
      return;
    }
    output[tid] = data[index_val * slice_size + offset];
  }
}

extern "C" int hip_gather(
    void* stream,
    const void* data,
    const void* indices,
    void* output,
    int64_t axis,
    int64_t data_num_elements,
    int64_t indices_num_elements,
    int64_t output_num_elements,
    int element_size_bytes) {
  if (output_num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (output_num_elements + kBlockSize - 1) / kBlockSize);

  if (axis != 0) {
    fprintf(stderr,
            "[custom_kernels] hip_gather: only axis=0 supported, got %lld\n",
            (long long)axis);
    return -1;
  }

  // For axis=0: output = [indices_shape..., data_shape[1:]]
  // slice_size = product of data dimensions after axis 0
  // data_axis_size = data_shape[0]
  int64_t slice_size = (indices_num_elements > 0)
      ? output_num_elements / indices_num_elements
      : output_num_elements;
  int64_t data_axis_size = (slice_size > 0)
      ? data_num_elements / slice_size
      : 0;

  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_gather: axis=0, elem_size=%d, "
          "data_num=%lld, indices_num=%lld, output_num=%lld, "
          "slice_size=%lld, data_axis_size=%lld, grid=%d, block=%d\n",
          element_size_bytes,
          (long long)data_num_elements, (long long)indices_num_elements,
          (long long)output_num_elements,
          (long long)slice_size, (long long)data_axis_size,
          grid_size, kBlockSize);

  switch (element_size_bytes) {
    case 2:
      hipLaunchKernelGGL(gather_axis0_kernel<uint16_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const uint16_t*>(data),
                         static_cast<const int64_t*>(indices),
                         static_cast<uint16_t*>(output),
                         slice_size, data_axis_size, output_num_elements);
      break;
    case 4:
      hipLaunchKernelGGL(gather_axis0_kernel<uint32_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const uint32_t*>(data),
                         static_cast<const int64_t*>(indices),
                         static_cast<uint32_t*>(output),
                         slice_size, data_axis_size, output_num_elements);
      break;
    case 8:
      hipLaunchKernelGGL(gather_axis0_kernel<int64_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int64_t*>(data),
                         static_cast<const int64_t*>(indices),
                         static_cast<int64_t*>(output),
                         slice_size, data_axis_size, output_num_elements);
      break;
    default:
      fprintf(stderr,
              "[custom_kernels] hip_gather: unsupported element_size=%d\n",
              element_size_bytes);
      return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_gather launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
