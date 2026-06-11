/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// Generic gather kernel for arbitrary axis.
// data    has logical shape [outer, axis_size, inner]
// indices has indices_num elements (flattened)
// output  has logical shape [outer, indices_num, inner]
// Each thread copies one element of the output tensor.
template <typename T>
__global__ void gather_kernel(
    const T* __restrict__ data,
    const int64_t* __restrict__ indices,
    T* __restrict__ output,
    int64_t outer_size,
    int64_t axis_size,
    int64_t inner_size,
    int64_t indices_num,
    int64_t total_elements) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements) return;
  int64_t inner_off = tid % inner_size;
  int64_t idx_off = (tid / inner_size) % indices_num;
  int64_t outer_off = tid / (inner_size * indices_num);
  int64_t index_val = indices[idx_off];
  if (index_val < 0) index_val += axis_size;
  if (index_val < 0 || index_val >= axis_size) {
    output[tid] = T(0);
    return;
  }
  int64_t data_off =
      (outer_off * axis_size + index_val) * inner_size + inner_off;
  output[tid] = data[data_off];
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
    int64_t axis_size,
    int64_t inner_size,
    int element_size_bytes) {
  if (output_num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (output_num_elements + kBlockSize - 1) / kBlockSize);

  // axis_size = data.shape[axis]; inner_size = product(data.shape[axis+1:])
  // outer_size derived from data layout: data has outer*axis_size*inner elems.
  int64_t outer_size = (axis_size > 0 && inner_size > 0)
      ? data_num_elements / (axis_size * inner_size)
      : 0;
  if (outer_size <= 0) {
    fprintf(stderr,
            "[custom_kernels] hip_gather: bad geometry axis=%lld data_num=%lld "
            "axis_size=%lld inner_size=%lld\n",
            (long long)axis, (long long)data_num_elements,
            (long long)axis_size, (long long)inner_size);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_gather: axis=%lld, elem_size=%d, "
          "data_num=%lld, indices_num=%lld, output_num=%lld, "
          "outer=%lld, axis_size=%lld, inner=%lld, grid=%d, block=%d\n",
          (long long)axis, element_size_bytes,
          (long long)data_num_elements, (long long)indices_num_elements,
          (long long)output_num_elements,
          (long long)outer_size, (long long)axis_size, (long long)inner_size,
          grid_size, kBlockSize);

  switch (element_size_bytes) {
    case 2:
      hipLaunchKernelGGL(gather_kernel<uint16_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const uint16_t*>(data),
                         static_cast<const int64_t*>(indices),
                         static_cast<uint16_t*>(output),
                         outer_size, axis_size, inner_size,
                         indices_num_elements, output_num_elements);
      break;
    case 4:
      hipLaunchKernelGGL(gather_kernel<uint32_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const uint32_t*>(data),
                         static_cast<const int64_t*>(indices),
                         static_cast<uint32_t*>(output),
                         outer_size, axis_size, inner_size,
                         indices_num_elements, output_num_elements);
      break;
    case 8:
      hipLaunchKernelGGL(gather_kernel<int64_t>,
                         dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                         static_cast<const int64_t*>(data),
                         static_cast<const int64_t*>(indices),
                         static_cast<int64_t*>(output),
                         outer_size, axis_size, inner_size,
                         indices_num_elements, output_num_elements);
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
