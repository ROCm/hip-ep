/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_runtime.h>

template <typename T>
__global__ void split_copy_kernel(const T *data, T *output, int64_t offset,
                                  int64_t input_axis_dim, int64_t output_axis_dim,
                                  int64_t inner_size, int64_t output_num_elements) {
  int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= output_num_elements)
    return;

  const int64_t outer = idx / (output_axis_dim * inner_size);
  const int64_t rem = idx % (output_axis_dim * inner_size);
  const int64_t axis_idx = rem / inner_size;
  const int64_t inner_idx = rem % inner_size;

  const int64_t src_idx =
      outer * (input_axis_dim * inner_size) + (offset + axis_idx) * inner_size +
      inner_idx;
  output[idx] = data[src_idx];
}

extern "C" int hip_split(void *stream, const void *data, void *output,
                         int64_t axis, int64_t offset, int64_t input_axis_dim,
                         int64_t output_axis_dim, int64_t inner_size,
                         int64_t output_num_elements, int64_t element_size_bytes) {
  (void)axis; // Axis is used by host-side lowering/runtime for metadata.

  if (!data || !output)
    return -1;
  if (output_num_elements <= 0)
    return 0;
  if (offset < 0 || input_axis_dim <= 0 || inner_size <= 0 ||
      element_size_bytes <= 0)
    return -1;

  if (output_axis_dim <= 0 || (offset + output_axis_dim) > input_axis_dim) {
    fprintf(stderr,
            "[custom_kernels] hip_split: invalid range offset=%lld out_axis=%lld "
            "input_axis=%lld\n",
            (long long)offset, (long long)output_axis_dim,
            (long long)input_axis_dim);
    return -1;
  }
  if ((output_axis_dim * inner_size) <= 0 ||
      (output_num_elements % (output_axis_dim * inner_size)) != 0) {
    fprintf(stderr,
            "[custom_kernels] hip_split: inconsistent output shape "
            "(output_num=%lld out_axis=%lld inner=%lld)\n",
            (long long)output_num_elements, (long long)output_axis_dim,
            (long long)inner_size);
    return -1;
  }

  constexpr int kBlockSize = 256;
  int grid = static_cast<int>((output_num_elements + kBlockSize - 1) / kBlockSize);
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_split: axis=%lld offset=%lld input_axis=%lld "
      "out_axis=%lld inner=%lld output_num=%lld elem_size=%lld grid=%d block=%d\n",
      (long long)axis, (long long)offset, (long long)input_axis_dim,
      (long long)output_axis_dim, (long long)inner_size,
      (long long)output_num_elements,
      (long long)element_size_bytes, grid, kBlockSize);

  switch (element_size_bytes) {
  case 1:
    hipLaunchKernelGGL(split_copy_kernel<uint8_t>, dim3(grid), dim3(kBlockSize),
                       0, hip_stream, static_cast<const uint8_t *>(data),
                       static_cast<uint8_t *>(output), offset, input_axis_dim,
                       output_axis_dim, inner_size, output_num_elements);
    break;
  case 2:
    hipLaunchKernelGGL(split_copy_kernel<uint16_t>, dim3(grid), dim3(kBlockSize),
                       0, hip_stream, static_cast<const uint16_t *>(data),
                       static_cast<uint16_t *>(output), offset, input_axis_dim,
                       output_axis_dim, inner_size, output_num_elements);
    break;
  case 4:
    hipLaunchKernelGGL(split_copy_kernel<uint32_t>, dim3(grid), dim3(kBlockSize),
                       0, hip_stream, static_cast<const uint32_t *>(data),
                       static_cast<uint32_t *>(output), offset, input_axis_dim,
                       output_axis_dim, inner_size, output_num_elements);
    break;
  case 8:
    hipLaunchKernelGGL(split_copy_kernel<uint64_t>, dim3(grid), dim3(kBlockSize),
                       0, hip_stream, static_cast<const uint64_t *>(data),
                       static_cast<uint64_t *>(output), offset, input_axis_dim,
                       output_axis_dim, inner_size, output_num_elements);
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_split: unsupported element_size=%lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_split launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

