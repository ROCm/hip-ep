/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

// General N-D gather along an arbitrary axis.  Output element at flat
// index `tid` is decoded into (pre_idx, k_idx, post_idx) where:
//   pre_idx  = tid / (indices_num * post_axis_size)
//   k_idx    = (tid / post_axis_size) % indices_num
//   post_idx = tid % post_axis_size
// then mapped to source data via:
//   src = pre_idx * (data_axis_size * post_axis_size)
//       + indices[k_idx] * post_axis_size
//       + post_idx
template <typename T>
__global__ void gather_general_kernel(
    const T* __restrict__ data,
    const int64_t* __restrict__ indices,
    T* __restrict__ output,
    int64_t pre_axis_size,
    int64_t data_axis_size,
    int64_t indices_num,
    int64_t post_axis_size,
    int64_t total_elements) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_elements)
    return;
  int64_t post_idx = tid % post_axis_size;
  int64_t after_post = tid / post_axis_size;
  int64_t k_idx = after_post % indices_num;
  int64_t pre_idx = after_post / indices_num;
  int64_t index_val = indices[k_idx];
  if (index_val < 0) index_val += data_axis_size;
  if (index_val < 0 || index_val >= data_axis_size) {
    output[tid] = T(0);
    return;
  }
  int64_t src = pre_idx * (data_axis_size * post_axis_size)
              + index_val * post_axis_size + post_idx;
  output[tid] = data[src];
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
    int element_size_bytes,
    int64_t pre_axis_size) {
  (void)axis;  // axis used to compute pre/post by the host caller
  if (output_num_elements <= 0) return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (output_num_elements + kBlockSize - 1) / kBlockSize);

  // Generic axis-N: split output into (pre, indices, post) using the
  // pre_axis_size passed by the lowering.  We compute:
  //   post_axis_size = output_num / (pre * indices_num)
  //   data_axis_size = data_num / (pre * post)
  if (pre_axis_size <= 0) pre_axis_size = 1;
  int64_t indices_safe = (indices_num_elements > 0) ? indices_num_elements : 1;
  int64_t post_axis_size =
      output_num_elements / (pre_axis_size * indices_safe);
  if (post_axis_size <= 0) post_axis_size = 1;
  int64_t data_axis_size =
      data_num_elements / (pre_axis_size * post_axis_size);
  if (data_axis_size <= 0) data_axis_size = 1;

  // Back-compat names for the debug log:
  int64_t slice_size = post_axis_size * indices_safe;

  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_gather: axis=0, elem_size=%d, "
          "data_num=%lld, indices_num=%lld, output_num=%lld, "
          "slice_size=%lld, data_axis_size=%lld, grid=%d, block=%d\n",
          element_size_bytes,
          (long long)data_num_elements, (long long)indices_num_elements,
          (long long)output_num_elements,
          (long long)slice_size, (long long)data_axis_size,
          grid_size, kBlockSize);

  // Drain any stale async error left by NAN_TRACE's hipMemcpy(D2H) so
  // hipGetLastError() below only reports errors from *this* launch.
  (void)hipGetLastError();

  switch (element_size_bytes) {
    case 2:
      hipLaunchKernelGGL(
          gather_general_kernel<uint16_t>, dim3(grid_size), dim3(kBlockSize),
          0, hip_stream, static_cast<const uint16_t *>(data),
          static_cast<const int64_t *>(indices),
          static_cast<uint16_t *>(output), pre_axis_size, data_axis_size,
          indices_safe, post_axis_size, output_num_elements);
      break;
    case 4:
      hipLaunchKernelGGL(
          gather_general_kernel<uint32_t>, dim3(grid_size), dim3(kBlockSize),
          0, hip_stream, static_cast<const uint32_t *>(data),
          static_cast<const int64_t *>(indices),
          static_cast<uint32_t *>(output), pre_axis_size, data_axis_size,
          indices_safe, post_axis_size, output_num_elements);
      break;
    case 8:
      hipLaunchKernelGGL(
          gather_general_kernel<int64_t>, dim3(grid_size), dim3(kBlockSize), 0,
          hip_stream, static_cast<const int64_t *>(data),
          static_cast<const int64_t *>(indices),
          static_cast<int64_t *>(output), pre_axis_size, data_axis_size,
          indices_safe, post_axis_size, output_num_elements);
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
