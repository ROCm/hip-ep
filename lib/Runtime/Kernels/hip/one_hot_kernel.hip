/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * ONNX OneHot: fill output with off_value, then write on_value at index
 * positions along the inserted axis.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kMaxRank = 8;

struct OneHotGeom {
  int64_t axis;
  int64_t depth;
  int64_t indices_rank;
  int64_t output_rank;
  int64_t ishape[kMaxRank];
  int64_t oshape[kMaxRank];
  int64_t ostride[kMaxRank];
};

template <typename T>
__global__ void one_hot_fill_kernel(T *__restrict__ output, int64_t num_elements,
                                    T off_value) {
  int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_elements)
    return;
  output[tid] = off_value;
}

template <typename T, typename IdxT>
__global__ void one_hot_scatter_kernel(const IdxT *__restrict__ indices,
                                       T on_value, T *__restrict__ output,
                                       int64_t num_indices, OneHotGeom geom) {
  int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_indices)
    return;

  int64_t idx_val = static_cast<int64_t>(indices[tid]);
  if (idx_val < -geom.depth || idx_val >= geom.depth)
    return;
  if (idx_val < 0)
    idx_val += geom.depth;

  int64_t coords[kMaxRank] = {0};
  int64_t rem = tid;
  for (int64_t d = geom.indices_rank - 1; d >= 0; --d) {
    int64_t dim = geom.ishape[d];
    coords[d] = (dim > 0) ? (rem % dim) : 0;
    if (dim > 0)
      rem /= dim;
  }

  int64_t offset = 0;
  int64_t in_dim = 0;
  for (int64_t d = 0; d < geom.output_rank; ++d) {
    int64_t c = (d == geom.axis) ? idx_val : coords[in_dim++];
    if (c < 0 || c >= geom.oshape[d])
      return;
    offset += c * geom.ostride[d];
  }
  output[offset] = on_value;
}

template <typename T, typename IdxT>
static int launch_one_hot(void *stream, const void *indices, const void *values,
                          void *output, int64_t axis, int64_t depth,
                          int64_t indices_rank, int64_t output_rank,
                          const int64_t *indices_shape,
                          const int64_t *output_shape, int64_t num_indices,
                          int64_t num_output_elements) {
  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;

  T host_values[2] = {};
  hipMemcpyAsync(host_values, values, 2 * sizeof(T), hipMemcpyDeviceToHost,
                 hip_stream);
  hipError_t err = hipStreamSynchronize(hip_stream);
  if (err != hipSuccess)
    return static_cast<int>(err);
  T off_value = host_values[0];
  T on_value = host_values[1];

  if (num_output_elements > 0) {
    int fill_grid = static_cast<int>(
        (num_output_elements + kBlockSize - 1) / kBlockSize);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(one_hot_fill_kernel<T>),
                     dim3(fill_grid), dim3(kBlockSize), 0, hip_stream,
                     static_cast<T *>(output), num_output_elements, off_value);
    err = hipGetLastError();
    if (err != hipSuccess)
      return static_cast<int>(err);
  }

  if (num_indices <= 0)
    return 0;

  OneHotGeom geom{};
  geom.axis = axis;
  geom.depth = depth;
  geom.indices_rank = indices_rank;
  geom.output_rank = output_rank;
  for (int64_t i = 0; i < indices_rank; ++i)
    geom.ishape[i] = indices_shape[i];
  for (int64_t i = 0; i < output_rank; ++i)
    geom.oshape[i] = output_shape[i];
  geom.ostride[output_rank - 1] = 1;
  for (int64_t i = output_rank - 2; i >= 0; --i)
    geom.ostride[i] = geom.ostride[i + 1] * geom.oshape[i + 1];

  int scatter_grid =
      static_cast<int>((num_indices + kBlockSize - 1) / kBlockSize);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(one_hot_scatter_kernel<T, IdxT>),
                   dim3(scatter_grid), dim3(kBlockSize), 0, hip_stream,
                   static_cast<const IdxT *>(indices), on_value,
                   static_cast<T *>(output), num_indices, geom);
  err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] one_hot_scatter_kernel launch failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  return 0;
}

template <typename T>
static int dispatch_indices(void *stream, const void *indices, const void *values,
                            void *output, int64_t axis, int64_t depth,
                            int64_t indices_rank, int64_t output_rank,
                            const int64_t *indices_shape,
                            const int64_t *output_shape, int64_t num_indices,
                            int64_t num_output_elements,
                            int indices_element_size_bytes) {
  if (indices_element_size_bytes == 4) {
    return launch_one_hot<T, int32_t>(
        stream, indices, values, output, axis, depth, indices_rank, output_rank,
        indices_shape, output_shape, num_indices, num_output_elements);
  }
  return launch_one_hot<T, int64_t>(
      stream, indices, values, output, axis, depth, indices_rank, output_rank,
      indices_shape, output_shape, num_indices, num_output_elements);
}

} // namespace

extern "C" int hip_one_hot(
    void *stream, const void *indices, const void *depth, const void *values,
    void *output, int64_t axis, int64_t indices_rank, int64_t output_rank,
    const int64_t *indices_shape, const int64_t *output_shape,
    int64_t num_indices, int64_t num_output_elements, int64_t depth_scalar,
    int element_size_bytes, int indices_element_size_bytes) {
  (void)depth;
  if (!indices || !values || !output || !indices_shape || !output_shape)
    return -1;
  if (indices_rank < 0 || indices_rank > 7 || output_rank <= indices_rank ||
      output_rank > 8)
    return -1;
  if (indices_element_size_bytes != 4 && indices_element_size_bytes != 8)
    return -1;

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_one_hot: axis=%lld, depth=%lld, idx_rank=%lld, "
      "out_rank=%lld, num_idx=%lld, num_out=%lld, elem=%d\n",
      (long long)axis, (long long)depth_scalar, (long long)indices_rank,
      (long long)output_rank, (long long)num_indices,
      (long long)num_output_elements, element_size_bytes);

  switch (element_size_bytes) {
  case 1:
    return dispatch_indices<uint8_t>(
        stream, indices, values, output, axis, depth_scalar, indices_rank,
        output_rank, indices_shape, output_shape, num_indices,
        num_output_elements, indices_element_size_bytes);
  case 2:
    return dispatch_indices<uint16_t>(
        stream, indices, values, output, axis, depth_scalar, indices_rank,
        output_rank, indices_shape, output_shape, num_indices,
        num_output_elements, indices_element_size_bytes);
  case 4:
    return dispatch_indices<uint32_t>(
        stream, indices, values, output, axis, depth_scalar, indices_rank,
        output_rank, indices_shape, output_shape, num_indices,
        num_output_elements, indices_element_size_bytes);
  case 8:
    return dispatch_indices<uint64_t>(
        stream, indices, values, output, axis, depth_scalar, indices_rank,
        output_rank, indices_shape, output_shape, num_indices,
        num_output_elements, indices_element_size_bytes);
  default:
    fprintf(stderr,
            "[custom_kernels] hip_one_hot: unsupported element_size=%d\n",
            element_size_bytes);
    return -1;
  }
}
