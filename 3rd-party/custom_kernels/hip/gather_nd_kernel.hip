/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * GatherND: pick slices of `data` along the first `K = indices.shape[-1]`
 * dims after `batch_dims`, one slice per row of `indices`.
 *
 * Output shape: indices.shape[:-1] + data.shape[batch_dims + K:]
 *               = [batch_dims | outer_indices_dims | slice_dims]
 *
 * Ported from onnxruntime/core/providers/cuda/tensor/gather_nd_impl.cu @
 * v1.22.2. ORT splits this into two kernels (compute per-slice offsets to
 * a scratch buffer, then gather). We fuse them into a single kernel so we
 * don't need scratch storage: each output thread reads K int64 indices
 * inline. K is small (typically 1-2 for vision models), so the extra reads
 * are negligible.
 *
 * Negative-index normalisation: ONNX allows `idx in [-D, D-1]`. We mirror
 * the ORT branch `if (index < 0) index += dim`.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kGatherNDMaxK = 8;
static constexpr int kGatherNDMaxRank = 8;

// Per-launch metadata. Packed into a small struct so we can pass it by value.
struct GatherNDParams {
  // For each k in [0, K), data stride along dim (batch_dims + k) -- i.e.
  // product(data_shape[batch_dims+k+1:]). The K-th entry (sizes[K-1]) is
  // therefore also slice_size = product(data_shape[batch_dims+K:]).
  int64_t sizes_from_slice_dims[kGatherNDMaxK];
  // For each k in [0, K), data_shape[batch_dims + k] -- used for negative
  // index normalization (idx<0 -> idx += dim).
  int64_t data_slice_dims[kGatherNDMaxK];
  int64_t input_batch_stride;     // product(data_shape[batch_dims:])
  int64_t num_slices_per_batch;   // product(indices_shape[batch_dims:-1])
  int64_t slice_size;             // product(data_shape[batch_dims+K:])
  int K;                          // == indices_shape[-1]
};

template <typename T>
__global__ void hip_gather_nd_kernel(const T *__restrict__ input,
                                     const int64_t *__restrict__ indices,
                                     T *__restrict__ output,
                                     GatherNDParams p,
                                     int64_t num_output_elements) {
  int64_t out_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= num_output_elements)
    return;

  int64_t slice_idx = out_idx / p.slice_size;
  int64_t inner = out_idx % p.slice_size;

  int64_t batch_idx = slice_idx / p.num_slices_per_batch;
  int64_t base_offset = batch_idx * p.input_batch_stride;

  // Walk the K-element indices row for this slice.
  const int64_t *slice_indices = indices + slice_idx * p.K;
  int64_t relative_slice_offset = 0;
  for (int k = 0; k < p.K; ++k) {
    int64_t idx = slice_indices[k];
    int64_t dim = p.data_slice_dims[k];
    if (idx < 0)
      idx += dim;
    // No bounds-clamp -- caller is responsible (ORT also asserts in debug).
    relative_slice_offset += idx * p.sizes_from_slice_dims[k];
  }

  output[out_idx] = input[base_offset + relative_slice_offset + inner];
}

extern "C" int hip_gather_nd(void *stream, const void *input,
                             const void *indices, void *output,
                             const int64_t *data_shape_host, int data_rank,
                             const int64_t *indices_shape_host,
                             int indices_rank, int batch_dims, int hip_dtype) {
  if (data_rank <= 0 || indices_rank <= 0) {
    fprintf(stderr,
            "[custom_kernels] hip_gather_nd: zero-rank input "
            "(data_rank=%d, indices_rank=%d)\n",
            data_rank, indices_rank);
    return -1;
  }
  if (data_rank > kGatherNDMaxRank || indices_rank > kGatherNDMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_gather_nd: rank exceeds %d "
            "(data_rank=%d, indices_rank=%d)\n",
            kGatherNDMaxRank, data_rank, indices_rank);
    return -1;
  }

  int K = static_cast<int>(indices_shape_host[indices_rank - 1]);
  if (K <= 0 || K > kGatherNDMaxK) {
    fprintf(stderr,
            "[custom_kernels] hip_gather_nd: K=%d out of range [1, %d]\n",
            K, kGatherNDMaxK);
    return -1;
  }
  if (batch_dims < 0 || batch_dims + K > data_rank) {
    fprintf(stderr,
            "[custom_kernels] hip_gather_nd: invalid batch_dims=%d "
            "(data_rank=%d, K=%d)\n",
            batch_dims, data_rank, K);
    return -1;
  }

  GatherNDParams p{};
  p.K = K;

  // input_batch_stride = product(data_shape[batch_dims:])
  int64_t input_batch_stride = 1;
  for (int d = batch_dims; d < data_rank; ++d)
    input_batch_stride *= data_shape_host[d];
  p.input_batch_stride = input_batch_stride;

  // sizes_from_slice_dims[k] = product(data_shape[batch_dims+k+1:])
  // data_slice_dims[k]       = data_shape[batch_dims+k]
  int64_t suffix = 1;
  for (int d = data_rank - 1; d >= batch_dims; --d) {
    int k = d - batch_dims;
    if (k < K) {
      p.sizes_from_slice_dims[k] = suffix;
      p.data_slice_dims[k] = data_shape_host[d];
    }
    suffix *= data_shape_host[d];
  }

  // slice_size = product(data_shape[batch_dims+K:])
  int64_t slice_size = 1;
  for (int d = batch_dims + K; d < data_rank; ++d)
    slice_size *= data_shape_host[d];
  p.slice_size = slice_size;

  // num_slices_per_batch = product(indices_shape[batch_dims:-1])
  int64_t num_slices_per_batch = 1;
  for (int d = batch_dims; d < indices_rank - 1; ++d)
    num_slices_per_batch *= indices_shape_host[d];
  p.num_slices_per_batch = num_slices_per_batch;

  // Total slices = product(indices_shape[:-1])
  int64_t total_slices = 1;
  for (int d = 0; d < indices_rank - 1; ++d)
    total_slices *= indices_shape_host[d];

  int64_t num_output_elements = total_slices * slice_size;
  if (num_output_elements == 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (num_output_elements + kBlockSize - 1) / kBlockSize);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_gather_nd: dtype=%d, data_rank=%d, "
      "indices_rank=%d, K=%d, batch_dims=%d, slice_size=%lld, "
      "num_slices=%lld, num_out=%lld\n",
      hip_dtype, data_rank, indices_rank, K, batch_dims, (long long)slice_size,
      (long long)total_slices, (long long)num_output_elements);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_gather_nd_kernel<__half>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<const int64_t *>(indices),
                       static_cast<__half *>(output), p, num_output_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_gather_nd_kernel<float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<const int64_t *>(indices),
                       static_cast<float *>(output), p, num_output_elements);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_gather_nd_kernel<int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<const int64_t *>(indices),
                       static_cast<int32_t *>(output), p, num_output_elements);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_gather_nd_kernel<int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<const int64_t *>(indices),
                       static_cast<int64_t *>(output), p, num_output_elements);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_gather_nd: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_gather_nd launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
