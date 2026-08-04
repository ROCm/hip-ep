/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * ONNX Compress: select slices (along axis) or flattened elements where a
 * 1-D boolean condition is true. Condition length may be shorter than the
 * axis extent; trailing slots are treated as false.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int kMaxRank = 8;
constexpr int kCompressScanBlock = 1024;

struct CompressAxisGeom {
  int64_t axis;
  int64_t rank;
  int64_t oshape[kMaxRank];
  int64_t istrides[kMaxRank];
};

// Ordered stream compaction: writes the indices of the true `condition`
// entries into `selected` in ASCENDING order, plus their total into `count`.
//
// ONNX Compress keeps the selected slices in condition order, so each kept
// slice must land at its rank among the true entries -- not at the position
// its thread happened to claim. An atomicAdd-only compaction is cheaper but
// permutes the slices, which is silently wrong whenever the payload is
// position-dependent (dropping the padded rows of a vision patch grid, say).
// Same single-block cooperative scan as nonzero_ordered_kernel:
//
//   1. each thread counts the true entries of its chunk [lo, hi)
//   2. thread 0 exclusive-scans the per-chunk counts and writes the total
//   3. each thread re-walks its chunk, writing at chunk_offset + local rank
__global__ void compress_scan_kernel(const uint8_t *__restrict__ condition,
                                     int64_t condition_len,
                                     int64_t *__restrict__ selected,
                                     int64_t *__restrict__ count) {
  __shared__ int s_chunk_count[kCompressScanBlock];
  __shared__ int s_chunk_offset[kCompressScanBlock];

  int tid = threadIdx.x;
  int64_t chunk =
      (condition_len + kCompressScanBlock - 1) / kCompressScanBlock;
  int64_t lo = static_cast<int64_t>(tid) * chunk;
  int64_t hi = lo + chunk;
  if (lo > condition_len)
    lo = condition_len;
  if (hi > condition_len)
    hi = condition_len;

  int my_count = 0;
  for (int64_t i = lo; i < hi; ++i) {
    if (condition[i])
      ++my_count;
  }
  s_chunk_count[tid] = my_count;
  __syncthreads();

  if (tid == 0) {
    int sum = 0;
    for (int t = 0; t < kCompressScanBlock; ++t) {
      s_chunk_offset[t] = sum;
      sum += s_chunk_count[t];
    }
    *count = sum;
  }
  __syncthreads();

  // `selected` holds condition_len slots and the total cannot exceed that, so
  // no bound check is needed here.
  int64_t pos = s_chunk_offset[tid];
  for (int64_t i = lo; i < hi; ++i) {
    if (condition[i])
      selected[pos++] = i;
  }
}

template <typename T>
__global__ void compress_flatten_kernel(const T *__restrict__ input,
                                        const int64_t *__restrict__ selected,
                                        int64_t selected_count, T *__restrict__ output) {
  int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= selected_count)
    return;
  output[tid] = input[selected[tid]];
}

template <typename T>
__global__ void compress_axis_kernel(const T *__restrict__ input,
                                   const int64_t *__restrict__ selected,
                                   int64_t selected_count, T *__restrict__ output,
                                   int64_t num_output, CompressAxisGeom geom) {
  int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= num_output)
    return;

  int64_t coords[kMaxRank] = {0};
  int64_t rem = tid;
  for (int64_t d = geom.rank - 1; d >= 0; --d) {
    int64_t dim = geom.oshape[d];
    coords[d] = (dim > 0) ? (rem % dim) : 0;
    if (dim > 0)
      rem /= dim;
  }

  int64_t o_axis = coords[geom.axis];
  if (o_axis < 0 || o_axis >= selected_count)
    return;

  coords[geom.axis] = selected[o_axis];
  int64_t offset = 0;
  for (int64_t d = 0; d < geom.rank; ++d)
    offset += coords[d] * geom.istrides[d];
  output[tid] = input[offset];
}

template <typename T>
static int launch_compress(void *stream, const void *input, const void *condition,
                           void *output, int64_t flatten, int64_t axis,
                           int64_t input_rank, int64_t output_rank,
                           const int64_t *input_shape,
                           const int64_t *output_shape, int64_t condition_len,
                           int64_t num_output_elements, void *workspace,
                           size_t workspace_bytes) {
  if (condition_len <= 0)
    return 0;
  if (!input || !condition || !output || !input_shape || !output_shape ||
      !workspace)
    return -1;
  if (workspace_bytes < static_cast<size_t>(condition_len) * sizeof(int64_t) +
                            sizeof(int64_t))
    return -1;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;

  int64_t *selected = static_cast<int64_t *>(workspace);
  auto *count_dev = reinterpret_cast<int64_t *>(
      static_cast<char *>(workspace) + condition_len * sizeof(int64_t));

  hipLaunchKernelGGL(compress_scan_kernel, dim3(1), dim3(kCompressScanBlock), 0,
                     hip_stream, static_cast<const uint8_t *>(condition),
                     condition_len, selected, count_dev);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] compress_scan_kernel launch failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }

  int64_t selected_count_host = 0;
  err = hipMemcpyAsync(&selected_count_host, count_dev, sizeof(int64_t),
                       hipMemcpyDeviceToHost, hip_stream);
  if (err != hipSuccess)
    return static_cast<int>(err);
  err = hipStreamSynchronize(hip_stream);
  if (err != hipSuccess)
    return static_cast<int>(err);

  int64_t selected_count = selected_count_host;
  if (selected_count <= 0)
    return 0;

  // The destination is written densely with `selected_count` slices along the
  // selection axis, so a runtime count above the extent the graph allocated
  // would write past the buffer. That only happens when the declared ONNX
  // extent disagrees with the data (the compiler sizes a dynamic extent from
  // the same count via hip.readback_dim); clamp rather than corrupt memory.
  int64_t out_extent = output_shape[flatten ? 0 : axis];
  if (selected_count > out_extent)
    selected_count = out_extent;

  if (flatten) {
    int grid = static_cast<int>((selected_count + kBlockSize - 1) / kBlockSize);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_flatten_kernel<T>),
                     dim3(grid), dim3(kBlockSize), 0, hip_stream,
                     static_cast<const T *>(input), selected, selected_count,
                     static_cast<T *>(output));
  } else {
    CompressAxisGeom geom{};
    geom.axis = axis;
    geom.rank = input_rank;
    for (int64_t i = 0; i < output_rank; ++i)
      geom.oshape[i] = output_shape[i];
    geom.oshape[axis] = selected_count;
    geom.istrides[input_rank - 1] = 1;
    for (int64_t i = input_rank - 2; i >= 0; --i)
      geom.istrides[i] = geom.istrides[i + 1] * input_shape[i + 1];

    int64_t axis_output_numel = selected_count;
    for (int64_t i = 0; i < output_rank; ++i) {
      if (i == axis)
        continue;
      axis_output_numel *= geom.oshape[i];
    }

    int grid = static_cast<int>(
        (axis_output_numel + kBlockSize - 1) / kBlockSize);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(compress_axis_kernel<T>),
                     dim3(grid), dim3(kBlockSize), 0, hip_stream,
                     static_cast<const T *>(input), selected, selected_count,
                     static_cast<T *>(output), axis_output_numel, geom);
  }

  err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] compress_kernel launch failed: %s\n",
            hipGetErrorString(err));
    return static_cast<int>(err);
  }
  (void)num_output_elements;
  return 0;
}

} // namespace

extern "C" int hip_compress(
    void *stream, const void *input, const void *condition, void *output,
    int64_t flatten, int64_t axis, int64_t input_rank, int64_t output_rank,
    const int64_t *input_shape, const int64_t *output_shape,
    int64_t condition_len, int64_t num_output_elements, void *workspace,
    size_t workspace_bytes, int element_size_bytes) {
  if (input_rank <= 0 || input_rank > kMaxRank || output_rank <= 0 ||
      output_rank > kMaxRank)
    return -1;

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_compress: flatten=%lld, axis=%lld, "
      "in_rank=%lld, out_rank=%lld, cond_len=%lld, num_out=%lld, elem=%d\n",
      (long long)flatten, (long long)axis, (long long)input_rank,
      (long long)output_rank, (long long)condition_len,
      (long long)num_output_elements, element_size_bytes);

  switch (element_size_bytes) {
  case 1:
    return launch_compress<uint8_t>(
        stream, input, condition, output, flatten, axis, input_rank,
        output_rank, input_shape, output_shape, condition_len,
        num_output_elements, workspace, workspace_bytes);
  case 2:
    return launch_compress<uint16_t>(
        stream, input, condition, output, flatten, axis, input_rank,
        output_rank, input_shape, output_shape, condition_len,
        num_output_elements, workspace, workspace_bytes);
  case 4:
    return launch_compress<uint32_t>(
        stream, input, condition, output, flatten, axis, input_rank,
        output_rank, input_shape, output_shape, condition_len,
        num_output_elements, workspace, workspace_bytes);
  case 8:
    return launch_compress<uint64_t>(
        stream, input, condition, output, flatten, axis, input_rank,
        output_rank, input_shape, output_shape, condition_len,
        num_output_elements, workspace, workspace_bytes);
  default:
    fprintf(stderr,
            "[custom_kernels] hip_compress: unsupported element_size=%d\n",
            element_size_bytes);
    return -1;
  }
}
