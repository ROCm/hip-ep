/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Slice (non-constant-indices / negative-step fallback).
 *
 * The compile-time-constant + positive-stride case is folded to
 * `tensor.extract_slice` upstream of the runtime (see
 * lib/Conversion/OnnxToHip/SliceConversion.cpp::SliceDecompose), so this
 * kernel only services slices whose `starts` / `ends` / `axes` / `steps`
 * are NOT graph-constant, or that have negative steps.
 *
 * The host wrapper does the heavy lifting:
 *   - D2H of the index tensors (typically tiny: a few int64s).
 *   - Per-axis resolution of `start` / `step` per ONNX spec (negative
 *     indices, clamping, axes that default to [0..r-1] when absent,
 *     steps that default to 1).
 *   - Output shape is provided by the IR (the lowering requires static
 *     output shape).
 *
 * Each kernel thread:
 *   1. Computes its output coordinate `(out_coord[0], ..., out_coord[r-1])`
 *      from its linear thread id via the output strides.
 *   2. For each dim d, computes `in_coord[d] = start[d] + out_coord[d] * step[d]`.
 *   3. Linearises `in_coord` via the input strides and copies the element.
 *
 * No bounds checks — the host wrapper is responsible for producing valid
 * (start, step, output_size) triples per axis.
 *
 * Bounded to rank <= 8 (matches kPadMaxRank / kGatherNDMaxRank).
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kSliceMaxRank = 8;

struct SliceParams {
  int64_t input_strides[kSliceMaxRank];
  int64_t output_strides[kSliceMaxRank];
  // start[d] and step[d] are signed; step may be negative when slicing
  // with a reverse stride.
  int64_t start_per_axis[kSliceMaxRank];
  int64_t step_per_axis[kSliceMaxRank];
  int rank;
};

template <typename T>
__global__ void hip_slice_kernel(const T *__restrict__ input,
                                 T *__restrict__ output, SliceParams p,
                                 int64_t num_output_elements) {
  int64_t out_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= num_output_elements)
    return;

  int64_t remaining = out_idx;
  int64_t in_offset = 0;
  for (int d = 0; d < p.rank; ++d) {
    int64_t out_stride = p.output_strides[d];
    int64_t out_coord = remaining / out_stride;
    remaining -= out_coord * out_stride;
    int64_t in_coord = p.start_per_axis[d] + out_coord * p.step_per_axis[d];
    in_offset += in_coord * p.input_strides[d];
  }

  output[out_idx] = input[in_offset];
}

static void compute_row_major_strides(const int64_t *shape, int rank,
                                      int64_t *strides_out) {
  int64_t suffix = 1;
  for (int d = rank - 1; d >= 0; --d) {
    strides_out[d] = suffix;
    suffix *= shape[d];
  }
}

extern "C" int hip_slice(void *stream, const void *input, void *output,
                         const int64_t *input_shape_host,
                         const int64_t *output_shape_host,
                         const int64_t *starts_per_axis_host,
                         const int64_t *steps_per_axis_host, int rank,
                         int hip_dtype) {
  if (rank <= 0 || rank > kSliceMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_slice: rank=%d out of range [1, %d]\n", rank,
            kSliceMaxRank);
    return -1;
  }

  SliceParams p{};
  p.rank = rank;
  compute_row_major_strides(input_shape_host, rank, p.input_strides);
  compute_row_major_strides(output_shape_host, rank, p.output_strides);
  for (int d = 0; d < rank; ++d) {
    p.start_per_axis[d] = starts_per_axis_host[d];
    p.step_per_axis[d] = steps_per_axis_host[d];
  }

  int64_t num_output_elements = 1;
  for (int d = 0; d < rank; ++d)
    num_output_elements *= output_shape_host[d];
  if (num_output_elements == 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (num_output_elements + kBlockSize - 1) / kBlockSize);

  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_slice: dtype=%d, rank=%d, "
                           "num_out=%lld\n",
                           hip_dtype, rank, (long long)num_output_elements);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<__half>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), p, num_output_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), p, num_output_elements);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<int32_t *>(output), p, num_output_elements);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<int64_t *>(output), p, num_output_elements);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_slice: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_slice launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
