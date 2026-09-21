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
 * Compile-time controls are resolved by the host wrapper. Runtime controls
 * are resolved by a one-thread device prepass, avoiding a GPU-to-host
 * synchronization. Both paths produce the same device-resident start, step,
 * and logical-extent tables consumed by the copy kernel.
 *   - Output shape is provided by the IR (the lowering requires static
 *     output shape).
 *
 * Each kernel thread:
 *   1. Computes its output coordinate `(out_coord[0], ..., out_coord[r-1])`
 *      from its linear thread id via the output strides.
 *   2. For each dim d, computes `in_coord[d] = start[d] + out_coord[d] *
 * step[d]`.
 *   3. Linearises `in_coord` via the input strides and copies the element.
 *
 * No bounds checks — the resolve path is responsible for producing valid
 * (start, step, output_size) triples per axis.
 *
 * Bounded to rank <= 8 (matches kPadMaxRank / kGatherNDMaxRank).
 */

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <cstdint>
#include <cstdio>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

static constexpr int kSliceMaxRank = 8;

struct SliceParams {
  int64_t input_strides[kSliceMaxRank];
  int64_t output_strides[kSliceMaxRank];
  int rank;
};

template <typename T>
__global__ void hip_slice_kernel(const T *__restrict__ input,
                                 T *__restrict__ output, SliceParams p,
                                 const int64_t *__restrict__ logical_extent,
                                 const int64_t *__restrict__ start_per_axis,
                                 const int64_t *__restrict__ step_per_axis,
                                 int64_t num_output_elements) {
  int64_t out_idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= num_output_elements)
    return;

  // Decode physical coord, then check whether every axis is within its
  // logical extent. Any over-allocated tail position writes zero.
  int64_t remaining = out_idx;
  int64_t coord[kSliceMaxRank];
  for (int d = 0; d < p.rank; ++d) {
    int64_t out_stride = p.output_strides[d];
    coord[d] = remaining / out_stride;
    remaining -= coord[d] * out_stride;
  }

  bool in_logical = true;
  for (int d = 0; d < p.rank; ++d) {
    if (coord[d] >= logical_extent[d]) {
      in_logical = false;
      break;
    }
  }
  if (!in_logical) {
    output[out_idx] = T(0);
    return;
  }

  int64_t in_offset = 0;
  for (int d = 0; d < p.rank; ++d) {
    int64_t in_coord = start_per_axis[d] + coord[d] * step_per_axis[d];
    in_offset += in_coord * p.input_strides[d];
  }

  output[out_idx] = input[in_offset];
}

struct SliceResolveParams {
  int64_t input_shape[kSliceMaxRank];
  int64_t output_shape[kSliceMaxRank];
  int64_t starts_host[kSliceMaxRank];
  int64_t ends_host[kSliceMaxRank];
  int64_t axes_host[kSliceMaxRank];
  int64_t steps_host[kSliceMaxRank];
  int rank;
  int count;
  bool starts_on_device;
  bool ends_on_device;
  bool axes_present;
  bool axes_on_device;
  bool steps_present;
  bool steps_on_device;
};

__device__ inline int64_t clamp_i64(int64_t value, int64_t low, int64_t high) {
  return value < low ? low : (value > high ? high : value);
}

__global__ void hip_slice_resolve_kernel(
    const int64_t *__restrict__ starts_device,
    const int64_t *__restrict__ ends_device,
    const int64_t *__restrict__ axes_device,
    const int64_t *__restrict__ steps_device, SliceResolveParams p,
    int64_t *__restrict__ start_per_axis, int64_t *__restrict__ step_per_axis,
    int64_t *__restrict__ logical_extent) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;

  for (int d = 0; d < p.rank; ++d) {
    start_per_axis[d] = 0;
    step_per_axis[d] = 1;
    logical_extent[d] = p.output_shape[d];
  }

  // Invalid axes are ignored. Duplicate axes intentionally use
  // last-writer-wins semantics because entries are processed serially.
  for (int k = 0; k < p.count; ++k) {
    int64_t axis = p.axes_present
                       ? (p.axes_on_device ? axes_device[k] : p.axes_host[k])
                       : k;
    if (axis < 0)
      axis += p.rank;
    if (axis < 0 || axis >= p.rank)
      continue;

    int64_t dim = p.input_shape[axis];
    int64_t start = p.starts_on_device ? starts_device[k] : p.starts_host[k];
    int64_t end = p.ends_on_device ? ends_device[k] : p.ends_host[k];
    int64_t step = p.steps_present
                       ? (p.steps_on_device ? steps_device[k] : p.steps_host[k])
                       : 1;
    if (step == 0)
      step = 1;

    if (start < 0)
      start += dim;
    if (end < 0)
      end += dim;
    if (step > 0) {
      start = clamp_i64(start, 0, dim);
      end = clamp_i64(end, 0, dim);
    } else {
      start = clamp_i64(start, 0, dim - 1);
      end = clamp_i64(end, -1, dim - 1);
    }

    int64_t expected = step > 0 ? (end - start + step - 1) / step
                                : (end - start + step + 1) / step;
    if (expected < 0)
      expected = 0;
    if (expected > p.output_shape[axis])
      expected = p.output_shape[axis];
    start_per_axis[axis] = start;
    step_per_axis[axis] = step;
    logical_extent[axis] = expected;
  }
}

static void compute_row_major_strides(const int64_t *shape, int rank,
                                      int64_t *strides_out) {
  int64_t suffix = 1;
  for (int d = rank - 1; d >= 0; --d) {
    strides_out[d] = suffix;
    suffix *= shape[d];
  }
}

extern "C" int
hip_slice_resolve(void *stream, const int64_t *starts_device,
                  const int64_t *starts_host, const int64_t *ends_device,
                  const int64_t *ends_host, const int64_t *axes_device,
                  const int64_t *axes_host, const int64_t *steps_device,
                  const int64_t *steps_host, const int64_t *input_shape_host,
                  const int64_t *output_shape_host, int count, int rank,
                  int64_t *start_per_axis_device, int64_t *step_per_axis_device,
                  int64_t *logical_extent_device) {
  if (rank <= 0 || rank > kSliceMaxRank || count <= 0 ||
      count > kSliceMaxRank || (!starts_device && !starts_host) ||
      (!ends_device && !ends_host))
    return -1;

  SliceResolveParams p{};
  p.rank = rank;
  p.count = count;
  p.starts_on_device = starts_host == nullptr;
  p.ends_on_device = ends_host == nullptr;
  p.axes_present = axes_device || axes_host;
  p.axes_on_device = p.axes_present && axes_host == nullptr;
  p.steps_present = steps_device || steps_host;
  p.steps_on_device = p.steps_present && steps_host == nullptr;
  for (int d = 0; d < rank; ++d) {
    p.input_shape[d] = input_shape_host[d];
    p.output_shape[d] = output_shape_host[d];
  }
  for (int k = 0; k < count; ++k) {
    if (starts_host)
      p.starts_host[k] = starts_host[k];
    if (ends_host)
      p.ends_host[k] = ends_host[k];
    if (axes_host)
      p.axes_host[k] = axes_host[k];
    if (steps_host)
      p.steps_host[k] = steps_host[k];
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  hipLaunchKernelGGL(hip_slice_resolve_kernel, dim3(1), dim3(1), 0, hip_stream,
                     starts_device, ends_device, axes_device, steps_device, p,
                     start_per_axis_device, step_per_axis_device,
                     logical_extent_device);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_slice_resolve launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int hip_slice(void *stream, const void *input, void *output,
                         const int64_t *input_shape_host,
                         const int64_t *output_shape_host,
                         const int64_t *logical_extent_device,
                         const int64_t *starts_per_axis_device,
                         const int64_t *steps_per_axis_device, int rank,
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

  int64_t num_output_elements = 1;
  for (int d = 0; d < rank; ++d)
    num_output_elements *= output_shape_host[d];
  if (num_output_elements == 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size =
      static_cast<int>((num_output_elements + kBlockSize - 1) / kBlockSize);

  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_slice: dtype=%d, rank=%d, "
                           "num_out=%lld\n",
                           hip_dtype, rank, (long long)num_output_elements);

  (void)hipGetLastError();
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_slice_kernel<__half>), dim3(grid_size),
        dim3(kBlockSize), 0, hip_stream, static_cast<const __half *>(input),
        static_cast<__half *>(output), p, logical_extent_device,
        starts_per_axis_device, steps_per_axis_device, num_output_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_slice_kernel<float>), dim3(grid_size),
        dim3(kBlockSize), 0, hip_stream, static_cast<const float *>(input),
        static_cast<float *>(output), p, logical_extent_device,
        starts_per_axis_device, steps_per_axis_device, num_output_elements);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_slice_kernel<int32_t>), dim3(grid_size),
        dim3(kBlockSize), 0, hip_stream, static_cast<const int32_t *>(input),
        static_cast<int32_t *>(output), p, logical_extent_device,
        starts_per_axis_device, steps_per_axis_device, num_output_elements);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(hip_slice_kernel<int64_t>), dim3(grid_size),
        dim3(kBlockSize), 0, hip_stream, static_cast<const int64_t *>(input),
        static_cast<int64_t *>(output), p, logical_extent_device,
        starts_per_axis_device, steps_per_axis_device, num_output_elements);
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
