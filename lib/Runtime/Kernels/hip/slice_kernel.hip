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
 * The index tensors stay on the device. Launch geometry depends only on the
 * output shape, which the lowering requires to be static, so nothing about
 * the launch needs their values -- they are only needed to turn an output
 * coordinate into an input coordinate. Resolving them per-block on the GPU
 * therefore costs a handful of scalar loads and removes the D2H + full
 * stream drain the host wrapper would otherwise pay on every call.
 *
 * Per block, thread 0 resolves the ONNX-13+ index rules (negative indices,
 * per-direction clamping, axes defaulting to [0..r-1], steps defaulting to
 * 1) into a shared per-axis (start, step, extent) plan. Every thread then:
 *   1. Computes its output coordinate from its linear id via output strides.
 *   2. Zero-fills if any coordinate is past that axis's logical extent (the
 *      over-allocated tail, see SliceAxisPlan::extent).
 *   3. Otherwise reads `input[sum_d (start[d] + coord[d]*step[d]) * in_stride[d]]`.
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

// Host-known geometry. Everything here is derived from the IR's static
// shapes, so it can be baked into the launch as a by-value argument.
struct SliceShape {
  int64_t data_shape[kSliceMaxRank];
  int64_t output_shape[kSliceMaxRank];
  int64_t input_strides[kSliceMaxRank];
  int64_t output_strides[kSliceMaxRank];
  int rank;
  // Number of entries in starts/ends (and in axes/steps when present).
  int num_index_entries;
};

// Device-resolved per-axis plan in INPUT space, one entry per data dimension.
struct SliceAxisPlan {
  // start[d] and step[d] are signed; step may be negative when slicing with
  // a reverse stride.
  int64_t start[kSliceMaxRank];
  int64_t step[kSliceMaxRank];
  // extent[d] = the actual per-axis slice output extent. For axes not
  // touched by `axes`, equals the data dim. When extent[d] < the physical
  // output dim, the host has OVER-ALLOCATED the output buffer because
  // `SliceToHip` could not constant-fold the per-axis extent at compile time
  // (typical for slices whose starts/ends are runtime Gather outputs inside
  // an outlined loop body). The kernel iterates physical positions and fills
  // the over-allocated tail with zeros so downstream consumers operating on
  // the full physical extent see clean deterministic data instead of pool
  // garbage.
  int64_t extent[kSliceMaxRank];
};

__device__ inline int64_t slice_clamp(int64_t v, int64_t lo, int64_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// Mirrors the ONNX-13+ Slice index rules. Malformed index metadata (an axis
// out of range, a repeated axis, a zero step, or a derived extent wider than
// the buffer the IR allocated) used to abort in the host wrapper, which could
// inspect the values. They are only visible on the device now, so each case
// is clamped to something in-bounds and reported through the shared device
// error flag -- the host observes it at the end of the inference without
// paying a synchronization here.
__device__ inline void resolve_slice_plan(const SliceShape &s,
                                          const int64_t *__restrict__ starts,
                                          const int64_t *__restrict__ ends,
                                          const int64_t *__restrict__ axes,
                                          const int64_t *__restrict__ steps,
                                          SliceAxisPlan &plan,
                                          int *device_error_flag) {
  bool touched[kSliceMaxRank];
  for (int d = 0; d < s.rank; ++d) {
    plan.start[d] = 0;
    plan.step[d] = 1;
    plan.extent[d] = s.output_shape[d];
    touched[d] = false;
  }

  for (int k = 0; k < s.num_index_entries; ++k) {
    int64_t axis = axes ? axes[k] : static_cast<int64_t>(k);
    if (axis < 0)
      axis += s.rank;
    if (axis < 0 || axis >= s.rank || touched[static_cast<int>(axis)]) {
      if (device_error_flag)
        *device_error_flag = -1;
      continue;
    }
    const int a = static_cast<int>(axis);
    touched[a] = true;

    const int64_t dim = s.data_shape[a];
    int64_t start = starts[k];
    int64_t end = ends[k];
    int64_t step = steps ? steps[k] : 1;
    if (step == 0) {
      if (device_error_flag)
        *device_error_flag = -1;
      step = 1;
    }

    if (start < 0)
      start += dim;
    if (end < 0)
      end += dim;
    if (step > 0) {
      start = slice_clamp(start, 0, dim);
      end = slice_clamp(end, 0, dim);
    } else {
      start = slice_clamp(start, 0, dim - 1);
      end = slice_clamp(end, -1, dim - 1);
    }

    int64_t extent = (step > 0) ? (end - start + step - 1) / step
                                : (end - start + step + 1) / step;
    if (extent < 0)
      extent = 0;
    // Slice cannot widen an axis, so an extent past the physical dim means
    // the IR's output shape and the runtime bounds disagree. Honouring it
    // would read past the end of the input.
    if (extent > s.output_shape[a]) {
      if (device_error_flag)
        *device_error_flag = -1;
      extent = s.output_shape[a];
    }

    plan.start[a] = start;
    plan.step[a] = step;
    plan.extent[a] = extent;
  }
}

template <typename T>
__global__ void hip_slice_kernel(const T *__restrict__ input,
                                 T *__restrict__ output,
                                 const int64_t *__restrict__ starts,
                                 const int64_t *__restrict__ ends,
                                 const int64_t *__restrict__ axes,
                                 const int64_t *__restrict__ steps,
                                 SliceShape s, int64_t num_output_elements,
                                 int *device_error_flag) {
  // Every thread must reach the barrier, so the plan is resolved before the
  // out-of-range threads drop out.
  __shared__ SliceAxisPlan plan;
  if (threadIdx.x == 0)
    resolve_slice_plan(s, starts, ends, axes, steps, plan, device_error_flag);
  __syncthreads();

  int64_t out_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= num_output_elements)
    return;

  // Decode physical coord, then check whether every axis is within its
  // logical extent. Any over-allocated tail position writes zero.
  int64_t remaining = out_idx;
  int64_t coord[kSliceMaxRank];
  for (int d = 0; d < s.rank; ++d) {
    int64_t out_stride = s.output_strides[d];
    coord[d] = remaining / out_stride;
    remaining -= coord[d] * out_stride;
  }

  bool in_logical = true;
  for (int d = 0; d < s.rank; ++d) {
    if (coord[d] >= plan.extent[d]) {
      in_logical = false;
      break;
    }
  }
  if (!in_logical) {
    output[out_idx] = T(0);
    return;
  }

  int64_t in_offset = 0;
  for (int d = 0; d < s.rank; ++d) {
    int64_t in_coord = plan.start[d] + coord[d] * plan.step[d];
    in_offset += in_coord * s.input_strides[d];
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
                         const void *starts_dev, const void *ends_dev,
                         const void *axes_dev, const void *steps_dev,
                         int num_index_entries, int rank, int hip_dtype,
                         void *device_error_flag) {
  if (rank <= 0 || rank > kSliceMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_slice: rank=%d out of range [1, %d]\n", rank,
            kSliceMaxRank);
    return -1;
  }
  if (num_index_entries <= 0 || num_index_entries > kSliceMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_slice: num_index_entries=%d out of range "
            "[1, %d]\n",
            num_index_entries, kSliceMaxRank);
    return -1;
  }
  if (!starts_dev || !ends_dev) {
    fprintf(stderr, "[custom_kernels] hip_slice: null starts/ends\n");
    return -1;
  }

  SliceShape s{};
  s.rank = rank;
  s.num_index_entries = num_index_entries;
  compute_row_major_strides(input_shape_host, rank, s.input_strides);
  compute_row_major_strides(output_shape_host, rank, s.output_strides);
  for (int d = 0; d < rank; ++d) {
    s.data_shape[d] = input_shape_host[d];
    s.output_shape[d] = output_shape_host[d];
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

  const int64_t *starts = static_cast<const int64_t *>(starts_dev);
  const int64_t *ends = static_cast<const int64_t *>(ends_dev);
  const int64_t *axes = static_cast<const int64_t *>(axes_dev);
  const int64_t *steps = static_cast<const int64_t *>(steps_dev);
  int *err_flag = static_cast<int *>(device_error_flag);

  CUSTOM_KERNELS_DEBUG_LOG("[custom_kernels] hip_slice: dtype=%d, rank=%d, "
                           "K=%d, num_out=%lld\n",
                           hip_dtype, rank, num_index_entries,
                           (long long)num_output_elements);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<__half>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), starts, ends, axes,
                       steps, s, num_output_elements, err_flag);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), starts, ends, axes, steps,
                       s, num_output_elements, err_flag);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<int32_t *>(output), starts, ends, axes,
                       steps, s, num_output_elements, err_flag);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<int64_t *>(output), starts, ends, axes,
                       steps, s, num_output_elements, err_flag);
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
