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
 * Index resolution lives on the device so wrap_slice does not D2H the
 * (tiny) index tensors or hipStreamSynchronize the inference stream:
 *
 *   Before: host memcpyAsync(D2H) starts/ends/axes/steps, stream sync,
 *           ONNX clamp on the CPU, launch with host (start, step,
 *           logical_extent) arrays.
 *   After:  launch immediately. Thread 0 of each block reads the index
 *           tensors, applies the ONNX-13+ negative-index + clamp rules
 *           into shared memory, and every output thread uses those
 *           resolved (start, step, logical_extent) triples.
 *
 * Each copy thread:
 *   1. Computes its output coordinate from its linear thread id.
 *   2. If any axis is past its logical extent, writes zero (over-allocated
 *      tail from SliceToHip's compile-time upper bound).
 *   3. Else in_coord[d] = start[d] + out_coord[d] * step[d], linearises
 *      via input strides, and copies the element.
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
  int64_t input_dims[kSliceMaxRank];
  int64_t output_dims[kSliceMaxRank];
  int rank;
};

struct ResolvedSlice {
  int64_t start_per_axis[kSliceMaxRank];
  int64_t step_per_axis[kSliceMaxRank];
  int64_t logical_extent[kSliceMaxRank];
  int error;
};

__device__ __forceinline__ int64_t slice_clamp(int64_t v, int64_t lo,
                                               int64_t hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

// ONNX-13+ Slice: per-axis start/step plus the logical output extent used
// to zero the over-allocated tail. Invalid axis / duplicate axis / step 0
// / extent past the physical buffer set `out.error` and keep writing with
// clamped values so the kernel cannot store out of bounds.
__device__ void resolve_onnx_slice(const int64_t *__restrict__ starts,
                                   const int64_t *__restrict__ ends,
                                   const int64_t *axes, const int64_t *steps,
                                   int64_t K, const SliceParams &p,
                                   ResolvedSlice *out) {
  out->error = 0;
  bool axis_set[kSliceMaxRank] = {};
  for (int d = 0; d < p.rank; ++d) {
    out->start_per_axis[d] = 0;
    out->step_per_axis[d] = 1;
    out->logical_extent[d] = p.output_dims[d];
  }
  for (int64_t k = 0; k < K; ++k) {
    int64_t axis = axes ? axes[k] : k;
    if (axis < 0)
      axis += p.rank;
    if (axis < 0 || axis >= p.rank) {
      out->error = -1;
      continue;
    }
    if (axis_set[axis]) {
      out->error = -1;
      continue;
    }
    axis_set[axis] = true;

    int64_t dim = p.input_dims[axis];
    int64_t start = starts[k];
    int64_t end = ends[k];
    int64_t step = steps ? steps[k] : 1;
    if (step == 0) {
      out->error = -1;
      continue;
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

    out->start_per_axis[axis] = start;
    out->step_per_axis[axis] = step;

    int64_t expected;
    if (step > 0)
      expected = (end - start + step - 1) / step;
    else
      expected = (end - start + step + 1) / step;
    if (expected < 0)
      expected = 0;
    if (expected > p.output_dims[axis]) {
      out->error = -1;
      expected = p.output_dims[axis];
    }
    out->logical_extent[axis] = expected;
  }
}

template <typename T>
__global__ void hip_slice_kernel(const T *__restrict__ input,
                                 T *__restrict__ output, SliceParams p,
                                 const int64_t *__restrict__ starts,
                                 const int64_t *__restrict__ ends,
                                 const int64_t *axes, const int64_t *steps,
                                 int64_t K, int64_t num_output_elements,
                                 int *device_error_flag) {
  __shared__ ResolvedSlice resolved;
  if (threadIdx.x == 0)
    resolve_onnx_slice(starts, ends, axes, steps, K, p, &resolved);
  __syncthreads();

  if (resolved.error && device_error_flag && threadIdx.x == 0 &&
      blockIdx.x == 0)
    *device_error_flag = -1;

  int64_t out_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= num_output_elements)
    return;

  int64_t remaining = out_idx;
  int64_t coord[kSliceMaxRank];
  for (int d = 0; d < p.rank; ++d) {
    int64_t out_stride = p.output_strides[d];
    coord[d] = remaining / out_stride;
    remaining -= coord[d] * out_stride;
  }

  bool in_logical = true;
  for (int d = 0; d < p.rank; ++d) {
    if (coord[d] >= resolved.logical_extent[d]) {
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
    int64_t in_coord =
        resolved.start_per_axis[d] + coord[d] * resolved.step_per_axis[d];
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
                         const int64_t *starts_dev, const int64_t *ends_dev,
                         const int64_t *axes_dev, const int64_t *steps_dev,
                         int64_t starts_num_elements, int rank, int hip_dtype,
                         void *device_error_flag) {
  if (rank <= 0 || rank > kSliceMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_slice: rank=%d out of range [1, %d]\n", rank,
            kSliceMaxRank);
    return -1;
  }
  if (!input || !output || !input_shape_host || !output_shape_host ||
      !starts_dev || !ends_dev) {
    fprintf(stderr, "[custom_kernels] hip_slice: null required pointer\n");
    return -1;
  }
  if (starts_num_elements <= 0 || starts_num_elements > kSliceMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_slice: starts_num_elements=%lld out of "
            "range [1, %d]\n",
            (long long)starts_num_elements, kSliceMaxRank);
    return -1;
  }

  SliceParams p{};
  p.rank = rank;
  compute_row_major_strides(input_shape_host, rank, p.input_strides);
  compute_row_major_strides(output_shape_host, rank, p.output_strides);
  for (int d = 0; d < rank; ++d) {
    p.input_dims[d] = input_shape_host[d];
    p.output_dims[d] = output_shape_host[d];
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
                           "K=%lld, num_out=%lld\n",
                           hip_dtype, rank, (long long)starts_num_elements,
                           (long long)num_output_elements);

  (void)hipGetLastError();
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<__half>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), p, starts_dev, ends_dev,
                       axes_dev, steps_dev, starts_num_elements,
                       num_output_elements,
                       static_cast<int *>(device_error_flag));
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), p, starts_dev, ends_dev,
                       axes_dev, steps_dev, starts_num_elements,
                       num_output_elements,
                       static_cast<int *>(device_error_flag));
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<int32_t *>(output), p, starts_dev, ends_dev,
                       axes_dev, steps_dev, starts_num_elements,
                       num_output_elements,
                       static_cast<int *>(device_error_flag));
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_slice_kernel<int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<int64_t *>(output), p, starts_dev, ends_dev,
                       axes_dev, steps_dev, starts_num_elements,
                       num_output_elements,
                       static_cast<int *>(device_error_flag));
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
