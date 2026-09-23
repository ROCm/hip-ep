/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Pad: copy `input` into a larger `output`, with the padded region filled
 * by one of four ONNX modes:
 *   - constant: write `pad_value` (default 0).
 *   - edge:     clamp out-of-range in_coord to the input edge.
 *   - reflect:  mirror the input at the edge (formula matches ORT
 *               _PadKernel<Reflect>).
 *   - wrap:     periodic extension. Equivalent to a modulo on the
 *               (out_coord - lower_pad). Not present in ORT's
 *               pad_impl.cu kernel; added here because the HipToLLVM
 *               PadLowering does emit mode_id=3 for ONNX `wrap`.
 *
 * Ported from onnxruntime/core/providers/cuda/tensor/pad_impl.cu @
 * v1.22.2 (`_PadKernel` -- generic ND, kept as a single fused kernel).
 *
 * Before: wrap_pad D2H'd pads / axes / constant_value and
 *         hipStreamSynchronize'd, then passed host lower_pads and a host
 *         pad_value into the launcher.
 * After:  no host wait. Thread 0 of each block maps ONNX-18 pads (and
 *         optional axes) into per-dim lower_pads and loads constant_value
 *         when mode is constant. Copy threads use those shared values.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kPadMaxRank = 8;

struct PadParams {
  int64_t input_shape[kPadMaxRank];
  int64_t output_shape[kPadMaxRank];
  int64_t input_strides[kPadMaxRank];
  int64_t output_strides[kPadMaxRank];
  int64_t lower_pads[kPadMaxRank];
  int rank;
};

__device__ void resolve_pad_lowers(const int64_t *__restrict__ pads,
                                   const int64_t *axes, int64_t n_axes,
                                   PadParams *p, int *error) {
  *error = 0;
  for (int d = 0; d < p->rank; ++d)
    p->lower_pads[d] = 0;
  bool axis_set[kPadMaxRank] = {};
  for (int64_t i = 0; i < n_axes; ++i) {
    int64_t axis = axes ? axes[i] : i;
    if (axis < 0)
      axis += p->rank;
    if (axis < 0 || axis >= p->rank) {
      *error = -1;
      continue;
    }
    if (axis_set[axis]) {
      *error = -1;
      continue;
    }
    axis_set[axis] = true;
    p->lower_pads[axis] = pads[i];
  }
}

// Returns false if out_coord falls outside the input region for some dim
// in `constant` mode (so the caller should write the pad_value instead).
__device__ inline bool compute_input_offset(
    const PadParams &p, int64_t output_index, int pad_mode,
    int64_t &input_index_out) {
  int64_t remaining = output_index;
  int64_t input_index = 0;

  for (int d = 0; d < p.rank; ++d) {
    int64_t stride = p.output_strides[d];
    int64_t out_coord = remaining / stride;
    remaining -= out_coord * stride;

    int64_t in_dim = p.input_shape[d];
    int64_t lower = p.lower_pads[d];
    int64_t in_coord;

    if (out_coord < lower) {
      switch (pad_mode) {
      case 0: // Constant
        return false;
      case 2: // Edge
        in_coord = 0;
        break;
      case 1: // Reflect
        in_coord = lower - out_coord;
        break;
      case 3: // Wrap
      default: {
        int64_t off = out_coord - lower; // negative
        in_coord = ((off % in_dim) + in_dim) % in_dim;
        break;
      }
      }
    } else if (out_coord >= lower + in_dim) {
      switch (pad_mode) {
      case 0: // Constant
        return false;
      case 2: // Edge
        in_coord = in_dim - 1;
        break;
      case 1: // Reflect
        in_coord = in_dim - 2 - (out_coord - (lower + in_dim));
        break;
      case 3: // Wrap
      default: {
        int64_t off = out_coord - lower;
        in_coord = ((off % in_dim) + in_dim) % in_dim;
        break;
      }
      }
    } else {
      in_coord = out_coord - lower;
    }

    input_index += in_coord * p.input_strides[d];
  }

  input_index_out = input_index;
  return true;
}

template <typename T>
__global__ void hip_pad_kernel(const T *__restrict__ input,
                               T *__restrict__ output, PadParams p,
                               const int64_t *__restrict__ pads,
                               const int64_t *axes, int64_t n_axes,
                               const T *constant_value, int pad_mode,
                               int64_t num_output_elements,
                               int *device_error_flag) {
  __shared__ PadParams resolved;
  __shared__ T pad_value;
  __shared__ int error;
  if (threadIdx.x == 0) {
    resolved = p;
    resolve_pad_lowers(pads, axes, n_axes, &resolved, &error);
    pad_value = (constant_value && pad_mode == 0) ? *constant_value : T(0);
  }
  __syncthreads();

  if (error && device_error_flag && threadIdx.x == 0 && blockIdx.x == 0)
    *device_error_flag = -1;

  int64_t out_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= num_output_elements)
    return;

  int64_t in_idx;
  if (compute_input_offset(resolved, out_idx, pad_mode, in_idx)) {
    output[out_idx] = input[in_idx];
  } else {
    output[out_idx] = pad_value;
  }
}

extern "C" int hip_pad(void *stream, const void *input, void *output,
                       const int64_t *input_shape_host,
                       const int64_t *output_shape_host, const int64_t *pads_dev,
                       const int64_t *axes_dev, int64_t pads_num_elements,
                       int64_t n_axes, int rank, int hip_dtype, int pad_mode,
                       const void *constant_value_dev,
                       void *device_error_flag) {
  if (rank <= 0)
    return 0;
  if (rank > kPadMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_pad: rank=%d exceeds kPadMaxRank=%d\n",
            rank, kPadMaxRank);
    return -1;
  }
  if (!input || !output || !input_shape_host || !output_shape_host ||
      !pads_dev)
    return -1;
  if (n_axes <= 0 || n_axes > kPadMaxRank ||
      pads_num_elements != 2 * n_axes)
    return -1;

  PadParams p{};
  p.rank = rank;
  int64_t num_output_elements = 1;
  for (int d = 0; d < rank; ++d) {
    p.input_shape[d] = input_shape_host[d];
    p.output_shape[d] = output_shape_host[d];
    p.lower_pads[d] = 0;
    num_output_elements *= output_shape_host[d];
  }

  int64_t in_stride = 1, out_stride = 1;
  for (int d = rank - 1; d >= 0; --d) {
    p.input_strides[d] = in_stride;
    p.output_strides[d] = out_stride;
    in_stride *= input_shape_host[d];
    out_stride *= output_shape_host[d];
  }

  if (num_output_elements == 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (num_output_elements + kBlockSize - 1) / kBlockSize);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_pad: dtype=%d, rank=%d, mode=%d, num_out=%lld\n",
      hip_dtype, rank, pad_mode, (long long)num_output_elements);

  (void)hipGetLastError();
  int *err_flag = static_cast<int *>(device_error_flag);
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<__half>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), p, pads_dev, axes_dev,
                       n_axes, static_cast<const __half *>(constant_value_dev),
                       pad_mode, num_output_elements, err_flag);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), p, pads_dev, axes_dev,
                       n_axes, static_cast<const float *>(constant_value_dev),
                       pad_mode, num_output_elements, err_flag);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<int32_t *>(output), p, pads_dev, axes_dev,
                       n_axes, static_cast<const int32_t *>(constant_value_dev),
                       pad_mode, num_output_elements, err_flag);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<int64_t *>(output), p, pads_dev, axes_dev,
                       n_axes, static_cast<const int64_t *>(constant_value_dev),
                       pad_mode, num_output_elements, err_flag);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_pad: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_pad launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
