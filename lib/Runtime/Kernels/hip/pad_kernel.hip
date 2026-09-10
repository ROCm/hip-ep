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
 * `pads`, `axes` and `constant_value` stay on the device. Launch geometry
 * depends only on the (static) output shape, so per block thread 0 scatters
 * `pads` through `axes` into a shared per-dimension `lower_pads` and loads
 * the constant value. Resolving on the device rather than in the host
 * wrapper avoids a D2H and the full stream drain needed to observe it.
 *
 * Ported from onnxruntime/core/providers/cuda/tensor/pad_impl.cu @
 * v1.22.2 (`_PadKernel` -- generic ND, kept as a single fused kernel).
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kPadMaxRank = 8;

// Host-known geometry, baked into the launch as a by-value argument.
struct PadShape {
  int64_t input_shape[kPadMaxRank];
  int64_t output_shape[kPadMaxRank];
  int64_t input_strides[kPadMaxRank];
  int64_t output_strides[kPadMaxRank];
  int rank;
  // Length of the `pads` tensor: 2 * (number of padded axes), begin-major.
  int num_pad_entries;
};

// ONNX-18 pads layout: [begin_0, ..., begin_K-1, end_0, ..., end_K-1] keyed
// by `axes` (defaulting to [0..rank-1]). Only the begin half is needed --
// the end padding is implied by output_shape. An axis outside [-rank, rank)
// is a malformed graph; it used to abort in the host wrapper, which could
// inspect the values, so it is skipped here and reported through the shared
// device error flag instead.
__device__ inline void resolve_pad_lower(const PadShape &p,
                                         const int64_t *__restrict__ pads,
                                         const int64_t *__restrict__ axes,
                                         int64_t *lower_pads,
                                         int *device_error_flag) {
  for (int d = 0; d < p.rank; ++d)
    lower_pads[d] = 0;

  const int num_axes = p.num_pad_entries / 2;
  for (int i = 0; i < num_axes; ++i) {
    int64_t axis = axes ? axes[i] : static_cast<int64_t>(i);
    if (axis < 0)
      axis += p.rank;
    if (axis < 0 || axis >= p.rank) {
      if (device_error_flag)
        *device_error_flag = -1;
      continue;
    }
    lower_pads[static_cast<int>(axis)] = pads[i];
  }
}

// Returns false if out_coord falls outside the input region for some dim
// in `constant` mode (so the caller should write the pad_value instead).
__device__ inline bool compute_input_offset(const PadShape &p,
                                            const int64_t *lower_pads,
                                            int64_t output_index, int pad_mode,
                                            int64_t &input_index_out) {
  int64_t remaining = output_index;
  int64_t input_index = 0;

  for (int d = 0; d < p.rank; ++d) {
    int64_t stride = p.output_strides[d];
    int64_t out_coord = remaining / stride;
    remaining -= out_coord * stride;

    int64_t in_dim = p.input_shape[d];
    int64_t lower = lower_pads[d];
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
                               T *__restrict__ output,
                               const int64_t *__restrict__ pads,
                               const int64_t *__restrict__ axes,
                               const T *__restrict__ pad_value_dev, PadShape p,
                               int pad_mode, int64_t num_output_elements,
                               int *device_error_flag) {
  // Every thread must reach the barrier, so the pad metadata is resolved
  // before the out-of-range threads drop out.
  __shared__ int64_t s_lower_pads[kPadMaxRank];
  __shared__ T s_pad_value;
  if (threadIdx.x == 0) {
    resolve_pad_lower(p, pads, axes, s_lower_pads, device_error_flag);
    s_pad_value =
        (pad_mode == 0 && pad_value_dev) ? *pad_value_dev : static_cast<T>(0);
  }
  __syncthreads();

  int64_t out_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= num_output_elements)
    return;

  int64_t in_idx;
  if (compute_input_offset(p, s_lower_pads, out_idx, pad_mode, in_idx)) {
    output[out_idx] = input[in_idx];
  } else {
    output[out_idx] = s_pad_value;
  }
}

extern "C" int hip_pad(void *stream, const void *input, void *output,
                       const int64_t *input_shape_host,
                       const int64_t *output_shape_host, const void *pads_dev,
                       const void *axes_dev, const void *pad_value_dev,
                       int num_pad_entries, int rank, int hip_dtype,
                       int pad_mode, void *device_error_flag) {
  if (rank <= 0)
    return 0;
  if (rank > kPadMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_pad: rank=%d exceeds kPadMaxRank=%d\n",
            rank, kPadMaxRank);
    return -1;
  }
  if (num_pad_entries <= 0 || num_pad_entries > 2 * kPadMaxRank ||
      (num_pad_entries % 2) != 0) {
    fprintf(stderr,
            "[custom_kernels] hip_pad: num_pad_entries=%d must be even and in "
            "[2, %d]\n",
            num_pad_entries, 2 * kPadMaxRank);
    return -1;
  }
  if (!pads_dev) {
    fprintf(stderr, "[custom_kernels] hip_pad: null pads\n");
    return -1;
  }

  PadShape p{};
  p.rank = rank;
  p.num_pad_entries = num_pad_entries;
  int64_t num_output_elements = 1;
  for (int d = 0; d < rank; ++d) {
    p.input_shape[d] = input_shape_host[d];
    p.output_shape[d] = output_shape_host[d];
    num_output_elements *= output_shape_host[d];
  }

  // Row-major strides for input and output.
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

  const int64_t *pads = static_cast<const int64_t *>(pads_dev);
  const int64_t *axes = static_cast<const int64_t *>(axes_dev);
  int *err_flag = static_cast<int *>(device_error_flag);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_pad: dtype=%d, rank=%d, mode=%d, num_out=%lld\n",
      hip_dtype, rank, pad_mode, (long long)num_output_elements);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<__half>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), pads, axes,
                       static_cast<const __half *>(pad_value_dev), p, pad_mode,
                       num_output_elements, err_flag);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), pads, axes,
                       static_cast<const float *>(pad_value_dev), p, pad_mode,
                       num_output_elements, err_flag);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<int32_t *>(output), pads, axes,
                       static_cast<const int32_t *>(pad_value_dev), p, pad_mode,
                       num_output_elements, err_flag);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<int64_t *>(output), pads, axes,
                       static_cast<const int64_t *>(pad_value_dev), p, pad_mode,
                       num_output_elements, err_flag);
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
