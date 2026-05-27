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
                               int pad_mode, T pad_value,
                               int64_t num_output_elements) {
  int64_t out_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= num_output_elements)
    return;

  int64_t in_idx;
  if (compute_input_offset(p, out_idx, pad_mode, in_idx)) {
    output[out_idx] = input[in_idx];
  } else {
    output[out_idx] = pad_value;
  }
}

extern "C" int hip_pad(void *stream, const void *input, void *output,
                       const int64_t *input_shape_host,
                       const int64_t *output_shape_host,
                       const int64_t *lower_pads_host, int rank,
                       int hip_dtype, int pad_mode,
                       const void *pad_value_host) {
  if (rank <= 0)
    return 0;
  if (rank > kPadMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_pad: rank=%d exceeds kPadMaxRank=%d\n",
            rank, kPadMaxRank);
    return -1;
  }

  PadParams p{};
  p.rank = rank;
  int64_t num_output_elements = 1;
  for (int d = 0; d < rank; ++d) {
    p.input_shape[d] = input_shape_host[d];
    p.output_shape[d] = output_shape_host[d];
    p.lower_pads[d] = lower_pads_host[d];
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

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_pad: dtype=%d, rank=%d, mode=%d, num_out=%lld\n",
      hip_dtype, rank, pad_mode, (long long)num_output_elements);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16: {
    __half pv = pad_value_host
                    ? *static_cast<const __half *>(pad_value_host)
                    : __float2half(0.0f);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<__half>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), p, pad_mode, pv,
                       num_output_elements);
    break;
  }
  case HIP_DTYPE_FLOAT32: {
    float pv = pad_value_host ? *static_cast<const float *>(pad_value_host)
                              : 0.0f;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), p, pad_mode, pv,
                       num_output_elements);
    break;
  }
  case HIP_DTYPE_INT32: {
    int32_t pv = pad_value_host
                     ? *static_cast<const int32_t *>(pad_value_host)
                     : 0;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<int32_t *>(output), p, pad_mode, pv,
                       num_output_elements);
    break;
  }
  case HIP_DTYPE_INT64: {
    int64_t pv = pad_value_host
                     ? *static_cast<const int64_t *>(pad_value_host)
                     : 0;
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<int64_t *>(output), p, pad_mode, pv,
                       num_output_elements);
    break;
  }
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
