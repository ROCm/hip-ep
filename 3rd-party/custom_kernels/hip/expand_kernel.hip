/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Expand: copy input over output, broadcasting any input dim of size 1 to
 * the corresponding output dim. ONNX Expand broadcasts with numpy semantics:
 * input is right-aligned against output; any "missing" leading input dims
 * are treated as 1.
 *
 * Ported from onnxruntime/core/providers/cuda/tensor/expand.cu @ v1.22.2.
 * Same one-thread-per-output-element + dim-walk structure as tile_kernel.hip;
 * differs only in how the in_coord is computed (broadcast vs modulo).
 *
 * The GPU `shape` input tensor is NOT read here -- the HipToLLVM Expand
 * lowering gives us host-side output_shape, which already includes whatever
 * the shape tensor would have provided.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>

static constexpr int kExpandMaxRank = 8;

struct ExpandShape {
  int64_t dims[kExpandMaxRank];
};

// One thread per output element. Walk output coordinates and use them to
// index into the (right-aligned, leading-padded) input. `aligned_in_shape`
// is pre-padded on the host so it has `output_rank` entries with leading
// 1s as needed.
template <typename T>
__global__ void hip_expand_kernel(const T *__restrict__ input,
                                  T *__restrict__ output,
                                  ExpandShape aligned_in_shape,
                                  ExpandShape output_shape,
                                  ExpandShape aligned_in_strides,
                                  int output_rank,
                                  int64_t num_output_elements) {
  int64_t out_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= num_output_elements)
    return;

  int64_t remaining = out_idx;
  int64_t in_idx = 0;

  // Right-to-left dim walk:
  //   out_coord = remaining % output_shape[d];
  //   remaining /= output_shape[d];
  //   in_coord  = (aligned_in_shape[d] == 1) ? 0 : out_coord;
  //   in_idx   += in_coord * aligned_in_strides[d];
  for (int d = output_rank - 1; d >= 0; --d) {
    int64_t out_dim = output_shape.dims[d];
    int64_t in_dim = aligned_in_shape.dims[d];
    int64_t out_coord = remaining % out_dim;
    remaining /= out_dim;
    int64_t in_coord = (in_dim == 1) ? 0 : out_coord;
    in_idx += in_coord * aligned_in_strides.dims[d];
  }

  output[out_idx] = input[in_idx];
}

extern "C" int hip_expand(void *stream, const void *input, void *output,
                          const int64_t *input_shape_host, int input_rank,
                          const int64_t *output_shape_host, int output_rank,
                          int hip_dtype) {
  if (output_rank <= 0)
    return 0;
  if (output_rank > kExpandMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_expand: output_rank=%d exceeds "
            "kExpandMaxRank=%d\n",
            output_rank, kExpandMaxRank);
    return -1;
  }
  if (input_rank > output_rank) {
    fprintf(stderr,
            "[custom_kernels] hip_expand: input_rank(%d) > output_rank(%d)\n",
            input_rank, output_rank);
    return -1;
  }

  // Right-align input shape against output shape (leading dims become 1).
  ExpandShape aligned_in_shape{};
  ExpandShape output_shape{};
  ExpandShape aligned_in_strides{};

  int rank_diff = output_rank - input_rank;
  int64_t num_output_elements = 1;
  for (int d = 0; d < output_rank; ++d) {
    int64_t in_dim = (d < rank_diff) ? 1 : input_shape_host[d - rank_diff];
    aligned_in_shape.dims[d] = in_dim;
    output_shape.dims[d] = output_shape_host[d];
    num_output_elements *= output_shape_host[d];
  }

  // Strides over the ORIGINAL (non-padded) input layout. A padded dim is
  // size 1, so its stride is 0 -- adding 0 won't move the read pointer.
  int64_t stride = 1;
  for (int d = output_rank - 1; d >= 0; --d) {
    if (d < rank_diff) {
      aligned_in_strides.dims[d] = 0; // padded dim
    } else {
      aligned_in_strides.dims[d] = stride;
      stride *= input_shape_host[d - rank_diff];
    }
  }

  if (num_output_elements == 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (num_output_elements + kBlockSize - 1) / kBlockSize);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_expand: dtype=%d, in_rank=%d, out_rank=%d, "
      "num_out=%lld, grid=%d, block=%d\n",
      hip_dtype, input_rank, output_rank, (long long)num_output_elements,
      grid_size, kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_expand_kernel<__half>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), aligned_in_shape,
                       output_shape, aligned_in_strides, output_rank,
                       num_output_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_expand_kernel<float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), aligned_in_shape,
                       output_shape, aligned_in_strides, output_rank,
                       num_output_elements);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_expand_kernel<int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<int32_t *>(output), aligned_in_shape,
                       output_shape, aligned_in_strides, output_rank,
                       num_output_elements);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_expand_kernel<int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<int64_t *>(output), aligned_in_shape,
                       output_shape, aligned_in_strides, output_rank,
                       num_output_elements);
    break;
  // Bitwise broadcast for 1-byte payloads (Equal/Mask outputs are uint8 or
  // bool / int8). int8 instantiation covers both since the kernel only
  // reads + writes 1-byte slots and never does arithmetic.
  case HIP_DTYPE_INT8:
  case HIP_DTYPE_UINT8:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_expand_kernel<int8_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int8_t *>(input),
                       static_cast<int8_t *>(output), aligned_in_shape,
                       output_shape, aligned_in_strides, output_rank,
                       num_output_elements);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_expand: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_expand launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
