/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Tile: copy input over the output shape, treating each output coordinate
 * as `out_coord[d] % input_shape[d]` along the input side.
 *
 * Ported from onnxruntime/core/providers/cuda/tensor/tile.cu @ v1.22.2.
 * The CUDA EP code uses TArray + fast_divmod for shape divmod; we use plain
 * int64 modulo against fixed-size stack arrays (rank capped at 8). The hot
 * path is therefore identical: one thread per output element, walk the
 * dimensions to translate output linear -> input linear, copy one element.
 *
 * The `repeats` GPU buffer is NOT read here -- the HipToLLVM Tile lowering
 * already gives us host-side input_shape + output_shape, which together
 * imply repeats[d] = output_shape[d] / input_shape[d]. Avoids a per-call
 * D2H of the repeats tensor.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cstdio>
#include <limits>

static constexpr int kTileMaxRank = 8;

struct TileShape {
  int64_t dims[kTileMaxRank];
};

// One thread per output element. Compute the per-dim output coordinate,
// then mod by input shape to get the input coordinate; assemble the input
// linear index from precomputed input row strides.
template <typename T>
__global__ void hip_tile_kernel(const T *__restrict__ input,
                                T *__restrict__ output,
                                TileShape input_shape,
                                TileShape output_shape,
                                TileShape input_strides,
                                int rank, int64_t num_output_elements) {
  int64_t out_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= num_output_elements)
    return;

  int64_t remaining = out_idx;
  int64_t in_idx = 0;

  // Stride iteration: for each dim (rank-1 down to 0):
  //   out_coord = remaining % output_shape[d];
  //   remaining /= output_shape[d];
  //   in_coord  = out_coord % input_shape[d];
  //   in_idx   += in_coord * input_strides[d];
  for (int d = rank - 1; d >= 0; --d) {
    int64_t out_dim = output_shape.dims[d];
    int64_t in_dim = input_shape.dims[d];
    int64_t out_coord = remaining % out_dim;
    remaining /= out_dim;
    int64_t in_coord = out_coord % in_dim;
    in_idx += in_coord * input_strides.dims[d];
  }

  output[out_idx] = input[in_idx];
}

extern "C" int hip_tile(void *stream, const void *input, void *output,
                        const int64_t *input_shape_host,
                        const int64_t *output_shape_host, int rank,
                        int hip_dtype) {
  if (rank <= 0)
    return 0;
  if (rank > kTileMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_tile: rank=%d exceeds kTileMaxRank=%d\n",
            rank, kTileMaxRank);
    return -1;
  }

  // Compute total output elements + input row strides on the host.
  TileShape input_shape{};
  TileShape output_shape{};
  TileShape input_strides{};
  int64_t num_output_elements = 1;
  for (int d = 0; d < rank; ++d) {
    if (input_shape_host[d] < 0 || output_shape_host[d] < 0 ||
        (output_shape_host[d] != 0 &&
         num_output_elements >
             std::numeric_limits<int64_t>::max() / output_shape_host[d])) {
      fprintf(stderr,
              "[custom_kernels] hip_tile: invalid or overflowing shape at "
              "dimension %d\n",
              d);
      return -1;
    }
    input_shape.dims[d] = input_shape_host[d];
    output_shape.dims[d] = output_shape_host[d];
    num_output_elements *= output_shape_host[d];
  }

  int64_t stride = 1;
  for (int d = rank - 1; d >= 0; --d) {
    if (input_shape_host[d] != 0 &&
        stride > std::numeric_limits<int64_t>::max() / input_shape_host[d]) {
      fprintf(stderr,
              "[custom_kernels] hip_tile: input stride overflow at dimension "
              "%d\n",
              d);
      return -1;
    }
    input_strides.dims[d] = stride;
    stride *= input_shape_host[d];
  }

  if (num_output_elements == 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  constexpr int kBlockSize = 256;
  uint64_t gridSize = (static_cast<uint64_t>(num_output_elements) +
                       kBlockSize - 1) /
                      kBlockSize;
  if (gridSize > std::numeric_limits<uint32_t>::max()) {
    fprintf(stderr, "[custom_kernels] hip_tile: grid dimension overflow\n");
    return -1;
  }
  uint32_t grid_size = static_cast<uint32_t>(gridSize);

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_tile: dtype=%d, rank=%d, num_out=%lld, "
      "grid=%d, block=%d\n",
      hip_dtype, rank, (long long)num_output_elements, grid_size, kBlockSize);

  (void)hipGetLastError();

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_tile_kernel<__half>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const __half *>(input),
                       static_cast<__half *>(output), input_shape,
                       output_shape, input_strides, rank, num_output_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_tile_kernel<float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const float *>(input),
                       static_cast<float *>(output), input_shape, output_shape,
                       input_strides, rank, num_output_elements);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_tile_kernel<int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int32_t *>(input),
                       static_cast<int32_t *>(output), input_shape,
                       output_shape, input_strides, rank, num_output_elements);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_tile_kernel<int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(input),
                       static_cast<int64_t *>(output), input_shape,
                       output_shape, input_strides, rank, num_output_elements);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_tile: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_tile launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
