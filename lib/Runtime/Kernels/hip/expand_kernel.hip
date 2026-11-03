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

// Under 16 live lanes a 256-wide block is mostly idle and the repeat loop is
// serial. That was slower than the general kernel for an inner of 1 and of
// 4, so those shapes use the span kernel. A wider inner keeps the load-once
// kernel.
static constexpr int64_t kExpandReplicateMinInner = 16;

// One broadcast axis, input dense, inner wide enough to fill lanes. Each
// thread loads one source element and writes it `repeat` times.
template <typename T>
__global__ void hip_expand_replicate_kernel(const T *__restrict__ input,
                                            T *__restrict__ output,
                                            int64_t inner, int64_t repeat,
                                            int64_t inner_tiles) {
  const int64_t tile = static_cast<int64_t>(blockIdx.x) % inner_tiles;
  const int64_t outer = static_cast<int64_t>(blockIdx.x) / inner_tiles;
  const int64_t i = tile * blockDim.x + threadIdx.x;
  if (i >= inner)
    return;
  const T v = input[outer * inner + i];
  T *dst = output + outer * repeat * inner + i;
  for (int64_t r = 0; r < repeat; ++r)
    dst[r * inner] = v;
}

// Same shape as the kernel above, for a narrow inner. One thread writes one
// output element, so a unit inner dimension stays coalesced instead of one
// lane walking `repeat`.
template <typename T>
__global__ void hip_expand_replicate_span_kernel(
    const T *__restrict__ input, T *__restrict__ output, int64_t inner,
    int64_t span, int64_t tiles, int64_t inner_mask, int grid_2d) {
  int64_t outer;
  int64_t t;
  if (grid_2d) {
    outer = static_cast<int64_t>(blockIdx.y);
    t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  } else {
    outer = static_cast<int64_t>(blockIdx.x) / tiles;
    const int64_t tile = static_cast<int64_t>(blockIdx.x) % tiles;
    t = tile * blockDim.x + threadIdx.x;
  }
  if (t >= span)
    return;
  const T *src = input + outer * inner;
  T *dst = output + outer * span;
  if (inner == 1) {
    dst[t] = src[0];
    return;
  }
  const int64_t col = inner_mask >= 0 ? (t & inner_mask) : (t % inner);
  dst[t] = src[col];
}

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

// Single broadcast axis over a dense input. Returns false when the shape is
// not that pattern, so the caller keeps the general kernel.
static bool expand_replicate_shape(const ExpandShape &in_shape,
                                   const ExpandShape &out_shape, int rank,
                                   int64_t *outer, int64_t *inner,
                                   int64_t *repeat) {
  int bcast = -1;
  int64_t rep = 1;
  for (int d = 0; d < rank; ++d) {
    const int64_t in_dim = in_shape.dims[d];
    const int64_t out_dim = out_shape.dims[d];
    if (in_dim == out_dim)
      continue;
    if (bcast >= 0 || in_dim != 1 || out_dim <= 1)
      return false;
    bcast = d;
    rep = out_dim;
  }
  if (bcast < 0)
    return false;
  int64_t in_elems = 1;
  int64_t out_inner = 1;
  for (int d = 0; d < bcast; ++d)
    in_elems *= out_shape.dims[d];
  for (int d = bcast + 1; d < rank; ++d)
    out_inner *= out_shape.dims[d];
  *outer = in_elems;
  *inner = out_inner;
  *repeat = rep;
  return true;
}

static int expand_launch_status(const char *what) {
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_expand %s: %s\n", what,
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

static int span_block_size(int64_t span) {
  int block = 32;
  while (static_cast<int64_t>(block) * 2 <= span && block < 256)
    block *= 2;
  if (static_cast<int64_t>(block) < span && block < 256)
    block *= 2;
  return block;
}

template <typename T>
static int launch_expand_replicate_span(hipStream_t stream, const void *input,
                                        void *output, int64_t outer,
                                        int64_t inner, int64_t repeat) {
  const int64_t span = repeat * inner;
  if (span <= 0 || outer <= 0)
    return 1;
  const int block = span_block_size(span);
  const int64_t tiles = (span + block - 1) / block;
  if (tiles <= 0 || outer > (0x7fffffff / tiles))
    return 1;
  const int64_t inner_mask =
      (inner & (inner - 1)) == 0 ? inner - 1 : static_cast<int64_t>(-1);
  const int grid_2d = outer <= 65535 && tiles <= 65535;
  hipGetLastError();
  if (grid_2d) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_expand_replicate_span_kernel<T>),
                       dim3(static_cast<unsigned>(tiles),
                            static_cast<unsigned>(outer)),
                       dim3(block), 0, stream, static_cast<const T *>(input),
                       static_cast<T *>(output), inner, span, tiles,
                       inner_mask, grid_2d);
  } else {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_expand_replicate_span_kernel<T>),
                       dim3(static_cast<unsigned>(outer * tiles)), dim3(block),
                       0, stream, static_cast<const T *>(input),
                       static_cast<T *>(output), inner, span, tiles,
                       inner_mask, grid_2d);
  }
  return 0;
}

template <typename T>
static int launch_expand_replicate(hipStream_t stream, const void *input,
                                   void *output, int64_t outer, int64_t inner,
                                   int64_t repeat) {
  if (inner < kExpandReplicateMinInner) {
    return launch_expand_replicate_span<T>(stream, input, output, outer, inner,
                                           repeat);
  }
  constexpr int kBlock = 256;
  const int64_t inner_tiles = (inner + kBlock - 1) / kBlock;
  const int64_t grid64 = outer * inner_tiles;
  if (grid64 <= 0 || grid64 > 0x7fffffff)
    return 1;
  hipGetLastError();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_expand_replicate_kernel<T>),
                     dim3(static_cast<unsigned>(grid64)), dim3(kBlock), 0,
                     stream, static_cast<const T *>(input),
                     static_cast<T *>(output), inner, repeat, inner_tiles);
  return 0;
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

  int64_t rep_outer = 0, rep_inner = 0, rep_repeat = 0;
  if (expand_replicate_shape(aligned_in_shape, output_shape, output_rank,
                             &rep_outer, &rep_inner, &rep_repeat)) {
    int rep_rc = 1;
    switch (hip_dtype) {
    case HIP_DTYPE_FLOAT16:
      rep_rc = launch_expand_replicate<__half>(hip_stream, input, output,
                                               rep_outer, rep_inner, rep_repeat);
      break;
    case HIP_DTYPE_FLOAT32:
      rep_rc = launch_expand_replicate<float>(hip_stream, input, output,
                                              rep_outer, rep_inner, rep_repeat);
      break;
    case HIP_DTYPE_INT32:
      rep_rc = launch_expand_replicate<int32_t>(
          hip_stream, input, output, rep_outer, rep_inner, rep_repeat);
      break;
    case HIP_DTYPE_INT64:
      rep_rc = launch_expand_replicate<int64_t>(
          hip_stream, input, output, rep_outer, rep_inner, rep_repeat);
      break;
    case HIP_DTYPE_INT8:
    case HIP_DTYPE_UINT8:
      rep_rc = launch_expand_replicate<int8_t>(
          hip_stream, input, output, rep_outer, rep_inner, rep_repeat);
      break;
    case HIP_DTYPE_INT16:
    case HIP_DTYPE_UINT16:
      rep_rc = launch_expand_replicate<int16_t>(
          hip_stream, input, output, rep_outer, rep_inner, rep_repeat);
      break;
    default:
      break;
    }
    if (rep_rc == 0)
      return expand_launch_status("replicate launch failed");
  }

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
  // 2-byte integer payloads. uint16 reuses the int16 instantiation: the
  // kernel only copies elements and never does arithmetic. wrap_equal
  // materialises a partial broadcast through this entry.
  case HIP_DTYPE_INT16:
  case HIP_DTYPE_UINT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_expand_kernel<int16_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int16_t *>(input),
                       static_cast<int16_t *>(output), aligned_in_shape,
                       output_shape, aligned_in_strides, output_rank,
                       num_output_elements);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_expand: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  return expand_launch_status("launch failed");
}
