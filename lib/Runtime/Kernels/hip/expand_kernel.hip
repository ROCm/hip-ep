/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Expand: copy input over output, broadcasting any input dim of size 1 to
 * the corresponding output dim. ONNX Expand broadcasts with numpy semantics:
 * input is right-aligned against output; any "missing" leading input dims
 * are treated as 1.
 *
 * The host collapses the shapes first: size-1 output dims are dropped and
 * adjacent dims that are both broadcast or both copied are merged, so a
 * per-channel broadcast such as [1,C,1,1] -> [N,C,H,W] becomes a rank-3
 * problem with one contiguous innermost run. Each thread then writes up to
 * 16 bytes of that run, and outer coordinates come from 32-bit
 * multiply-shift division. Integer division has no hardware instruction on
 * AMD GPUs, so a 64-bit divide per dim per element made the kernel
 * index-bound rather than bandwidth-bound. Shapes beyond 32-bit indexing
 * keep a 64-bit per-element path.
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
static constexpr int kBlockSize = 256;
static constexpr int kVectorBytes = 16;

// Unsigned division by a launch-invariant divisor as a multiply-high plus
// shift. Exact for dividends and divisors below 2^31.
struct FastDivmod {
  uint32_t divisor;
  uint32_t multiplier;
  uint32_t shift;

  static FastDivmod make(uint32_t d) {
    uint32_t s = 0;
    while ((uint64_t(1) << s) < d)
      ++s;
    uint64_t m = ((uint64_t(1) << 32) * ((uint64_t(1) << s) - d)) / d + 1;
    return {d, static_cast<uint32_t>(m), s};
  }

  __device__ uint32_t div(uint32_t n) const {
    return (__umulhi(n, multiplier) + n) >> shift;
  }
};

// Collapsed problem: `outer` dims (outermost first) above one contiguous
// innermost run of `inner` elements. A broadcast dim has input stride 0.
struct ExpandParams32 {
  FastDivmod outer[kExpandMaxRank];
  uint32_t outer_strides[kExpandMaxRank];
  FastDivmod inner_vectors;
  int outer_rank;
  bool inner_broadcast;
};

template <typename T, int V> struct alignas(sizeof(T) * V) VectorOf {
  T v[V];
};

// One thread per V-element vector of the innermost run.
template <typename T, int V>
__global__ void hip_expand_kernel32(const T *__restrict__ input,
                                    T *__restrict__ output, ExpandParams32 p,
                                    uint32_t num_vectors) {
  uint32_t vec = blockIdx.x * blockDim.x + threadIdx.x;
  if (vec >= num_vectors)
    return;

  uint32_t row = p.inner_vectors.div(vec);
  uint32_t in_idx =
      p.inner_broadcast ? 0 : (vec - row * p.inner_vectors.divisor) * V;
  for (int d = p.outer_rank - 1; d > 0; --d) {
    uint32_t quotient = p.outer[d].div(row);
    in_idx += (row - quotient * p.outer[d].divisor) * p.outer_strides[d];
    row = quotient;
  }
  if (p.outer_rank > 0)
    in_idx += row * p.outer_strides[0];

  T *dst = output + static_cast<size_t>(vec) * V;
  if (V == 1) {
    *dst = input[in_idx];
  } else {
    VectorOf<T, V> value;
    if (p.inner_broadcast) {
      T x = input[in_idx];
#pragma unroll
      for (int i = 0; i < V; ++i)
        value.v[i] = x;
    } else {
      value = *reinterpret_cast<const VectorOf<T, V> *>(input + in_idx);
    }
    *reinterpret_cast<VectorOf<T, V> *>(dst) = value;
  }
}

struct ExpandShape {
  int64_t dims[kExpandMaxRank];
};

// One thread per output element with 64-bit index math, for tensors whose
// element count or extents do not fit the 32-bit path.
template <typename T>
__global__ void hip_expand_kernel64(const T *__restrict__ input,
                                    T *__restrict__ output,
                                    ExpandShape output_shape,
                                    ExpandShape in_strides, int rank,
                                    int64_t num_output_elements) {
  int64_t out_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (out_idx >= num_output_elements)
    return;

  int64_t remaining = out_idx;
  int64_t in_idx = 0;
  for (int d = rank - 1; d >= 0; --d) {
    int64_t out_dim = output_shape.dims[d];
    in_idx += (remaining % out_dim) * in_strides.dims[d];
    remaining /= out_dim;
  }
  output[out_idx] = input[in_idx];
}

static bool isAligned(const void *ptr, size_t bytes) {
  return reinterpret_cast<uintptr_t>(ptr) % bytes == 0;
}

template <typename T>
static void launchExpand(hipStream_t stream, const T *input, T *output,
                         const int64_t *dims, const bool *broadcast,
                         const int64_t *in_strides, int rank,
                         int64_t num_output_elements,
                         int64_t num_input_elements) {
  constexpr int kVec = kVectorBytes / sizeof(T);
  int64_t inner = dims[rank - 1];
  bool inner_broadcast = broadcast[rank - 1];

  bool fits32 = num_output_elements < (int64_t(1) << 31) &&
                num_input_elements < (int64_t(1) << 31);
  if (fits32) {
    bool vectorize = kVec > 1 && inner % kVec == 0 &&
                     isAligned(output, kVectorBytes) &&
                     (inner_broadcast || isAligned(input, kVectorBytes));
    int vec = vectorize ? kVec : 1;

    ExpandParams32 p{};
    p.outer_rank = rank - 1;
    for (int d = 0; d < rank - 1; ++d) {
      p.outer[d] = FastDivmod::make(static_cast<uint32_t>(dims[d]));
      p.outer_strides[d] = static_cast<uint32_t>(in_strides[d]);
    }
    p.inner_vectors = FastDivmod::make(static_cast<uint32_t>(inner / vec));
    p.inner_broadcast = inner_broadcast;
    uint32_t num_vectors = static_cast<uint32_t>(num_output_elements / vec);
    int grid = static_cast<int>((num_vectors + kBlockSize - 1) / kBlockSize);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_expand: collapsed_rank=%d, inner=%lld, "
        "inner_broadcast=%d, vec=%d, grid=%d\n",
        rank, (long long)inner, (int)inner_broadcast, vec, grid);
    if (vectorize)
      hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_expand_kernel32<T, kVec>),
                         dim3(grid), dim3(kBlockSize), 0, stream, input,
                         output, p, num_vectors);
    else
      hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_expand_kernel32<T, 1>),
                         dim3(grid), dim3(kBlockSize), 0, stream, input,
                         output, p, num_vectors);
    return;
  }

  ExpandShape shape{};
  ExpandShape strides{};
  for (int d = 0; d < rank; ++d) {
    shape.dims[d] = dims[d];
    strides.dims[d] = in_strides[d];
  }
  int64_t grid = (num_output_elements + kBlockSize - 1) / kBlockSize;
  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_expand: 64-bit path, collapsed_rank=%d, "
      "num_out=%lld, grid=%lld\n",
      rank, (long long)num_output_elements, (long long)grid);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_expand_kernel64<T>),
                     dim3(static_cast<unsigned>(grid)), dim3(kBlockSize), 0,
                     stream, input, output, shape, strides, rank,
                     num_output_elements);
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

  // Right-align the input against the output, drop size-1 output dims, and
  // merge neighbours with the same broadcast state.
  int64_t dims[kExpandMaxRank];
  bool broadcast[kExpandMaxRank];
  int rank = 0;
  int rank_diff = output_rank - input_rank;
  int64_t num_output_elements = 1;
  int64_t num_input_elements = 1;
  for (int d = 0; d < output_rank; ++d) {
    int64_t out_dim = output_shape_host[d];
    int64_t in_dim = (d < rank_diff) ? 1 : input_shape_host[d - rank_diff];
    if (in_dim != 1 && in_dim != out_dim) {
      fprintf(stderr,
              "[custom_kernels] hip_expand: input dim %lld does not "
              "broadcast to output dim %lld\n",
              (long long)in_dim, (long long)out_dim);
      return -1;
    }
    num_output_elements *= out_dim;
    num_input_elements *= in_dim;
    if (out_dim == 1)
      continue;
    bool is_broadcast = in_dim == 1;
    if (rank > 0 && broadcast[rank - 1] == is_broadcast) {
      dims[rank - 1] *= out_dim;
    } else {
      dims[rank] = out_dim;
      broadcast[rank] = is_broadcast;
      ++rank;
    }
  }
  if (num_output_elements == 0)
    return 0;
  if (rank == 0) {
    dims[0] = 1;
    broadcast[0] = false;
    rank = 1;
  }

  int64_t in_strides[kExpandMaxRank];
  int64_t stride = 1;
  for (int d = rank - 1; d >= 0; --d) {
    in_strides[d] = broadcast[d] ? 0 : stride;
    if (!broadcast[d])
      stride *= dims[d];
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_expand: dtype=%d, in_rank=%d, out_rank=%d, "
      "num_out=%lld\n",
      hip_dtype, input_rank, output_rank, (long long)num_output_elements);

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    launchExpand(hip_stream, static_cast<const __half *>(input),
                 static_cast<__half *>(output), dims, broadcast, in_strides,
                 rank, num_output_elements, num_input_elements);
    break;
  case HIP_DTYPE_FLOAT32:
    launchExpand(hip_stream, static_cast<const float *>(input),
                 static_cast<float *>(output), dims, broadcast, in_strides,
                 rank, num_output_elements, num_input_elements);
    break;
  case HIP_DTYPE_INT32:
    launchExpand(hip_stream, static_cast<const int32_t *>(input),
                 static_cast<int32_t *>(output), dims, broadcast, in_strides,
                 rank, num_output_elements, num_input_elements);
    break;
  case HIP_DTYPE_INT64:
    launchExpand(hip_stream, static_cast<const int64_t *>(input),
                 static_cast<int64_t *>(output), dims, broadcast, in_strides,
                 rank, num_output_elements, num_input_elements);
    break;
  // Bitwise broadcast for 1-byte payloads (Equal/Mask outputs are uint8 or
  // bool / int8). int8 instantiation covers both since the kernel only
  // reads + writes 1-byte slots and never does arithmetic.
  case HIP_DTYPE_INT8:
  case HIP_DTYPE_UINT8:
    launchExpand(hip_stream, static_cast<const int8_t *>(input),
                 static_cast<int8_t *>(output), dims, broadcast, in_strides,
                 rank, num_output_elements, num_input_elements);
    break;
  // 2-byte payloads. uint16 and bf16 reuse the int16 instantiation: the
  // kernel only copies elements and never does arithmetic. wrap_equal
  // materialises a partial broadcast through this entry.
  case HIP_DTYPE_INT16:
  case HIP_DTYPE_UINT16:
  case HIP_DTYPE_BFLOAT16:
    launchExpand(hip_stream, static_cast<const int16_t *>(input),
                 static_cast<int16_t *>(output), dims, broadcast, in_strides,
                 rank, num_output_elements, num_input_elements);
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
