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

namespace {

// Division by a runtime constant as a multiply-high and a shift. Valid for
// numerators below 2^31; the launcher checks that before taking the fast path.
struct TDiv {
  uint32_t mul; // 0 => a pure shift (divisor is a power of two)
  uint32_t shr;
};

__device__ __forceinline__ uint32_t tFastDiv(uint32_t n, TDiv d) {
  return d.mul ? (__umulhi(n, d.mul) >> d.shr) : (n >> d.shr);
}

TDiv makeTDiv(int64_t denom) {
  const uint32_t d = static_cast<uint32_t>(denom < 1 ? 1 : denom);
  TDiv r{0u, 0u};
  uint32_t l = 0;
  while ((1u << l) < d)
    ++l;
  if ((1u << l) == d) {
    r.shr = l;
    return r;
  }
  const uint32_t p = 31 + l;
  r.mul = static_cast<uint32_t>(((1ull << p) + d - 1) / d);
  r.shr = p - 32;
  return r;
}

constexpr int kPadFastMaxRank = 3;
constexpr int kPadFastPerThread = 4;

struct PadFastParams {
  int32_t in[kPadFastMaxRank];
  int32_t out[kPadFastMaxRank];
  int32_t lower[kPadFastMaxRank];
  int32_t in_stride[kPadFastMaxRank];
  TDiv r_out[kPadFastMaxRank];
  int32_t groups; // ceil(out[rank-1] / kPadFastPerThread)
  TDiv r_groups;
};

// Maps an output coordinate to an input coordinate, or -1 for a constant fill.
// Same formulas as compute_input_offset.
template <int MODE>
__device__ __forceinline__ int32_t padMap(int32_t o, int32_t lower,
                                          int32_t n) {
  int32_t i = o - lower;
  if (i < 0) {
    if (MODE == 0)
      return -1;
    if (MODE == 2)
      return 0;
    if (MODE == 1)
      return -i;
    return ((i % n) + n) % n;
  }
  if (i >= n) {
    if (MODE == 0)
      return -1;
    if (MODE == 2)
      return n - 1;
    if (MODE == 1)
      return n - 2 - (i - n);
    return i % n;
  }
  return i;
}

// Each thread writes kPadFastPerThread consecutive outputs of one innermost
// row. The row's outer coordinates are decoded once.
template <typename T, int MODE, int RANK>
__global__ void pad_fast_kernel(const T *__restrict__ input,
                                T *__restrict__ output, PadFastParams p,
                                T pad_value, uint32_t num_threads) {
  const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= num_threads)
    return;

  const uint32_t row = tFastDiv(tid, p.r_groups);
  const int32_t col0 =
      static_cast<int32_t>(tid - row * p.groups) * kPadFastPerThread;

  bool row_valid = true;
  int32_t base = 0;
  uint32_t rem = row;
#pragma unroll
  for (int d = RANK - 2; d >= 0; --d) {
    const uint32_t q = tFastDiv(rem, p.r_out[d]);
    const int32_t c = static_cast<int32_t>(rem - q * p.out[d]);
    rem = q;
    const int32_t ic = padMap<MODE>(c, p.lower[d], p.in[d]);
    if (ic < 0)
      row_valid = false;
    base += ic * p.in_stride[d];
  }

  const int32_t inner = p.out[RANK - 1];
  T *out_row = output + static_cast<size_t>(row) * inner;
#pragma unroll
  for (int j = 0; j < kPadFastPerThread; ++j) {
    const int32_t c = col0 + j;
    if (c >= inner)
      break;
    const int32_t ic = padMap<MODE>(c, p.lower[RANK - 1], p.in[RANK - 1]);
    out_row[c] = (row_valid && ic >= 0) ? input[base + ic] : pad_value;
  }
}

struct PadDim {
  int64_t in, out, lower;
};

// Merges runs of adjacent unpadded dims (lower == 0 and in == out). A padded
// dim is never merged with a neighbour: reflect/edge/wrap act per element of
// that dim, so folding it into another dim would change the result.
int mergePadDims(const PadParams &p, PadDim *dims) {
  int n = 0;
  bool prev_unpadded = false;
  for (int d = 0; d < p.rank; ++d) {
    const PadDim cur{p.input_shape[d], p.output_shape[d], p.lower_pads[d]};
    const bool unpadded = cur.lower == 0 && cur.in == cur.out;
    if (unpadded && cur.in == 1)
      continue;
    if (unpadded && prev_unpadded) {
      dims[n - 1].in *= cur.in;
      dims[n - 1].out *= cur.out;
    } else {
      dims[n++] = cur;
    }
    prev_unpadded = unpadded;
  }
  if (n == 0)
    dims[n++] = PadDim{1, 1, 0};
  return n;
}

template <typename T, int MODE, int RANK>
void launchPadFast(hipStream_t stream, const T *input, T *output,
                   const PadFastParams &fp, T pad_value,
                   uint32_t num_threads) {
  constexpr int kBlock = 256;
  const unsigned grid = (num_threads + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(pad_fast_kernel<T, MODE, RANK>),
                     dim3(grid), dim3(kBlock), 0, stream, input, output, fp,
                     pad_value, num_threads);
}

template <typename T, int MODE>
void launchPadFastRank(int rank, hipStream_t stream, const T *input,
                       T *output, const PadFastParams &fp, T pad_value,
                       uint32_t num_threads) {
  switch (rank) {
  case 1:
    launchPadFast<T, MODE, 1>(stream, input, output, fp, pad_value,
                              num_threads);
    break;
  case 2:
    launchPadFast<T, MODE, 2>(stream, input, output, fp, pad_value,
                              num_threads);
    break;
  default:
    launchPadFast<T, MODE, 3>(stream, input, output, fp, pad_value,
                              num_threads);
    break;
  }
}

// Launches the fast kernel when the merged problem fits it. Returns false so
// the caller falls back to hip_pad_kernel.
template <typename T>
bool tryPadFast(hipStream_t stream, const T *input, T *output,
                const PadParams &p, int pad_mode, T pad_value) {
  if (pad_mode < 0 || pad_mode > 3)
    return false;
  PadDim dims[kPadMaxRank];
  const int rank = mergePadDims(p, dims);
  if (rank > kPadFastMaxRank)
    return false;

  constexpr int64_t kLimit = int64_t(1) << 31;
  int64_t in_elems = 1, out_elems = 1;
  for (int d = 0; d < rank; ++d) {
    if (dims[d].in <= 0 || dims[d].out <= 0 || dims[d].in >= kLimit ||
        dims[d].out >= kLimit || dims[d].lower <= -kLimit ||
        dims[d].lower >= kLimit)
      return false;
    in_elems *= dims[d].in;
    out_elems *= dims[d].out;
    if (in_elems >= kLimit || out_elems >= kLimit)
      return false;
  }

  PadFastParams fp{};
  int64_t stride = 1;
  for (int d = rank - 1; d >= 0; --d) {
    fp.in[d] = static_cast<int32_t>(dims[d].in);
    fp.out[d] = static_cast<int32_t>(dims[d].out);
    fp.lower[d] = static_cast<int32_t>(dims[d].lower);
    fp.in_stride[d] = static_cast<int32_t>(stride);
    fp.r_out[d] = makeTDiv(dims[d].out);
    stride *= dims[d].in;
  }
  const int64_t inner = dims[rank - 1].out;
  const int64_t groups = (inner + kPadFastPerThread - 1) / kPadFastPerThread;
  fp.groups = static_cast<int32_t>(groups);
  fp.r_groups = makeTDiv(groups);
  const int64_t threads = (out_elems / inner) * groups;
  if (threads >= kLimit)
    return false;
  const uint32_t num_threads = static_cast<uint32_t>(threads);

  switch (pad_mode) {
  case 0:
    launchPadFastRank<T, 0>(rank, stream, input, output, fp, pad_value,
                            num_threads);
    break;
  case 1:
    launchPadFastRank<T, 1>(rank, stream, input, output, fp, pad_value,
                            num_threads);
    break;
  case 2:
    launchPadFastRank<T, 2>(rank, stream, input, output, fp, pad_value,
                            num_threads);
    break;
  default:
    launchPadFastRank<T, 3>(rank, stream, input, output, fp, pad_value,
                            num_threads);
    break;
  }
  return true;
}

template <typename T>
void launchPad(hipStream_t stream, const void *input, void *output,
               const PadParams &p, int pad_mode, T pad_value,
               int64_t num_output_elements) {
  const T *in = static_cast<const T *>(input);
  T *out = static_cast<T *>(output);
  if (tryPadFast<T>(stream, in, out, p, pad_mode, pad_value))
    return;
  constexpr int kBlockSize = 256;
  const int grid_size =
      static_cast<int>((num_output_elements + kBlockSize - 1) / kBlockSize);
  hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_pad_kernel<T>), dim3(grid_size),
                     dim3(kBlockSize), 0, stream, in, out, p, pad_mode,
                     pad_value, num_output_elements);
}

} // namespace

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

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_pad: dtype=%d, rank=%d, mode=%d, num_out=%lld\n",
      hip_dtype, rank, pad_mode, (long long)num_output_elements);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16: {
    __half pv = pad_value_host
                    ? *static_cast<const __half *>(pad_value_host)
                    : __float2half(0.0f);
    launchPad<__half>(hip_stream, input, output, p, pad_mode, pv,
                      num_output_elements);
    break;
  }
  case HIP_DTYPE_FLOAT32: {
    float pv = pad_value_host ? *static_cast<const float *>(pad_value_host)
                              : 0.0f;
    launchPad<float>(hip_stream, input, output, p, pad_mode, pv,
                     num_output_elements);
    break;
  }
  case HIP_DTYPE_INT32: {
    int32_t pv = pad_value_host
                     ? *static_cast<const int32_t *>(pad_value_host)
                     : 0;
    launchPad<int32_t>(hip_stream, input, output, p, pad_mode, pv,
                       num_output_elements);
    break;
  }
  case HIP_DTYPE_INT64: {
    int64_t pv = pad_value_host
                     ? *static_cast<const int64_t *>(pad_value_host)
                     : 0;
    launchPad<int64_t>(hip_stream, input, output, p, pad_mode, pv,
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
