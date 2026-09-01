/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

// =============================================================================
// QuantizeLinear / DequantizeLinear
// =============================================================================
//
// One thread per element. Input and output are elementwise the same shape, so
// the thread's linear index addresses both directly and only the scale /
// zero_point index has to be derived.
//
// ONNX defines three quantization granularities. They are distinguished by the
// shape of `scale` rather than by a flag on the ABI:
//
//   per-tensor  scale holds a single element
//                 -> index 0
//   per-axis    scale is 1-D of length input_shape[axis]
//                 -> (i / inner) % axis_dim, inner = product of dims > axis
//   blocked     scale matches input on every dim except `axis`, where its
//               extent is ceil(input_shape[axis] / block_size)
//                 -> same row-major coord as the input, with the axis coord
//                    divided by block_size and refolded against scale's dims
//
// The launcher resolves the mode once on the host so per-tensor -- by far the
// most common case -- costs no integer division on device. The blocked walk
// mirrors hip_gather_block_quantized, which folds the same coord/block_size
// index.
//
// Quantize evaluates the spec formula `y = saturate(round(x / scale) + zp)` in
// that exact order. Dequantize evaluates `y = (x - zp) * scale` with the
// subtraction kept in int32.
//
// Intermediate arithmetic is fp32 for every storage-type combination; see the
// `precision` handling in hip_quantize_linear for how that relates to the ONNX
// attribute of the same name.

#define HIP_QDQ_MAX_RANK 8

namespace {

// The ONNX `precision` attribute is a TensorProto DataType enum, NOT a
// hip_dtype_t, and the two numberings collide (TensorProto FLOAT == 1 ==
// HIP_DTYPE_FLOAT16). Only these two values describe what this kernel computes.
constexpr int kOnnxPrecisionUnset = 0; // spec: follow the scale's type
constexpr int kOnnxPrecisionFloat = 1; // TensorProto FLOAT

// Returned by the dispatch helpers when no template instantiation matches, so
// the entry point can report all three dtypes in one message.
constexpr int kUnsupportedDtype = -2;

enum QdqGranularity {
  kQdqPerTensor = 0,
  kQdqPerAxis = 1,
  kQdqBlocked = 2,
};

struct QdqDimsND {
  int64_t v[HIP_QDQ_MAX_RANK];
};

// Everything the kernels need beyond the buffers, resolved on the host.
struct QdqLaunchPlan {
  int mode;
  int rank;
  int axis;
  int block_size;
  int64_t inner;    // per-axis only
  int64_t axis_dim; // per-axis only
  int64_t total;
  QdqDimsND input_shape; // blocked only
  QdqDimsND scale_shape; // blocked only
};

// ---- float load/store helpers --------------------------------------------

template <typename T> struct FloatIO;

template <> struct FloatIO<float> {
  __device__ static inline float load(float x) { return x; }
  __device__ static inline float store(float x) { return x; }
};

template <> struct FloatIO<__half> {
  __device__ static inline float load(__half x) { return __half2float(x); }
  __device__ static inline __half store(float x) { return __float2half(x); }
};

// ---- quantized storage carriers -------------------------------------------

// ONNX INT4 / UINT4 stores two values per byte, low nibble first, over the
// flattened row-major sequence, while every shape it declares stays logical.
// These carriers stand in for the storage type so the packing lives in the
// load and store helpers below rather than in each kernel.
struct PackedInt4 {
  uint8_t byte;
};
struct PackedUint4 {
  uint8_t byte;
};

// ---- saturation bounds ----------------------------------------------------

// Representable range per quantized storage type, as int32 (all six fit).
// Spelled out rather than taken from std::numeric_limits, which is not
// available in device code.
template <typename QuantT> struct QuantRange;

template <> struct QuantRange<int8_t> {
  static constexpr int32_t lo = -128, hi = 127;
};
template <> struct QuantRange<uint8_t> {
  static constexpr int32_t lo = 0, hi = 255;
};
template <> struct QuantRange<int16_t> {
  static constexpr int32_t lo = -32768, hi = 32767;
};
template <> struct QuantRange<uint16_t> {
  static constexpr int32_t lo = 0, hi = 65535;
};
// The carrier is a byte wide, but the value it holds is a nibble.
template <> struct QuantRange<PackedInt4> {
  static constexpr int32_t lo = -8, hi = 7;
};
template <> struct QuantRange<PackedUint4> {
  static constexpr int32_t lo = 0, hi = 15;
};

// ---- quantized element load ----------------------------------------------

__device__ inline uint32_t qdq_nibble(uint8_t byte, int64_t index) {
  return (byte >> ((index & 1) * 4)) & 0xF;
}

// Widens one logical element to int32, which represents every supported
// storage type exactly and is where dequantize subtracts the zero point.
template <typename QuantT> struct QuantLoad {
  __device__ static inline int32_t load(const QuantT *p, int64_t i) {
    return static_cast<int32_t>(p[i]);
  }
};

template <> struct QuantLoad<PackedInt4> {
  __device__ static inline int32_t load(const PackedInt4 *p, int64_t i) {
    int32_t v = static_cast<int32_t>(qdq_nibble(p[i >> 1].byte, i));
    return v >= 8 ? v - 16 : v; // sign-extend the nibble
  }
};

template <> struct QuantLoad<PackedUint4> {
  __device__ static inline int32_t load(const PackedUint4 *p, int64_t i) {
    return static_cast<int32_t>(qdq_nibble(p[i >> 1].byte, i));
  }
};

// ---- scale / zero_point index --------------------------------------------

// `inner` and `axis_dim` are read only in per-axis mode, the shapes only in
// blocked mode; the host leaves the unused fields at harmless defaults.
__device__ inline int64_t qdq_param_index(int64_t lin, int mode, int rank,
                                          int axis, int block_size,
                                          int64_t inner, int64_t axis_dim,
                                          const QdqDimsND &input_shape,
                                          const QdqDimsND &scale_shape) {
  if (mode == kQdqPerTensor)
    return 0;
  if (mode == kQdqPerAxis)
    return (lin / inner) % axis_dim;

  int64_t coord[HIP_QDQ_MAX_RANK];
  int64_t rem = lin;
  for (int d = rank - 1; d >= 0; --d) {
    coord[d] = rem % input_shape.v[d];
    rem /= input_shape.v[d];
  }
  int64_t scale_lin = 0;
  for (int d = 0; d < rank; ++d) {
    int64_t c = (d == axis) ? (coord[d] / block_size) : coord[d];
    scale_lin = scale_lin * scale_shape.v[d] + c;
  }
  return scale_lin;
}

// ---- kernels -------------------------------------------------------------

// The spec formula for the single logical element `i`, whose scale and zero
// point sit at `p`. Shared by the byte-per-value and the packed kernels so
// their rounding and clamping cannot drift apart.
template <typename InT, typename ScaleT, typename QuantT>
__device__ inline int32_t quantize_one(const InT *input, const ScaleT *scale,
                                       const QuantT *zero_point, int64_t i,
                                       int64_t p) {
  float x = FloatIO<InT>::load(input[i]);
  float s = FloatIO<ScaleT>::load(scale[p]);
  int32_t zp = zero_point ? QuantLoad<QuantT>::load(zero_point, p) : 0;

  // rintf() rounds to nearest with ties to even, which is what the spec asks
  // for; roundf() (ties away from zero) would be wrong. The division stays a
  // division on purpose: folding it into a multiply by a precomputed 1/s adds
  // a second rounding that can flip the result by one LSB at tie boundaries.
  float q = rintf(x / s) + static_cast<float>(zp);

  // Clamp in fp32 before narrowing, because converting an out-of-range or NaN
  // float to an integer is undefined in C++. This also pins NaN to `lo`
  // deterministically, since IEEE fmaxf/fminf return the non-NaN operand.
  // Every bound and every sum above is exact in fp32 for 4/8/16-bit storage.
  q = fmaxf(q, static_cast<float>(QuantRange<QuantT>::lo));
  q = fminf(q, static_cast<float>(QuantRange<QuantT>::hi));
  return static_cast<int32_t>(q);
}

template <typename InT, typename ScaleT, typename QuantT>
__global__ void quantize_linear_kernel(
    const InT *__restrict__ input, const ScaleT *__restrict__ scale,
    const QuantT *__restrict__ zero_point, QuantT *__restrict__ output,
    int64_t total, int mode, int rank, int axis, int block_size, int64_t inner,
    int64_t axis_dim, QdqDimsND input_shape, QdqDimsND scale_shape) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;

  int64_t p = qdq_param_index(tid, mode, rank, axis, block_size, inner,
                              axis_dim, input_shape, scale_shape);
  output[tid] = static_cast<QuantT>(
      quantize_one<InT, ScaleT, QuantT>(input, scale, zero_point, tid, p));
}

// One thread per output BYTE rather than per element, because a packed byte is
// shared by two logical values. A thread-per-element mapping would have to
// read-modify-write that byte and would race with its neighbour.
template <typename InT, typename ScaleT, typename QuantT>
__global__ void quantize_linear_packed_kernel(
    const InT *__restrict__ input, const ScaleT *__restrict__ scale,
    const QuantT *__restrict__ zero_point, QuantT *__restrict__ output,
    int64_t total, int mode, int rank, int axis, int block_size, int64_t inner,
    int64_t axis_dim, QdqDimsND input_shape, QdqDimsND scale_shape) {
  int64_t byte_idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  int64_t base = byte_idx * 2;
  if (base >= total)
    return;

  uint32_t packed = 0;
  for (int half = 0; half < 2; ++half) {
    int64_t i = base + half;
    // An odd element count pads the final byte; leave its high nibble zero.
    if (i >= total)
      break;
    int64_t p = qdq_param_index(i, mode, rank, axis, block_size, inner,
                                axis_dim, input_shape, scale_shape);
    int32_t q =
        quantize_one<InT, ScaleT, QuantT>(input, scale, zero_point, i, p);
    // Already clamped to the nibble's range, so the mask only drops the sign
    // extension that a negative value carries in int32.
    packed |= (static_cast<uint32_t>(q) & 0xF) << (half * 4);
  }
  output[byte_idx].byte = static_cast<uint8_t>(packed);
}

template <typename QuantT, typename ScaleT, typename OutT>
__global__ void dequantize_linear_kernel(
    const QuantT *__restrict__ input, const ScaleT *__restrict__ scale,
    const QuantT *__restrict__ zero_point, OutT *__restrict__ output,
    int64_t total, int mode, int rank, int axis, int block_size, int64_t inner,
    int64_t axis_dim, QdqDimsND input_shape, QdqDimsND scale_shape) {
  int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total)
    return;

  int64_t p = qdq_param_index(tid, mode, rank, axis, block_size, inner,
                              axis_dim, input_shape, scale_shape);
  // Both loads take a logical index; QuantLoad hides whether the buffer is
  // one element per byte or two.
  int32_t x = QuantLoad<QuantT>::load(input, tid);
  int32_t zp = zero_point ? QuantLoad<QuantT>::load(zero_point, p) : 0;
  float s = FloatIO<ScaleT>::load(scale[p]);

  // Subtract in int32 so nothing is lost before the widening; doing it in the
  // scale's type would round twice for 16-bit storage with an fp16 scale.
  output[tid] = FloatIO<OutT>::store(static_cast<float>(x - zp) * s);
}

// ---- host-side plan ------------------------------------------------------

// Returns 0 to launch, 1 when there is nothing to do (empty tensor), <0 on
// invalid arguments (already reported). The wrap_ layer validates the same
// granularity rules; repeating them here keeps the kernel safe when it is
// driven directly from a test harness, and the extent check below is what
// stops a malformed scale tensor from being indexed out of bounds.
int plan_qdq(const char *name, const void *input, const void *scale,
             const void *output, const int64_t *input_shape, int input_rank,
             const int64_t *scale_shape, int scale_rank, int axis,
             int block_size, QdqLaunchPlan *out) {
  if (!input || !scale || !output || !input_shape || !scale_shape) {
    fprintf(stderr, "[custom_kernels] %s: null required argument\n", name);
    return -1;
  }
  if (input_rank < 0 || input_rank > HIP_QDQ_MAX_RANK || scale_rank < 0 ||
      scale_rank > HIP_QDQ_MAX_RANK) {
    fprintf(stderr,
            "[custom_kernels] %s: rank out of range (max=%d input=%d scale=%d)\n",
            name, HIP_QDQ_MAX_RANK, input_rank, scale_rank);
    return -1;
  }
  if (block_size < 0) {
    fprintf(stderr, "[custom_kernels] %s: negative block_size %d\n", name,
            block_size);
    return -1;
  }

  int64_t total = 1;
  for (int d = 0; d < input_rank; ++d)
    total *= input_shape[d];
  if (total <= 0)
    return 1;

  int64_t scale_total = 1;
  for (int d = 0; d < scale_rank; ++d)
    scale_total *= scale_shape[d];
  if (scale_total <= 0) {
    fprintf(stderr, "[custom_kernels] %s: scale tensor is empty\n", name);
    return -1;
  }

  QdqDimsND input_nd = {}, scale_nd = {};
  for (int d = 0; d < input_rank; ++d)
    input_nd.v[d] = input_shape[d];
  for (int d = 0; d < scale_rank; ++d)
    scale_nd.v[d] = scale_shape[d];

  int mode;
  int64_t inner = 1;
  int64_t axis_dim = 1;
  if (scale_total == 1) {
    // Covers a rank-0 scale as well as the 1-element 1-D spelling.
    mode = kQdqPerTensor;
  } else {
    // Every remaining granularity indexes along `axis`, so it must be a real
    // dimension by now. Negatives are the host wrapper's job to normalize.
    if (axis < 0 || axis >= input_rank) {
      fprintf(stderr,
              "[custom_kernels] %s: axis %d out of range for input_rank %d\n",
              name, axis, input_rank);
      return -1;
    }
    if (block_size > 0) {
      if (scale_rank != input_rank) {
        fprintf(stderr,
                "[custom_kernels] %s: blocked quantization needs scale_rank == "
                "input_rank (got %d vs %d)\n",
                name, scale_rank, input_rank);
        return -1;
      }
      for (int d = 0; d < input_rank; ++d) {
        if (d != axis && scale_shape[d] != input_shape[d]) {
          fprintf(stderr,
                  "[custom_kernels] %s: blocked scale dim %d is %lld, expected "
                  "%lld to match the input\n",
                  name, d, (long long)scale_shape[d],
                  (long long)input_shape[d]);
          return -1;
        }
      }
      // ONNX allows a partial trailing block, so the scale extent only has to
      // cover the input extent. Without this, coord/block_size could index
      // past the scale buffer.
      if (scale_shape[axis] * static_cast<int64_t>(block_size) <
          input_shape[axis]) {
        fprintf(stderr,
                "[custom_kernels] %s: scale extent %lld * block_size %d does "
                "not cover input extent %lld on axis %d\n",
                name, (long long)scale_shape[axis], block_size,
                (long long)input_shape[axis], axis);
        return -1;
      }
      mode = kQdqBlocked;
    } else if (scale_rank == 1) {
      if (scale_shape[0] != input_shape[axis]) {
        fprintf(stderr,
                "[custom_kernels] %s: per-axis scale length %lld does not "
                "match input extent %lld on axis %d\n",
                name, (long long)scale_shape[0], (long long)input_shape[axis],
                axis);
        return -1;
      }
      axis_dim = input_shape[axis];
      for (int d = axis + 1; d < input_rank; ++d)
        inner *= input_shape[d];
      mode = kQdqPerAxis;
    } else {
      fprintf(stderr,
              "[custom_kernels] %s: cannot classify granularity (scale_rank=%d "
              "holds %lld elements, block_size=%d); expected a scalar scale, a "
              "1-D per-axis scale, or block_size > 0 for blocked\n",
              name, scale_rank, (long long)scale_total, block_size);
      return -1;
    }
  }

  out->mode = mode;
  out->rank = input_rank;
  out->axis = axis;
  out->block_size = block_size;
  out->inner = inner;
  out->axis_dim = axis_dim;
  out->total = total;
  out->input_shape = input_nd;
  out->scale_shape = scale_nd;
  return 0;
}

int qdq_grid_size(const char *name, int64_t total, int block, int *grid) {
  int64_t grid_long = (total + block - 1) / block;
  if (grid_long > static_cast<int64_t>(0x7FFFFFFF)) {
    fprintf(stderr,
            "[custom_kernels] %s: total=%lld too large for one launch "
            "(grid_x overflow)\n",
            name, (long long)total);
    return -1;
  }
  *grid = static_cast<int>(grid_long);
  return 0;
}

// Storage width a quantized hip_dtype_t implies, or 0 when it names no
// integer storage type. Used to cross-check the caller's declared bit width.
int qdq_storage_bits(int dtype) {
  switch (dtype) {
  case HIP_DTYPE_INT8:
  case HIP_DTYPE_UINT8:
    return 8;
  case HIP_DTYPE_INT16:
  case HIP_DTYPE_UINT16:
    return 16;
  default:
    return 0;
  }
}

// ---- quantize dispatch ---------------------------------------------------

template <typename InT, typename ScaleT, typename QuantT>
int launch_quantize(hipStream_t stream, const void *input, const void *scale,
                    const void *zero_point, void *output,
                    const QdqLaunchPlan &p) {
  constexpr int kBlockSize = 256;
  int grid = 0;
  if (qdq_grid_size("hip_quantize_linear", p.total, kBlockSize, &grid) != 0)
    return -1;
  hipLaunchKernelGGL((quantize_linear_kernel<InT, ScaleT, QuantT>), dim3(grid),
                     dim3(kBlockSize), 0, stream,
                     static_cast<const InT *>(input),
                     static_cast<const ScaleT *>(scale),
                     static_cast<const QuantT *>(zero_point),
                     static_cast<QuantT *>(output), p.total, p.mode, p.rank,
                     p.axis, p.block_size, p.inner, p.axis_dim, p.input_shape,
                     p.scale_shape);
  return 0;
}

template <typename InT, typename ScaleT, typename QuantT>
int launch_quantize_packed(hipStream_t stream, const void *input,
                           const void *scale, const void *zero_point,
                           void *output, const QdqLaunchPlan &p) {
  constexpr int kBlockSize = 256;
  int grid = 0;
  // One thread per output byte, so half as many as there are elements.
  int64_t bytes = (p.total + 1) / 2;
  if (qdq_grid_size("hip_quantize_linear", bytes, kBlockSize, &grid) != 0)
    return -1;
  hipLaunchKernelGGL((quantize_linear_packed_kernel<InT, ScaleT, QuantT>),
                     dim3(grid), dim3(kBlockSize), 0, stream,
                     static_cast<const InT *>(input),
                     static_cast<const ScaleT *>(scale),
                     static_cast<const QuantT *>(zero_point),
                     static_cast<QuantT *>(output), p.total, p.mode, p.rank,
                     p.axis, p.block_size, p.inner, p.axis_dim, p.input_shape,
                     p.scale_shape);
  return 0;
}

template <typename InT, typename ScaleT>
int dispatch_quantize_storage(hipStream_t stream, const void *input,
                              const void *scale, const void *zero_point,
                              void *output, const QdqLaunchPlan &p,
                              int out_dtype, int out_bits) {
  // A 4-bit target keeps the 8-bit dtype it will be imported as, so out_dtype
  // contributes only the signedness and out_bits picks the packed carrier.
  if (out_bits == 4) {
    switch (out_dtype) {
    case HIP_DTYPE_INT8:
      return launch_quantize_packed<InT, ScaleT, PackedInt4>(
          stream, input, scale, zero_point, output, p);
    case HIP_DTYPE_UINT8:
      return launch_quantize_packed<InT, ScaleT, PackedUint4>(
          stream, input, scale, zero_point, output, p);
    default:
      return kUnsupportedDtype;
    }
  }
  switch (out_dtype) {
  case HIP_DTYPE_INT8:
    return launch_quantize<InT, ScaleT, int8_t>(stream, input, scale,
                                                zero_point, output, p);
  case HIP_DTYPE_UINT8:
    return launch_quantize<InT, ScaleT, uint8_t>(stream, input, scale,
                                                 zero_point, output, p);
  case HIP_DTYPE_INT16:
    return launch_quantize<InT, ScaleT, int16_t>(stream, input, scale,
                                                 zero_point, output, p);
  case HIP_DTYPE_UINT16:
    return launch_quantize<InT, ScaleT, uint16_t>(stream, input, scale,
                                                  zero_point, output, p);
  default:
    return kUnsupportedDtype;
  }
}

template <typename InT>
int dispatch_quantize_scale(hipStream_t stream, const void *input,
                            const void *scale, const void *zero_point,
                            void *output, const QdqLaunchPlan &p,
                            int scale_dtype, int out_dtype, int out_bits) {
  switch (scale_dtype) {
  case HIP_DTYPE_FLOAT32:
    return dispatch_quantize_storage<InT, float>(
        stream, input, scale, zero_point, output, p, out_dtype, out_bits);
  case HIP_DTYPE_FLOAT16:
    return dispatch_quantize_storage<InT, __half>(
        stream, input, scale, zero_point, output, p, out_dtype, out_bits);
  default:
    return kUnsupportedDtype;
  }
}

// ---- dequantize dispatch -------------------------------------------------

template <typename QuantT, typename ScaleT, typename OutT>
int launch_dequantize(hipStream_t stream, const void *input, const void *scale,
                      const void *zero_point, void *output,
                      const QdqLaunchPlan &p) {
  constexpr int kBlockSize = 256;
  int grid = 0;
  if (qdq_grid_size("hip_dequantize_linear", p.total, kBlockSize, &grid) != 0)
    return -1;
  hipLaunchKernelGGL((dequantize_linear_kernel<QuantT, ScaleT, OutT>),
                     dim3(grid), dim3(kBlockSize), 0, stream,
                     static_cast<const QuantT *>(input),
                     static_cast<const ScaleT *>(scale),
                     static_cast<const QuantT *>(zero_point),
                     static_cast<OutT *>(output), p.total, p.mode, p.rank,
                     p.axis, p.block_size, p.inner, p.axis_dim, p.input_shape,
                     p.scale_shape);
  return 0;
}

template <typename QuantT, typename ScaleT>
int dispatch_dequantize_out(hipStream_t stream, const void *input,
                            const void *scale, const void *zero_point,
                            void *output, const QdqLaunchPlan &p,
                            int out_dtype) {
  switch (out_dtype) {
  case HIP_DTYPE_FLOAT32:
    return launch_dequantize<QuantT, ScaleT, float>(stream, input, scale,
                                                    zero_point, output, p);
  case HIP_DTYPE_FLOAT16:
    return launch_dequantize<QuantT, ScaleT, __half>(stream, input, scale,
                                                     zero_point, output, p);
  default:
    return kUnsupportedDtype;
  }
}

template <typename QuantT>
int dispatch_dequantize_scale(hipStream_t stream, const void *input,
                              const void *scale, const void *zero_point,
                              void *output, const QdqLaunchPlan &p,
                              int scale_dtype, int out_dtype) {
  switch (scale_dtype) {
  case HIP_DTYPE_FLOAT32:
    return dispatch_dequantize_out<QuantT, float>(stream, input, scale,
                                                  zero_point, output, p,
                                                  out_dtype);
  case HIP_DTYPE_FLOAT16:
    return dispatch_dequantize_out<QuantT, __half>(stream, input, scale,
                                                   zero_point, output, p,
                                                   out_dtype);
  default:
    return kUnsupportedDtype;
  }
}

} // namespace

extern "C" int
hip_quantize_linear(void *stream, const void *input, const void *scale,
                    const void *zero_point, void *output,
                    const int64_t *input_shape, int input_rank,
                    const int64_t *scale_shape, int scale_rank, int axis,
                    int block_size, int precision, int in_dtype,
                    int scale_dtype, int out_dtype, int out_bits) {
  QdqLaunchPlan plan;
  int rc = plan_qdq("hip_quantize_linear", input, scale, output, input_shape,
                    input_rank, scale_shape, scale_rank, axis, block_size,
                    &plan);
  if (rc != 0)
    return rc < 0 ? -1 : 0;

  // This kernel always divides in fp32, so an explicit TensorProto FLOAT
  // request is honoured exactly and an unset attribute is honoured for an fp32
  // scale (and computed more precisely than required for an fp16 one, matching
  // what ORT's reference does). Any other request would silently get a
  // different precision than it asked for, so reject it instead.
  if (precision != kOnnxPrecisionUnset && precision != kOnnxPrecisionFloat) {
    fprintf(stderr,
            "[custom_kernels] hip_quantize_linear: unsupported precision=%d "
            "(ONNX TensorProto enum); only 0 (unset) and 1 (FLOAT) are "
            "implemented\n",
            precision);
    return -1;
  }

  // out_bits must either restate the width out_dtype already implies or drop
  // to 4, the one case out_dtype cannot express. Anything else means the
  // caller and this kernel disagree on the output stride, which would write
  // silently wrong data rather than fail.
  int dtype_bits = qdq_storage_bits(out_dtype);
  if (out_bits != 4 && out_bits != dtype_bits) {
    fprintf(stderr,
            "[custom_kernels] hip_quantize_linear: out_bits=%d contradicts "
            "out_dtype=%d (%d-bit storage); expected 4 or %d\n",
            out_bits, out_dtype, dtype_bits, dtype_bits);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_quantize_linear: total=%lld, mode=%d, "
      "input_rank=%d, scale_rank=%d, axis=%d, block_size=%d, precision=%d, "
      "in_dtype=%d, scale_dtype=%d, out_dtype=%d, out_bits=%d, has_zp=%d\n",
      (long long)plan.total, plan.mode, input_rank, scale_rank, plan.axis,
      block_size, precision, in_dtype, scale_dtype, out_dtype, out_bits,
      zero_point != nullptr ? 1 : 0);

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();

  int lrc;
  switch (in_dtype) {
  case HIP_DTYPE_FLOAT32:
    lrc = dispatch_quantize_scale<float>(hip_stream, input, scale, zero_point,
                                         output, plan, scale_dtype, out_dtype,
                                         out_bits);
    break;
  case HIP_DTYPE_FLOAT16:
    lrc = dispatch_quantize_scale<__half>(hip_stream, input, scale, zero_point,
                                          output, plan, scale_dtype, out_dtype,
                                          out_bits);
    break;
  default:
    lrc = kUnsupportedDtype;
    break;
  }
  if (lrc == kUnsupportedDtype) {
    fprintf(stderr,
            "[custom_kernels] hip_quantize_linear: unsupported dtype "
            "combination (input=%d scale=%d output=%d bits=%d); expected "
            "float32 or float16 input and scale, and int8/uint8/int16/uint16 "
            "output (int8/uint8 when bits=4)\n",
            in_dtype, scale_dtype, out_dtype, out_bits);
    return -1;
  }
  if (lrc != 0)
    return lrc;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_quantize_linear launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

extern "C" int
hip_dequantize_linear(void *stream, const void *input, const void *scale,
                      const void *zero_point, void *output,
                      const int64_t *input_shape, int input_rank,
                      const int64_t *scale_shape, int scale_rank, int axis,
                      int block_size, int in_dtype, int scale_dtype,
                      int out_dtype, int in_bits) {
  QdqLaunchPlan plan;
  int rc = plan_qdq("hip_dequantize_linear", input, scale, output, input_shape,
                    input_rank, scale_shape, scale_rank, axis, block_size,
                    &plan);
  if (rc != 0)
    return rc < 0 ? -1 : 0;

  // in_bits must either restate the width in_dtype already implies or drop to
  // 4, the one case in_dtype cannot express. Anything else means the caller
  // and this kernel disagree on the input stride, which would read silently
  // wrong data rather than fail.
  int dtype_bits = qdq_storage_bits(in_dtype);
  if (in_bits != 4 && in_bits != dtype_bits) {
    fprintf(stderr,
            "[custom_kernels] hip_dequantize_linear: in_bits=%d contradicts "
            "in_dtype=%d (%d-bit storage); expected 4 or %d\n",
            in_bits, in_dtype, dtype_bits, dtype_bits);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_dequantize_linear: total=%lld, mode=%d, "
      "input_rank=%d, scale_rank=%d, axis=%d, block_size=%d, in_dtype=%d, "
      "in_bits=%d, scale_dtype=%d, out_dtype=%d, has_zp=%d\n",
      (long long)plan.total, plan.mode, input_rank, scale_rank, plan.axis,
      block_size, in_dtype, in_bits, scale_dtype, out_dtype,
      zero_point != nullptr ? 1 : 0);

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);
  (void)hipGetLastError();

  // A 4-bit source keeps the 8-bit dtype it was imported as, so in_dtype
  // contributes only the signedness and in_bits picks the packed carrier.
  int lrc;
  if (in_bits == 4) {
    switch (in_dtype) {
    case HIP_DTYPE_INT8:
      lrc = dispatch_dequantize_scale<PackedInt4>(hip_stream, input, scale,
                                                  zero_point, output, plan,
                                                  scale_dtype, out_dtype);
      break;
    case HIP_DTYPE_UINT8:
      lrc = dispatch_dequantize_scale<PackedUint4>(hip_stream, input, scale,
                                                   zero_point, output, plan,
                                                   scale_dtype, out_dtype);
      break;
    default:
      lrc = kUnsupportedDtype;
      break;
    }
  } else {
    switch (in_dtype) {
    case HIP_DTYPE_INT8:
      lrc = dispatch_dequantize_scale<int8_t>(hip_stream, input, scale,
                                              zero_point, output, plan,
                                              scale_dtype, out_dtype);
      break;
    case HIP_DTYPE_UINT8:
      lrc = dispatch_dequantize_scale<uint8_t>(hip_stream, input, scale,
                                               zero_point, output, plan,
                                               scale_dtype, out_dtype);
      break;
    case HIP_DTYPE_INT16:
      lrc = dispatch_dequantize_scale<int16_t>(hip_stream, input, scale,
                                               zero_point, output, plan,
                                               scale_dtype, out_dtype);
      break;
    case HIP_DTYPE_UINT16:
      lrc = dispatch_dequantize_scale<uint16_t>(hip_stream, input, scale,
                                                zero_point, output, plan,
                                                scale_dtype, out_dtype);
      break;
    default:
      lrc = kUnsupportedDtype;
      break;
    }
  }
  if (lrc == kUnsupportedDtype) {
    fprintf(stderr,
            "[custom_kernels] hip_dequantize_linear: unsupported dtype "
            "combination (input=%d bits=%d scale=%d output=%d); expected "
            "int8/uint8/int16/uint16 input (int8/uint8 when bits=4) and "
            "float32 or float16 scale and output\n",
            in_dtype, in_bits, scale_dtype, out_dtype);
    return -1;
  }
  if (lrc != 0)
    return lrc;

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_dequantize_linear launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
