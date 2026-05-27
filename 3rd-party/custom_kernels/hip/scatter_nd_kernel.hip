/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * ScatterND: produce `output` with the shape of `data` whose values are
 * `data` copied, then `updates` overwritten / reduced into at positions
 * specified by `indices`. INT64 indices only.
 *
 * Output layout:
 *   output.shape  = data.shape
 *   K             = indices.shape[-1]                 (in [1, data_rank])
 *   num_slices    = product(indices.shape[:-1])       = product(updates.shape[:indices_rank-1])
 *   slice_size    = product(data.shape[K:])           = product(updates.shape[indices_rank-1:])
 *   updates.shape = indices.shape[:-1] + data.shape[K:]
 *
 * Per ONNX spec the host wrapper first does an `output := data` D2D
 * copy, then this kernel applies the updates. Threads cover the
 * "flat updates" space:
 *
 *   total_threads = num_slices * slice_size
 *   slice_id      = tid / slice_size
 *   inner         = tid % slice_size
 *   target_off    = sum_d ( indices[slice_id * K + d] * data_stride[d] ) + inner
 *
 * Reduction modes (one templated kernel per mode):
 *   0 = "none" : direct write       — duplicate indices => last-writer-wins
 *                                     (matches ONNX's "undefined" guarantee).
 *   1 = "add" : atomic add          — built-in atomicAdd for f32/i32/u64,
 *                                     16-bit-packed CAS for f16.
 *   2 = "mul" : CAS-emulated mul
 *   3 = "min" : atomicMin (int) or  CAS-emulated for floats
 *   4 = "max" : atomicMax (int) or  CAS-emulated for floats
 *
 * The CAS-based emulators preserve full IEEE-754 semantics including
 * NaN propagation rules consistent with `min(a, NaN) = NaN`. For fp16,
 * we CAS the 32-bit-aligned word containing the target half and rewrite
 * the appropriate half-word; this relies on fp16 tensors being at least
 * 2-byte aligned (always true for ONNX tensors).
 *
 * Bounded to rank <= 8.
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

static constexpr int kScatterNDMaxRank = 8;
static constexpr int kScatterNDMaxK = 8;

enum ScatterNDReduction {
  SCATTER_ND_NONE = 0,
  SCATTER_ND_ADD = 1,
  SCATTER_ND_MUL = 2,
  SCATTER_ND_MIN = 3,
  SCATTER_ND_MAX = 4,
};

struct ScatterNDParams {
  // For each k in [0, K), product(data_shape[k+1:]) -- the stride that
  // index[k] multiplies. The K-th entry (sizes[K-1]) is therefore also
  // slice_size = product(data_shape[K:]).
  int64_t sizes_from_slice_dims[kScatterNDMaxK];
  // data_shape[k] for negative-index normalisation (idx<0 => idx += dim).
  int64_t data_slice_dims[kScatterNDMaxK];
  int64_t slice_size; // product(data_shape[K:])
  int K;              // == indices_shape[-1]
};

// ----------------------------------------------------------------------------
// CAS-emulated atomic operations.
//
// HIP provides native atomicAdd for f32/__half/int/uint64; native atomicMin/Max
// for int/uint64. Everything else is CAS-emulated below. The CAS loops always
// compare the FULL 32-/64-bit word so that no concurrent writer can interleave.

template <typename T> struct CasWord;
template <> struct CasWord<int32_t> {
  using type = uint32_t;
};
template <> struct CasWord<int64_t> {
  using type = unsigned long long;
};
template <> struct CasWord<float> {
  using type = uint32_t;
};
// __half is handled specially via packed 32-bit CAS (see below).

template <typename T> __device__ inline typename CasWord<T>::type bit_cast_to_word(T v) {
  typename CasWord<T>::type w;
  static_assert(sizeof(T) == sizeof(w), "size mismatch");
  // __builtin_memcpy compiles to no-op when the value is already in a
  // register of the right width; std::memcpy is __host__-only in HIP.
  __builtin_memcpy(&w, &v, sizeof(T));
  return w;
}

template <typename T> __device__ inline T bit_cast_from_word(typename CasWord<T>::type w) {
  T v;
  static_assert(sizeof(T) == sizeof(w), "size mismatch");
  __builtin_memcpy(&v, &w, sizeof(T));
  return v;
}

// Element-wise reduction functor invoked from inside the CAS loop.
enum class Op { Mul, Min, Max };

// Integer types — no NaN concerns.
template <typename T> __device__ inline T apply_op(T cur, T upd, Op op) {
  switch (op) {
  case Op::Mul:
    return cur * upd;
  case Op::Min:
    return cur < upd ? cur : upd;
  case Op::Max:
    return cur > upd ? cur : upd;
  }
  return cur;
}

// fp32 — NaN propagates through min/max per IEEE 754-2019 (`min(a, NaN) = NaN`),
// matching ORT's CUDA EP behaviour. fminf/fmaxf would silently swallow NaN.
__device__ inline float apply_op(float cur, float upd, Op op) {
  switch (op) {
  case Op::Mul:
    return cur * upd;
  case Op::Min:
    if (isnan(cur) || isnan(upd))
      return __builtin_nanf("");
    return cur < upd ? cur : upd;
  case Op::Max:
    if (isnan(cur) || isnan(upd))
      return __builtin_nanf("");
    return cur > upd ? cur : upd;
  }
  return cur;
}

__device__ inline __half apply_op(__half cur, __half upd, Op op) {
  float c = __half2float(cur);
  float u = __half2float(upd);
  return __float2half(apply_op(c, u, op));
}

template <typename T>
__device__ inline void cas_atomic_apply(T *addr, T val, Op op) {
  using W = typename CasWord<T>::type;
  W *waddr = reinterpret_cast<W *>(addr);
  W old = *waddr, assumed;
  do {
    assumed = old;
    T cur = bit_cast_from_word<T>(assumed);
    T upd = apply_op<T>(cur, val, op);
    W desired = bit_cast_to_word<T>(upd);
    old = atomicCAS(waddr, assumed, desired);
  } while (assumed != old);
}

// ---------- fp16-specific helpers (packed 32-bit CAS) ----------
// We CAS the 32-bit-aligned word that contains the target half. If the half
// lives in the low 16 bits of the word we rewrite the low half; otherwise
// the high half. Relies on fp16 tensors being at least 2-byte aligned.

__device__ inline __half pack_load_half(uint32_t word, bool is_high) {
  uint16_t bits =
      is_high ? static_cast<uint16_t>((word >> 16) & 0xFFFFu)
              : static_cast<uint16_t>(word & 0xFFFFu);
  return __ushort_as_half(bits);
}

__device__ inline uint32_t pack_store_half(uint32_t word, __half v, bool is_high) {
  uint32_t bits = static_cast<uint32_t>(__half_as_ushort(v));
  if (is_high)
    return (word & 0x0000FFFFu) | (bits << 16);
  else
    return (word & 0xFFFF0000u) | bits;
}

template <Op OP>
__device__ inline void atomic_apply_f16(__half *addr, __half val) {
  uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  bool is_high = (a & 0x3u) == 2u;
  uint32_t *base = reinterpret_cast<uint32_t *>(a & ~uintptr_t(0x3u));
  uint32_t old = *base, assumed;
  do {
    assumed = old;
    __half cur = pack_load_half(assumed, is_high);
    __half upd = apply_op(cur, val, OP);
    uint32_t desired = pack_store_half(assumed, upd, is_high);
    old = atomicCAS(base, assumed, desired);
  } while (assumed != old);
}

__device__ inline void atomic_add_f16(__half *addr, __half val) {
  // Same packed-CAS pattern as atomic_apply_f16 but with addition baked in,
  // since add doesn't go through the Op-enum dispatch (the non-fp16 types
  // use HIP's native atomicAdd intrinsic instead of CAS).
  uintptr_t a = reinterpret_cast<uintptr_t>(addr);
  bool is_high = (a & 0x3u) == 2u;
  uint32_t *base = reinterpret_cast<uint32_t *>(a & ~uintptr_t(0x3u));
  uint32_t old = *base, assumed;
  do {
    assumed = old;
    __half cur = pack_load_half(assumed, is_high);
    float sum = __half2float(cur) + __half2float(val);
    __half upd = __float2half(sum);
    uint32_t desired = pack_store_half(assumed, upd, is_high);
    old = atomicCAS(base, assumed, desired);
  } while (assumed != old);
}

// ---------- The five reduction primitives, dispatched per dtype ----------
// One explicit overload per supported dtype. We could use `if constexpr`
// but hipcc's default C++ standard on Windows is C++14; the explicit
// overloads compile cleanly under both.

__device__ inline void scatter_nd_apply(int32_t *addr, int32_t val,
                                        int reduction_id) {
  switch (reduction_id) {
  case SCATTER_ND_NONE:
    *addr = val;
    return;
  case SCATTER_ND_ADD:
    atomicAdd(addr, val);
    return;
  case SCATTER_ND_MUL:
    cas_atomic_apply<int32_t>(addr, val, Op::Mul);
    return;
  case SCATTER_ND_MIN:
    atomicMin(addr, val);
    return;
  case SCATTER_ND_MAX:
    atomicMax(addr, val);
    return;
  }
}

__device__ inline void scatter_nd_apply(int64_t *addr, int64_t val,
                                        int reduction_id) {
  switch (reduction_id) {
  case SCATTER_ND_NONE:
    *addr = val;
    return;
  case SCATTER_ND_ADD:
    // No native int64 atomicAdd on HIP; the unsigned-long-long overload
    // produces correct two's-complement results regardless of operand sign.
    atomicAdd(reinterpret_cast<unsigned long long *>(addr),
              static_cast<unsigned long long>(val));
    return;
  case SCATTER_ND_MUL:
    cas_atomic_apply<int64_t>(addr, val, Op::Mul);
    return;
  case SCATTER_ND_MIN:
    // Signed-int64 atomicMin would require a custom CAS because casting
    // through unsigned long long flips the ordering for negative values.
    cas_atomic_apply<int64_t>(addr, val, Op::Min);
    return;
  case SCATTER_ND_MAX:
    cas_atomic_apply<int64_t>(addr, val, Op::Max);
    return;
  }
}

__device__ inline void scatter_nd_apply(float *addr, float val,
                                        int reduction_id) {
  switch (reduction_id) {
  case SCATTER_ND_NONE:
    *addr = val;
    return;
  case SCATTER_ND_ADD:
    atomicAdd(addr, val);
    return;
  case SCATTER_ND_MUL:
    cas_atomic_apply<float>(addr, val, Op::Mul);
    return;
  case SCATTER_ND_MIN:
    cas_atomic_apply<float>(addr, val, Op::Min);
    return;
  case SCATTER_ND_MAX:
    cas_atomic_apply<float>(addr, val, Op::Max);
    return;
  }
}

// fp16: packed-CAS for everything except the "none" path (no native
// atomic word width matches `__half`).
__device__ inline void scatter_nd_apply(__half *addr, __half val,
                                        int reduction_id) {
  switch (reduction_id) {
  case SCATTER_ND_NONE:
    *addr = val;
    return;
  case SCATTER_ND_ADD:
    atomic_add_f16(addr, val);
    return;
  case SCATTER_ND_MUL:
    atomic_apply_f16<Op::Mul>(addr, val);
    return;
  case SCATTER_ND_MIN:
    atomic_apply_f16<Op::Min>(addr, val);
    return;
  case SCATTER_ND_MAX:
    atomic_apply_f16<Op::Max>(addr, val);
    return;
  }
}

// ----------------------------------------------------------------------------
// Kernel: one thread per (updates_slice_idx, inner) pair.

template <typename T>
__global__ void hip_scatter_nd_kernel(const int64_t *__restrict__ indices,
                                      const T *__restrict__ updates,
                                      T *__restrict__ output,
                                      ScatterNDParams p,
                                      int64_t total_threads,
                                      int reduction_id) {
  int64_t tid =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid >= total_threads)
    return;

  int64_t slice_id = tid / p.slice_size;
  int64_t inner = tid % p.slice_size;

  // Walk the K-element index tuple for this slice.
  //
  // Out-of-range index policy matches ORT's CUDA EP (`scatter_nd_impl.cu`):
  // throwing from a device kernel is impractical, so OOB indices are
  // CLAMPED into range rather than skipped.
  //   idx >=  dim      -> dim - 1   (clamp to top)
  //   idx <  -dim      -> 0         (clamp to bottom)
  //   -dim <= idx < 0  -> idx + dim (normal negative-index wrap)
  // Note: ORT CPU treats OOB as an error; this divergence is documented
  // upstream as "consistent with other GPU backends".
  const int64_t *slice_indices = indices + slice_id * p.K;
  int64_t base_offset = 0;
  for (int k = 0; k < p.K; ++k) {
    int64_t idx = slice_indices[k];
    int64_t dim = p.data_slice_dims[k];
    if (idx >= 0) {
      if (idx >= dim)
        idx = dim - 1;
    } else {
      if (idx < -dim)
        idx = 0;
      else
        idx += dim;
    }
    base_offset += idx * p.sizes_from_slice_dims[k];
  }

  T val = updates[tid];
  scatter_nd_apply(output + base_offset + inner, val, reduction_id);
}

// ----------------------------------------------------------------------------
// Host launcher.

extern "C" int hip_scatter_nd(void *stream, const void *data,
                              const void *indices, const void *updates,
                              void *output, const int64_t *data_shape_host,
                              int data_rank, const int64_t *indices_shape_host,
                              int indices_rank, int reduction_id,
                              int hip_dtype) {
  if (data_rank <= 0 || indices_rank <= 0) {
    fprintf(stderr,
            "[custom_kernels] hip_scatter_nd: zero-rank input "
            "(data_rank=%d, indices_rank=%d)\n",
            data_rank, indices_rank);
    return -1;
  }
  if (data_rank > kScatterNDMaxRank ||
      indices_rank > kScatterNDMaxRank) {
    fprintf(stderr,
            "[custom_kernels] hip_scatter_nd: rank exceeds %d "
            "(data_rank=%d, indices_rank=%d)\n",
            kScatterNDMaxRank, data_rank, indices_rank);
    return -1;
  }

  int K = static_cast<int>(indices_shape_host[indices_rank - 1]);
  if (K <= 0 || K > kScatterNDMaxK || K > data_rank) {
    fprintf(stderr,
            "[custom_kernels] hip_scatter_nd: K=%d out of range "
            "[1, min(%d, data_rank=%d)]\n",
            K, kScatterNDMaxK, data_rank);
    return -1;
  }
  if (reduction_id < 0 || reduction_id > SCATTER_ND_MAX) {
    fprintf(stderr,
            "[custom_kernels] hip_scatter_nd: invalid reduction_id=%d\n",
            reduction_id);
    return -1;
  }

  ScatterNDParams p{};
  p.K = K;

  // sizes_from_slice_dims[k] = product(data_shape[k+1:])
  // data_slice_dims[k]       = data_shape[k]
  int64_t suffix = 1;
  int64_t total_data_elements = 1;
  for (int d = data_rank - 1; d >= 0; --d) {
    if (d < K) {
      p.sizes_from_slice_dims[d] = suffix;
      p.data_slice_dims[d] = data_shape_host[d];
    }
    suffix *= data_shape_host[d];
    total_data_elements *= data_shape_host[d];
  }
  // slice_size = product(data_shape[K:])
  int64_t slice_size = 1;
  for (int d = K; d < data_rank; ++d)
    slice_size *= data_shape_host[d];
  p.slice_size = slice_size;

  // num_slices = product(indices_shape[:-1])
  int64_t num_slices = 1;
  for (int d = 0; d < indices_rank - 1; ++d)
    num_slices *= indices_shape_host[d];

  int64_t total_threads = num_slices * slice_size;
  if (total_data_elements == 0)
    return 0;

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  // Step 1: copy data -> output. DPS guarantees these are distinct buffers,
  // so a plain D2D async copy is safe.
  int elem_bytes = 0;
  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    elem_bytes = 2;
    break;
  case HIP_DTYPE_FLOAT32:
    elem_bytes = 4;
    break;
  case HIP_DTYPE_INT32:
    elem_bytes = 4;
    break;
  case HIP_DTYPE_INT64:
    elem_bytes = 8;
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_scatter_nd: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  hipError_t err = hipMemcpyAsync(
      output, data, static_cast<size_t>(total_data_elements * elem_bytes),
      hipMemcpyDeviceToDevice, hip_stream);
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_scatter_nd: data->output D2D failed: %s\n",
            hipGetErrorString(err));
    return -1;
  }

  if (total_threads == 0)
    return 0;

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_scatter_nd: dtype=%d, data_rank=%d, "
      "indices_rank=%d, K=%d, slice_size=%lld, num_slices=%lld, "
      "reduction=%d\n",
      hip_dtype, data_rank, indices_rank, K, (long long)slice_size,
      (long long)num_slices, reduction_id);

  constexpr int kBlockSize = 256;
  int grid_size = static_cast<int>(
      (total_threads + kBlockSize - 1) / kBlockSize);

  switch (hip_dtype) {
  case HIP_DTYPE_FLOAT16:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_scatter_nd_kernel<__half>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(indices),
                       static_cast<const __half *>(updates),
                       static_cast<__half *>(output), p, total_threads,
                       reduction_id);
    break;
  case HIP_DTYPE_FLOAT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_scatter_nd_kernel<float>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(indices),
                       static_cast<const float *>(updates),
                       static_cast<float *>(output), p, total_threads,
                       reduction_id);
    break;
  case HIP_DTYPE_INT32:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_scatter_nd_kernel<int32_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(indices),
                       static_cast<const int32_t *>(updates),
                       static_cast<int32_t *>(output), p, total_threads,
                       reduction_id);
    break;
  case HIP_DTYPE_INT64:
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hip_scatter_nd_kernel<int64_t>),
                       dim3(grid_size), dim3(kBlockSize), 0, hip_stream,
                       static_cast<const int64_t *>(indices),
                       static_cast<const int64_t *>(updates),
                       static_cast<int64_t *>(output), p, total_threads,
                       reduction_id);
    break;
  default:
    fprintf(stderr, "[custom_kernels] hip_scatter_nd: unsupported dtype=%d\n",
            hip_dtype);
    return -1;
  }

  err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_scatter_nd launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
