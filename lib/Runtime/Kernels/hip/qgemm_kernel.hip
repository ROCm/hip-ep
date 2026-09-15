/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// FusedOp(A, B, C) = Q(alpha * DQ(A)' @ DQ(B)' + beta * DQ(C))
//
// a = (A - z_a) * s_a
// b = (B - z_b[n]) * s_b[n]
// c = (C - z_c) * s_c
// Y = saturate(round((alpha * a' @ b' + beta * c) / s_y) + z_y)
//
// Neither scale depends on k, so both leave the sum:
//
//   acc[m,n] = sum over k of (A[m,k] - z_a) * (B[k,n] - z_b[n])
//   Y[m,n]   = saturate(round(M_ab * s_b[n] * acc[m,n]
//                             + M_c * (C[m,n] - z_c))
//                       + z_y)
//   M_ab     = alpha * s_a * s_b / s_y    (folded by the caller)
//   M_c      = beta * s_c / s_y           (folded by the caller)
//
// acc is an exact integer, so the only rounding is the multiply by the
// coefficient and the final round -- strictly less error than the unfused path,
// which accumulates K products in float.
//
// A per-output-channel B is what M_ab cannot absorb: s_b and z_b are then
// device arrays indexed by n, and the per-tensor B_scale contributes its 1.0
// identity to M_ab instead. Both cases are one coefficient per column, read
// once in the epilogue.
//
// Zero points as correction terms
// -------------------------------
// A - z_a does not fit back into A's width, so subtracting inside the reduction
// would force every B value through a widening subtract before it can be
// multiplied. Expanding the product instead keeps the inner loop on raw
// operands:
//
//   acc[n] = accRaw[n] - z_b[n]*rowA - z_a*colB[n] + K*z_a*z_b[n]
//   accRaw[n] = sum_k A[k]*B[k,n],  rowA = sum_k A[k],  colB[n] = sum_k B[k,n]
//
// Each correction term is only accumulated when its zero point can be non-zero,
// so the symmetric weights that dominate real QDQ graphs cost nothing extra.
//
// Accumulator width
// -----------------
// |A - z_a| reaches 65535 for a 16-bit activation and |B - z_b| reaches 255
// even for a 4-bit weight whose zero point sits at the far end of the storage
// range, which bounds one term at 1.7e7 -- past int32 after two terms. Every
// accumulator is therefore int64, which nothing here makes expensive.

#include "debug_log.h"
#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

namespace {

//===----------------------------------------------------------------------===//
// Element types and shared readers
//===----------------------------------------------------------------------===//

template <typename T> struct QGemmTraits;

template <> struct QGemmTraits<int8_t> {
  static constexpr bool kSigned = true;
  __device__ static float lowest() { return -128.0f; }
  __device__ static float highest() { return 127.0f; }
};

template <> struct QGemmTraits<uint8_t> {
  static constexpr bool kSigned = false;
  __device__ static float lowest() { return 0.0f; }
  __device__ static float highest() { return 255.0f; }
};

template <> struct QGemmTraits<int16_t> {
  static constexpr bool kSigned = true;
  __device__ static float lowest() { return -32768.0f; }
  __device__ static float highest() { return 32767.0f; }
};

template <> struct QGemmTraits<uint16_t> {
  static constexpr bool kSigned = false;
  __device__ static float lowest() { return 0.0f; }
  __device__ static float highest() { return 65535.0f; }
};

// The HIP compile flags include `-Xclang -fno-cuda-host-device-constexpr`
// (see lib/Runtime/Kernels/cmake/hip_utils.cmake), which makes
// `std::numeric_limits<T>` members host-only, hence the explicit bounds above.
template <typename T> __device__ inline T q_saturate(float v) {
  return static_cast<T>(
      fminf(QGemmTraits<T>::highest(), fmaxf(QGemmTraits<T>::lowest(), v)));
}

// Widen one element of an 8-, 16- or 32-bit quantized buffer.
//
// The dtype is a runtime argument rather than a template parameter because C
// is read once per output: templating it would multiply the instantiation count
// by five to save a branch that costs nothing outside a reduction.
__device__ inline int32_t qgemm_load(const void *__restrict__ buf, int64_t idx,
                                     int dtype) {
  switch (dtype) {
  case HIP_DTYPE_INT8:
    return static_cast<const int8_t *>(buf)[idx];
  case HIP_DTYPE_UINT8:
    return static_cast<const uint8_t *>(buf)[idx];
  case HIP_DTYPE_INT16:
    return static_cast<const int16_t *>(buf)[idx];
  case HIP_DTYPE_UINT16:
    return static_cast<const uint16_t *>(buf)[idx];
  default: // HIP_DTYPE_INT32, the only remaining accepted code
    return static_cast<const int32_t *>(buf)[idx];
  }
}

// Read one LOGICAL element of B or B_zero_points.
//
// `bits` is 4 or 8. At 4, two values share a byte, low nibble first over the
// flattened row-major sequence -- the ONNX INT4/UINT4 convention. `lin` counts
// logical elements, never bytes.
//
// `is_signed` decides how a nibble widens, and the two disagree on every value
// above 7, so it must track B's element type rather than being assumed.
__device__ inline int32_t qgemm_read_quant(const uint8_t *__restrict__ buf,
                                           int64_t lin, int bits,
                                           bool is_signed) {
  if (bits == 4) {
    const uint8_t byte = buf[lin >> 1];
    const uint8_t nibble = (lin & 1) ? (byte >> 4) : (byte & 0xF);
    if (is_signed) {
      // Shifting into the high nibble of an int8 and back propagates bit 3.
      return static_cast<int32_t>(static_cast<int8_t>(nibble << 4)) >> 4;
    }
    return static_cast<int32_t>(nibble);
  }
  if (is_signed)
    return static_cast<int32_t>(static_cast<int8_t>(buf[lin]));
  return static_cast<int32_t>(buf[lin]);
}

__device__ inline void qgemm_store(void *__restrict__ buf, int64_t idx,
                                   int dtype, float v) {
  switch (dtype) {
  case HIP_DTYPE_INT8:
    static_cast<int8_t *>(buf)[idx] = q_saturate<int8_t>(v);
    break;
  case HIP_DTYPE_UINT8:
    static_cast<uint8_t *>(buf)[idx] = q_saturate<uint8_t>(v);
    break;
  case HIP_DTYPE_INT16:
    static_cast<int16_t *>(buf)[idx] = q_saturate<int16_t>(v);
    break;
  default: // HIP_DTYPE_UINT16
    static_cast<uint16_t *>(buf)[idx] = q_saturate<uint16_t>(v);
    break;
  }
}

//===----------------------------------------------------------------------===//
// Reference kernel (M > 1)
//===----------------------------------------------------------------------===//
//
// One thread per output element, no tiling and no reuse: every thread re-reads
// a full row of A and a full column of B from global memory, so the same A
// element is fetched N times and the same B element M times. That is the
// arithmetic spelled out literally, which is what makes it the baseline the
// GEMV path below is checked against.

// threadIdx.x indexes n so that consecutive threads write consecutive Y
// elements; n is the contiguous axis of the output.
constexpr int kBlockN = 16;
constexpr int kBlockM = 16;

__global__ __launch_bounds__(kBlockN *kBlockM) void qgemm_naive_kernel(
    const void *__restrict__ A, const uint8_t *__restrict__ B,
    const void *__restrict__ C, const float *__restrict__ B_scales,
    const uint8_t *__restrict__ B_zero_points, void *__restrict__ Y, int64_t M,
    int64_t N, int64_t K, bool trans_a, bool trans_b, int a_dtype, int b_dtype,
    int c_dtype, int y_dtype, int b_bits, bool b_signed, int64_t c_dim0,
    int64_t c_dim1, float M_ab, float M_c, int32_t a_zp, int32_t b_zp,
    int32_t c_zp, int32_t y_zp) {
  (void)b_dtype; // carried as b_signed, which is what the nibble read needs
  const int64_t n = static_cast<int64_t>(blockIdx.x) * kBlockN + threadIdx.x;
  const int64_t m = static_cast<int64_t>(blockIdx.y) * kBlockM + threadIdx.y;
  if (m >= M || n >= N)
    return;

  // Both halves of a per-channel B index by n alone, so they are read once per
  // thread rather than once per k.
  const int32_t bzp = B_zero_points
                          ? qgemm_read_quant(B_zero_points, n, b_bits, b_signed)
                          : b_zp;
  const float coeff = B_scales ? M_ab * B_scales[n] : M_ab;

  int64_t acc = 0;
  for (int64_t k = 0; k < K; ++k) {
    // trans_a stores A as [K, M] and trans_b stores B as [N, K]; only the
    // stride changes, the logical extents do not.
    const int32_t a =
        qgemm_load(A, trans_a ? k * M + m : m * K + k, a_dtype) - a_zp;
    const int32_t b =
        qgemm_read_quant(B, trans_b ? n * K + k : k * N + n, b_bits, b_signed) -
        bzp;
    acc += static_cast<int64_t>(a) * b;
  }

  float scaled = coeff * static_cast<float>(acc);
  if (C) {
    // Unidirectional broadcast to [M, N]: an extent of 1 repeats along its
    // axis, which is the whole of ONNX Gemm's bias rule once the caller has
    // normalized C to rank 2.
    const int64_t cm = (c_dim0 == 1) ? 0 : m;
    const int64_t cn = (c_dim1 == 1) ? 0 : n;
    scaled += M_c * static_cast<float>(
                        qgemm_load(C, cm * c_dim1 + cn, c_dtype) - c_zp);
  }
  qgemm_store(Y, m * N + n, y_dtype, rintf(scaled) + static_cast<float>(y_zp));
}

//===----------------------------------------------------------------------===//
// GEMV (M == 1, the LLM decode shape)
//===----------------------------------------------------------------------===//
//
// The reference kernel collapses at M == 1: the grid is N/16 blocks of 16 live
// threads, each walking all of K one byte at a time, so a 4096-column decode
// leaves most of the device idle and every A element is re-read per column.
//
// This path fixes both. A thread owns kCols consecutive columns, which turns
// the per-k read of B into whole dwords instead of a single byte, and A is
// staged once per k-tile in LDS rather than re-read per column.
//
// Concurrent k slices are what the shape is short of, and K is the only
// dimension left to spread: the column window cancels out of the throughput,
// since a narrower one yields proportionally more windows. K is therefore split
// twice.
//
//   * kSplits slices run side by side inside one block and merge through LDS
//     atomics. Atomics rather than per-slice buffers because they keep the
//     merge footprint independent of kSplits, which is what lets the slice
//     count grow. kSplits is capped by K / kTile, so kTile is tunable and wants
//     to be small -- it only sizes the staging tile for A.
//   * A narrow N still leaves too few blocks to fill the device even then
//     (N = 32 is one block however the window is chosen), so the grid splits K
//     again across blockIdx.y, accumulating into a scratch buffer that a second
//     pass requantizes. That pass costs two extra launches and a memset, so it
//     is only used when the column windows alone cannot fill the device.
//
// The geometry parameters are templated because their balance is a per-shape
// property; qgemm_gemv_config measures it on the device.
//
// There is no trans_a flag here: at M == 1 both layouts of A are the same
// K-element vector, so the index is unchanged.

// Everything the epilogue needs that does not come out of the reduction. Passed
// by value, so it must stay trivially copyable.
struct QGemmRequant {
  const float *b_scales;
  const uint8_t *b_zero_points;
  const void *C;
  float m_ab;
  float m_c;
  int64_t K;
  int64_t c_dim1;
  int32_t a_zp;
  int32_t b_zp;
  int32_t c_zp;
  int32_t y_zp;
  int32_t b_bits;
  int32_t c_dtype;
  bool b_signed;
};

template <typename TY>
__device__ inline TY qgemm_gemv_epilogue(int64_t acc, int64_t row_a,
                                         int64_t col_b, int64_t n,
                                         const QGemmRequant &r) {
  const int32_t b_zp =
      r.b_zero_points
          ? qgemm_read_quant(r.b_zero_points, n, r.b_bits, r.b_signed)
          : r.b_zp;
  const int64_t corrected = acc - static_cast<int64_t>(b_zp) * row_a -
                            static_cast<int64_t>(r.a_zp) * col_b +
                            r.K * static_cast<int64_t>(r.a_zp) * b_zp;

  const float coeff = r.b_scales ? r.m_ab * r.b_scales[n] : r.m_ab;
  float scaled = coeff * static_cast<float>(corrected);
  if (r.C) {
    // M == 1 collapses the row index, so only C's column extent can broadcast.
    const int64_t cn = (r.c_dim1 == 1) ? 0 : n;
    scaled +=
        r.m_c * static_cast<float>(qgemm_load(r.C, cn, r.c_dtype) - r.c_zp);
  }
  return q_saturate<TY>(rintf(scaled) + static_cast<float>(r.y_zp));
}

template <int kCols> struct QGemmGemvSums {
  int64_t acc[kCols];
  int64_t col_b[kCols];
  int64_t row_a;
};

// Slices carry a uniform tile count so the block's barriers stay aligned: a
// slice whose range ran out just walks empty tiles instead of leaving the loop
// early.
struct QGemmGemvSlice {
  int64_t k_begin;
  int64_t k_end;
  int64_t tiles;
};

template <int kSplits, int kTile>
__device__ inline QGemmGemvSlice qgemm_gemv_slice(int64_t k_begin,
                                                  int64_t k_end, int slice) {
  const int64_t extent = k_end - k_begin;
  int64_t chunk = (extent + kSplits - 1) / kSplits;
  chunk = (chunk + kTile - 1) / kTile * kTile;
  const int64_t begin = k_begin + static_cast<int64_t>(slice) * chunk;
  const int64_t capped = (begin + chunk < k_end) ? begin + chunk : k_end;
  return {begin, capped < begin ? begin : capped, chunk / kTile};
}

__device__ inline void qgemm_atomic_add_i64(int64_t *addr, int64_t v) {
  atomicAdd(reinterpret_cast<unsigned long long *>(addr),
            static_cast<unsigned long long>(v));
}

// One logical B element out of a dword holding 32 / kBits of them.
template <int kBits, typename TB>
__device__ inline int32_t qgemm_gemv_unpack(uint32_t packed, int lane) {
  if constexpr (kBits == 4) {
    const uint32_t nibble = (packed >> (lane * 4)) & 0xFu;
    if constexpr (QGemmTraits<TB>::kSigned)
      return static_cast<int32_t>(static_cast<int8_t>(nibble << 4)) >> 4;
    else
      return static_cast<int32_t>(nibble);
  } else {
    const uint32_t raw = (packed >> (lane * 8)) & 0xFFu;
    if constexpr (QGemmTraits<TB>::kSigned)
      return static_cast<int32_t>(static_cast<int8_t>(raw));
    else
      return static_cast<int32_t>(raw);
  }
}

// kCols logical B values. Held as dwords so the staging loads stay plain
// aligned dword reads that the vectorizer can fold into dwordx2 / dwordx4.
template <int kWords> struct QGemmGemvPack {
  uint32_t w[kWords];
};

template <int kBits, typename TB, int kCols>
__device__ inline void
qgemm_gemv_step(int32_t a_val, const QGemmGemvPack<kCols * kBits / 32> &packed,
                bool need_row_a, bool need_col_b, QGemmGemvSums<kCols> &sums) {
  constexpr int kPerWord = 32 / kBits;
  if (need_row_a)
    sums.row_a += a_val;
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    const int32_t b_val =
        qgemm_gemv_unpack<kBits, TB>(packed.w[c / kPerWord], c % kPerWord);
    sums.acc[c] += static_cast<int64_t>(a_val) * b_val;
    if (need_col_b)
      sums.col_b[c] += b_val;
  }
}

template <int kTile>
__device__ inline int qgemm_gemv_tile(int64_t k_end, int64_t k0) {
  if (k0 >= k_end)
    return 0;
  const int64_t remaining = k_end - k0;
  return static_cast<int>(remaining < kTile ? remaining : kTile);
}

template <int kBlock, typename TA>
__device__ inline void qgemm_gemv_stage_a(const TA *__restrict__ A, int64_t k0,
                                          int tile, int32_t *a_lds) {
  for (int t = threadIdx.x; t < tile; t += kBlock)
    a_lds[t] = static_cast<int32_t>(A[k0 + t]);
}

// B stored [K, N]: one k step hands a thread its whole column group, so the
// wave reads kBlock * kCols contiguous bytes per step.
template <int kBits, int kBlock, int kCols, int kUnroll, int kTile,
          typename TA, typename TB>
__device__ void qgemm_gemv_vector_kn(const TA *__restrict__ A,
                                     const uint8_t *__restrict__ B, int64_t N,
                                     const QGemmGemvSlice &slice,
                                     int64_t n_base, bool need_row_a,
                                     bool need_col_b, int32_t *a_lds,
                                     QGemmGemvSums<kCols> &sums) {
  constexpr int kWords = kCols * kBits / 32;
  // Unreachable: the caller's vector_kn_ok requires kCols * b_bits >= 32. The
  // guard exists so a narrow-column config still compiles against kBits == 4.
  if constexpr (kWords >= 1) {
    const int64_t row_bytes = N * kBits / 8;
    const uint8_t *b_col = B + n_base * kBits / 8;

    for (int64_t ti = 0; ti < slice.tiles; ++ti) {
      const int64_t k0 = slice.k_begin + ti * kTile;
      const int tile = qgemm_gemv_tile<kTile>(slice.k_end, k0);
      qgemm_gemv_stage_a<kBlock>(A, k0, tile, a_lds);
      __syncthreads();

      const uint8_t *b_row = b_col + k0 * row_bytes;
      int t = 0;
      for (; t + kUnroll <= tile; t += kUnroll) {
        QGemmGemvPack<kWords> staged[kUnroll];
#pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
          const uint32_t *row = reinterpret_cast<const uint32_t *>(b_row);
#pragma unroll
          for (int w = 0; w < kWords; ++w)
            staged[u].w[w] = row[w];
          b_row += row_bytes;
        }
#pragma unroll
        for (int u = 0; u < kUnroll; ++u)
          qgemm_gemv_step<kBits, TB, kCols>(a_lds[t + u], staged[u], need_row_a,
                                            need_col_b, sums);
      }
      for (; t < tile; ++t) {
        QGemmGemvPack<kWords> packed;
        const uint32_t *row = reinterpret_cast<const uint32_t *>(b_row);
#pragma unroll
        for (int w = 0; w < kWords; ++w)
          packed.w[w] = row[w];
        b_row += row_bytes;
        qgemm_gemv_step<kBits, TB, kCols>(a_lds[t], packed, need_row_a,
                                          need_col_b, sums);
      }

      __syncthreads();
    }
  }
}

// B stored [N, K], which is what ONNX Gemm's transB = 1 produces and the layout
// every fully-connected weight arrives in. Here k is the contiguous axis, so a
// dword carries 32 / kBits steps of ONE column instead of one step of several.
// The wave's lanes then sit K bytes apart, but each walks its own cache line to
// the end before moving on, so the line is fully consumed either way.
//
// kUnroll has no role in this layout: successive iterations read the next
// dwords of the same lines, which already keeps several requests in flight, and
// staging them would only cost registers.
template <int kBits, int kBlock, int kCols, int kTile, typename TA,
          typename TB>
__device__ void qgemm_gemv_vector_nk(const TA *__restrict__ A,
                                     const uint8_t *__restrict__ B, int64_t K,
                                     const QGemmGemvSlice &slice,
                                     int64_t n_base, bool need_row_a,
                                     bool need_col_b, int32_t *a_lds,
                                     QGemmGemvSums<kCols> &sums) {
  constexpr int kStep = 32 / kBits;
  const int64_t col_bytes = K * kBits / 8;
  const uint8_t *b_base = B + n_base * col_bytes;

  for (int64_t ti = 0; ti < slice.tiles; ++ti) {
    const int64_t k0 = slice.k_begin + ti * kTile;
    const int tile = qgemm_gemv_tile<kTile>(slice.k_end, k0);
    qgemm_gemv_stage_a<kBlock>(A, k0, tile, a_lds);
    __syncthreads();

    const uint8_t *b_tile = b_base + k0 * kBits / 8;
    int t = 0;
    for (; t + kStep <= tile; t += kStep) {
      uint32_t packed[kCols];
#pragma unroll
      for (int c = 0; c < kCols; ++c)
        packed[c] = *reinterpret_cast<const uint32_t *>(
            b_tile + c * col_bytes + t * kBits / 8);
#pragma unroll
      for (int j = 0; j < kStep; ++j) {
        const int32_t a_val = a_lds[t + j];
        if (need_row_a)
          sums.row_a += a_val;
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
          const int32_t b_val = qgemm_gemv_unpack<kBits, TB>(packed[c], j);
          sums.acc[c] += static_cast<int64_t>(a_val) * b_val;
          if (need_col_b)
            sums.col_b[c] += b_val;
        }
      }
    }
    // A slice clipped by K can leave fewer than kStep steps in its last tile.
    for (; t < tile; ++t) {
      const int32_t a_val = a_lds[t];
      if (need_row_a)
        sums.row_a += a_val;
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        const int32_t b_val = qgemm_read_quant(
            B, (n_base + c) * K + k0 + t, kBits, QGemmTraits<TB>::kSigned);
        sums.acc[c] += static_cast<int64_t>(a_val) * b_val;
        if (need_col_b)
          sums.col_b[c] += b_val;
      }
    }

    __syncthreads();
  }
}

// Fallback for what the dword loads cannot serve: an unaligned or indivisible
// N or K, and the last block of a window that runs off the end of N.
template <bool kTransB, int kBlock, int kCols, int kTile, typename TA,
          typename TB>
__device__ void qgemm_gemv_scalar(const TA *__restrict__ A,
                                  const uint8_t *__restrict__ B, int64_t N,
                                  int64_t K, const QGemmGemvSlice &slice,
                                  int64_t n_base, int b_bits, bool need_row_a,
                                  bool need_col_b, int32_t *a_lds,
                                  QGemmGemvSums<kCols> &sums) {
  for (int64_t ti = 0; ti < slice.tiles; ++ti) {
    const int64_t k0 = slice.k_begin + ti * kTile;
    const int tile = qgemm_gemv_tile<kTile>(slice.k_end, k0);
    qgemm_gemv_stage_a<kBlock>(A, k0, tile, a_lds);
    __syncthreads();

    for (int t = 0; t < tile; ++t) {
      const int32_t a_val = a_lds[t];
      const int64_t k = k0 + t;
      if (need_row_a)
        sums.row_a += a_val;
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        const int64_t n = n_base + c;
        if (n >= N)
          continue;
        const int64_t index = kTransB ? n * K + k : k * N + n;
        const int32_t b_val =
            qgemm_read_quant(B, index, b_bits, QGemmTraits<TB>::kSigned);
        sums.acc[c] += static_cast<int64_t>(a_val) * b_val;
        if (need_col_b)
          sums.col_b[c] += b_val;
      }
    }

    __syncthreads();
  }
}

// `partial` is null for the single-pass form, where this block owns all of K
// for its columns and writes Y itself. Otherwise blockIdx.y selects a slice of
// K and the totals go to scratch for qgemm_gemv_reduce_kernel to requantize.
template <typename TA, typename TB, typename TY, int kBlock, int kCols,
          int kUnroll, int kSplits, int kTile>
__global__ __launch_bounds__(kBlock *kSplits) void qgemm_gemv_kernel(
    const TA *__restrict__ A, const uint8_t *__restrict__ B,
    TY *__restrict__ Y, int64_t *__restrict__ partial, int64_t N, int64_t K,
    int64_t grid_chunk, QGemmRequant requant, int b_bits, bool need_row_a,
    bool need_col_b, bool trans_b) {
  constexpr int kWindow = kBlock * kCols;
  static_assert(kCols % 4 == 0, "the vector path stages whole dwords");
  static_assert(kUnroll <= kTile, "the unrolled body must fit inside a tile");
  static_assert(kBlock * kSplits <= 1024, "block exceeds the HIP thread limit");
  static_assert(kSplits * kTile * sizeof(int32_t) +
                        2 * kWindow * sizeof(int64_t) + sizeof(int64_t) <=
                    64 * 1024,
                "geometry exceeds the per-block LDS budget");

  __shared__ int32_t a_lds[kSplits][kTile];
  __shared__ int64_t red_acc[kWindow];
  __shared__ int64_t red_col_b[kWindow];
  __shared__ int64_t red_row_a;

  const int lane = threadIdx.x;
  const int slice_id = threadIdx.y;
  const int tid = slice_id * kBlock + lane;

  for (int i = tid; i < kWindow; i += kBlock * kSplits) {
    red_acc[i] = 0;
    red_col_b[i] = 0;
  }
  if (tid == 0)
    red_row_a = 0;

  const int64_t block_base = static_cast<int64_t>(blockIdx.x) * kWindow;
  const int64_t n_base = block_base + static_cast<int64_t>(lane) * kCols;

  const int64_t grid_begin = static_cast<int64_t>(blockIdx.y) * grid_chunk;
  const int64_t grid_end =
      (grid_begin + grid_chunk < K) ? grid_begin + grid_chunk : K;
  const QGemmGemvSlice slice =
      qgemm_gemv_slice<kSplits, kTile>(grid_begin, grid_end, slice_id);
  QGemmGemvSums<kCols> sums = {};

  // Block-uniform by construction: the loops below carry __syncthreads(), so a
  // per-thread choice here would hang the block. Every dword load needs its
  // lane in range, its row length to keep the next row aligned, and the base
  // pointer aligned, or it would straddle the end of a row.
  const int64_t align = 32 / b_bits; // logical elements per dword
  const bool base_ok = (reinterpret_cast<uintptr_t>(B) % sizeof(uint32_t)) == 0;
  const bool window_ok = block_base + kWindow <= N;
  const bool vector_kn_ok = !trans_b && base_ok && window_ok &&
                            kCols * b_bits >= 32 && (N % align) == 0 &&
                            (kCols % align) == 0;
  const bool vector_nk_ok =
      trans_b && base_ok && window_ok && (K % align) == 0 && (kTile % align) == 0;

  // Also publishes the zeroed merge buffers to the whole block.
  __syncthreads();

  if (vector_kn_ok) {
    if (b_bits == 8)
      qgemm_gemv_vector_kn<8, kBlock, kCols, kUnroll, kTile, TA, TB>(
          A, B, N, slice, n_base, need_row_a, need_col_b, a_lds[slice_id],
          sums);
    else
      qgemm_gemv_vector_kn<4, kBlock, kCols, kUnroll, kTile, TA, TB>(
          A, B, N, slice, n_base, need_row_a, need_col_b, a_lds[slice_id],
          sums);
  } else if (vector_nk_ok) {
    if (b_bits == 8)
      qgemm_gemv_vector_nk<8, kBlock, kCols, kTile, TA, TB>(
          A, B, K, slice, n_base, need_row_a, need_col_b, a_lds[slice_id],
          sums);
    else
      qgemm_gemv_vector_nk<4, kBlock, kCols, kTile, TA, TB>(
          A, B, K, slice, n_base, need_row_a, need_col_b, a_lds[slice_id],
          sums);
  } else if (trans_b) {
    qgemm_gemv_scalar<true, kBlock, kCols, kTile, TA, TB>(
        A, B, N, K, slice, n_base, b_bits, need_row_a, need_col_b,
        a_lds[slice_id], sums);
  } else {
    qgemm_gemv_scalar<false, kBlock, kCols, kTile, TA, TB>(
        A, B, N, K, slice, n_base, b_bits, need_row_a, need_col_b,
        a_lds[slice_id], sums);
  }

  // Column-major so the lanes of a wave land on consecutive int64 slots; the
  // only contention is between slices, which the LDS unit serializes anyway.
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    const int slot = c * kBlock + lane;
    qgemm_atomic_add_i64(&red_acc[slot], sums.acc[c]);
    qgemm_atomic_add_i64(&red_col_b[slot], sums.col_b[c]);
  }
  // rowA does not depend on n, so every lane of a slice holds the same value.
  if (lane == 0)
    qgemm_atomic_add_i64(&red_row_a, sums.row_a);
  __syncthreads();

  if (slice_id != 0)
    return;

  const int64_t row_a = red_row_a;
  if (!partial) {
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      const int64_t n = n_base + c;
      if (n >= N)
        continue;
      const int slot = c * kBlock + lane;
      Y[n] = qgemm_gemv_epilogue<TY>(red_acc[slot], row_a, red_col_b[slot], n,
                                     requant);
    }
    return;
  }

#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    const int64_t n = n_base + c;
    if (n >= N)
      continue;
    const int slot = c * kBlock + lane;
    qgemm_atomic_add_i64(&partial[n], red_acc[slot]);
    qgemm_atomic_add_i64(&partial[N + n], red_col_b[slot]);
  }
  // Every column window of a k slice recomputed the same rowA, so exactly one
  // of them may contribute it.
  if (lane == 0 && blockIdx.x == 0)
    qgemm_atomic_add_i64(&partial[2 * N], row_a);
}

constexpr int kGemvReduceBlock = 256;

template <typename TY>
__global__ __launch_bounds__(kGemvReduceBlock) void qgemm_gemv_reduce_kernel(
    const int64_t *__restrict__ partial, TY *__restrict__ Y, int64_t N,
    QGemmRequant requant) {
  const int64_t n = static_cast<int64_t>(blockIdx.x) * kGemvReduceBlock +
                    threadIdx.x;
  if (n >= N)
    return;
  Y[n] = qgemm_gemv_epilogue<TY>(partial[n], partial[2 * N], partial[N + n], n,
                                 requant);
}

//===----------------------------------------------------------------------===//
// Host launch and dtype dispatch
//===----------------------------------------------------------------------===//

// Bundled so the dispatch levels below stay readable: A has 4 element types, B
// has 2 and Y has 4, and threading two dozen scalars through 32 switch arms by
// hand buries the one thing each arm actually says.
struct QGemmArgs {
  hipStream_t stream;
  const void *A;
  const uint8_t *B;
  const void *C;
  const float *b_scales;
  const uint8_t *b_zero_points;
  void *Y;
  int64_t M;
  int64_t N;
  int64_t K;
  bool trans_a;
  bool trans_b;
  int a_dtype;
  int b_dtype;
  int c_dtype;
  int y_dtype;
  int b_bits;
  bool b_signed;
  int64_t c_dim0;
  int64_t c_dim1;
  float m_ab;
  float m_c;
  int32_t a_zp;
  int32_t b_zp;
  int32_t c_zp;
  int32_t y_zp;
  void *workspace;
  size_t workspace_bytes;
};

//===----------------------------------------------------------------------===//
// GEMV geometry autotune
//===----------------------------------------------------------------------===//

// Entries must stay in lockstep with the switch in qgemm_gemv_launch_config;
// a config's id is its index here.
//
// Ordered by descending splits / tile ratio, because concurrent k slices are
// what this shape is short of and splits is capped by K / tile. A small block
// raises that cap further (splits is capped by 1024 / block too), but never
// below 32, or a wave would run with idle lanes.
//
// cols reaches 8 only where it has to: a 4-bit B needs 8 logical values to fill
// a dword, so the [K, N] vector path is unavailable to the cols == 4 entries.
// Nothing wider is in the table: a column costs its thread two int64
// accumulators, so a 16-wide tile spends 64 registers before it loads anything.
//
// The low-splits entries at the end are not only launch fallbacks: they leave
// the largest grid-level k split available, which is what a narrow N needs.
struct QGemmGemvConfig {
  int block;
  int cols;
  int unroll;
  int splits;
  int tile;
};

constexpr QGemmGemvConfig kQGemmGemvConfigs[] = {
    {32, 4, 8, 32, 64},  {32, 8, 8, 32, 64}, {32, 4, 8, 32, 32},
    {32, 8, 8, 32, 32},  {64, 4, 8, 16, 64}, {64, 8, 8, 16, 64},
    {32, 4, 16, 32, 64}, {64, 4, 8, 16, 128}, {128, 4, 8, 8, 64},
    {64, 8, 8, 8, 64},   {64, 4, 8, 4, 64},  {64, 4, 8, 1, 256},
};
constexpr int kNumQGemmGemvConfigs =
    static_cast<int>(sizeof(kQGemmGemvConfigs) / sizeof(kQGemmGemvConfigs[0]));

// The answer whenever autotune is off or no config in the table would launch.
constexpr int kQGemmGemvDefaultConfig = 0;
// Narrowest entry: one k slice, 64 threads, the smallest LDS footprint in the
// table. The launch fallback needs a geometry no device can refuse.
constexpr int kQGemmGemvSafeConfig = 11;

// Two blocks per CU is the point past which extra grid-level k slices stop
// buying occupancy and start paying for the scratch pass.
constexpr int kGemvTargetBlocksPerCU = 2;
constexpr int kGemvMaxGridSplits = 64;

int qgemm_device_cu_count() {
  static const int count = [] {
    int device = 0;
    if (hipGetDevice(&device) != hipSuccess)
      return 1;
    hipDeviceProp_t props;
    if (hipGetDeviceProperties(&props, device) != hipSuccess)
      return 1;
    return props.multiProcessorCount > 0 ? props.multiProcessorCount : 1;
  }();
  return count;
}

size_t qgemm_gemv_scratch_bytes(int64_t N) {
  return static_cast<size_t>(2 * N + 1) * sizeof(int64_t);
}

// How many k slices to hand the grid. One means the single-pass form, which
// needs no scratch and no second launch.
template <int kWindow, int kSplits, int kTile>
int qgemm_gemv_grid_splits(const QGemmArgs &g) {
  const int64_t windows = (g.N + kWindow - 1) / kWindow;
  const int64_t target =
      static_cast<int64_t>(qgemm_device_cu_count()) * kGemvTargetBlocksPerCU;
  if (windows >= target)
    return 1;
  if (!g.workspace || g.workspace_bytes < qgemm_gemv_scratch_bytes(g.N))
    return 1;

  // A grid slice narrower than the in-block splits can consume would leave
  // slices walking empty tiles, which costs barriers and buys nothing.
  const int64_t cap = g.K / (static_cast<int64_t>(kSplits) * kTile);
  if (cap < 2)
    return 1;

  int64_t want = (target + windows - 1) / windows;
  if (want > cap)
    want = cap;
  if (want > kGemvMaxGridSplits)
    want = kGemvMaxGridSplits;
  return want > 1 ? static_cast<int>(want) : 1;
}

template <typename TA, typename TB, typename TY, int kBlock, int kCols,
          int kUnroll, int kSplits, int kTile>
int qgemm_gemv_launch(const QGemmArgs &g, const QGemmRequant &requant,
                      bool need_row_a, bool need_col_b) {
  constexpr int kWindow = kBlock * kCols;
  const int grid_splits = qgemm_gemv_grid_splits<kWindow, kSplits, kTile>(g);
  auto *partial =
      grid_splits > 1 ? static_cast<int64_t *>(g.workspace) : nullptr;

  (void)hipGetLastError();
  if (partial) {
    // Half-zeroed accumulators would corrupt the result, so a memset that fails
    // drops the whole op rather than the extra parallelism silently.
    const hipError_t cleared = hipMemsetAsync(
        partial, 0, qgemm_gemv_scratch_bytes(g.N), g.stream);
    if (cleared != hipSuccess)
      return static_cast<int>(cleared);
  }

  // Rounded up to kTile so a slice boundary never lands mid-tile.
  int64_t grid_chunk = g.K;
  if (grid_splits > 1) {
    grid_chunk = (g.K + grid_splits - 1) / grid_splits;
    grid_chunk = (grid_chunk + kTile - 1) / kTile * kTile;
  }

  dim3 grid(static_cast<unsigned>((g.N + kWindow - 1) / kWindow),
            static_cast<unsigned>(grid_splits));
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qgemm_gemv_kernel<TA, TB, TY, kBlock, kCols, kUnroll,
                                        kSplits, kTile>),
      grid, dim3(kBlock, kSplits), 0, g.stream, static_cast<const TA *>(g.A),
      g.B, static_cast<TY *>(g.Y), partial, g.N, g.K, grid_chunk, requant,
      g.b_bits, need_row_a, need_col_b, g.trans_b);
  const hipError_t launched = hipGetLastError();
  if (launched != hipSuccess || !partial)
    return static_cast<int>(launched);

  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(qgemm_gemv_reduce_kernel<TY>),
      dim3(static_cast<unsigned>((g.N + kGemvReduceBlock - 1) /
                                 kGemvReduceBlock)),
      dim3(kGemvReduceBlock), 0, g.stream, partial, static_cast<TY *>(g.Y), g.N,
      requant);
  return static_cast<int>(hipGetLastError());
}

template <typename TA, typename TB, typename TY>
int qgemm_gemv_launch_config(int config_id, const QGemmArgs &g,
                             const QGemmRequant &requant, bool need_row_a,
                             bool need_col_b) {
#define QGEMM_GEMV_CASE(ID, BLOCK, COLS, UNROLL, SPLITS, TILE)                 \
  case ID:                                                                     \
    return qgemm_gemv_launch<TA, TB, TY, BLOCK, COLS, UNROLL, SPLITS, TILE>(   \
        g, requant, need_row_a, need_col_b)

  switch (config_id) {
    QGEMM_GEMV_CASE(0, 32, 4, 8, 32, 64);
    QGEMM_GEMV_CASE(1, 32, 8, 8, 32, 64);
    QGEMM_GEMV_CASE(2, 32, 4, 8, 32, 32);
    QGEMM_GEMV_CASE(3, 32, 8, 8, 32, 32);
    QGEMM_GEMV_CASE(4, 64, 4, 8, 16, 64);
    QGEMM_GEMV_CASE(5, 64, 8, 8, 16, 64);
    QGEMM_GEMV_CASE(6, 32, 4, 16, 32, 64);
    QGEMM_GEMV_CASE(7, 64, 4, 8, 16, 128);
    QGEMM_GEMV_CASE(8, 128, 4, 8, 8, 64);
    QGEMM_GEMV_CASE(9, 64, 8, 8, 8, 64);
    QGEMM_GEMV_CASE(10, 64, 4, 8, 4, 64);
    QGEMM_GEMV_CASE(11, 64, 4, 8, 1, 256);
  default:
    return qgemm_gemv_launch<TA, TB, TY, 32, 4, 8, 32, 64>(
        g, requant, need_row_a, need_col_b);
  }
#undef QGEMM_GEMV_CASE
}

// Read once and latched: the tuner sits on the decode dispatch path, where a
// getenv per call would serialize every token through the CRT environment lock.
bool qgemm_gemv_autotune_enabled() {
  static const bool enabled = [] {
    std::string raw;
#ifdef _WIN32
    char *value = nullptr;
    size_t len = 0;
    if (_dupenv_s(&value, &len, "HIPDNN_QGEMM_GEMV_AUTOTUNE") == 0 && value) {
      raw = value;
      free(value);
    }
#else
    if (const char *value = getenv("HIPDNN_QGEMM_GEMV_AUTOTUNE"))
      raw = value;
#endif
    return !(raw == "0" || raw == "off" || raw == "OFF");
  }();
  return enabled;
}

// Distinct per element type: the candidates differ in either width or
// signedness.
template <typename T> constexpr int qgemm_dtype_tag() {
  return static_cast<int>(sizeof(T)) * 2 + (QGemmTraits<T>::kSigned ? 1 : 0);
}

// Everything that moves the per-config ranking: the shape, the types the kernel
// was instantiated for, and the layout and correction terms that decide how
// much work one k step carries.
std::string qgemm_gemv_cache_key(const QGemmArgs &g, int dtypes,
                                 bool need_row_a, bool need_col_b) {
  std::string key = std::to_string(g.N);
  key += '_';
  key += std::to_string(g.K);
  key += '_';
  key += std::to_string(dtypes);
  key += '_';
  key += std::to_string(g.b_bits);
  key += g.trans_b ? 't' : 'n';
  key += g.b_scales ? 'p' : '.';
  key += need_row_a ? 'r' : '.';
  key += need_col_b ? 'c' : '.';
  return key;
}

// Sweeping over the live output buffer is safe because every config writes the
// same Y for every n, and the caller relaunches the winner afterwards.
template <typename TA, typename TB, typename TY>
int qgemm_gemv_tune(const QGemmArgs &g, const QGemmRequant &requant,
                    bool need_row_a, bool need_col_b) {
  constexpr int kIters = 5;

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (hipEventCreate(&start) != hipSuccess)
    return kQGemmGemvDefaultConfig;
  if (hipEventCreate(&stop) != hipSuccess) {
    (void)hipEventDestroy(start);
    return kQGemmGemvDefaultConfig;
  }

  int best = kQGemmGemvDefaultConfig;
  float best_ms = 0.0f;
  bool have_best = false;

  for (int cid = 0; cid < kNumQGemmGemvConfigs; ++cid) {
    // The warmup doubles as the probe: a config this device refuses drops out
    // instead of contributing a bogus timing.
    bool ok = qgemm_gemv_launch_config<TA, TB, TY>(cid, g, requant, need_row_a,
                                                   need_col_b) == hipSuccess;
    if (!ok || hipEventRecord(start, g.stream) != hipSuccess)
      continue;

    for (int i = 0; i < kIters && ok; ++i)
      ok = qgemm_gemv_launch_config<TA, TB, TY>(cid, g, requant, need_row_a,
                                                need_col_b) == hipSuccess;
    if (!ok || hipEventRecord(stop, g.stream) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess)
      continue;

    float ms = 0.0f;
    if (hipEventElapsedTime(&ms, start, stop) != hipSuccess)
      continue;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] qgemm gemv config[%2d] block=%3d cols=%d unroll=%2d "
        "splits=%2d tile=%3d : %.4f ms\n",
        cid, kQGemmGemvConfigs[cid].block, kQGemmGemvConfigs[cid].cols,
        kQGemmGemvConfigs[cid].unroll, kQGemmGemvConfigs[cid].splits,
        kQGemmGemvConfigs[cid].tile, ms / kIters);

    if (!have_best || ms < best_ms) {
      best_ms = ms;
      best = cid;
      have_best = true;
    }
  }

  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  return best;
}

std::unordered_map<std::string, int> g_qgemm_gemv_cache;
std::mutex g_qgemm_gemv_cache_mutex;

template <typename TA, typename TB, typename TY>
int qgemm_gemv_config(const QGemmArgs &g, const QGemmRequant &requant,
                      bool need_row_a, bool need_col_b) {
  if (!qgemm_gemv_autotune_enabled())
    return kQGemmGemvDefaultConfig;

  constexpr int kDtypes = qgemm_dtype_tag<TA>() * 100 +
                          qgemm_dtype_tag<TB>() * 10 + qgemm_dtype_tag<TY>();
  const std::string key =
      qgemm_gemv_cache_key(g, kDtypes, need_row_a, need_col_b);

  std::lock_guard<std::mutex> lock(g_qgemm_gemv_cache_mutex);
  const auto it = g_qgemm_gemv_cache.find(key);
  if (it != g_qgemm_gemv_cache.end())
    return it->second;

  const int best =
      qgemm_gemv_tune<TA, TB, TY>(g, requant, need_row_a, need_col_b);
  g_qgemm_gemv_cache[key] = best;

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] qgemm gemv tune \"%s\" -> config[%d]\n", key.c_str(),
      best);
  return best;
}

template <typename TA, typename TB, typename TY>
int qgemm_launch_gemv(const QGemmArgs &g) {
  const QGemmRequant requant{g.b_scales,
                             g.b_zero_points,
                             g.C,
                             g.m_ab,
                             g.m_c,
                             g.K,
                             g.c_dim1,
                             g.a_zp,
                             g.b_zp,
                             g.c_zp,
                             g.y_zp,
                             g.b_bits,
                             g.c_dtype,
                             g.b_signed};
  // A per-channel zero point is read in the epilogue, so its correction term
  // must be carried whether or not any single entry happens to be zero.
  const bool need_row_a = g.b_zero_points != nullptr || g.b_zp != 0;
  const bool need_col_b = g.a_zp != 0;

  const int config =
      qgemm_gemv_config<TA, TB, TY>(g, requant, need_row_a, need_col_b);
  const int err = qgemm_gemv_launch_config<TA, TB, TY>(config, g, requant,
                                                       need_row_a, need_col_b);
  if (err == hipSuccess || config == kQGemmGemvSafeConfig)
    return err;
  // The table's widest geometries ask for 1024 threads, which a device may
  // refuse for resources. Retry narrow rather than failing the op, and clear
  // the sticky error or the clean retry would still report it.
  (void)hipGetLastError();
  return qgemm_gemv_launch_config<TA, TB, TY>(kQGemmGemvSafeConfig, g, requant,
                                              need_row_a, need_col_b);
}

template <typename TA, typename TB>
int qgemm_dispatch_y(const QGemmArgs &g) {
  switch (g.y_dtype) {
  case HIP_DTYPE_INT8:
    return qgemm_launch_gemv<TA, TB, int8_t>(g);
  case HIP_DTYPE_UINT8:
    return qgemm_launch_gemv<TA, TB, uint8_t>(g);
  case HIP_DTYPE_INT16:
    return qgemm_launch_gemv<TA, TB, int16_t>(g);
  default: // HIP_DTYPE_UINT16
    return qgemm_launch_gemv<TA, TB, uint16_t>(g);
  }
}

template <typename TA> int qgemm_dispatch_b(const QGemmArgs &g) {
  if (g.b_dtype == HIP_DTYPE_INT8)
    return qgemm_dispatch_y<TA, int8_t>(g);
  return qgemm_dispatch_y<TA, uint8_t>(g);
}

int qgemm_dispatch_a(const QGemmArgs &g) {
  switch (g.a_dtype) {
  case HIP_DTYPE_INT8:
    return qgemm_dispatch_b<int8_t>(g);
  case HIP_DTYPE_UINT8:
    return qgemm_dispatch_b<uint8_t>(g);
  case HIP_DTYPE_INT16:
    return qgemm_dispatch_b<int16_t>(g);
  default: // HIP_DTYPE_UINT16
    return qgemm_dispatch_b<uint16_t>(g);
  }
}

int qgemm_launch_naive(const QGemmArgs &g) {
  const dim3 block(kBlockN, kBlockM);
  const dim3 grid(static_cast<unsigned>((g.N + kBlockN - 1) / kBlockN),
                  static_cast<unsigned>((g.M + kBlockM - 1) / kBlockM));

  (void)hipGetLastError();
  hipLaunchKernelGGL(qgemm_naive_kernel, grid, block, 0, g.stream, g.A, g.B,
                     g.C, g.b_scales, g.b_zero_points, g.Y, g.M, g.N, g.K,
                     g.trans_a, g.trans_b, g.a_dtype, g.b_dtype, g.c_dtype,
                     g.y_dtype, g.b_bits, g.b_signed, g.c_dim0, g.c_dim1,
                     g.m_ab, g.m_c, g.a_zp, g.b_zp, g.c_zp, g.y_zp);
  return static_cast<int>(hipGetLastError());
}

bool is_quantized_8_or_16(int dtype) {
  return dtype == HIP_DTYPE_INT8 || dtype == HIP_DTYPE_UINT8 ||
         dtype == HIP_DTYPE_INT16 || dtype == HIP_DTYPE_UINT16;
}

} // namespace

extern "C" size_t hip_qgemm_workspace_bytes(int64_t M, int64_t N, int64_t K) {
  (void)K;
  // Only the GEMV path splits k across the grid, and its scratch is one
  // accumulator plus one colB per column plus a single rowA.
  if (M != 1 || N <= 0)
    return 0;
  return qgemm_gemv_scratch_bytes(N);
}

extern "C" int hip_qgemm(void *stream, const void *A, const void *B,
                         const void *C, const void *B_scales,
                         const void *B_zero_points, void *Y, int64_t M,
                         int64_t N, int64_t K, int trans_a, int trans_b,
                         int a_dtype, int b_dtype, int c_dtype, int y_dtype,
                         int b_bits, int64_t c_dim0, int64_t c_dim1,
                         float M_ab, float M_c, int64_t a_zp, int64_t b_zp,
                         int64_t c_zp, int64_t y_zp, void *workspace,
                         size_t workspace_bytes) {
  if (!A || !B || !Y) {
    fprintf(stderr, "[custom_kernels] hip_qgemm: null tensor argument\n");
    return -1;
  }
  if (M <= 0 || N <= 0 || K <= 0)
    return 0;

  // The kernel indexes a scale and a zero point by the same n; neither has a
  // meaningful stand-in, so one without the other is a caller bug.
  if (static_cast<bool>(B_scales) != static_cast<bool>(B_zero_points)) {
    fprintf(stderr, "[custom_kernels] hip_qgemm: B_scales and B_zero_points "
                    "must be given together\n");
    return -1;
  }
  if (b_bits != 4 && b_bits != 8) {
    fprintf(stderr, "[custom_kernels] hip_qgemm: unsupported b_bits=%d\n",
            b_bits);
    return -1;
  }

  bool b_signed;
  switch (b_dtype) {
  case HIP_DTYPE_INT8:
    b_signed = true;
    break;
  case HIP_DTYPE_UINT8:
    b_signed = false;
    break;
  default:
    fprintf(stderr,
            "[custom_kernels] hip_qgemm: unsupported b_dtype=%d (expected "
            "8-bit storage)\n",
            b_dtype);
    return -1;
  }
  if (!is_quantized_8_or_16(a_dtype) || !is_quantized_8_or_16(y_dtype)) {
    fprintf(stderr,
            "[custom_kernels] hip_qgemm: unsupported a_dtype=%d / y_dtype=%d "
            "(expected 8- or 16-bit quantized)\n",
            a_dtype, y_dtype);
    return -1;
  }
  // int32 is accepted only for C: an ONNX quantizer emits a Gemm bias at
  // s_c = s_a * s_b, which needs the wider accumulator type.
  if (C && !is_quantized_8_or_16(c_dtype) && c_dtype != HIP_DTYPE_INT32) {
    fprintf(stderr,
            "[custom_kernels] hip_qgemm: unsupported c_dtype=%d (expected 8-, "
            "16- or 32-bit quantized)\n",
            c_dtype);
    return -1;
  }
  if (C && (c_dim0 <= 0 || c_dim1 <= 0)) {
    fprintf(stderr,
            "[custom_kernels] hip_qgemm: invalid C extents %lldx%lld\n",
            (long long)c_dim0, (long long)c_dim1);
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_qgemm: M=%lld N=%lld K=%lld trans=(%d,%d) "
      "dtypes=(%d,%d,%d,%d) b_bits=%d per_channel=%s M_ab=%g M_c=%g "
      "zp=(%lld,%lld,%lld,%lld) C=%s scratch=%zu B\n",
      (long long)M, (long long)N, (long long)K, trans_a, trans_b, a_dtype,
      b_dtype, c_dtype, y_dtype, b_bits, B_scales ? "yes" : "no", (double)M_ab,
      (double)M_c, (long long)a_zp, (long long)b_zp, (long long)c_zp,
      (long long)y_zp, C ? "yes" : "null", workspace_bytes);

  const QGemmArgs args{static_cast<hipStream_t>(stream),
                       A,
                       static_cast<const uint8_t *>(B),
                       C,
                       static_cast<const float *>(B_scales),
                       static_cast<const uint8_t *>(B_zero_points),
                       Y,
                       M,
                       N,
                       K,
                       trans_a != 0,
                       trans_b != 0,
                       a_dtype,
                       b_dtype,
                       c_dtype,
                       y_dtype,
                       b_bits,
                       b_signed,
                       c_dim0,
                       c_dim1,
                       M_ab,
                       M_c,
                       static_cast<int32_t>(a_zp),
                       static_cast<int32_t>(b_zp),
                       static_cast<int32_t>(c_zp),
                       static_cast<int32_t>(y_zp),
                       workspace,
                       workspace_bytes};

  const int err = (M == 1) ? qgemm_dispatch_a(args) : qgemm_launch_naive(args);
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_qgemm launch failed: %s\n",
            hipGetErrorString(static_cast<hipError_t>(err)));
  }
  return err;
}
