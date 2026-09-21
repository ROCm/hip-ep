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
// coefficient and the final round.
//
// A per-output-channel B is what M_ab cannot absorb: s_b and z_b are then
// device arrays indexed by n, and the per-tensor B_scale contributes its 1.0
// identity to M_ab instead.
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
// Each correction term is only accumulated when its zero point can be non-zero.
//
// Accumulator width
// -----------------
// One subtracted term reaches 1.7e7 -- |A - z_a| up to 65535 for a 16-bit
// activation, |B - z_b| up to 255 even for a 4-bit weight -- so it outgrows
// int32 after two terms. Accumulating raw operands instead lets the tiled path
// split a 16-bit A into two byte planes that each fit an int32 dot-product
// accumulator and combine them in int64 once per output. The GEMV path is
// bandwidth-bound and accumulates in int64 directly.
//
// Kernel paths
// ------------
//   M == 1                      -> GEMV, k split inside the block and the grid
//   M > 1                       -> tiled GEMM on 8-bit integer dot products
//   M > 1, too few output tiles -> the same tile, with k split across the grid

#include "debug_log.h"
#include "hip_arch_compat.h"
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

// Widen one element of an 8-, 16- or 32-bit quantized buffer. The dtype is a
// runtime argument rather than a template parameter because C is read once per
// output, off every reduction.
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

// Read one LOGICAL element of B or B_zero_points: `lin` counts logical
// elements, never bytes. At bits == 4 two values share a byte, low nibble first
// over the flattened row-major sequence -- the ONNX INT4/UINT4 convention --
// and `is_signed` must track B's element type, since the two interpretations
// disagree on every nibble above 7.
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

//===----------------------------------------------------------------------===//
// Requantization epilogue
//===----------------------------------------------------------------------===//
//
// Shared by every path: each one only has to deliver the three sums of the
// identity above for one output element, and this turns them into Y.

// Everything the epilogue needs that does not come out of the reduction. Passed
// by value, so it must stay trivially copyable.
struct QGemmRequant {
  const float *b_scales;
  const uint8_t *b_zero_points;
  const void *C;
  float m_ab;
  float m_c;
  int64_t K;
  int64_t c_dim0;
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
__device__ inline TY qgemm_requant(int64_t acc, int64_t row_a, int64_t col_b,
                                   int64_t m, int64_t n,
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
    // Unidirectional broadcast to [M, N]: an extent of 1 repeats along its
    // axis, which is the whole of ONNX Gemm's bias rule once the caller has
    // normalized C to rank 2.
    const int64_t cm = (r.c_dim0 == 1) ? 0 : m;
    const int64_t cn = (r.c_dim1 == 1) ? 0 : n;
    scaled += r.m_c * static_cast<float>(
                          qgemm_load(r.C, cm * r.c_dim1 + cn, r.c_dtype) -
                          r.c_zp);
  }
  return q_saturate<TY>(rintf(scaled) + static_cast<float>(r.y_zp));
}

__device__ inline void qgemm_atomic_add_i64(int64_t *addr, int64_t v) {
  atomicAdd(reinterpret_cast<unsigned long long *>(addr),
            static_cast<unsigned long long>(v));
}

//===----------------------------------------------------------------------===//
// GEMV (M == 1)
//===----------------------------------------------------------------------===//
//
// A tile has no reuse left to schedule at M == 1 -- one A element is consumed
// once per column and never again -- so what remains is a bandwidth problem.
//
// This kernel serves B stored [K, N], a [N, K] B whose columns are too short
// to hand every lane of a wave a dword, and whatever the dword loads cannot
// take at all. A long-column [N, K] B goes to qgemm_gemv_nk_kernel below, which
// walks k contiguously because k is the contiguous axis there.
//
// A thread owns kCols consecutive columns, which turns the per-k read of B into
// whole dwords instead of a single byte, and A is staged once per k-tile in LDS
// rather than re-read per column.
//
// Concurrent k slices are what the shape is short of, and K is the only
// dimension left to spread: the column window cancels out of the throughput,
// since a narrower one yields proportionally more windows. K is therefore split
// twice.
//
//   * kSplits slices run side by side inside one block and merge through LDS
//     atomics, which keeps the merge footprint independent of kSplits. kSplits
//     is capped by K / kTile, so kTile wants to be small -- it only sizes the
//     staging tile for A.
//   * A narrow N still leaves too few blocks to fill the device even then, so
//     the grid splits K again across blockIdx.y, accumulating into a scratch
//     buffer that a second pass requantizes. That pass costs two extra launches
//     and a memset, so it is only used when the column windows alone cannot
//     fill the device.
//
// The geometry parameters are templated because their balance is a per-shape
// property; qgemm_gemv_config measures it on the device.
//
// There is no trans_a flag here: at M == 1 both layouts of A are the same
// K-element vector, so the index is unchanged.

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

// B stored [N, K] with a column too short for qgemm_gemv_nk_kernel to give
// every lane of a wave a dword of it. Here k is the contiguous axis, so a dword
// carries 32 / kBits steps of ONE column instead of one step of several, and
// the wave's lanes sit K bytes apart. That wastes most of each fetched line
// once a column is long, which is why the [N, K] kernel takes over there, but
// on a short column the line is small enough that the lanes consume it.
//
// kUnroll has no role in this layout: successive iterations read the next
// dwords of the same lines, which already keeps several requests in flight.
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
  const bool vector_nk_ok = trans_b && base_ok && window_ok &&
                            (K % align) == 0 && (kTile % align) == 0;

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
      // M == 1 collapses the row index, and C's row extent is then 1 too, so
      // the epilogue's broadcast leaves the bias index at the column alone.
      Y[n] = qgemm_requant<TY>(red_acc[slot], row_a, red_col_b[slot], 0, n,
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
  Y[n] = qgemm_requant<TY>(partial[n], partial[2 * N], partial[N + n], 0, n,
                           requant);
}

//===----------------------------------------------------------------------===//
// Tiled GEMM (M > 1)
//===----------------------------------------------------------------------===//
//
// Two-level tiling -- a block stages a slab of A and a slab of B in LDS, a
// thread owns a [kThreadM x kThreadN] micro-tile of the output -- plus the two
// things the quantized operands ask for:
//
//   * The products run on hipdnn_dot4 (v_dot4_i32_iu8), four 8-bit multiplies
//     per instruction into one int32 accumulator. A 16-bit A does not fit that
//     operand, so it is split into its two bytes. With A = hi*256 + lo,
//
//       acc  = 256 * sum_k hi[k]*B[k] + sum_k lo[k]*B[k]
//       rowA = 256 * sum_k hi[k]      + sum_k lo[k]
//
//     each half runs on its own int32 accumulator -- |lo*B| <= 255*128 keeps K
//     under 65792 -- and the two are combined in int64 in the epilogue, so the
//     result stays exact. hi carries A's signedness (an arithmetic shift for
//     int16, a logical one for uint16) and lo is always an unsigned byte.
//   * A 4-bit B is unpacked to whole bytes while staging, so the inner loop is
//     the same instruction sequence at either width. The unpack is paid once
//     per slab and reused kTileM / kThreadM times.
//
// The zero points stay out of the reduction as the rowA / colB corrections of
// the identity at the top of this file, which is what leaves dot4 its raw 8-bit
// operands; both ride the same dot pipeline against an all-ones vector.

// dot4(x, kOnes4) sums the four bytes of x.
constexpr int kOnes4 = 0x01010101;

// One block owns a [kTileM x kTileN] output tile, one thread a
// [kThreadM x kThreadN] micro-tile of it.
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreadM = 4;
constexpr int kThreadN = 4;
constexpr int kTiledBlock = (kTileM / kThreadM) * (kTileN / kThreadN); // 256
// k values per dot4 operand, and so per LDS dword.
constexpr int kDotK = 4;
constexpr int kTileQuads = kTileK / kDotK;
// Rows of a [K, N] B slab that one pass over the block can stage.
constexpr int kBStageRows = kTiledBlock / kTileN; // 4

// A 16-bit A is accumulated as two byte planes (hi, lo); an 8-bit one needs a
// single plane holding the value itself. Indices into the plane dimension:
constexpr int kPlaneHi = 0;
constexpr int kPlaneLo = 1;

template <typename TA> struct QGemmAPlanes {
  static constexpr bool kWide = sizeof(TA) == 2;
  static constexpr int kCount = kWide ? 2 : 1;
};

// The LDS slabs are k-quad-major dwords: logical element (k, row) lives in byte
// k % kDotK of dword [k / kDotK][row]. That is the layout dot4 wants -- four
// k-consecutive bytes in one register -- and it also puts a micro-tile's four
// rows in four CONSECUTIVE dwords, so one fragment is a single 16-byte LDS
// read.
using QGemmSlab = int32_t;
using QGemmByte = uint8_t;

// The slabs hold raw bytes: hipdnn_dot4's template flags decide how each
// operand's bytes are interpreted, so staging never has to know the signedness.
template <int kCount>
__device__ inline int32_t qgemm_pack_bytes(const int32_t (&v)[kCount]) {
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < kCount; ++i)
    packed |= (static_cast<uint32_t>(v[i]) & 0xFFu) << (i * 8);
  return static_cast<int32_t>(packed);
}

// kThreadM / kThreadN being kDotK and the slabs being 16-byte aligned make a
// fragment one aligned dwordx4 read.
template <int kCount>
__device__ inline void qgemm_read_frag(const QGemmSlab *slab,
                                       int32_t (&out)[kCount]) {
  static_assert(kCount == 4, "a fragment is one 16-byte LDS read");
  const int4 v = *reinterpret_cast<const int4 *>(slab);
  out[0] = v.x;
  out[1] = v.y;
  out[2] = v.z;
  out[3] = v.w;
}

template <int kPlanes> struct QGemmMicroTile {
  int32_t acc[kThreadM][kThreadN][kPlanes];
  int32_t row_a[kThreadM][kPlanes];
  int32_t col_b[kThreadN];
};

// A split 16-bit A recombines as hi*256 + lo; a single plane is the value.
template <int kPlanes>
__device__ inline int64_t qgemm_combine(const int32_t (&planes)[kPlanes]) {
  if constexpr (kPlanes == 2)
    return int64_t{256} * planes[kPlaneHi] + planes[kPlaneLo];
  else
    return planes[kPlaneHi];
}

__device__ inline void qgemm_micro_tile_origin(int tid, int &thread_m,
                                               int &thread_n) {
  thread_m = (tid / (kTileN / kThreadN)) * kThreadM;
  thread_n = (tid % (kTileN / kThreadN)) * kThreadN;
}

// Reduces A[m_base.., k_begin:k_end] x B[k_begin:k_end, n_base..] into this
// thread's micro-tile, leaving the zero points to the epilogue. K is still the
// full logical extent because it is also the operand stride.
//
// The k range is a parameter so split-K can hand each block a slice. Staging
// zero-fills anything past an edge and zero is the identity for all three sums,
// so a slice needs no tail handling; it only has to start on a kTileK boundary
// to keep the staging maps aligned.
//
// trans_a / trans_b and b_bits stay runtime flags: they only pick an index
// during staging, which is block-uniform and off the dot4 inner loop.
template <typename TA, typename TB>
__device__ QGemmMicroTile<QGemmAPlanes<TA>::kCount> qgemm_accumulate_k_range(
    const TA *__restrict__ A, const uint8_t *__restrict__ B, int64_t M,
    int64_t N, int64_t K, int64_t m_base, int64_t n_base, int64_t k_begin,
    int64_t k_end, int b_bits, bool need_row_a, bool need_col_b, bool trans_a,
    bool trans_b) {
  constexpr bool kAWide = QGemmAPlanes<TA>::kWide;
  constexpr int kAPlanes = QGemmAPlanes<TA>::kCount;
  constexpr bool kASigned = QGemmTraits<TA>::kSigned;
  constexpr bool kBSigned = QGemmTraits<TB>::kSigned;

  static_assert(kThreadM == kDotK && kThreadN == kDotK,
                "a micro-tile row and column are each one fragment");
  static_assert(kTileM * kTileK == kTiledBlock * kDotK,
                "the quad staging map covers a slab exactly once");
  static_assert(kTileK % kBStageRows == 0,
                "the [K, N] staging map covers every byte lane exactly once");

  __shared__ __align__(16) QGemmSlab a_lds[kAPlanes][kTileQuads][kTileM];
  __shared__ __align__(16) QGemmSlab b_lds[kTileQuads][kTileN];

  const int tid = threadIdx.x;
  int thread_m, thread_n;
  qgemm_micro_tile_origin(tid, thread_m, thread_n);

  QGemmMicroTile<kAPlanes> out = {};

  // Staging maps, each coalescing the reads of the layout it serves.
  //
  // `quad_*` hands a thread the kDotK k-consecutive elements of one row, which
  // is exactly one LDS dword. That is what A wants, and also what a [N, K] B
  // wants since its k runs are contiguous; kTileM == kTileN, so the one map
  // covers both slabs.
  //
  // `b_*` walks a [K, N] B, where the k stride is N: one element per thread per
  // k, so the kTileN threads sharing a k cover a contiguous run of columns.
  static_assert(kTileM == kTileN, "the quad staging map is shared by A and B");
  const int quad_row = tid / kTileQuads;
  const int quad_idx = tid % kTileQuads;
  const int b_n = tid % kTileN;
  const int b_k_base = tid / kTileN;

  for (int64_t k0 = k_begin; k0 < k_end; k0 += kTileK) {
    const int64_t quad_k = k0 + quad_idx * kDotK;
    {
      const int64_t a_row = m_base + quad_row;
      int32_t planes[kAPlanes][kDotK];
#pragma unroll
      for (int i = 0; i < kDotK; ++i) {
        const int64_t k = quad_k + i;
        const int64_t idx = trans_a ? k * M + a_row : a_row * K + k;
        const int v = (a_row < M && k < k_end) ? static_cast<int>(A[idx]) : 0;
        if constexpr (kAWide) {
          // Promoting to int first makes `>> 8` arithmetic for int16 and
          // logical for uint16, so hi inherits A's signedness while lo stays
          // an unsigned byte -- exactly the split A = hi*256 + lo needs.
          planes[kPlaneHi][i] = v >> 8;
          planes[kPlaneLo][i] = v & 0xFF;
        } else {
          planes[kPlaneHi][i] = v;
        }
      }
#pragma unroll
      for (int p = 0; p < kAPlanes; ++p)
        a_lds[p][quad_idx][quad_row] = qgemm_pack_bytes(planes[p]);
    }

    if (trans_b) {
      const int64_t b_row = n_base + quad_row;
      int32_t v[kDotK];
#pragma unroll
      for (int i = 0; i < kDotK; ++i) {
        const int64_t k = quad_k + i;
        v[i] = (b_row < N && k < k_end)
                   ? qgemm_read_quant(B, b_row * K + k, b_bits, kBSigned)
                   : 0;
      }
      b_lds[quad_idx][quad_row] = qgemm_pack_bytes(v);
    } else {
      // k is the strided axis of a [K, N] B, so a thread's elements land one
      // byte each in that many different dwords; the kDotK byte lanes of a
      // dword come from as many distinct threads.
      const int64_t b_col = n_base + b_n;
#pragma unroll
      for (int rep = 0; rep < kTileK / kBStageRows; ++rep) {
        const int b_k = b_k_base + rep * kBStageRows;
        const int64_t k = k0 + b_k;
        const int32_t v =
            (b_col < N && k < k_end)
                ? qgemm_read_quant(B, k * N + b_col, b_bits, kBSigned)
                : 0;
        reinterpret_cast<QGemmByte *>(&b_lds[b_k / kDotK][b_n])[b_k % kDotK] =
            static_cast<QGemmByte>(v);
      }
    }

    __syncthreads();

#pragma unroll
    for (int q = 0; q < kTileQuads; ++q) {
      int32_t a_frag[kAPlanes][kThreadM];
      int32_t b_frag[kThreadN];
#pragma unroll
      for (int p = 0; p < kAPlanes; ++p)
        qgemm_read_frag(&a_lds[p][q][thread_m], a_frag[p]);
      qgemm_read_frag(&b_lds[q][thread_n], b_frag);

#pragma unroll
      for (int i = 0; i < kThreadM; ++i) {
#pragma unroll
        for (int j = 0; j < kThreadN; ++j) {
          out.acc[i][j][kPlaneHi] = hipdnn_dot4<kASigned, kBSigned>(
              a_frag[kPlaneHi][i], b_frag[j], out.acc[i][j][kPlaneHi]);
          if constexpr (kAWide)
            out.acc[i][j][kPlaneLo] = hipdnn_dot4<false, kBSigned>(
                a_frag[kPlaneLo][i], b_frag[j], out.acc[i][j][kPlaneLo]);
        }
      }

      // Block-uniform branches, so the skipped dots cost nothing when the
      // corresponding zero point is 0 -- or, under split-K, when this block
      // was not the one elected to contribute that correction.
      if (need_row_a) {
#pragma unroll
        for (int i = 0; i < kThreadM; ++i) {
          out.row_a[i][kPlaneHi] = hipdnn_dot4<kASigned, false>(
              a_frag[kPlaneHi][i], kOnes4, out.row_a[i][kPlaneHi]);
          if constexpr (kAWide)
            out.row_a[i][kPlaneLo] = hipdnn_dot4<false, false>(
                a_frag[kPlaneLo][i], kOnes4, out.row_a[i][kPlaneLo]);
        }
      }
      if (need_col_b) {
#pragma unroll
        for (int j = 0; j < kThreadN; ++j)
          out.col_b[j] =
              hipdnn_dot4<kBSigned, false>(b_frag[j], kOnes4, out.col_b[j]);
      }
    }

    __syncthreads();
  }

  return out;
}

// Y[m, n] = requantize(sum_k A[m, k] * B[k, n])
//
// Grid: (ceil(N/kTileN), ceil(M/kTileM)). One block owns a whole output tile
// and the whole reduction, so it can requantize in place.
template <typename TA, typename TB, typename TY>
__global__ __launch_bounds__(kTiledBlock) void qgemm_tiled_kernel(
    const TA *__restrict__ A, const uint8_t *__restrict__ B,
    TY *__restrict__ Y, int64_t M, int64_t N, int64_t K, QGemmRequant requant,
    int b_bits, bool need_row_a, bool need_col_b, bool trans_a, bool trans_b) {
  const int64_t m_base = static_cast<int64_t>(blockIdx.y) * kTileM;
  const int64_t n_base = static_cast<int64_t>(blockIdx.x) * kTileN;

  const auto mt = qgemm_accumulate_k_range<TA, TB>(
      A, B, M, N, K, m_base, n_base, 0, K, b_bits, need_row_a, need_col_b,
      trans_a, trans_b);

  int thread_m, thread_n;
  qgemm_micro_tile_origin(threadIdx.x, thread_m, thread_n);

#pragma unroll
  for (int i = 0; i < kThreadM; ++i) {
    const int64_t m = m_base + thread_m + i;
    if (m >= M)
      continue;
    const int64_t row_a = qgemm_combine(mt.row_a[i]);
#pragma unroll
    for (int j = 0; j < kThreadN; ++j) {
      const int64_t n = n_base + thread_n + j;
      if (n >= N)
        continue;
      Y[m * N + n] = qgemm_requant<TY>(qgemm_combine(mt.acc[i][j]), row_a,
                                       mt.col_b[j], m, n, requant);
    }
  }
}

//===----------------------------------------------------------------------===//
// Split-K (low output parallelism)
//===----------------------------------------------------------------------===//
//
// The tiled grid has no k dimension, so its parallelism is bounded by the
// output alone: ceil(N/kTileN) * ceil(M/kTileM) blocks. A narrow output leaves
// most of the device idle with each block walking the entire reduction.
//
// Split-K restores the missing dimension: each block reduces a k slice into a
// global int64 accumulator, and a second kernel requantizes once every slice
// has landed. Integer addition is associative and exact, so the atomics make
// the result bit-identical to the single-pass path however the slices
// interleave.

// One block per (n-tile, m-tile, k slice).
template <typename TA, typename TB>
__global__ __launch_bounds__(kTiledBlock) void qgemm_splitk_partial_kernel(
    const TA *__restrict__ A, const uint8_t *__restrict__ B,
    int64_t *__restrict__ acc_ws, int64_t *__restrict__ row_a_ws,
    int64_t *__restrict__ col_b_ws, int64_t M, int64_t N, int64_t K,
    int64_t k_chunk, int b_bits, bool need_row_a, bool need_col_b,
    bool trans_a, bool trans_b) {
  const int64_t k_begin = static_cast<int64_t>(blockIdx.z) * k_chunk;
  if (k_begin >= K)
    return;
  const int64_t k_end = (k_begin + k_chunk < K) ? k_begin + k_chunk : K;

  const int64_t m_base = static_cast<int64_t>(blockIdx.y) * kTileM;
  const int64_t n_base = static_cast<int64_t>(blockIdx.x) * kTileN;

  // rowA does not depend on n and colB does not depend on m, so every block
  // column would otherwise add the same rowA slice again (and every block row
  // the same colB slice). Elect one of each; the flags are block-uniform, so
  // the blocks not elected skip the corresponding dots entirely.
  const bool add_row_a = need_row_a && blockIdx.x == 0;
  const bool add_col_b = need_col_b && blockIdx.y == 0;

  const auto mt = qgemm_accumulate_k_range<TA, TB>(
      A, B, M, N, K, m_base, n_base, k_begin, k_end, b_bits, add_row_a,
      add_col_b, trans_a, trans_b);

  int thread_m, thread_n;
  qgemm_micro_tile_origin(threadIdx.x, thread_m, thread_n);

#pragma unroll
  for (int i = 0; i < kThreadM; ++i) {
    const int64_t m = m_base + thread_m + i;
    if (m >= M)
      continue;
    // Every thread sharing this m holds the same rowA slice, so within the
    // elected block only the first micro-tile column commits it.
    if (add_row_a && thread_n == 0)
      qgemm_atomic_add_i64(&row_a_ws[m], qgemm_combine(mt.row_a[i]));
#pragma unroll
    for (int j = 0; j < kThreadN; ++j) {
      const int64_t n = n_base + thread_n + j;
      if (n < N)
        qgemm_atomic_add_i64(&acc_ws[m * N + n], qgemm_combine(mt.acc[i][j]));
    }
  }

  // Mirror of the rowA election: only the first micro-tile row commits colB.
  if (add_col_b && thread_m == 0) {
#pragma unroll
    for (int j = 0; j < kThreadN; ++j) {
      const int64_t n = n_base + thread_n + j;
      if (n < N)
        qgemm_atomic_add_i64(&col_b_ws[n], mt.col_b[j]);
    }
  }
}

constexpr int kSplitKEpilogueBlock = 256;

// Requantizes the completed split-K sums, one thread per output element. rowA /
// colB are read unconditionally: when their zero point is zero the partial pass
// never wrote them, but the workspace was zeroed beforehand and the epilogue
// multiplies them by that same zero point.
template <typename TY>
__global__ __launch_bounds__(kSplitKEpilogueBlock) void
qgemm_splitk_epilogue_kernel(const int64_t *__restrict__ acc_ws,
                             const int64_t *__restrict__ row_a_ws,
                             const int64_t *__restrict__ col_b_ws,
                             TY *__restrict__ Y, int64_t N, int64_t total,
                             QGemmRequant requant) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * kSplitKEpilogueBlock +
                      threadIdx.x;
  if (idx >= total)
    return;

  // acc_ws and Y share the [M, N] row-major order, so one flat index addresses
  // both; only the two correction vectors need unpacking.
  const int64_t m = idx / N;
  const int64_t n = idx - m * N;
  Y[idx] = qgemm_requant<TY>(acc_ws[idx], row_a_ws[m], col_b_ws[n], m, n,
                             requant);
}

//===----------------------------------------------------------------------===//
// GEMV for B stored [N, K]
//===----------------------------------------------------------------------===//
//
// The layout every fully-connected weight arrives in, so this is the decode
// hot path. The kernel above walks it with a lane per column group, which puts
// the lanes of a wave K bytes apart: one dword load per lane, each on its own
// cache line, 4 bytes taken from a 128-byte fetch. The line is only repaid if
// it survives until that lane's next step, and it does not -- the concurrent k
// slices of one block keep far more lines live than the vector cache holds.
//
// Here a whole wave owns a column group and the lanes walk k instead, so the
// 32 dwords of one load are 128 contiguous bytes: one fetch, fully consumed.
// The cost is that a column's sum ends up spread across the wave and has to be
// folded with cross-lane shuffles, which is paid once per column group rather
// than once per k step.
//
// No LDS and no barrier: A is re-read per wave, but it is the activation
// vector, which every block reads and the cache keeps.
//
// kCols columns share a wave so that one A load and one rowA reduction serve
// all of them; their dword loads are each independently contiguous.

constexpr int kGemvNkLanes = 32;

// One dword per lane per column, folded into the running sums. The lanes of a
// wave then cover 32 consecutive dwords of each column, which is one fetch per
// column with every byte of it used.
template <int kBits, int kCols, typename TA, typename TB>
__device__ void qgemm_gemv_nk_columns(const TA *__restrict__ A,
                                      const uint8_t *__restrict__ B,
                                      int64_t k_begin, int64_t k_end,
                                      const int64_t *col_offsets, int lane,
                                      bool need_row_a, bool need_col_b,
                                      int64_t *acc, int64_t *col_b,
                                      int64_t &row_a) {
  constexpr int kStep = 32 / kBits; // logical values per dword
  constexpr int64_t kStride = static_cast<int64_t>(kGemvNkLanes) * kStep;
  // A term is at most 65535 * 128, so a step contributes under 2^25 to an
  // accumulator and 32 bits hold a few dozen of them. Folding the int32
  // partials into the int64 totals on that period keeps the inner multiply-add
  // off the 64-bit path, where every term would cost a carry chain.
  constexpr int kFold = (kBits == 4) ? 256 : 32;

  int32_t p_acc[kCols] = {};
  int32_t p_col_b[kCols] = {};
  int32_t p_row_a = 0;
  int folded = 0;

  auto fold = [&]() {
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      acc[c] += p_acc[c];
      p_acc[c] = 0;
      col_b[c] += p_col_b[c];
      p_col_b[c] = 0;
    }
    row_a += p_row_a;
    p_row_a = 0;
  };

  // Decode reads every weight once per token, so this loop is bandwidth bound
  // and the width of the load is what it lives on. A dword per column per step
  // leaves it issuing four narrow loads where one 16-byte load would do; the
  // wide phase below covers whole 16-byte strides and the narrow phase keeps
  // the remainder, so no divisibility is assumed about K or a split-K slice.
  constexpr int kWideStep = 128 / kBits; // logical values per 16-byte load
  constexpr int64_t kWideStride = static_cast<int64_t>(kGemvNkLanes) * kWideStep;
  constexpr int kDwordStep = 32 / kBits; // logical values per dword

  const int64_t span = k_end - k_begin;
  // A 16-byte load needs the column base and the lane's own offset aligned;
  // both follow from the byte stride of a column and the slice's start.
  const bool wide_ok =
      ((static_cast<int64_t>(kGemvNkLanes) * kWideStep * kBits / 8) % 16) == 0 &&
      ((k_begin * kBits / 8) % 16) == 0 &&
      (reinterpret_cast<uintptr_t>(B) % 16) == 0 && col_offsets[0] % 16 == 0 &&
      col_offsets[kCols - 1] % 16 == 0;
  const int64_t k_mid =
      wide_ok ? k_begin + (span / kWideStride) * kWideStride : k_begin;

  // Step pointers rather than recompute an int64 offset per column per
  // iteration. The address arithmetic costs more latency than the
  // multiply-accumulates it feeds, and none of it depends on anything the loop
  // body produces.
  constexpr int64_t kWideByteStep = kWideStride * kBits / 8;
  const int64_t k_wide0 = k_begin + static_cast<int64_t>(lane) * kWideStep;
  const uint8_t *bp[kCols];
#pragma unroll
  for (int c = 0; c < kCols; ++c)
    bp[c] = B + col_offsets[c] + k_wide0 * kBits / 8;
  const TA *ap = A + k_wide0;

  for (int64_t k = k_wide0; k < k_mid;
       k += kWideStride, ap += kWideStride) {
    uint4 packed[kCols];
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      packed[c] = *reinterpret_cast<const uint4 *>(bp[c]);
      bp[c] += kWideByteStep;
    }

#pragma unroll
    for (int d = 0; d < 4; ++d) {
      uint32_t w[kCols];
#pragma unroll
      for (int c = 0; c < kCols; ++c)
        w[c] = (d == 0)   ? packed[c].x
               : (d == 1) ? packed[c].y
               : (d == 2) ? packed[c].z
                          : packed[c].w;
#pragma unroll
      for (int j = 0; j < kDwordStep; ++j) {
        const int32_t a_val = static_cast<int32_t>(ap[d * kDwordStep + j]);
        if (need_row_a)
          p_row_a += a_val;
#pragma unroll
        for (int c = 0; c < kCols; ++c) {
          const int32_t b_val = qgemm_gemv_unpack<kBits, TB>(w[c], j);
          p_acc[c] += a_val * b_val;
          if (need_col_b)
            p_col_b[c] += b_val;
        }
      }
    }

    // One wide step carries four dwords' worth of terms, so it consumes four
    // of the accumulator's headroom before a fold is due.
    folded += 4;
    if (folded >= kFold) {
      fold();
      folded = 0;
    }
  }

  constexpr int64_t kNarrowByteStep = kStride * kBits / 8;
  const int64_t k_narrow0 = k_mid + static_cast<int64_t>(lane) * kStep;
  const uint8_t *np[kCols];
#pragma unroll
  for (int c = 0; c < kCols; ++c)
    np[c] = B + col_offsets[c] + k_narrow0 * kBits / 8;
  const TA *nap = A + k_narrow0;

  for (int64_t k = k_narrow0; k + kStep <= k_end;
       k += kStride, nap += kStride) {
    uint32_t packed[kCols];
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      packed[c] = *reinterpret_cast<const uint32_t *>(np[c]);
      np[c] += kNarrowByteStep;
    }

#pragma unroll
    for (int j = 0; j < kStep; ++j) {
      const int32_t a_val = static_cast<int32_t>(nap[j]);
      if (need_row_a)
        p_row_a += a_val;
#pragma unroll
      for (int c = 0; c < kCols; ++c) {
        const int32_t b_val = qgemm_gemv_unpack<kBits, TB>(packed[c], j);
        p_acc[c] += a_val * b_val;
        if (need_col_b)
          p_col_b[c] += b_val;
      }
    }

    if (++folded == kFold) {
      fold();
      folded = 0;
    }
  }
  fold();
}

__device__ inline int64_t qgemm_wave_sum(int64_t v) {
#pragma unroll
  for (int off = kGemvNkLanes / 2; off > 0; off >>= 1)
    v += __shfl_down(v, off, kGemvNkLanes);
  return v;
}

// `partial` is null for the single-pass form, where the wave owns all of K for
// its columns and writes Y itself. Otherwise blockIdx.y selects a slice of K
// and the totals go to scratch for qgemm_gemv_reduce_kernel to requantize.
template <typename TA, typename TB, typename TY, int kCols, int kWaves>
__global__ __launch_bounds__(kGemvNkLanes *kWaves) void qgemm_gemv_nk_kernel(
    const TA *__restrict__ A, const uint8_t *__restrict__ B,
    TY *__restrict__ Y, int64_t *__restrict__ partial, int64_t N, int64_t K,
    int64_t grid_chunk, QGemmRequant requant, int b_bits, bool need_row_a,
    bool need_col_b) {
  const int lane = threadIdx.x;
  const int wave = threadIdx.y;
  const int64_t n0 =
      (static_cast<int64_t>(blockIdx.x) * kWaves + wave) * kCols;
  if (n0 >= N)
    return;

  const int64_t col_bytes = K * b_bits / 8;
  // A column past N would read out of bounds, so the lane reads column 0
  // instead and the store below drops it. That keeps the load uniform.
  int64_t col_offsets[kCols];
#pragma unroll
  for (int c = 0; c < kCols; ++c)
    col_offsets[c] = (n0 + c < N ? n0 + c : 0) * col_bytes;

  const int64_t k_begin = static_cast<int64_t>(blockIdx.y) * grid_chunk;
  const int64_t k_end =
      (k_begin + grid_chunk < K) ? k_begin + grid_chunk : K;

  int64_t acc[kCols] = {};
  int64_t col_b[kCols] = {};
  int64_t row_a = 0;

  if (b_bits == 8)
    qgemm_gemv_nk_columns<8, kCols, TA, TB>(A, B, k_begin, k_end, col_offsets,
                                            lane, need_row_a, need_col_b, acc,
                                            col_b, row_a);
  else
    qgemm_gemv_nk_columns<4, kCols, TA, TB>(A, B, k_begin, k_end, col_offsets,
                                            lane, need_row_a, need_col_b, acc,
                                            col_b, row_a);

#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    acc[c] = qgemm_wave_sum(acc[c]);
    if (need_col_b)
      col_b[c] = qgemm_wave_sum(col_b[c]);
  }
  if (need_row_a)
    row_a = qgemm_wave_sum(row_a);

  if (lane != 0)
    return;

  if (!partial) {
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      const int64_t n = n0 + c;
      if (n < N)
        Y[n] = qgemm_requant<TY>(acc[c], row_a, col_b[c], 0, n, requant);
    }
    return;
  }

#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    const int64_t n = n0 + c;
    if (n >= N)
      continue;
    // One wave per (column, k slice), so these never contend with each other;
    // the atomic is what merges the slices.
    qgemm_atomic_add_i64(&partial[n], acc[c]);
    qgemm_atomic_add_i64(&partial[N + n], col_b[c]);
  }
  // Every column of a k slice recomputed the same rowA, so exactly one of them
  // may contribute it.
  if (need_row_a && blockIdx.x == 0 && wave == 0)
    qgemm_atomic_add_i64(&partial[2 * N], row_a);
}

//===----------------------------------------------------------------------===//
// Host launch and dtype dispatch
//===----------------------------------------------------------------------===//

// Bundled so the dispatch levels below stay readable: the dtype triple alone
// costs 32 switch arms.
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
// raises that cap further, but never below 32, or a wave would run with idle
// lanes. cols reaches 8 only where it has to: a 4-bit B needs 8 logical values
// to fill a dword, so the [K, N] vector path is unavailable to the cols == 4
// entries. The low-splits entries at the end double as launch fallbacks and as
// the geometries that leave the largest grid-level k split available.
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

// Column groups per block, and how many of them a block carries. Four waves
// keeps the tail waste under a wave on the narrow shapes while still giving
// the scheduler whole-block granularity.
constexpr int kGemvNkCols = 2;
constexpr int kGemvNkWaves = 4;
constexpr int kGemvNkGroup = kGemvNkCols * kGemvNkWaves;

// Whether the [N, K] kernel can take the shape. Its loads are whole dwords cut
// out of one column, so a column must hold an integral number of them -- and
// at least one per lane, or the wave that owns the column runs with most of
// its lanes idle and the [K, N] kernel's column-per-lane mapping wins instead.
bool qgemm_gemv_nk_ok(const QGemmArgs &g) {
  if (!g.trans_b)
    return false;
  const int64_t align = 32 / g.b_bits; // logical elements per dword
  return (reinterpret_cast<uintptr_t>(g.B) % sizeof(uint32_t)) == 0 &&
         (g.K % align) == 0 && g.K / align >= kGemvNkLanes;
}

// Column groups are the only other source of blocks here, so a narrow N still
// needs K split across the grid. Unlike the [K, N] path there is no in-block
// slice count competing for the same K, so the cap is one whole wave step per
// slice.
int qgemm_gemv_nk_grid_splits(const QGemmArgs &g, int64_t step) {
  const int64_t groups = (g.N + kGemvNkGroup - 1) / kGemvNkGroup;
  const int64_t target =
      static_cast<int64_t>(qgemm_device_cu_count()) * kGemvTargetBlocksPerCU;
  if (groups >= target)
    return 1;
  if (!g.workspace || g.workspace_bytes < qgemm_gemv_scratch_bytes(g.N))
    return 1;

  const int64_t cap = g.K / (kGemvNkLanes * step);
  if (cap < 2)
    return 1;

  int64_t want = (target + groups - 1) / groups;
  if (want > cap)
    want = cap;
  if (want > kGemvMaxGridSplits)
    want = kGemvMaxGridSplits;
  return want > 1 ? static_cast<int>(want) : 1;
}

template <typename TA, typename TB, typename TY>
int qgemm_gemv_nk_launch(const QGemmArgs &g, const QGemmRequant &requant,
                         bool need_row_a, bool need_col_b) {
  const int64_t step = 32 / g.b_bits; // logical elements per dword
  const int splits = qgemm_gemv_nk_grid_splits(g, step);
  auto *partial = splits > 1 ? static_cast<int64_t *>(g.workspace) : nullptr;

  (void)hipGetLastError();
  if (partial) {
    // Half-zeroed accumulators would corrupt the result, so a memset that fails
    // drops the whole op rather than the extra parallelism silently.
    const hipError_t cleared = hipMemsetAsync(
        partial, 0, qgemm_gemv_scratch_bytes(g.N), g.stream);
    if (cleared != hipSuccess)
      return static_cast<int>(cleared);
  }

  // Rounded up to a whole wave step so a slice boundary lands where a lane
  // starts a dword.
  int64_t grid_chunk = g.K;
  if (splits > 1) {
    const int64_t unit = kGemvNkLanes * step;
    grid_chunk = (g.K + splits - 1) / splits;
    grid_chunk = (grid_chunk + unit - 1) / unit * unit;
  }

  const dim3 grid(
      static_cast<unsigned>((g.N + kGemvNkGroup - 1) / kGemvNkGroup),
      static_cast<unsigned>(splits));
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
          qgemm_gemv_nk_kernel<TA, TB, TY, kGemvNkCols, kGemvNkWaves>),
      grid, dim3(kGemvNkLanes, kGemvNkWaves), 0, g.stream,
      static_cast<const TA *>(g.A), g.B, static_cast<TY *>(g.Y), partial, g.N,
      g.K, grid_chunk, requant, g.b_bits, need_row_a, need_col_b);
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

// Read once and latched: a getenv per call would serialize every dispatch
// through the CRT environment lock.
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
int qgemm_launch_gemv(const QGemmArgs &g, const QGemmRequant &requant,
                      bool need_row_a, bool need_col_b) {
  if (qgemm_gemv_nk_ok(g))
    return qgemm_gemv_nk_launch<TA, TB, TY>(g, requant, need_row_a,
                                            need_col_b);

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

//===----------------------------------------------------------------------===//
// Tiled launch and split-K planning
//===----------------------------------------------------------------------===//

// Slice ceiling. Past it the per-slice staging prologue and the atomic traffic
// grow faster than the added parallelism pays back.
constexpr int kSplitKMaxSplits = 32;
// A slice still has to cover enough k-tiles to amortize that prologue and the
// atomics it ends with.
constexpr int64_t kSplitKMinKPerSplit = 4 * kTileK;
// Blocks per CU the plain grid must already reach for split-K to be pointless.
constexpr int kSplitKTargetBlocksPerCU = 2;

struct QGemmSplitKPlan {
  bool enabled;
  int splits;
  int64_t k_chunk;
};

QGemmSplitKPlan qgemm_plan_splitk(int64_t M, int64_t N, int64_t K) {
  QGemmSplitKPlan plan{false, 1, K};
  // M == 1 takes the GEMV path, which splits k on its own.
  if (M <= 1)
    return plan;

  const int64_t base_blocks =
      ((N + kTileN - 1) / kTileN) * ((M + kTileM - 1) / kTileM);
  const int64_t target =
      static_cast<int64_t>(qgemm_device_cu_count()) * kSplitKTargetBlocksPerCU;
  if (base_blocks >= target)
    return plan;

  int64_t want = (target + base_blocks - 1) / base_blocks;
  if (want > kSplitKMaxSplits)
    want = kSplitKMaxSplits;
  const int64_t cap = K / kSplitKMinKPerSplit;
  if (want > cap)
    want = cap;
  if (want < 2)
    return plan;

  // Round the slice up to a whole k-tile so each one starts on the boundary the
  // staging maps assume, then recount so the last slice is not empty.
  int64_t chunk = (K + want - 1) / want;
  chunk = (chunk + kTileK - 1) / kTileK * kTileK;
  const int64_t splits = (K + chunk - 1) / chunk;
  if (splits < 2)
    return plan;

  plan.enabled = true;
  plan.splits = static_cast<int>(splits);
  plan.k_chunk = chunk;
  return plan;
}

// One int64 accumulator per output plus the two correction vectors. Split-K
// only engages below kSplitKTargetBlocksPerCU tiles per CU, which bounds M * N
// at that many tiles.
size_t qgemm_splitk_scratch_bytes(int64_t M, int64_t N) {
  return static_cast<size_t>(M * N + M + N) * sizeof(int64_t);
}

template <typename TA, typename TB, typename TY>
int qgemm_launch_tiled(const QGemmArgs &g, const QGemmRequant &requant,
                       bool need_row_a, bool need_col_b) {
  const auto *a_ptr = static_cast<const TA *>(g.A);
  auto *y_ptr = static_cast<TY *>(g.Y);
  const dim3 grid_xy(static_cast<unsigned>((g.N + kTileN - 1) / kTileN),
                     static_cast<unsigned>((g.M + kTileM - 1) / kTileM));

  const QGemmSplitKPlan plan = qgemm_plan_splitk(g.M, g.N, g.K);
  const size_t scratch =
      plan.enabled ? qgemm_splitk_scratch_bytes(g.M, g.N) : 0;
  bool use_splitk = plan.enabled && g.workspace && g.workspace_bytes >= scratch;

  (void)hipGetLastError();
  if (use_splitk &&
      hipMemsetAsync(g.workspace, 0, scratch, g.stream) != hipSuccess) {
    // Half-zeroed accumulators would corrupt the result, so drop to the
    // single-pass kernel, which needs no scratch. Clear the sticky error too,
    // or the fallback's own clean launch would still be reported as a failure.
    (void)hipGetLastError();
    use_splitk = false;
  }

  if (use_splitk) {
    auto *acc_ws = static_cast<int64_t *>(g.workspace);
    auto *row_a_ws = acc_ws + g.M * g.N;
    auto *col_b_ws = row_a_ws + g.M;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] qgemm tiled: split-K M=%lld N=%lld K=%lld splits=%d "
        "k_chunk=%lld scratch=%zu B\n",
        (long long)g.M, (long long)g.N, (long long)g.K, plan.splits,
        (long long)plan.k_chunk, scratch);

    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(qgemm_splitk_partial_kernel<TA, TB>),
        dim3(grid_xy.x, grid_xy.y, static_cast<unsigned>(plan.splits)),
        dim3(kTiledBlock), 0, g.stream, a_ptr, g.B, acc_ws, row_a_ws, col_b_ws,
        g.M, g.N, g.K, plan.k_chunk, g.b_bits, need_row_a, need_col_b,
        g.trans_a, g.trans_b);
    const hipError_t launched = hipGetLastError();
    if (launched != hipSuccess)
      return static_cast<int>(launched);

    const int64_t total = g.M * g.N;
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(qgemm_splitk_epilogue_kernel<TY>),
        dim3(static_cast<unsigned>((total + kSplitKEpilogueBlock - 1) /
                                   kSplitKEpilogueBlock)),
        dim3(kSplitKEpilogueBlock), 0, g.stream, acc_ws, row_a_ws, col_b_ws,
        y_ptr, g.N, total, requant);
    return static_cast<int>(hipGetLastError());
  }

  hipLaunchKernelGGL(HIP_KERNEL_NAME(qgemm_tiled_kernel<TA, TB, TY>), grid_xy,
                     dim3(kTiledBlock), 0, g.stream, a_ptr, g.B, y_ptr, g.M,
                     g.N, g.K, requant, g.b_bits, need_row_a, need_col_b,
                     g.trans_a, g.trans_b);
  return static_cast<int>(hipGetLastError());
}

template <typename TA, typename TB, typename TY>
int qgemm_launch(const QGemmArgs &g) {
  const QGemmRequant requant{g.b_scales,
                             g.b_zero_points,
                             g.C,
                             g.m_ab,
                             g.m_c,
                             g.K,
                             g.c_dim0,
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

  if (g.M == 1)
    return qgemm_launch_gemv<TA, TB, TY>(g, requant, need_row_a, need_col_b);
  return qgemm_launch_tiled<TA, TB, TY>(g, requant, need_row_a, need_col_b);
}

template <typename TA, typename TB>
int qgemm_dispatch_y(const QGemmArgs &g) {
  switch (g.y_dtype) {
  case HIP_DTYPE_INT8:
    return qgemm_launch<TA, TB, int8_t>(g);
  case HIP_DTYPE_UINT8:
    return qgemm_launch<TA, TB, uint8_t>(g);
  case HIP_DTYPE_INT16:
    return qgemm_launch<TA, TB, int16_t>(g);
  default: // HIP_DTYPE_UINT16
    return qgemm_launch<TA, TB, uint16_t>(g);
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

bool is_quantized_8_or_16(int dtype) {
  return dtype == HIP_DTYPE_INT8 || dtype == HIP_DTYPE_UINT8 ||
         dtype == HIP_DTYPE_INT16 || dtype == HIP_DTYPE_UINT16;
}

} // namespace

extern "C" size_t hip_qgemm_workspace_bytes(int64_t M, int64_t N, int64_t K) {
  if (M <= 0 || N <= 0 || K <= 0)
    return 0;
  // Either path only needs scratch where it splits k across the grid. The GEMV
  // one decides that per launch, from a geometry this function does not know,
  // so it always reserves: one accumulator plus one colB per column plus a
  // single rowA. The tiled one asks the same planner the launch will.
  if (M == 1)
    return qgemm_gemv_scratch_bytes(N);
  const QGemmSplitKPlan plan = qgemm_plan_splitk(M, N, K);
  return plan.enabled ? qgemm_splitk_scratch_bytes(M, N) : 0;
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

  const int err = qgemm_dispatch_a(args);
  if (err != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_qgemm launch failed: %s\n",
            hipGetErrorString(static_cast<hipError_t>(err)));
  }
  return err;
}
