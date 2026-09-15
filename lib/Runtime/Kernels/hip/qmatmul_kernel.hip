/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// FusedOp(A,B) = Q(DQ(A) @ DQ(B))
//
// a = (A - z_a) * s_a
// b = (B - z_b) * s_b
// Y = saturate(round((a @ b) / s_y) + z_y)
//
// (a @ b)[m,n] = s_a*s_b * sum_k (A[m,k] - z_a) * (B[k,n] - z_b)
//              = s_a*s_b * (acc[m,n] - z_b*rowA[m] - z_a*colB[n] + K*z_a*z_b)
//   acc[m,n] = sum_k A[m,k]*B[k,n]
//   rowA[m]  = sum_k A[m,k]
//   colB[n]  = sum_k B[k,n]
// --->
// let M = s_a * s_b / s_y   (folded by lowering)
// Y = saturate(round(M * (acc - z_b*rowA - z_a*colB + K*z_a*z_b)) + z_y)
//
// A - z_a does not fit back into 8 bits, so the zero points cannot be folded
// into the operands; they are carried as the rowA / colB correction terms
// instead. Each term is only accumulated when its zero point is non-zero, so
// the symmetric weights that dominate real QDQ graphs cost nothing extra.
//
// 16-bit activations (A16W8)
// --------------------------
// A may be 16-bit while B stays 8-bit. Two things break at that width: the
// int32 accumulator overflows at K around 256 (a single u16*u8 product already
// reaches 1.7e7), and hipdnn_dot4 is a 4x8-bit instruction that cannot see a
// 16-bit operand at all.
//
// Both are solved by splitting A into its two bytes. With
//
//   A = hi * 256 + lo,   hi = A >> 8,   lo = A & 0xFF
//
// (an arithmetic shift for int16 and a logical one for uint16, so hi carries
// A's signedness and lo is always unsigned), the dot product separates into
// two 8-bit dot products:
//
//   acc  = sum_k A[k]*B[k] = 256 * sum_k hi[k]*B[k] + sum_k lo[k]*B[k]
//   rowA = sum_k A[k]      = 256 * sum_k hi[k]      + sum_k lo[k]
//
// Each half runs on dot4 with its own int32 accumulator -- |lo*B| <= 255*128
// keeps K under 65792, far past any real shape -- and the two halves are
// combined in int64 at the epilogue. The result is exact, and the cost is two
// dot4 instructions per term instead of one rather than the 4x-plus slowdown a
// scalar int64 inner loop would bring.
//
// Kernel paths
// ------------
// All three share the identity above and differ only in how the reduction is
// parallelized:
//
//   M == 1                       -> GEMV, one thread per column group
//   M > 1                        -> tiled GEMM, one block per output tile
//   M > 1, too few output tiles  -> split-K over the tiled inner loop
//
// The GEMV path walks B along n rather than k, so dot4 has no four k-consecutive
// bytes to work with and it accumulates in int64 directly instead of splitting
// A.

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
// Element types, epilogue, and tile layout
//===----------------------------------------------------------------------===//

// All-ones byte vector: dot4(x, kOnes4) sums the four bytes of x, which is how
// rowA / colB ride along on the same integer dot pipeline as the products.
constexpr int kOnes4 = 0x01010101;

template <typename T> struct QMatMulTraits;

template <> struct QMatMulTraits<int8_t> {
  static constexpr bool kSigned = true;
  __device__ static float lowest() { return static_cast<float>(INT8_MIN); }
  __device__ static float highest() { return static_cast<float>(INT8_MAX); }
};

template <> struct QMatMulTraits<uint8_t> {
  static constexpr bool kSigned = false;
  __device__ static float lowest() { return 0.0f; }
  __device__ static float highest() { return static_cast<float>(UINT8_MAX); }
};

template <> struct QMatMulTraits<int16_t> {
  static constexpr bool kSigned = true;
  __device__ static float lowest() { return static_cast<float>(INT16_MIN); }
  __device__ static float highest() { return static_cast<float>(INT16_MAX); }
};

template <> struct QMatMulTraits<uint16_t> {
  static constexpr bool kSigned = false;
  __device__ static float lowest() { return 0.0f; }
  __device__ static float highest() { return static_cast<float>(UINT16_MAX); }
};

// The HIP compile flags include `-Xclang -fno-cuda-host-device-constexpr`
// (see lib/Runtime/Kernels/cmake/hip_utils.cmake), which makes
// `std::numeric_limits<T>` members host-only, hence the explicit bounds above.
template <typename T> __device__ inline T q_saturate(float v) {
  return static_cast<T>(
      fminf(QMatMulTraits<T>::highest(), fmaxf(QMatMulTraits<T>::lowest(), v)));
}

__device__ inline int qmatmul_read_quant(const uint8_t *__restrict__ data,
                                        int64_t index, int bits,
                                        bool is_signed) {
  if (bits == 4) {
    const uint8_t packed = data[index >> 1];
    const uint8_t nibble = (index & 1) ? packed >> 4 : packed & 0xF;
    return is_signed
               ? static_cast<int>(static_cast<int8_t>(nibble << 4)) >> 4
               : static_cast<int>(nibble);
  }
  return is_signed ? static_cast<int>(static_cast<int8_t>(data[index]))
                   : static_cast<int>(data[index]);
}

struct QMatMulRequant {
  float m_scale;
  float ay_ratio;
  const float *b_scales;
  const uint8_t *b_zero_points;
  int32_t a_zp;
  int32_t b_zp;
  int32_t y_zp;
  int32_t b_bits;
  bool b_signed;
  int64_t K;
  int64_t k_zazb;
};

template <typename TY>
__device__ inline TY
qmatmul_epilogue(int64_t acc, int64_t row_a, int64_t col_b, int64_t n,
                 QMatMulRequant requant) {
  const float scale = requant.b_scales
                          ? requant.ay_ratio * requant.b_scales[n]
                          : requant.m_scale;
  int32_t b_zp = requant.b_zp;
  int64_t k_zazb = requant.k_zazb;
  if (requant.b_scales && requant.b_bits == 8) {
    b_zp = qmatmul_read_quant(requant.b_zero_points, n, 8,
                              requant.b_signed);
    k_zazb = requant.K * static_cast<int64_t>(requant.a_zp) * b_zp;
  }
  const int64_t corrected =
      acc - static_cast<int64_t>(b_zp) * row_a -
      static_cast<int64_t>(requant.a_zp) * col_b + k_zazb;
  return q_saturate<TY>(rintf(scale * static_cast<float>(corrected)) +
                        static_cast<float>(requant.y_zp));
}

// Tile geometry shared by the tiled and split-K paths: one block owns a
// [kTileM x kTileN] output tile, one thread a [kThreadM x kThreadN] micro-tile.
constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreadM = 4;
constexpr int kThreadN = 4;
constexpr int kTiledBlock = (kTileM / kThreadM) * (kTileN / kThreadN); // 256
// Rows of the transposed B tile that one pass over the block can stage.
constexpr int kBStageRows = kTiledBlock / kTileN; // 4

// The LDS tiles hold raw bytes: hipdnn_dot4's template flags decide how each
// operand's bytes are interpreted, so staging never has to know the signedness.
using QMatMulByte = uint8_t;

// A 16-bit A is staged as two byte planes (hi, lo); an 8-bit one needs a single
// plane holding the value itself. Indices into the plane dimension:
constexpr int kPlaneHi = 0;
constexpr int kPlaneLo = 1;

template <typename TA> struct QMatMulAPlanes {
  static constexpr bool kWide = sizeof(TA) == 2;
  static constexpr int kCount = kWide ? 2 : 1;
};

// Reads 4 k-consecutive bytes of an LDS tile row as one dot4 operand. The tile
// rows are kTileK bytes wide and the array is 16-byte aligned, so every
// multiple-of-4 offset is a legal aligned int load.
__device__ inline int lds_pack4(const QMatMulByte *row, int k) {
  return *reinterpret_cast<const int *>(row + k);
}

template <int kPlanes> struct QMatMulMicroTile {
  int32_t acc[kPlanes][kThreadM][kThreadN];
  int32_t row_a[kPlanes][kThreadM];
  int32_t col_b[kThreadN];
};

__device__ inline void qmatmul_micro_tile_origin(int tid, int &thread_m,
                                                 int &thread_n) {
  thread_m = (tid / (kTileN / kThreadN)) * kThreadM;
  thread_n = (tid % (kTileN / kThreadN)) * kThreadN;
}

//===----------------------------------------------------------------------===//
// Shared k-range accumulation (LDS staging + dot4)
//===----------------------------------------------------------------------===//

// Reduces A[m_base.., k_begin:k_end] x B[k_begin:k_end, n_base..] into this
// thread's micro-tile. A and B are already offset to the block's batch; K is
// still the full logical extent because it is also the operand stride.
//
// The range is a parameter rather than always [0, K) so split-K can hand each
// block a slice. Staging zero-fills anything past k_end and zero is the
// identity for all three sums, so a slice needs no separate tail handling; it
// only has to start on a kTileK boundary to keep the staging maps coalesced.
//
// Both LDS tiles are k-major because dot4 needs four k-consecutive bytes in one
// register. For a [K, N] B that means staging it transposed; for a [N, K] one
// (trans_b) the tile is already in that order and staging is a straight copy.
//
// trans_a / trans_b are runtime flags rather than template parameters: they
// only pick a load stride during staging, which is block-uniform and off the
// dot4 inner loop, and templating them would quadruple the instantiations the
// dtype pair already costs.
template <typename TA, typename TB>
__device__ QMatMulMicroTile<QMatMulAPlanes<TA>::kCount>
qmatmul_accumulate_k_range(const TA *__restrict__ A, const TB *__restrict__ B,
                           int64_t M, int64_t N, int64_t K, int64_t m_base,
                           int64_t n_base, int64_t k_begin, int64_t k_end,
                           bool need_row_a, bool need_col_b, bool trans_a,
                           bool trans_b, int b_bits, bool b_signed,
                           const uint8_t *__restrict__ b_zero_points) {
  constexpr bool kAWide = QMatMulAPlanes<TA>::kWide;
  constexpr int kAPlanes = QMatMulAPlanes<TA>::kCount;
  constexpr bool kASigned = QMatMulTraits<TA>::kSigned;
  constexpr bool kBSigned = QMatMulTraits<TB>::kSigned;

  __shared__ __align__(16) QMatMulByte a_lds[kAPlanes][kTileM][kTileK];
  __shared__ __align__(16) QMatMulByte b_lds[kTileN][kTileK];

  const int tid = threadIdx.x;
  int thread_m, thread_n;
  qmatmul_micro_tile_origin(tid, thread_m, thread_n);

  // Per-plane accumulators. |lo * B| <= 255*128 bounds the widest plane, so
  // int32 holds K up to 65792 -- past any shape the runtime admits.
  QMatMulMicroTile<kAPlanes> out = {};

  // Staging maps, each coalescing the reads of the layout it serves.
  //
  // `row4_*` walks a tile row-major, one 4-byte run per thread, which is what
  // A wants and also what a [N, K] B wants since its k-runs are contiguous.
  // kTileM == kTileN, so the one map covers both tiles.
  //
  // `b_*` walks a [K, N] B, where the k stride is N: one byte per thread per
  // k, so the 64 threads sharing a k cover a contiguous 64-byte run.
  static_assert(kTileM == kTileN, "row4 staging map is shared by A and B");
  const int row4_r = (tid * 4) / kTileK;
  const int row4_c = (tid * 4) % kTileK;
  const int b_n = tid % kTileN;
  const int b_k_base = tid / kTileN;
  const int64_t row4_n = n_base + row4_r;
  const int64_t col_n = n_base + b_n;
  const int row4_zp =
      b_bits == 4 && row4_n < N
          ? qmatmul_read_quant(b_zero_points, row4_n, 4, b_signed)
          : 0;
  const int col_zp =
      b_bits == 4 && col_n < N
          ? qmatmul_read_quant(b_zero_points, col_n, 4, b_signed)
          : 0;

  for (int64_t k0 = k_begin; k0 < k_end; k0 += kTileK) {
    // Out-of-range elements are staged as 0, the identity for all three sums.
    const int64_t a_row = m_base + row4_r;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t k = k0 + row4_c + i;
      const bool in_range = a_row < M && k < k_end;
      // A [K, M] costs a strided read here, unlike the contiguous [M, K] one.
      // Left as is: TransposeMatMulFold folds Transpose into whichever operand
      // the graph transposes, and attention's Q @ K^T only ever stamps transB.
      const int64_t a_idx = trans_a ? k * M + a_row : a_row * K + k;
      if constexpr (kAWide) {
        // Promoting to int first makes `>> 8` arithmetic for int16 and logical
        // for uint16, so hi inherits A's signedness while lo stays an unsigned
        // byte -- exactly the split A = hi*256 + lo needs.
        const int v = in_range ? static_cast<int>(A[a_idx]) : 0;
        a_lds[kPlaneHi][row4_r][row4_c + i] = static_cast<QMatMulByte>(v >> 8);
        a_lds[kPlaneLo][row4_r][row4_c + i] =
            static_cast<QMatMulByte>(v & 0xFF);
      } else {
        a_lds[kPlaneHi][row4_r][row4_c + i] =
            in_range ? static_cast<QMatMulByte>(A[a_idx]) : QMatMulByte{0};
      }
    }

    if (trans_b) {
      const int64_t b_row = n_base + row4_r;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int64_t k = k0 + row4_c + i;
        int b_value = 0;
        if (b_row < N && k < k_end) {
          b_value = b_bits == 4
                        ? qmatmul_read_quant(
                              reinterpret_cast<const uint8_t *>(B),
                              b_row * K + k, 4, b_signed) -
                              row4_zp
                        : static_cast<int>(B[b_row * K + k]);
        }
        b_lds[row4_r][row4_c + i] =
            static_cast<QMatMulByte>(b_value);
      }
    } else {
      const int64_t b_col = n_base + b_n;
#pragma unroll
      for (int rep = 0; rep < kTileK / kBStageRows; ++rep) {
        const int b_k = b_k_base + rep * kBStageRows;
        const int64_t k = k0 + b_k;
        int b_value = 0;
        if (b_col < N && k < k_end) {
          b_value = b_bits == 4
                        ? qmatmul_read_quant(
                              reinterpret_cast<const uint8_t *>(B),
                              k * N + b_col, 4, b_signed) -
                              col_zp
                        : static_cast<int>(B[k * N + b_col]);
        }
        b_lds[b_n][b_k] = static_cast<QMatMulByte>(b_value);
      }
    }

    __syncthreads();

#pragma unroll
    for (int kk = 0; kk < kTileK; kk += 4) {
      int a_frag[kAPlanes][kThreadM];
      int b_frag[kThreadN];
#pragma unroll
      for (int p = 0; p < kAPlanes; ++p) {
#pragma unroll
        for (int i = 0; i < kThreadM; ++i)
          a_frag[p][i] = lds_pack4(a_lds[p][thread_m + i], kk);
      }
#pragma unroll
      for (int j = 0; j < kThreadN; ++j)
        b_frag[j] = lds_pack4(b_lds[thread_n + j], kk);

#pragma unroll
      for (int i = 0; i < kThreadM; ++i) {
#pragma unroll
        for (int j = 0; j < kThreadN; ++j) {
          out.acc[kPlaneHi][i][j] = hipdnn_dot4<kASigned, kBSigned>(
              a_frag[kPlaneHi][i], b_frag[j], out.acc[kPlaneHi][i][j]);
          if constexpr (kAWide)
            out.acc[kPlaneLo][i][j] = hipdnn_dot4<false, kBSigned>(
                a_frag[kPlaneLo][i], b_frag[j], out.acc[kPlaneLo][i][j]);
        }
      }

      // Block-uniform branches, so the skipped dots cost nothing when the
      // corresponding zero point is 0 -- or, under split-K, when this block
      // was not the one elected to contribute that correction.
      if (need_row_a) {
#pragma unroll
        for (int i = 0; i < kThreadM; ++i) {
          out.row_a[kPlaneHi][i] = hipdnn_dot4<kASigned, false>(
              a_frag[kPlaneHi][i], kOnes4, out.row_a[kPlaneHi][i]);
          if constexpr (kAWide)
            out.row_a[kPlaneLo][i] = hipdnn_dot4<false, false>(
                a_frag[kPlaneLo][i], kOnes4, out.row_a[kPlaneLo][i]);
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

//===----------------------------------------------------------------------===//
// Tiled GEMM (M > 1)
//===----------------------------------------------------------------------===//

// Y[b, m, n] = requantize(sum_k A[b, m, k] * B[b, k, n])
//
// Grid: (ceil(N/kTileN), ceil(M/kTileM), batch_count). One block owns a whole
// output tile and the whole reduction, so it can requantize in place.
template <typename TA, typename TB, typename TY>
__global__ __launch_bounds__(kTiledBlock) void qmatmul_tiled_kernel(
    const TA *__restrict__ A, const TB *__restrict__ B, TY *__restrict__ Y,
    const uint8_t *__restrict__ b_zero_points, int64_t M, int64_t N, int64_t K,
    int64_t b_batch_stride, QMatMulRequant requant, int b_bits, bool b_signed,
    bool need_row_a, bool need_col_b, bool trans_a, bool trans_b) {
  constexpr bool kAWide = QMatMulAPlanes<TA>::kWide;

  const int64_t batch = blockIdx.z;
  A += batch * M * K;
  B += batch * b_batch_stride;
  Y += batch * M * N;

  const int64_t m_base = static_cast<int64_t>(blockIdx.y) * kTileM;
  const int64_t n_base = static_cast<int64_t>(blockIdx.x) * kTileN;

  const auto mt = qmatmul_accumulate_k_range<TA, TB>(
      A, B, M, N, K, m_base, n_base, 0, K, need_row_a, need_col_b, trans_a,
      trans_b, b_bits, b_signed, b_zero_points);

  int thread_m, thread_n;
  qmatmul_micro_tile_origin(threadIdx.x, thread_m, thread_n);

#pragma unroll
  for (int i = 0; i < kThreadM; ++i) {
    const int64_t m = m_base + thread_m + i;
    if (m >= M)
      continue;
    int64_t row_a64;
    if constexpr (kAWide)
      row_a64 = int64_t{256} * mt.row_a[kPlaneHi][i] + mt.row_a[kPlaneLo][i];
    else
      row_a64 = mt.row_a[kPlaneHi][i];
#pragma unroll
    for (int j = 0; j < kThreadN; ++j) {
      const int64_t n = n_base + thread_n + j;
      if (n >= N)
        continue;
      int64_t acc64;
      if constexpr (kAWide)
        acc64 = int64_t{256} * mt.acc[kPlaneHi][i][j] + mt.acc[kPlaneLo][i][j];
      else
        acc64 = mt.acc[kPlaneHi][i][j];
      Y[m * N + n] =
          qmatmul_epilogue<TY>(acc64, row_a64, mt.col_b[j], n, requant);
    }
  }
}

//===----------------------------------------------------------------------===//
// Split-K (low output-parallelism shapes)
//===----------------------------------------------------------------------===//
//
// The tiled grid has no K dimension, so its parallelism is bounded by the
// output alone: ceil(N/kTileN) * ceil(M/kTileM) * batch_count blocks. A
// low-rank shape such as M=128, N=32 therefore launches two blocks, leaves the
// rest of the device idle, and makes each block walk the entire reduction.
//
// Split-K restores the missing dimension: each block reduces a k slice into a
// global int64 accumulator, and a second kernel requantizes once every slice
// has landed. Integer addition is associative and exact, so the atomics make
// the result bit-identical to the single-pass path regardless of the order the
// slices complete in -- there is no reduction-order drift to reason about.

__device__ inline void qmatmul_atomic_add_i64(int64_t *addr, int64_t v) {
  atomicAdd(reinterpret_cast<unsigned long long *>(addr),
            static_cast<unsigned long long>(v));
}

// Split-K partial pass: one block per (n-tile, m-tile, batch x split).
template <typename TA, typename TB>
__global__ __launch_bounds__(kTiledBlock) void qmatmul_splitk_partial_kernel(
    const TA *__restrict__ A, const TB *__restrict__ B,
    const uint8_t *__restrict__ b_zero_points,
    int64_t *__restrict__ acc_ws, int64_t *__restrict__ row_a_ws,
    int64_t *__restrict__ col_b_ws, int64_t M, int64_t N, int64_t K,
    int64_t b_batch_stride, int splits, int64_t k_chunk, bool need_row_a,
    bool need_col_b, bool trans_a, bool trans_b, int b_bits, bool b_signed) {
  constexpr bool kAWide = QMatMulAPlanes<TA>::kWide;

  const int64_t split = blockIdx.z % splits;
  const int64_t batch = blockIdx.z / splits;

  const int64_t k_begin = split * k_chunk;
  if (k_begin >= K)
    return;
  const int64_t k_end = (k_begin + k_chunk < K) ? (k_begin + k_chunk) : K;

  A += batch * M * K;
  B += batch * b_batch_stride;
  acc_ws += batch * M * N;
  row_a_ws += batch * M;
  col_b_ws += batch * N;

  const int64_t m_base = static_cast<int64_t>(blockIdx.y) * kTileM;
  const int64_t n_base = static_cast<int64_t>(blockIdx.x) * kTileN;

  // rowA does not depend on n and colB does not depend on m, so every block
  // column would otherwise add the same rowA slice again (and every block row
  // the same colB slice). Elect one of each; the flags are block-uniform, so
  // the non-electing blocks skip the corresponding dots entirely.
  const bool add_row_a = need_row_a && blockIdx.x == 0;
  const bool add_col_b = need_col_b && blockIdx.y == 0;

  const auto mt = qmatmul_accumulate_k_range<TA, TB>(
      A, B, M, N, K, m_base, n_base, k_begin, k_end, add_row_a, add_col_b,
      trans_a, trans_b, b_bits, b_signed, b_zero_points);

  int thread_m, thread_n;
  qmatmul_micro_tile_origin(threadIdx.x, thread_m, thread_n);

#pragma unroll
  for (int i = 0; i < kThreadM; ++i) {
    const int64_t m = m_base + thread_m + i;
    if (m >= M)
      continue;
    // Every thread sharing this m holds the same rowA slice, so within the
    // elected block only the first micro-tile column commits it.
    if (add_row_a && thread_n == 0) {
      int64_t row_a64;
      if constexpr (kAWide)
        row_a64 = int64_t{256} * mt.row_a[kPlaneHi][i] + mt.row_a[kPlaneLo][i];
      else
        row_a64 = mt.row_a[kPlaneHi][i];
      qmatmul_atomic_add_i64(&row_a_ws[m], row_a64);
    }
#pragma unroll
    for (int j = 0; j < kThreadN; ++j) {
      const int64_t n = n_base + thread_n + j;
      if (n >= N)
        continue;
      int64_t acc64;
      if constexpr (kAWide)
        acc64 = int64_t{256} * mt.acc[kPlaneHi][i][j] + mt.acc[kPlaneLo][i][j];
      else
        acc64 = mt.acc[kPlaneHi][i][j];
      qmatmul_atomic_add_i64(&acc_ws[m * N + n], acc64);
    }
  }

  // Mirror of the rowA election: only the first micro-tile row commits colB.
  if (add_col_b && thread_m == 0) {
#pragma unroll
    for (int j = 0; j < kThreadN; ++j) {
      const int64_t n = n_base + thread_n + j;
      if (n < N)
        qmatmul_atomic_add_i64(&col_b_ws[n], mt.col_b[j]);
    }
  }
}

constexpr int kSplitKEpilogueBlock = 256;

// Requantizes the completed split-K sums, one thread per output element.
//
// rowA / colB are read unconditionally. When their zero point is zero the
// partial pass never wrote them, but the workspace was zeroed beforehand and
// the epilogue multiplies them by that same zero point, so the branch would
// buy nothing.
template <typename TY>
__global__ __launch_bounds__(kSplitKEpilogueBlock) void
qmatmul_splitk_epilogue_kernel(const int64_t *__restrict__ acc_ws,
                               const int64_t *__restrict__ row_a_ws,
                               const int64_t *__restrict__ col_b_ws,
                               TY *__restrict__ Y, int64_t M, int64_t N,
                               int64_t total, QMatMulRequant requant) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * kSplitKEpilogueBlock + threadIdx.x;
  if (idx >= total)
    return;

  // acc_ws and Y share the [batch, M, N] row-major order, so one flat index
  // addresses both; only the two correction vectors need unpacking.
  const int64_t mn = M * N;
  const int64_t batch = idx / mn;
  const int64_t rem = idx - batch * mn;
  const int64_t m = rem / N;
  const int64_t n = rem - m * N;

  Y[idx] =
      qmatmul_epilogue<TY>(acc_ws[idx], row_a_ws[batch * M + m],
                           col_b_ws[batch * N + n], n, requant);
}

// Slice ceiling. Past it the per-slice staging prologue and the atomic traffic
// grow faster than the added parallelism pays back.
constexpr int kSplitKMaxSplits = 32;
// A slice still has to cover enough k-tiles to amortize that prologue and the
// atomic it ends with.
constexpr int64_t kSplitKMinKPerSplit = 4 * kTileK;
// Blocks per CU the plain grid must already reach for split-K to be pointless.
constexpr int kSplitKTargetBlocksPerCU = 2;

// Compute-unit count of the currently selected device, cached per device
// ordinal for the same reason hipdnn_device_wave_size caches it: one host may
// drive several GPUs of different widths. 0 means "could not probe", which
// disables split-K rather than planning against a guessed width.
int qmatmul_device_cu_count() {
  constexpr int kMaxCachedDevices = 16;
  // Zero-initialized (0 == not probed yet), so no thread-safe-statics guard is
  // emitted; concurrent probes race only to write the same value.
  static int cached[kMaxCachedDevices] = {};

  auto probe = [](int d) {
    hipDeviceProp_t p;
    return (hipGetDeviceProperties(&p, d) == hipSuccess) ? p.multiProcessorCount
                                                         : 0;
  };

  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess || dev < 0)
    return 0;
  if (dev >= kMaxCachedDevices)
    return probe(dev);

  int c = cached[dev];
  if (c == 0) {
    c = probe(dev);
    cached[dev] = c;
  }
  return c;
}

struct QMatMulSplitKPlan {
  bool enabled;
  int splits;
  int64_t k_chunk;
};

QMatMulSplitKPlan qmatmul_plan_splitk(int64_t M, int64_t N, int64_t K,
                                      int64_t batch_count) {
  QMatMulSplitKPlan plan{false, 1, K};
  // M == 1 takes the GEMV kernel, which already parallelizes over N.
  if (M <= 1)
    return plan;

  const int cus = qmatmul_device_cu_count();
  if (cus <= 0)
    return plan;

  const int64_t base_blocks = ((N + kTileN - 1) / kTileN) *
                              ((M + kTileM - 1) / kTileM) * batch_count;
  const int64_t target = static_cast<int64_t>(cus) * kSplitKTargetBlocksPerCU;
  if (base_blocks >= target)
    return plan;

  int64_t want = (target + base_blocks - 1) / base_blocks;
  if (want > kSplitKMaxSplits)
    want = kSplitKMaxSplits;
  const int64_t k_cap = K / kSplitKMinKPerSplit;
  if (want > k_cap)
    want = k_cap;
  if (want < 2)
    return plan;

  // Round the slice up to a whole k-tile so each one starts on the boundary
  // the staging maps assume, then recompute the count so the last slice is not
  // empty.
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

size_t qmatmul_splitk_workspace_bytes(int64_t M, int64_t N,
                                      int64_t batch_count) {
  return static_cast<size_t>(batch_count) * (M * N + M + N) * sizeof(int64_t);
}

//===----------------------------------------------------------------------===//
// GEMV (M == 1, the LLM decode shape)
//===----------------------------------------------------------------------===//

// A thread owns kCols consecutive columns, so a [K, N] 8-bit B hands it whole
// dwords per k step instead of a single byte, and kUnroll steps are staged in
// registers before any multiply. Reading one byte per step made the loop wait on
// every load, which cost a full memory round trip per k.
//
// kSplits slices of the reduction then run side by side in one block and merge
// through LDS. The output alone bounds this shape's parallelism -- a narrow N
// reaches only a handful of CUs however the columns are divided -- so k is the
// only dimension left that can raise the requests in flight.
//
// kBlock * kCols is the block's column window and therefore sets the grid:
// widening a thread's register work without widening that window would shrink
// the grid and idle CUs on a narrow N.
//
// The four are template parameters because their balance is a per-shape
// property; qmatmul_gemv_config measures it on the device.
constexpr int kGemvTileK = 256;

template <int kCols> struct QMatMulGemvSums {
  int64_t acc[kCols];
  int64_t col_b[kCols];
  int64_t row_a;
};

// Slices carry a uniform tile count so the block's barriers stay aligned: a
// slice whose range ran out just walks empty tiles instead of leaving the loop
// early.
struct QMatMulGemvSlice {
  int64_t k_begin;
  int64_t k_end;
  int64_t tiles;
};

template <int kSplits>
__device__ inline QMatMulGemvSlice qmatmul_gemv_slice(int64_t K, int slice) {
  int64_t chunk = (K + kSplits - 1) / kSplits;
  chunk = (chunk + kGemvTileK - 1) / kGemvTileK * kGemvTileK;
  const int64_t begin = static_cast<int64_t>(slice) * chunk;
  const int64_t capped = (begin + chunk < K) ? begin + chunk : K;
  return {begin, capped < begin ? begin : capped, chunk / kGemvTileK};
}

template <typename TB>
__device__ inline int32_t qmatmul_gemv_byte(uint32_t packed, int lane) {
  const uint32_t raw = (packed >> (lane * 8)) & 0xFFu;
  if constexpr (QMatMulTraits<TB>::kSigned)
    return static_cast<int32_t>(static_cast<int8_t>(raw));
  else
    return static_cast<int32_t>(raw);
}

// kCols bytes of one B row. Held as dwords so the staging loads stay plain
// aligned dword reads that the vectorizer can fold into dwordx2 / dwordx4.
template <int kWords> struct QMatMulGemvPack {
  uint32_t w[kWords];
};

template <typename TB, int kCols>
__device__ inline void
qmatmul_gemv_step(int32_t a_val, const QMatMulGemvPack<kCols / 4> &packed,
                  bool need_row_a, bool need_col_b,
                  QMatMulGemvSums<kCols> &sums) {
  if (need_row_a)
    sums.row_a += a_val;
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    const int32_t b_val = qmatmul_gemv_byte<TB>(packed.w[c / 4], c % 4);
    sums.acc[c] += a_val * b_val;
    if (need_col_b)
      sums.col_b[c] += b_val;
  }
}

__device__ inline int qmatmul_gemv_tile(int64_t k_end, int64_t k0) {
  if (k0 >= k_end)
    return 0;
  const int64_t remaining = k_end - k0;
  return static_cast<int>(remaining < kGemvTileK ? remaining : kGemvTileK);
}

template <int kBlock, typename TA>
__device__ inline void qmatmul_gemv_stage_a(const TA *__restrict__ A,
                                            int64_t k0, int tile,
                                            int32_t *a_lds) {
  for (int t = threadIdx.x; t < tile; t += kBlock)
    a_lds[t] = static_cast<int32_t>(A[k0 + t]);
}

template <int kBlock, int kCols, int kUnroll, typename TA, typename TB>
__device__ void qmatmul_gemv_vector_loop(const TA *__restrict__ A,
                                         const TB *__restrict__ B, int64_t N,
                                         const QMatMulGemvSlice &slice,
                                         int64_t n_base, bool need_row_a,
                                         bool need_col_b, int32_t *a_lds,
                                         QMatMulGemvSums<kCols> &sums) {
  constexpr int kWords = kCols / 4;
  const uint8_t *b_col = reinterpret_cast<const uint8_t *>(B) + n_base;

  for (int64_t ti = 0; ti < slice.tiles; ++ti) {
    const int64_t k0 = slice.k_begin + ti * kGemvTileK;
    const int tile = qmatmul_gemv_tile(slice.k_end, k0);
    qmatmul_gemv_stage_a<kBlock>(A, k0, tile, a_lds);
    __syncthreads();

    const uint8_t *b_row = b_col + k0 * N;
    int t = 0;
    for (; t + kUnroll <= tile; t += kUnroll) {
      QMatMulGemvPack<kWords> staged[kUnroll];
#pragma unroll
      for (int u = 0; u < kUnroll; ++u) {
        const uint32_t *row = reinterpret_cast<const uint32_t *>(b_row);
#pragma unroll
        for (int w = 0; w < kWords; ++w)
          staged[u].w[w] = row[w];
        b_row += N;
      }
#pragma unroll
      for (int u = 0; u < kUnroll; ++u)
        qmatmul_gemv_step<TB, kCols>(a_lds[t + u], staged[u], need_row_a,
                                     need_col_b, sums);
    }
    for (; t < tile; ++t) {
      QMatMulGemvPack<kWords> packed;
      const uint32_t *row = reinterpret_cast<const uint32_t *>(b_row);
#pragma unroll
      for (int w = 0; w < kWords; ++w)
        packed.w[w] = row[w];
      b_row += N;
      qmatmul_gemv_step<TB, kCols>(a_lds[t], packed, need_row_a, need_col_b,
                                   sums);
    }

    __syncthreads();
  }
}

// Fallback for the layouts the dword load cannot serve: a [N, K] B, packed W4,
// and a last block that runs off the end of N.
template <int kBBits, bool kTransB, int kBlock, int kCols, typename TA,
          typename TB>
__device__ void
qmatmul_gemv_scalar_loop(const TA *__restrict__ A, const TB *__restrict__ B,
                         const uint8_t *__restrict__ b_zero_points, int64_t N,
                         int64_t K, const QMatMulGemvSlice &slice,
                         int64_t n_base, bool need_row_a, bool need_col_b,
                         bool b_signed, int32_t *a_lds,
                         QMatMulGemvSums<kCols> &sums) {
  int column_zp[kCols] = {};
  if constexpr (kBBits == 4) {
#pragma unroll
    for (int c = 0; c < kCols; ++c) {
      const int64_t n = n_base + c;
      if (n < N)
        column_zp[c] = qmatmul_read_quant(b_zero_points, n, kBBits, b_signed);
    }
  }

  for (int64_t ti = 0; ti < slice.tiles; ++ti) {
    const int64_t k0 = slice.k_begin + ti * kGemvTileK;
    const int tile = qmatmul_gemv_tile(slice.k_end, k0);
    qmatmul_gemv_stage_a<kBlock>(A, k0, tile, a_lds);
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
        int32_t b_val;
        if constexpr (kBBits == 4)
          b_val = qmatmul_read_quant(reinterpret_cast<const uint8_t *>(B),
                                     index, kBBits, b_signed) -
                  column_zp[c];
        else
          b_val = static_cast<int32_t>(B[index]);
        sums.acc[c] += a_val * b_val;
        if (need_col_b)
          sums.col_b[c] += b_val;
      }
    }

    __syncthreads();
  }
}

// Y[b, 0, n] = requantize(sum_k A[b, 0, k] * B[b, k, n])
//
// A is identical across threads, so it is staged in LDS once per k-tile instead
// of being re-read per column. Each k slice needs a different tile at the same
// time, hence one staging buffer per slice.
//
// A tiled GEMM degenerates badly at M == 1 (63 of 64 rows in each block tile
// are padding), and this shape dominates LLM decode, so it gets its own path.
//
// There is no trans_a flag: at M == 1 both layouts of A are the same K-element
// vector, so the index is unchanged.
template <typename TA, typename TB, typename TY, int kBlock, int kCols,
          int kUnroll, int kSplits>
__global__ __launch_bounds__(kBlock *kSplits) void qmatmul_gemv_kernel(
    const TA *__restrict__ A, const TB *__restrict__ B, TY *__restrict__ Y,
    const uint8_t *__restrict__ b_zero_points, int64_t N, int64_t K,
    int64_t b_batch_stride, QMatMulRequant requant, int b_bits, bool b_signed,
    bool need_row_a, bool need_col_b, bool trans_b) {
  constexpr int kWindow = kBlock * kCols;
  static_assert(kCols % 4 == 0, "the vector path stages whole dwords");
  static_assert(kGemvTileK % kBlock == 0, "A staging map assumes even split");
  static_assert(kBlock * kSplits <= 1024, "block exceeds the HIP thread limit");
  static_assert(kSplits * kGemvTileK * sizeof(int32_t) +
                        2 * kSplits * kWindow * sizeof(int64_t) +
                        kSplits * sizeof(int64_t) <=
                    64 * 1024,
                "geometry exceeds the per-block LDS budget");

  __shared__ int32_t a_lds[kSplits][kGemvTileK];
  __shared__ int64_t red_acc[kSplits][kWindow];
  __shared__ int64_t red_col_b[kSplits][kWindow];
  __shared__ int64_t red_row_a[kSplits];

  const int lane = threadIdx.x;
  const int slice_id = threadIdx.y;

  const int64_t batch = blockIdx.y;
  A += batch * K;
  B += batch * b_batch_stride;
  Y += batch * N;

  const int64_t block_base = static_cast<int64_t>(blockIdx.x) * kWindow;
  const int64_t n_base = block_base + static_cast<int64_t>(lane) * kCols;

  const QMatMulGemvSlice slice = qmatmul_gemv_slice<kSplits>(K, slice_id);
  QMatMulGemvSums<kCols> sums = {};

  // Block-uniform by construction: the loops below carry __syncthreads(), so a
  // per-thread choice here would hang the block. The staged load also needs
  // every lane in range and every B row aligned to its width, or it would
  // straddle the end of a row.
  const bool vector_ok = !trans_b && b_bits == 8 && block_base + kWindow <= N &&
                         (N % kCols) == 0 &&
                         (reinterpret_cast<uintptr_t>(B) % kCols) == 0;

  if (vector_ok)
    qmatmul_gemv_vector_loop<kBlock, kCols, kUnroll, TA, TB>(
        A, B, N, slice, n_base, need_row_a, need_col_b, a_lds[slice_id], sums);
  else if (b_bits == 4 && trans_b)
    qmatmul_gemv_scalar_loop<4, true, kBlock, kCols>(
        A, B, b_zero_points, N, K, slice, n_base, need_row_a, need_col_b,
        b_signed, a_lds[slice_id], sums);
  else if (b_bits == 4)
    qmatmul_gemv_scalar_loop<4, false, kBlock, kCols>(
        A, B, b_zero_points, N, K, slice, n_base, need_row_a, need_col_b,
        b_signed, a_lds[slice_id], sums);
  else if (trans_b)
    qmatmul_gemv_scalar_loop<8, true, kBlock, kCols>(
        A, B, b_zero_points, N, K, slice, n_base, need_row_a, need_col_b,
        b_signed, a_lds[slice_id], sums);
  else
    qmatmul_gemv_scalar_loop<8, false, kBlock, kCols>(
        A, B, b_zero_points, N, K, slice, n_base, need_row_a, need_col_b,
        b_signed, a_lds[slice_id], sums);

  // Column-major so the lanes of a wave land on consecutive int64 slots.
#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    const int slot = c * kBlock + lane;
    red_acc[slice_id][slot] = sums.acc[c];
    red_col_b[slice_id][slot] = sums.col_b[c];
  }
  // rowA does not depend on n, so every lane of a slice holds the same value.
  if (lane == 0)
    red_row_a[slice_id] = sums.row_a;
  __syncthreads();

  if (slice_id != 0)
    return;

  int64_t row_a = 0;
#pragma unroll
  for (int s = 0; s < kSplits; ++s)
    row_a += red_row_a[s];

#pragma unroll
  for (int c = 0; c < kCols; ++c) {
    const int64_t n = n_base + c;
    if (n >= N)
      continue;
    const int slot = c * kBlock + lane;
    int64_t acc = 0;
    int64_t col_b = 0;
#pragma unroll
    for (int s = 0; s < kSplits; ++s) {
      acc += red_acc[s][slot];
      col_b += red_col_b[s][slot];
    }
    Y[n] = qmatmul_epilogue<TY>(acc, row_a, col_b, n, requant);
  }
}

//===----------------------------------------------------------------------===//
// Host launch and dtype dispatch
//===----------------------------------------------------------------------===//

// Bundled so the three dispatch levels below stay readable: A has 4 element
// types, B has 2 and Y has 4, and threading a dozen scalars through 32 switch
// arms by hand buries the one thing each arm actually says.
struct QMatMulArgs {
  hipStream_t stream;
  const void *A;
  const void *B;
  void *Y;
  const float *b_scales;
  const uint8_t *b_zero_points;
  int64_t M;
  int64_t N;
  int64_t K;
  int64_t batch_count;
  int64_t b_batch_stride;
  bool trans_a;
  bool trans_b;
  int b_bits;
  bool b_signed;
  float m_scale;
  float ay_ratio;
  int32_t a_zp;
  int32_t b_zp;
  int32_t y_zp;
  void *workspace;
  size_t workspace_bytes;
};

//===----------------------------------------------------------------------===//
// GEMV geometry autotune
//===----------------------------------------------------------------------===//

// Entries must stay in lockstep with the switch in
// qmatmul_gemv_launch_config; a config's id is its index here.
//
// cols stops at 8 on the evidence recorded beside kGemvConfigs in
// matmul_nbits_kernel.hip: over ~50 tunes there, a wider per-thread column tile
// spilled registers and never won.
struct QMatMulGemvConfig {
  int block;
  int cols;
  int unroll;
  int splits;
};

constexpr QMatMulGemvConfig kQMatMulGemvConfigs[] = {
    {64, 4, 8, 4},  {64, 4, 8, 1},  {64, 4, 8, 2},   {64, 4, 8, 8},
    {64, 4, 16, 4}, {64, 4, 4, 4},  {64, 8, 8, 2},   {64, 8, 8, 4},
    {32, 4, 8, 4},  {32, 4, 8, 8},  {32, 4, 16, 8},  {128, 4, 8, 2},
};
constexpr int kNumQMatMulGemvConfigs = static_cast<int>(
    sizeof(kQMatMulGemvConfigs) / sizeof(kQMatMulGemvConfigs[0]));

// The answer whenever autotune is off or no config in the table would launch.
constexpr int kQMatMulGemvDefaultConfig = 0;

template <typename TA, typename TB, typename TY, int kBlock, int kCols,
          int kUnroll, int kSplits>
int qmatmul_gemv_launch(const QMatMulArgs &g, const QMatMulRequant &requant,
                        bool need_row_a, bool need_col_b) {
  constexpr int kWindow = kBlock * kCols;
  dim3 grid(static_cast<unsigned>((g.N + kWindow - 1) / kWindow),
            static_cast<unsigned>(g.batch_count));
  (void)hipGetLastError();
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
          qmatmul_gemv_kernel<TA, TB, TY, kBlock, kCols, kUnroll, kSplits>),
      grid, dim3(kBlock, kSplits), 0, g.stream, static_cast<const TA *>(g.A),
      static_cast<const TB *>(g.B), static_cast<TY *>(g.Y), g.b_zero_points,
      g.N, g.K, g.b_batch_stride, requant, g.b_bits, g.b_signed, need_row_a,
      need_col_b, g.trans_b);
  return static_cast<int>(hipGetLastError());
}

template <typename TA, typename TB, typename TY>
int qmatmul_gemv_launch_config(int config_id, const QMatMulArgs &g,
                               const QMatMulRequant &requant, bool need_row_a,
                               bool need_col_b) {
#define QMATMUL_GEMV_CASE(ID, BLOCK, COLS, UNROLL, SPLITS)                     \
  case ID:                                                                     \
    return qmatmul_gemv_launch<TA, TB, TY, BLOCK, COLS, UNROLL, SPLITS>(       \
        g, requant, need_row_a, need_col_b)

  switch (config_id) {
    QMATMUL_GEMV_CASE(0, 64, 4, 8, 4);
    QMATMUL_GEMV_CASE(1, 64, 4, 8, 1);
    QMATMUL_GEMV_CASE(2, 64, 4, 8, 2);
    QMATMUL_GEMV_CASE(3, 64, 4, 8, 8);
    QMATMUL_GEMV_CASE(4, 64, 4, 16, 4);
    QMATMUL_GEMV_CASE(5, 64, 4, 4, 4);
    QMATMUL_GEMV_CASE(6, 64, 8, 8, 2);
    QMATMUL_GEMV_CASE(7, 64, 8, 8, 4);
    QMATMUL_GEMV_CASE(8, 32, 4, 8, 4);
    QMATMUL_GEMV_CASE(9, 32, 4, 8, 8);
    QMATMUL_GEMV_CASE(10, 32, 4, 16, 8);
    QMATMUL_GEMV_CASE(11, 128, 4, 8, 2);
  default:
    return qmatmul_gemv_launch<TA, TB, TY, 64, 4, 8, 4>(g, requant, need_row_a,
                                                        need_col_b);
  }
#undef QMATMUL_GEMV_CASE
}

// Read once and latched: the tuner sits on the decode dispatch path, where a
// getenv per call would serialize every token through the CRT environment lock.
bool qmatmul_gemv_autotune_enabled() {
  static const bool enabled = [] {
    std::string raw;
#ifdef _WIN32
    char *v = nullptr;
    size_t len = 0;
    if (_dupenv_s(&v, &len, "HIPDNN_QMATMUL_GEMV_AUTOTUNE") == 0 && v) {
      raw = v;
      free(v);
    }
#else
    if (const char *v = getenv("HIPDNN_QMATMUL_GEMV_AUTOTUNE"))
      raw = v;
#endif
    return !(raw == "0" || raw == "off" || raw == "OFF");
  }();
  return enabled;
}

// Distinct per element type: the four candidates differ in either width or
// signedness.
template <typename T> constexpr int qmatmul_dtype_tag() {
  return static_cast<int>(sizeof(T)) * 2 + (QMatMulTraits<T>::kSigned ? 1 : 0);
}

// Everything that moves the per-config ranking: the shape, the types the kernel
// was instantiated for, and the layout and correction terms that decide how
// much work one k step carries.
std::string qmatmul_gemv_cache_key(const QMatMulArgs &g, int dtypes,
                                   bool need_row_a, bool need_col_b) {
  std::string key = std::to_string(g.N);
  key += '_';
  key += std::to_string(g.K);
  key += '_';
  key += std::to_string(g.batch_count);
  key += '_';
  key += std::to_string(dtypes);
  key += '_';
  key += std::to_string(g.b_bits);
  key += g.trans_b ? 't' : 'n';
  key += need_row_a ? 'r' : '.';
  key += need_col_b ? 'c' : '.';
  return key;
}

// Sweeping over the live output buffer is safe because every config writes the
// same Y for every n, and the caller relaunches the winner afterwards.
template <typename TA, typename TB, typename TY>
int qmatmul_gemv_tune(const QMatMulArgs &g, const QMatMulRequant &requant,
                      bool need_row_a, bool need_col_b) {
  constexpr int kIters = 5;

  hipEvent_t start = nullptr;
  hipEvent_t stop = nullptr;
  if (hipEventCreate(&start) != hipSuccess)
    return kQMatMulGemvDefaultConfig;
  if (hipEventCreate(&stop) != hipSuccess) {
    (void)hipEventDestroy(start);
    return kQMatMulGemvDefaultConfig;
  }

  int best = kQMatMulGemvDefaultConfig;
  float best_ms = 0.0f;
  bool have_best = false;

  for (int cid = 0; cid < kNumQMatMulGemvConfigs; ++cid) {
    // The warmup doubles as the probe: a config this device refuses drops out
    // instead of contributing a bogus timing.
    bool ok = qmatmul_gemv_launch_config<TA, TB, TY>(
                  cid, g, requant, need_row_a, need_col_b) == hipSuccess;
    if (!ok || hipEventRecord(start, g.stream) != hipSuccess)
      continue;

    for (int i = 0; i < kIters && ok; ++i)
      ok = qmatmul_gemv_launch_config<TA, TB, TY>(cid, g, requant, need_row_a,
                                                  need_col_b) == hipSuccess;
    if (!ok || hipEventRecord(stop, g.stream) != hipSuccess ||
        hipEventSynchronize(stop) != hipSuccess)
      continue;

    float ms = 0.0f;
    if (hipEventElapsedTime(&ms, start, stop) != hipSuccess)
      continue;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] qmatmul gemv config[%2d] block=%3d cols=%d "
        "unroll=%2d splits=%d : %.4f ms\n",
        cid, kQMatMulGemvConfigs[cid].block, kQMatMulGemvConfigs[cid].cols,
        kQMatMulGemvConfigs[cid].unroll, kQMatMulGemvConfigs[cid].splits,
        ms / kIters);

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

std::unordered_map<std::string, int> g_qmatmul_gemv_cache;
std::mutex g_qmatmul_gemv_cache_mutex;

template <typename TA, typename TB, typename TY>
int qmatmul_gemv_config(const QMatMulArgs &g, const QMatMulRequant &requant,
                        bool need_row_a, bool need_col_b) {
  if (!qmatmul_gemv_autotune_enabled())
    return kQMatMulGemvDefaultConfig;

  constexpr int kDtypes = qmatmul_dtype_tag<TA>() * 100 +
                          qmatmul_dtype_tag<TB>() * 10 +
                          qmatmul_dtype_tag<TY>();
  const std::string key =
      qmatmul_gemv_cache_key(g, kDtypes, need_row_a, need_col_b);

  std::lock_guard<std::mutex> lock(g_qmatmul_gemv_cache_mutex);
  const auto it = g_qmatmul_gemv_cache.find(key);
  if (it != g_qmatmul_gemv_cache.end())
    return it->second;

  const int best =
      qmatmul_gemv_tune<TA, TB, TY>(g, requant, need_row_a, need_col_b);
  g_qmatmul_gemv_cache[key] = best;

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] qmatmul gemv tune \"%s\" -> config[%d]\n", key.c_str(),
      best);
  return best;
}

// Picks between the three paths: GEMV at M == 1, then split-K when the plan
// asks for it and the caller supplied enough scratch, otherwise tiled.
template <typename TA, typename TB, typename TY>
int launch_qmatmul(const QMatMulArgs &g) {
  const bool per_column = g.b_scales != nullptr;
  const QMatMulRequant requant{
      g.m_scale,
      g.ay_ratio,
      g.b_scales,
      g.b_zero_points,
      g.a_zp,
      per_column ? 0 : g.b_zp,
      g.y_zp,
      g.b_bits,
      g.b_signed,
      g.K,
      per_column ? 0 : static_cast<int64_t>(g.K) * g.a_zp * g.b_zp};
  const bool need_row_a = per_column ? g.b_bits == 8 : g.b_zp != 0;
  const bool need_col_b = g.a_zp != 0;

  const auto *a_ptr = static_cast<const TA *>(g.A);
  const auto *b_ptr = static_cast<const TB *>(g.B);
  auto *y_ptr = static_cast<TY *>(g.Y);

  if (g.M == 1) {
    const int config =
        qmatmul_gemv_config<TA, TB, TY>(g, requant, need_row_a, need_col_b);
    return qmatmul_gemv_launch_config<TA, TB, TY>(config, g, requant,
                                                  need_row_a, need_col_b);
  }

  const QMatMulSplitKPlan plan =
      qmatmul_plan_splitk(g.M, g.N, g.K, g.batch_count);
  const size_t ws_needed =
      plan.enabled ? qmatmul_splitk_workspace_bytes(g.M, g.N, g.batch_count)
                   : 0;
  bool use_splitk =
      plan.enabled && g.workspace && g.workspace_bytes >= ws_needed;

  if (use_splitk &&
      hipMemsetAsync(g.workspace, 0, ws_needed, g.stream) != hipSuccess) {
    // Half-zeroed accumulators would corrupt the result, so drop to the
    // single-pass kernel, which needs no scratch. Clear the sticky error too,
    // or the fallback's own clean launch would still be reported as a failure.
    (void)hipGetLastError();
    use_splitk = false;
  }

  if (use_splitk) {
    auto *acc_ws = static_cast<int64_t *>(g.workspace);
    auto *row_a_ws = acc_ws + g.batch_count * g.M * g.N;
    auto *col_b_ws = row_a_ws + g.batch_count * g.M;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_qmatmul: split-K M=%lld N=%lld K=%lld "
        "splits=%d k_chunk=%lld scratch=%zu B\n",
        (long long)g.M, (long long)g.N, (long long)g.K, plan.splits,
        (long long)plan.k_chunk, ws_needed);

    // grid.z stays small: split-K only engages when the plain grid is under
    // kSplitKTargetBlocksPerCU blocks/CU, which bounds batch_count by that
    // same figure, so batch_count * splits cannot approach the 65535 limit.
    dim3 grid(static_cast<unsigned>((g.N + kTileN - 1) / kTileN),
              static_cast<unsigned>((g.M + kTileM - 1) / kTileM),
              static_cast<unsigned>(g.batch_count * plan.splits));
    (void)hipGetLastError();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(qmatmul_splitk_partial_kernel<TA, TB>),
                       grid, dim3(kTiledBlock), 0, g.stream, a_ptr, b_ptr,
                       g.b_zero_points, acc_ws, row_a_ws, col_b_ws, g.M, g.N,
                       g.K, g.b_batch_stride, plan.splits, plan.k_chunk,
                       need_row_a, need_col_b, g.trans_a, g.trans_b, g.b_bits,
                       g.b_signed);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess)
      return static_cast<int>(err);

    const int64_t total = g.batch_count * g.M * g.N;
    (void)hipGetLastError();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(qmatmul_splitk_epilogue_kernel<TY>),
        dim3(static_cast<unsigned>((total + kSplitKEpilogueBlock - 1) /
                                   kSplitKEpilogueBlock)),
        dim3(kSplitKEpilogueBlock), 0, g.stream, acc_ws, row_a_ws, col_b_ws,
        y_ptr, g.M, g.N, total, requant);
    return static_cast<int>(hipGetLastError());
  }

  dim3 grid(static_cast<unsigned>((g.N + kTileN - 1) / kTileN),
            static_cast<unsigned>((g.M + kTileM - 1) / kTileM),
            static_cast<unsigned>(g.batch_count));
  (void)hipGetLastError();
  hipLaunchKernelGGL(HIP_KERNEL_NAME(qmatmul_tiled_kernel<TA, TB, TY>), grid,
                     dim3(kTiledBlock), 0, g.stream, a_ptr, b_ptr, y_ptr,
                     g.b_zero_points, g.M, g.N, g.K, g.b_batch_stride, requant,
                     g.b_bits, g.b_signed, need_row_a, need_col_b, g.trans_a,
                     g.trans_b);
  return static_cast<int>(hipGetLastError());
}

// Y is 8- or 16-bit: a 16-bit activation is usually requantized back to 16-bit
// to keep the precision it was chosen for, but an 8-bit output is legal too.
template <typename TA, typename TB>
int dispatch_y(int y_dtype, const QMatMulArgs &args) {
  switch (y_dtype) {
  case HIP_DTYPE_INT8:
    return launch_qmatmul<TA, TB, int8_t>(args);
  case HIP_DTYPE_UINT8:
    return launch_qmatmul<TA, TB, uint8_t>(args);
  case HIP_DTYPE_INT16:
    return launch_qmatmul<TA, TB, int16_t>(args);
  case HIP_DTYPE_UINT16:
    return launch_qmatmul<TA, TB, uint16_t>(args);
  default:
    return -1;
  }
}

// B stays 8-bit because only A is byte-split, leaving a 16-bit weight no path
// through the dot4 pipeline. Splitting B the same way would work -- four
// plane products instead of two, with the widest still bounded by 255*255 --
// but is not implemented: 16-bit integer weights save neither space nor
// bandwidth over fp16, which has hardware matrix instructions.
template <typename TA>
int dispatch_b(int b_dtype, int y_dtype, const QMatMulArgs &args) {
  if (args.b_bits == 4)
    return dispatch_y<TA, int8_t>(y_dtype, args);

  switch (b_dtype) {
  case HIP_DTYPE_INT8:
    return dispatch_y<TA, int8_t>(y_dtype, args);
  case HIP_DTYPE_UINT8:
    return dispatch_y<TA, uint8_t>(y_dtype, args);
  default:
    return -1;
  }
}

int dispatch_a(int a_dtype, int b_dtype, int y_dtype, const QMatMulArgs &args) {
  switch (a_dtype) {
  case HIP_DTYPE_INT8:
    return dispatch_b<int8_t>(b_dtype, y_dtype, args);
  case HIP_DTYPE_UINT8:
    return dispatch_b<uint8_t>(b_dtype, y_dtype, args);
  case HIP_DTYPE_INT16:
    return dispatch_b<int16_t>(b_dtype, y_dtype, args);
  case HIP_DTYPE_UINT16:
    return dispatch_b<uint16_t>(b_dtype, y_dtype, args);
  default:
    return -1;
  }
}

} // namespace

//===----------------------------------------------------------------------===//
// C API
//===----------------------------------------------------------------------===//

extern "C" size_t hip_qmatmul_workspace_bytes(int64_t M, int64_t N, int64_t K,
                                              int64_t batch_count) {
  if (M <= 0 || N <= 0 || K <= 0 || batch_count <= 0)
    return 0;
  const QMatMulSplitKPlan plan = qmatmul_plan_splitk(M, N, K, batch_count);
  return plan.enabled ? qmatmul_splitk_workspace_bytes(M, N, batch_count) : 0;
}

extern "C" int hip_qmatmul(void *stream, const void *A, const void *B, void *Y,
                           const void *B_scales, const void *B_zero_points,
                           int64_t M, int64_t N, int64_t K, int64_t batch_count,
                           int64_t b_batch_stride, int trans_a, int trans_b,
                           int a_dtype, int b_dtype, int y_dtype, int b_bits,
                           float M_scale, float AY_ratio, int64_t a_zp,
                           int64_t b_zp, int64_t y_zp, void *workspace,
                           size_t workspace_bytes) {
  if (!A || !B || !Y) {
    fprintf(stderr, "[custom_kernels] hip_qmatmul: null tensor argument\n");
    return -1;
  }
  if (M <= 0 || N <= 0 || K <= 0 || batch_count <= 0)
    return 0;
  const bool per_column = B_scales != nullptr;
  if (per_column != (B_zero_points != nullptr)) {
    fprintf(stderr,
            "[custom_kernels] hip_qmatmul: B scales and zero points must both "
            "be present\n");
    return -1;
  }
  if ((per_column && b_bits != 4 && b_bits != 8) ||
      (!per_column && b_bits != 8)) {
    fprintf(stderr,
            "[custom_kernels] hip_qmatmul: unsupported B quantization "
            "(per_column=%d, b_bits=%d)\n",
            per_column, b_bits);
    return -1;
  }
  if (b_bits == 4 && b_batch_stride != 0) {
    fprintf(stderr,
            "[custom_kernels] hip_qmatmul: batched packed B is unsupported\n");
    return -1;
  }

  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_qmatmul: M=%lld N=%lld K=%lld batch=%lld "
      "b_batch_stride=%lld trans=(%d,%d) dtypes=(%d,%d,%d) b_bits=%d "
      "per_column=%d M_scale=%g AY_ratio=%g zp=(%lld,%lld,%lld)\n",
      (long long)M, (long long)N, (long long)K, (long long)batch_count,
      (long long)b_batch_stride, trans_a, trans_b, a_dtype, b_dtype, y_dtype,
      b_bits, per_column, (double)M_scale, (double)AY_ratio, (long long)a_zp,
      (long long)b_zp, (long long)y_zp);

  const QMatMulArgs args{static_cast<hipStream_t>(stream),
                         A,
                         B,
                         Y,
                         static_cast<const float *>(B_scales),
                         static_cast<const uint8_t *>(B_zero_points),
                         M,
                         N,
                         K,
                         batch_count,
                         b_batch_stride,
                         trans_a != 0,
                         trans_b != 0,
                         b_bits,
                         b_dtype == HIP_DTYPE_INT8,
                         M_scale,
                         AY_ratio,
                         static_cast<int32_t>(a_zp),
                         static_cast<int32_t>(b_zp),
                         static_cast<int32_t>(y_zp),
                         workspace,
                         workspace_bytes};
  const int rc = dispatch_a(a_dtype, b_dtype, y_dtype, args);

  if (rc == -1) {
    fprintf(stderr,
            "[custom_kernels] hip_qmatmul: unsupported dtype triple "
            "(a=%d, b=%d, y=%d)\n",
            a_dtype, b_dtype, y_dtype);
    return rc;
  }
  if (rc != hipSuccess) {
    fprintf(stderr, "[custom_kernels] hip_qmatmul launch failed: %s\n",
            hipGetErrorString(static_cast<hipError_t>(rc)));
  }
  return rc;
}
