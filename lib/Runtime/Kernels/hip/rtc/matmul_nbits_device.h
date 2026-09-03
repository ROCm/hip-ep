/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Device half of matmul_nbits, split out so the AOT build and hipRTC compile
// the same text; if the two diverge their code objects stop being comparable.
//
// This header is not self-contained by design: it relies on hip_arch_compat.h
// having been included first. The AOT build gets that from
// matmul_nbits_kernel.hip; for hipRTC the build concatenates the two files,
// since hipRTC resolves no include paths.

#ifndef HIPDNN_EP_RTC_MATMUL_NBITS_DEVICE_H
#define HIPDNN_EP_RTC_MATMUL_NBITS_DEVICE_H

#include <cstdint>

#if defined(__HIPCC_RTC__)
  // hipRTC supplies the math functions but not the <math.h> macros, and <cmath>
  // conflicts with clang's HIP mode.
  #define INFINITY __builtin_inff()
#endif

// half16 holds one WMMA A/B fragment; its vector width matches
// HIPDNN_WMMA_FRAG_ELEMS (16 on gfx11, 8 on the gfx12-style encoding
// gfx1170/gfx12xx use -- see hip_arch_compat.h). The name is kept for
// continuity even though the width is arch-dependent.
typedef _Float16 half16 __attribute__((ext_vector_type(HIPDNN_WMMA_FRAG_ELEMS)));
typedef float    float8 __attribute__((ext_vector_type(8)));

// Shared with the host dispatch, which sizes its grid from it.
static constexpr int kBlockDim = 16;

__global__ void matmul_nbits_kernel(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    const __half* __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size) {
  int64_t n = static_cast<int64_t>(blockIdx.x) * kBlockDim + threadIdx.x;
  int64_t m = static_cast<int64_t>(blockIdx.y) * kBlockDim + threadIdx.y;
  int64_t batch = blockIdx.z;

  if (m >= M || n >= N) return;

  int64_t k_blocks = (K + block_size - 1) / block_size;
  int64_t blob_size = block_size / 2;

  float acc = 0.0f;
  const __half* a_row = A + batch * M * K + m * K;
  // Packed 4-bit weight rows are [N, k_blocks, block_size/2] bytes. The row
  // stride must use k_blocks (= ceil(K/block_size)), NOT K/2: when K is not a
  // multiple of block_size the last block is padded, so the true row is
  // k_blocks*(block_size/2) bytes, which exceeds K/2. Using K/2 misaligns
  // every row n>0 by the padding delta and corrupts the output. When
  // K % block_size == 0 this equals K/2 (backward compatible). Verified
  // against the CPU reference for K=4304, block_size=32 (down_proj).
  const int64_t b_row_bytes = k_blocks * blob_size;
  const uint8_t* b_row = B + n * b_row_bytes;
  const __half* s_row = scales + n * k_blocks;

  for (int64_t blk = 0; blk < k_blocks; blk++) {
    float scale_val = static_cast<float>(s_row[blk]);
    float zp_val = (zp != nullptr)
                       ? static_cast<float>(zp[n * k_blocks + blk])
                       : 8.0f;

    int64_t k_start = blk * block_size;
    int64_t k_end = k_start + block_size;
    if (k_end > K) {
      k_end = K;
    }
    const uint8_t* b_block = b_row + blk * blob_size;

    for (int64_t k = k_start; k < k_end; k++) {
      int64_t in_blk = k - k_start;
      int64_t byte_off = in_blk / 2;
      int nibble = static_cast<int>(in_blk & 1);
      uint8_t packed = b_block[byte_off];
      float qval = (nibble == 0) ? static_cast<float>(packed & 0xF)
                                 : static_cast<float>(packed >> 4);

      acc += static_cast<float>(a_row[k]) * (qval - zp_val) * scale_val;
    }
  }

  if (bias) {
    acc += static_cast<float>(bias[n]);
  }
  output[batch * M * N + m * N + n] = static_cast<__half>(acc);
}


__global__ void matmul_nbits_kernel_i8(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    const __half* __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size) {
  int64_t n = static_cast<int64_t>(blockIdx.x) * kBlockDim + threadIdx.x;
  int64_t m = static_cast<int64_t>(blockIdx.y) * kBlockDim + threadIdx.y;
  int64_t batch = blockIdx.z;

  if (m >= M || n >= N) return;

  int64_t k_blocks = (K + block_size - 1) / block_size;

  float acc = 0.0f;
  const __half*  a_row = A + batch * M * K + m * K;
  const uint8_t* b_row = B + n * K;
  const __half*  s_row = scales + n * k_blocks;

  for (int64_t blk = 0; blk < k_blocks; blk++) {
    float scale_val = static_cast<float>(s_row[blk]);
    float zp_val    = (zp != nullptr)
                          ? static_cast<float>(zp[n * k_blocks + blk])
                          : 128.0f;

    int64_t k_start = blk * block_size;
    int64_t k_end   = k_start + block_size;
    if (k_end > K) k_end = K;
    const uint8_t* b_block = b_row + k_start;

    for (int64_t k = k_start; k < k_end; k++) {
      float qval = static_cast<float>(b_block[k - k_start]);
      acc += static_cast<float>(a_row[k]) * (qval - zp_val) * scale_val;
    }
  }

  if (bias) acc += static_cast<float>(bias[n]);
  output[batch * M * N + m * N + n] = static_cast<__half>(acc);
}


template <int BLOCK_SIZE, int TILE_N>
__global__ void matmul_nbits_gemv_kernel_i8(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    const __half* __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size)
{
    const int64_t n_base = static_cast<int64_t>(blockIdx.x) * TILE_N;
    const int64_t m      = blockIdx.y;
    const int64_t batch  = blockIdx.z;
    const int tid        = threadIdx.x;

    if (m >= M) return;

    const int64_t k_blocks = (K + block_size - 1) / block_size;
    const int     bs_shift = __builtin_ctzll(
                                 static_cast<unsigned long long>(block_size));

    // Row-major A only (no COL_MAJOR variant -- int8 has no WMMA path).
    const __half* a_base   = A + batch * M * K + m * K;

    // Per-tile-N column pointers (clamped to N-1 for OOB lanes; the
    // out-of-bounds tile guards in the writeback skip those results).
    const uint8_t* b_base_arr[TILE_N];
    const __half*  s_rows[TILE_N];
    int64_t        n_idx[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        int64_t n  = n_base + tn;
        int64_t sn = (n < N) ? n : 0;
        n_idx[tn]  = sn;
        b_base_arr[tn] = B + sn * K;
        s_rows[tn] = scales + sn * k_blocks;
    }

    float partial[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++)
        partial[tn] = 0.0f;

    const bool has_zp = (zp != nullptr);

    // ---- Phase 1: 16-byte (128-bit / uint4) vectorized main loop ----
    // 16 weights per global memory transaction; matches the 16-nibble loop
    // in the int4 kernel but at twice the bytes-per-iter (since each
    // weight is now a full byte).
    const int64_t num_u128 = K / 16;
    for (int64_t idx = static_cast<int64_t>(tid); idx < num_u128;
         idx += static_cast<int64_t>(BLOCK_SIZE)) {
        const int64_t k16 = idx * 16;

        float a_vals[16];
        float a_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            a_vals[i] = static_cast<float>(a_base[k16 + i]);
            a_sum += a_vals[i];
        }

        // 16-byte chunks lie in one group iff block_size >= 16 (precondition).
        const int64_t grp = k16 >> bs_shift;

        // 16-byte (uint4) loads from each of TILE_N B columns.
        uint4 b_pk[TILE_N];
        float sv[TILE_N];
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            b_pk[tn] = *reinterpret_cast<const uint4*>(
                            &b_base_arr[tn][k16]);
            sv[tn]   = static_cast<float>(s_rows[tn][grp]);
        }

#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            // Decode 16 uint8 weights from the four 32-bit lanes of the
            // uint4 vector. Each uint32 holds 4 bytes (little-endian).
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; i++)
                dot += a_vals[i] *
                       static_cast<float>((b_pk[tn].x >> (i * 8)) & 0xFFu);
#pragma unroll
            for (int i = 0; i < 4; i++)
                dot += a_vals[4 + i] *
                       static_cast<float>((b_pk[tn].y >> (i * 8)) & 0xFFu);
#pragma unroll
            for (int i = 0; i < 4; i++)
                dot += a_vals[8 + i] *
                       static_cast<float>((b_pk[tn].z >> (i * 8)) & 0xFFu);
#pragma unroll
            for (int i = 0; i < 4; i++)
                dot += a_vals[12 + i] *
                       static_cast<float>((b_pk[tn].w >> (i * 8)) & 0xFFu);

            if (has_zp) {
                float zv = static_cast<float>(zp[n_idx[tn] * k_blocks + grp]);
                partial[tn] += dot * sv[tn] - a_sum * zv * sv[tn];
            } else {
                // ONNX MatMulNBits default zp for bits=8 is 2^(bits-1) = 128.
                partial[tn] += dot * sv[tn] - a_sum * 128.0f * sv[tn];
            }
        }
    }

    // ---- Phase 2: 8-byte (uint2 / 64-bit) middle loop (K%16 remainder) ----
    {
        const int64_t k_after_p1 = num_u128 * 16;
        const int64_t num_u64_m  = (K - k_after_p1) / 8;
        for (int64_t idx2 = static_cast<int64_t>(tid); idx2 < num_u64_m;
             idx2 += static_cast<int64_t>(BLOCK_SIZE)) {
            const int64_t k8 = k_after_p1 + idx2 * 8;

            float a_vals[8];
            float a_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                a_vals[i] = static_cast<float>(a_base[k8 + i]);
                a_sum += a_vals[i];
            }

            const int64_t grp = k8 >> bs_shift;

#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float sv = static_cast<float>(s_rows[tn][grp]);
                uint2 packed = *reinterpret_cast<const uint2*>(
                                    &b_base_arr[tn][k8]);

                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; i++)
                    dot += a_vals[i] *
                           static_cast<float>((packed.x >> (i * 8)) & 0xFFu);
#pragma unroll
                for (int i = 0; i < 4; i++)
                    dot += a_vals[4 + i] *
                           static_cast<float>((packed.y >> (i * 8)) & 0xFFu);

                if (has_zp) {
                    float zv = static_cast<float>(
                                   zp[n_idx[tn] * k_blocks + grp]);
                    partial[tn] += dot * sv - a_sum * zv * sv;
                } else {
                    partial[tn] += dot * sv - a_sum * 128.0f * sv;
                }
            }
        }
    }

    // ---- Phase 3: scalar tail (K%8 remainder) ----
    {
        const int64_t k_tail = (K / 8) * 8;
        for (int64_t k = k_tail + static_cast<int64_t>(tid);
             k < K; k += static_cast<int64_t>(BLOCK_SIZE)) {
            const float   a_val = static_cast<float>(a_base[k]);
            const int64_t grp   = k >> bs_shift;

#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float sv = static_cast<float>(s_rows[tn][grp]);
                float zv = has_zp
                              ? static_cast<float>(
                                    zp[n_idx[tn] * k_blocks + grp])
                              : 128.0f;
                float qval = static_cast<float>(b_base_arr[tn][k]);
                partial[tn] += a_val * (qval - zv) * sv;
            }
        }
    }

    // --- Reduction: warp shuffle then shared memory (mirrors int4 path) ---
    const int wave_size = __builtin_amdgcn_wavefrontsize();
    const int warp_id   = tid / wave_size;
    const int lane_id   = tid % wave_size;
    const int num_warps = BLOCK_SIZE / wave_size;

#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        for (int offset = wave_size >> 1; offset > 0; offset >>= 1)
            partial[tn] += __shfl_down(partial[tn], offset);
    }

    constexpr int MAX_WARPS = BLOCK_SIZE / 32;
    __shared__ float warp_sums[TILE_N * MAX_WARPS];

    if (lane_id == 0) {
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++)
            warp_sums[tn * MAX_WARPS + warp_id] = partial[tn];
    }
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            float val = (lane_id < num_warps)
                            ? warp_sums[tn * MAX_WARPS + lane_id] : 0.0f;
            for (int offset = wave_size >> 1; offset > 0; offset >>= 1)
                val += __shfl_down(val, offset);

            if (lane_id == 0) {
                int64_t n = n_base + tn;
                if (n < N) {
                    if (bias)
                        val += static_cast<float>(bias[n]);
                    output[batch * M * N + m * N + n] =
                        static_cast<__half>(val);
                }
            }
        }
    }
}


template <int BLOCK_SIZE, int TILE_N, bool COL_MAJOR = false>
__global__ void matmul_nbits_gemv_kernel(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    const __half* __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size)
{
    const int64_t n_base = static_cast<int64_t>(blockIdx.x) * TILE_N;
    const int64_t m      = blockIdx.y;
    const int64_t batch  = COL_MAJOR ? 0 : blockIdx.z;
    const int tid        = threadIdx.x;

    if (m >= M) return;

    const int64_t k_blocks = (K + block_size - 1) / block_size;
    const int     bs_shift = __builtin_ctzll(
                                 static_cast<unsigned long long>(block_size));

    const __half* a_base;
    int64_t       a_stride;
    if constexpr (COL_MAJOR) {
        a_base   = A + m;
        a_stride = M;
    } else {
        a_base   = A + batch * M * K + m * K;
        a_stride = 1;
    }

    [[maybe_unused]] const __half* zp_fp16 =
        COL_MAJOR ? reinterpret_cast<const __half*>(zp) : nullptr;

    const uint8_t* b_base_arr[TILE_N];
    const __half*  s_rows[TILE_N];
    int64_t        n_idx[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        int64_t n  = n_base + tn;
        int64_t sn = (n < N) ? n : 0;
        n_idx[tn]  = sn;
        // Packed 4-bit weight rows are [N, k_blocks, block_size/2] bytes. The
        // row stride must use k_blocks (= ceil(K/block_size)), NOT K/2: when K
        // is not a multiple of block_size the last block is padded, so the true
        // row is k_blocks*(block_size/2) bytes, which exceeds K/2. Using K/2
        // misaligns every row n>0 by the padding delta and corrupts the output.
        // When K % block_size == 0 this equals K/2 (backward compatible).
        b_base_arr[tn] = B + sn * (k_blocks * (block_size / 2));
        s_rows[tn] = scales + sn * k_blocks;
    }

    float partial[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++)
        partial[tn] = 0.0f;

    const bool has_zp = (zp != nullptr);

    // Helper: read zero-point for (n_idx, grp) — FP16 when COL_MAJOR
    auto read_zp = [&](int64_t nidx, int64_t grp) __attribute__((always_inline)) -> float {
        if constexpr (COL_MAJOR)
            return static_cast<float>(zp_fp16[nidx * k_blocks + grp]);
        else
            return static_cast<float>(zp[nidx * k_blocks + grp]);
    };

    // Helper: read A element at K-offset
    auto read_a = [&](int64_t k_off) __attribute__((always_inline)) -> float {
        return static_cast<float>(a_base[k_off * a_stride]);
    };

    // ---- Phase 1: 32-nibble (128-bit) vectorized main loop ----
    // Use int for loop vars — K <= 32768, idx <= K/32 = 1024, fits 32-bit.
    // uint4 (128-bit) loads process 32 nibbles (one full block_size=32 group).
    //
    // NOTE: K-loop unroll x2 was tried (processing two K-groups per iteration
    // to overlap B loads) but regressed performance by ~5% due to doubled
    // register pressure causing spilling on RDNA 3. Reverted to single-group.
    const int num_u128 = static_cast<int>(K / 32);
    for (int idx = tid; idx < num_u128; idx += BLOCK_SIZE) {
        const int k32 = idx * 32;

        float a_vals[32];
        float a_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            a_vals[i] = read_a(k32 + i);
            a_sum += a_vals[i];
        }

        const int grp = k32 >> bs_shift;
        const float a_sum_x_default_zp = a_sum * 8.0f;

        uint4   b_pk[TILE_N];
        float   sv[TILE_N];
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            b_pk[tn] = *reinterpret_cast<const uint4*>(
                            &b_base_arr[tn][idx * 16]);
            sv[tn]   = static_cast<float>(s_rows[tn][grp]);
        }

#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++)
                dot = __fmaf_rn(a_vals[i],
                       static_cast<float>((b_pk[tn].x >> (i * 4)) & 0xF), dot);
#pragma unroll
            for (int i = 0; i < 8; i++)
                dot = __fmaf_rn(a_vals[8 + i],
                       static_cast<float>((b_pk[tn].y >> (i * 4)) & 0xF), dot);
#pragma unroll
            for (int i = 0; i < 8; i++)
                dot = __fmaf_rn(a_vals[16 + i],
                       static_cast<float>((b_pk[tn].z >> (i * 4)) & 0xF), dot);
#pragma unroll
            for (int i = 0; i < 8; i++)
                dot = __fmaf_rn(a_vals[24 + i],
                       static_cast<float>((b_pk[tn].w >> (i * 4)) & 0xF), dot);

            if (has_zp) {
                float zv = read_zp(n_idx[tn], grp);
                partial[tn] += (dot - a_sum * zv) * sv[tn];
            } else {
                partial[tn] += (dot - a_sum_x_default_zp) * sv[tn];
            }
        }
    }

    // ---- Phase 1b: 16-nibble (64-bit) remainder (K%32 ≥ 16) ----
    {
        const int k16_base = num_u128 * 32;
        const int num_u64 = (static_cast<int>(K) - k16_base) / 16;
        for (int idx = tid; idx < num_u64; idx += BLOCK_SIZE) {
            const int k16 = k16_base + idx * 16;

            float a_vals[16];
            float a_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                a_vals[i] = read_a(k16 + i);
                a_sum += a_vals[i];
            }

            const int grp = k16 >> bs_shift;
            const float a_sum_x_default_zp = a_sum * 8.0f;

            uint2   b_pk[TILE_N];
            float   sv[TILE_N];
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                b_pk[tn] = *reinterpret_cast<const uint2*>(
                                &b_base_arr[tn][(k16_base / 2) + idx * 8]);
                sv[tn]   = static_cast<float>(s_rows[tn][grp]);
            }

#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; i++)
                    dot = __fmaf_rn(a_vals[i],
                           static_cast<float>((b_pk[tn].x >> (i * 4)) & 0xF), dot);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    dot = __fmaf_rn(a_vals[8 + i],
                           static_cast<float>((b_pk[tn].y >> (i * 4)) & 0xF), dot);

                if (has_zp) {
                    float zv = read_zp(n_idx[tn], grp);
                    partial[tn] += (dot - a_sum * zv) * sv[tn];
                } else {
                    partial[tn] += (dot - a_sum_x_default_zp) * sv[tn];
                }
            }
        }
    }

    // ---- Phase 2: 8-nibble (32-bit) middle (K%8 remainder after uint4+uint2) ----
    {
        const int k8_base   = (static_cast<int>(K) / 16) * 16;
        const int num_u32_m = (static_cast<int>(K) - k8_base) / 8;
        for (int idx2 = tid; idx2 < num_u32_m; idx2 += BLOCK_SIZE) {
            const int k8 = k8_base + idx2 * 8;

            float a_vals[8];
            float a_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                a_vals[i] = read_a(k8 + i);
                a_sum += a_vals[i];
            }

            const int grp = k8 >> bs_shift;
            const float a_sum_x_default_zp = a_sum * 8.0f;

#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float sv = static_cast<float>(s_rows[tn][grp]);
                uint32_t packed = *reinterpret_cast<const uint32_t*>(
                                      &b_base_arr[tn][k8 / 2]);

                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; i++)
                    dot = __fmaf_rn(a_vals[i],
                           static_cast<float>((packed >> (i * 4)) & 0xF), dot);

                if (has_zp) {
                    float zv = read_zp(n_idx[tn], grp);
                    partial[tn] += (dot - a_sum * zv) * sv;
                } else {
                    partial[tn] += (dot - a_sum_x_default_zp) * sv;
                }
            }
        }
    }

    // ---- Phase 3: scalar tail (K%8 remainder) ----
    {
        const int k_tail = (static_cast<int>(K) / 8) * 8;
        for (int k = k_tail + tid; k < static_cast<int>(K);
             k += BLOCK_SIZE) {
            const float a_val  = read_a(k);
            const int   grp    = k >> bs_shift;
            const int   nibble = k & 1;

#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float sv = static_cast<float>(s_rows[tn][grp]);
                float zv = has_zp ? read_zp(n_idx[tn], grp) : 8.0f;
                uint8_t packed = b_base_arr[tn][k / 2];
                float   qval   = (nibble == 0)
                                   ? static_cast<float>(packed & 0xF)
                                   : static_cast<float>(packed >> 4);
                partial[tn] += a_val * (qval - zv) * sv;
            }
        }
    }

    // --- Reduction: warp shuffle then shared memory ---
    // RDNA 3/3.5 (gfx11xx) runs wave32 by default. Compile-time constant
    // for warp_size allows the compiler to dead-strip unused branches.
    // Compile-time wavefront size (32 on RDNA, 64 on CDNA/MI350). For a block
    // smaller than a hardware wave (e.g. BLOCK_SIZE=32 on wave64) only the first
    // BLOCK_SIZE lanes are live, so the single-wave shuffle tree below must start
    // at BLOCK_SIZE/2 -- starting at WARP_SIZE/2 would fold in inactive lanes.
    constexpr int WARP_SIZE = HIPDNN_WAVE_SIZE;
    const int warp_id   = tid / WARP_SIZE;
    const int lane_id   = tid % WARP_SIZE;
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    constexpr int RED_W = BLOCK_SIZE < WARP_SIZE ? BLOCK_SIZE : WARP_SIZE;

#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        for (int offset = RED_W >> 1; offset > 0; offset >>= 1)
            partial[tn] += __shfl_down(partial[tn], offset);
    }

    // Single warp (BLOCK_SIZE==32 in wave32): shuffle has the final result.
    if constexpr (BLOCK_SIZE <= WARP_SIZE) {
        if (lane_id == 0) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                int64_t n = n_base + tn;
                if (n < N) {
                    float val = partial[tn];
                    if (bias)
                        val += static_cast<float>(bias[n]);
                    // NOTE: nontemporal stores were tried here but regressed
                    // perf by ~3% on gfx1150 LPDDR5X — the output is too small
                    // (28KB) to pollute cache, and NT writes compete with read
                    // traffic on the shared memory bus.
                    if constexpr (COL_MAJOR)
                        output[n * M + m] = static_cast<__half>(val);
                    else
                        output[batch * M * N + m * N + n] =
                            static_cast<__half>(val);
                }
            }
        }
    } else {
        __shared__ float warp_sums[TILE_N * NUM_WARPS];

        if (lane_id == 0) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++)
                warp_sums[tn * NUM_WARPS + warp_id] = partial[tn];
        }
        __syncthreads();

        if (warp_id == 0) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float val = (lane_id < NUM_WARPS)
                                ? warp_sums[tn * NUM_WARPS + lane_id] : 0.0f;
                for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
                    val += __shfl_down(val, offset);

                if (lane_id == 0) {
                    int64_t n = n_base + tn;
                    if (n < N) {
                        if (bias)
                            val += static_cast<float>(bias[n]);
                        if constexpr (COL_MAJOR)
                            output[n * M + m] = static_cast<__half>(val);
                        else
                            output[batch * M * N + m * N + n] =
                                static_cast<__half>(val);
                    }
                }
            }
        }
    }
}


// Deinterleaved-int8 activation layout, per 8-element chunk c (k = 8c + p):
//   bytes[8c+0..3] = {aq[8c+0], aq[8c+2], aq[8c+4], aq[8c+6]}   (a_lo, even p)
//   bytes[8c+4..7] = {aq[8c+1], aq[8c+3], aq[8c+5], aq[8c+7]}   (a_hi, odd p)
// This matches the byte order produced by masking a packed-nibble uint32 with
// 0x0F0F0F0F (even nibbles) and (>>4)&0x0F0F0F0F (odd nibbles). One block per
// K-group; group boundaries are 8-aligned (block_size is a power of two >= 32),
// so no chunk straddles a group.
template <int BLOCK_SIZE>
__global__ void matmul_nbits_quant_act_i8_kernel(
    const __half* __restrict__ A,
    int8_t*       __restrict__ a_qb,
    float*        __restrict__ a_scale,
    int64_t K, int64_t block_size)
{
    const int64_t grp = blockIdx.x;
    const int64_t g0  = grp * block_size;
    const int64_t g1  = min(g0 + block_size, K);
    const int     tid = threadIdx.x;

    // Pass 1: per-group max|A| reduction (warp shuffle + LDS).
    float local_max = 0.0f;
    for (int64_t k = g0 + tid; k < g1; k += BLOCK_SIZE)
        local_max = fmaxf(local_max, fabsf(static_cast<float>(A[k])));

    // Wave-portable (32 on RDNA, 64 on CDNA/MI350). RED_W caps the shuffle to
    // the active lanes for a sub-wave block, and the per-wave fan-in index uses
    // tid / WARP_SIZE rather than a hard-coded /32.
    constexpr int WARP_SIZE = HIPDNN_WAVE_SIZE;
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    constexpr int RED_W = BLOCK_SIZE < WARP_SIZE ? BLOCK_SIZE : WARP_SIZE;
    for (int off = RED_W >> 1; off > 0; off >>= 1)
        local_max = fmaxf(local_max, __shfl_down(local_max, off));

    __shared__ float smax[NUM_WARPS > 0 ? NUM_WARPS : 1];
    if ((tid & (WARP_SIZE - 1)) == 0)
        smax[tid / WARP_SIZE] = local_max;
    __syncthreads();
    if (tid == 0) {
        float m = 0.0f;
#pragma unroll
        for (int i = 0; i < NUM_WARPS; i++)
            m = fmaxf(m, smax[i]);
        smax[0] = m;
    }
    __syncthreads();

    const float gmax      = smax[0];
    const float scale     = (gmax > 0.0f) ? (gmax * (1.0f / 127.0f)) : 1.0f;
    const float inv_scale = (gmax > 0.0f) ? (127.0f / gmax) : 0.0f;
    if (tid == 0)
        a_scale[grp] = scale;

    // Pass 2: quantize + deinterleaved-pack write (byte writes, no conflict).
    for (int64_t k = g0 + tid; k < g1; k += BLOCK_SIZE) {
        int q = __float2int_rn(static_cast<float>(A[k]) * inv_scale);
        q = max(-127, min(127, q));
        const int64_t c = k >> 3;
        const int     p = static_cast<int>(k & 7);
        const int64_t byte_off =
            (p & 1) ? (c * 8 + 4 + (p >> 1)) : (c * 8 + (p >> 1));
        a_qb[byte_off] = static_cast<int8_t>(q);
    }
}


// dp4a GEMV: mirrors matmul_nbits_gemv_kernel's main uint4 phase, but the
// per-element dequant/FMA chain is replaced by two sudot4 per 8 nibbles.
// Row-major only (decode); requires K % 32 == 0. Reads the per-group int8
// activation (a_qb) + scales (a_scale) prepared ONCE per token by
// matmul_nbits_quant_act_i8_kernel above -- the quant is amortized across all
// N/TILE_N column blocks (fusing it into the GEMV instead re-quantizes A in
// every block, which is a net loss: measured ~1.3-2x slower on decode shapes
// because the redundant fp16 read + convert per block dwarfs the one launch it
// would save).
template <int BLOCK_SIZE, int TILE_N>
__global__ void matmul_nbits_gemv_dp4a_kernel(
    const int8_t*  __restrict__ a_qb,
    const float*   __restrict__ a_scale,
    const uint8_t* __restrict__ B,
    const __half*  __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const __half*  __restrict__ bias,
    __half*        __restrict__ output,
    int64_t N, int64_t K, int64_t block_size)
{
    const int64_t n_base = static_cast<int64_t>(blockIdx.x) * TILE_N;
    const int     tid    = threadIdx.x;
    const int64_t k_blocks = (K + block_size - 1) / block_size;
    const int     bs_shift =
        __builtin_ctzll(static_cast<unsigned long long>(block_size));
    const bool has_zp = (zp != nullptr);

    const uint8_t* b_base_arr[TILE_N];
    const __half*  s_rows[TILE_N];
    int64_t        n_idx[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        int64_t n  = n_base + tn;
        int64_t sn = (n < N) ? n : 0;
        n_idx[tn]  = sn;
        b_base_arr[tn] = B + sn * (K / 2);
        s_rows[tn] = scales + sn * k_blocks;
    }

    float partial[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++)
        partial[tn] = 0.0f;

    const int num_u128 = static_cast<int>(K / 32);
    for (int idx = tid; idx < num_u128; idx += BLOCK_SIZE) {
        const int   k32 = idx * 32;
        const int   grp = k32 >> bs_shift;
        const float a_sc = a_scale[grp];

        // 4 deinterleaved chunks (a_lo/a_hi) covering the 32 activations.
        const int8_t* aq = a_qb + static_cast<int64_t>(idx) * 32;
        int a_lo[4], a_hi[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            a_lo[j] = *reinterpret_cast<const int*>(aq + j * 8);
            a_hi[j] = *reinterpret_cast<const int*>(aq + j * 8 + 4);
        }
        // sum(aq) over the 32 elements: signed-int8 dot against all-ones.
        int a_sum_q = 0;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            a_sum_q = hipdnn_sudot4(a_lo[j], 0x01010101, a_sum_q);
            a_sum_q = hipdnn_sudot4(a_hi[j], 0x01010101, a_sum_q);
        }

#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            uint4 b = *reinterpret_cast<const uint4*>(&b_base_arr[tn][idx * 16]);
            const unsigned comp[4] = {b.x, b.y, b.z, b.w};
            int dot = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int lo_w = static_cast<int>(comp[j] & 0x0F0F0F0Fu);
                const int hi_w = static_cast<int>((comp[j] >> 4) & 0x0F0F0F0Fu);
                dot = hipdnn_sudot4(a_lo[j], lo_w, dot);
                dot = hipdnn_sudot4(a_hi[j], hi_w, dot);
            }
            const float sv = static_cast<float>(s_rows[tn][grp]);
            const int   zv = has_zp
                                 ? static_cast<int>(zp[n_idx[tn] * k_blocks + grp])
                                 : 8;
            partial[tn] += a_sc * sv * static_cast<float>(dot - a_sum_q * zv);
        }
    }

    // --- Reduction: warp shuffle then shared memory (mirrors int4 path) ---
    // Compile-time wavefront size (32 on RDNA, 64 on CDNA/MI350). For a block
    // smaller than a hardware wave (e.g. BLOCK_SIZE=32 on wave64) only the first
    // BLOCK_SIZE lanes are live, so the single-wave shuffle tree below must start
    // at BLOCK_SIZE/2 -- starting at WARP_SIZE/2 would fold in inactive lanes.
    constexpr int WARP_SIZE = HIPDNN_WAVE_SIZE;
    const int warp_id   = tid / WARP_SIZE;
    const int lane_id   = tid % WARP_SIZE;
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    constexpr int RED_W = BLOCK_SIZE < WARP_SIZE ? BLOCK_SIZE : WARP_SIZE;

#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        for (int offset = RED_W >> 1; offset > 0; offset >>= 1)
            partial[tn] += __shfl_down(partial[tn], offset);
    }

    if constexpr (BLOCK_SIZE <= WARP_SIZE) {
        if (lane_id == 0) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                int64_t n = n_base + tn;
                if (n < N) {
                    float val = partial[tn];
                    if (bias)
                        val += static_cast<float>(bias[n]);
                    output[n] = static_cast<__half>(val);
                }
            }
        }
    } else {
        __shared__ float warp_sums[TILE_N * NUM_WARPS];
        if (lane_id == 0) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++)
                warp_sums[tn * NUM_WARPS + warp_id] = partial[tn];
        }
        __syncthreads();
        if (warp_id == 0) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float val = (lane_id < NUM_WARPS)
                                ? warp_sums[tn * NUM_WARPS + lane_id] : 0.0f;
                for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
                    val += __shfl_down(val, offset);
                if (lane_id == 0) {
                    int64_t n = n_base + tn;
                    if (n < N) {
                        if (bias)
                            val += static_cast<float>(bias[n]);
                        output[n] = static_cast<__half>(val);
                    }
                }
            }
        }
    }
}


__global__ void matmul_nbits_kernel_u3(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    const __half* __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size) {
  int64_t n = static_cast<int64_t>(blockIdx.x) * kBlockDim + threadIdx.x;
  int64_t m = static_cast<int64_t>(blockIdx.y) * kBlockDim + threadIdx.y;
  int64_t batch = blockIdx.z;

  if (m >= M || n >= N) return;

  int64_t k_blocks  = (K + block_size - 1) / block_size;
  int64_t row_bytes = (K * 3 + 7) / 8;

  float acc = 0.0f;
  const __half*  a_row = A + batch * M * K + m * K;
  const uint8_t* b_row = B + n * row_bytes;
  const __half*  s_row = scales + n * k_blocks;

  for (int64_t k = 0; k < K; k++) {
    int64_t grp = k / block_size;
    float scale_val = static_cast<float>(s_row[grp]);
    float zp_val = (zp != nullptr)
                       ? static_cast<float>(zp[n * k_blocks + grp])
                       : 4.0f;

    int64_t bit   = k * 3;
    int64_t byte0 = bit >> 3;
    int     shift = static_cast<int>(bit & 7);
    uint16_t combined = b_row[byte0];
    if (byte0 + 1 < row_bytes)
      combined |= static_cast<uint16_t>(b_row[byte0 + 1]) << 8;
    float qval = static_cast<float>((combined >> shift) & 0x7);

    acc += static_cast<float>(a_row[k]) * (qval - zp_val) * scale_val;
  }

  if (bias) {
    acc += static_cast<float>(bias[n]);
  }
  output[batch * M * N + m * N + n] = static_cast<__half>(acc);
}


template <int BLOCK_SIZE, int TILE_N>
__global__ void matmul_nbits_gemv_kernel_u3(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    const __half* __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size)
{
    const int64_t n_base = static_cast<int64_t>(blockIdx.x) * TILE_N;
    const int64_t m      = blockIdx.y;
    const int64_t batch  = blockIdx.z;
    const int tid        = threadIdx.x;

    if (m >= M) return;

    const int64_t k_blocks  = (K + block_size - 1) / block_size;
    const int64_t row_bytes = (K * 3) / 8;  // K % 32 == 0 implies K % 8 == 0
    const int      bs_shift = __builtin_ctzll(
                                  static_cast<unsigned long long>(block_size));

    const __half* a_base = A + batch * M * K + m * K;

    const uint8_t* b_base_arr[TILE_N];
    const __half*  s_rows[TILE_N];
    int64_t        n_idx[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        int64_t n  = n_base + tn;
        int64_t sn = (n < N) ? n : 0;
        n_idx[tn]  = sn;
        b_base_arr[tn] = B + sn * row_bytes;
        s_rows[tn] = scales + sn * k_blocks;
    }

    float partial[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++)
        partial[tn] = 0.0f;

    const bool has_zp = (zp != nullptr);

    // ---- Phase 1: 32-lane (96-bit / uint3) vectorized main loop ----
    const int num_chunks = static_cast<int>(K / 32);
    for (int idx = tid; idx < num_chunks; idx += BLOCK_SIZE) {
        const int k32 = idx * 32;

        float a_vals[32];
        float a_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            a_vals[i] = static_cast<float>(a_base[k32 + i]);
            a_sum += a_vals[i];
        }

        const int grp = k32 >> bs_shift;
        const float a_sum_x_default_zp = a_sum * 4.0f;  // 2^(bits-1)

        uint3 b_pk[TILE_N];
        float sv[TILE_N];
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            b_pk[tn] = *reinterpret_cast<const uint3*>(
                            &b_base_arr[tn][idx * 12]);
            sv[tn]   = static_cast<float>(s_rows[tn][grp]);
        }

#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            const uint32_t w0 = b_pk[tn].x, w1 = b_pk[tn].y, w2 = b_pk[tn].z;
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; i++) {
                const int bit     = i * 3;
                const int wordIdx = bit >> 5;
                const int bitOff  = bit & 31;
                uint32_t lo = (wordIdx == 0) ? w0 : (wordIdx == 1) ? w1 : w2;
                uint32_t qv = (lo >> bitOff) & 0x7u;
                if (bitOff > 29) {
                    uint32_t hi = (wordIdx == 0) ? w1 : w2;
                    const int lowBits = 32 - bitOff;
                    qv |= (hi & ((1u << (3 - lowBits)) - 1u)) << lowBits;
                }
                dot = __fmaf_rn(a_vals[i], static_cast<float>(qv), dot);
            }

            if (has_zp) {
                float zv = static_cast<float>(zp[n_idx[tn] * k_blocks + grp]);
                partial[tn] += (dot - a_sum * zv) * sv[tn];
            } else {
                partial[tn] += (dot - a_sum_x_default_zp) * sv[tn];
            }
        }
    }

    // ---- Phase 2: scalar tail (K % 32 remainder) ----
    {
        const int k_tail = num_chunks * 32;
        for (int k = k_tail + tid; k < static_cast<int>(K); k += BLOCK_SIZE) {
            const float   a_val = static_cast<float>(a_base[k]);
            const int     grp   = k >> bs_shift;
            const int64_t bit   = static_cast<int64_t>(k) * 3;
            const int64_t byte0 = bit >> 3;
            const int     shift = static_cast<int>(bit & 7);

#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float sv = static_cast<float>(s_rows[tn][grp]);
                float zv = has_zp
                              ? static_cast<float>(zp[n_idx[tn] * k_blocks + grp])
                              : 4.0f;
                uint16_t combined = b_base_arr[tn][byte0];
                if (byte0 + 1 < row_bytes)
                    combined |= static_cast<uint16_t>(b_base_arr[tn][byte0 + 1]) << 8;
                float qval = static_cast<float>((combined >> shift) & 0x7);
                partial[tn] += a_val * (qval - zv) * sv;
            }
        }
    }

    // --- Reduction: warp shuffle then shared memory (mirrors int4 path) ---
    // Compile-time wavefront size (32 on RDNA, 64 on CDNA/MI350). For a block
    // smaller than a hardware wave (e.g. BLOCK_SIZE=32 on wave64) only the first
    // BLOCK_SIZE lanes are live, so the single-wave shuffle tree below must start
    // at BLOCK_SIZE/2 -- starting at WARP_SIZE/2 would fold in inactive lanes.
    constexpr int WARP_SIZE = HIPDNN_WAVE_SIZE;
    const int warp_id   = tid / WARP_SIZE;
    const int lane_id   = tid % WARP_SIZE;
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    constexpr int RED_W = BLOCK_SIZE < WARP_SIZE ? BLOCK_SIZE : WARP_SIZE;

#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        for (int offset = RED_W >> 1; offset > 0; offset >>= 1)
            partial[tn] += __shfl_down(partial[tn], offset);
    }

    if constexpr (BLOCK_SIZE <= WARP_SIZE) {
        if (lane_id == 0) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                int64_t n = n_base + tn;
                if (n < N) {
                    float val = partial[tn];
                    if (bias)
                        val += static_cast<float>(bias[n]);
                    output[batch * M * N + m * N + n] =
                        static_cast<__half>(val);
                }
            }
        }
    } else {
        __shared__ float warp_sums[TILE_N * NUM_WARPS];

        if (lane_id == 0) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++)
                warp_sums[tn * NUM_WARPS + warp_id] = partial[tn];
        }
        __syncthreads();

        if (warp_id == 0) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float val = (lane_id < NUM_WARPS)
                                ? warp_sums[tn * NUM_WARPS + lane_id] : 0.0f;
                for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
                    val += __shfl_down(val, offset);

                if (lane_id == 0) {
                    int64_t n = n_base + tn;
                    if (n < N) {
                        if (bias)
                            val += static_cast<float>(bias[n]);
                        output[batch * M * N + m * N + n] =
                            static_cast<__half>(val);
                    }
                }
            }
        }
    }
}


__global__ void matmul_nbits_kernel_u2(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    const __half* __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size) {
  int64_t n = static_cast<int64_t>(blockIdx.x) * kBlockDim + threadIdx.x;
  int64_t m = static_cast<int64_t>(blockIdx.y) * kBlockDim + threadIdx.y;
  int64_t batch = blockIdx.z;

  if (m >= M || n >= N) return;

  int64_t k_blocks  = (K + block_size - 1) / block_size;
  int64_t row_bytes = (K * 2 + 7) / 8;

  float acc = 0.0f;
  const __half*  a_row = A + batch * M * K + m * K;
  const uint8_t* b_row = B + n * row_bytes;
  const __half*  s_row = scales + n * k_blocks;

  for (int64_t k = 0; k < K; k++) {
    int64_t grp = k / block_size;
    float scale_val = static_cast<float>(s_row[grp]);
    float zp_val = (zp != nullptr)
                       ? static_cast<float>(zp[n * k_blocks + grp])
                       : 2.0f;

    int64_t bit   = k * 2;
    int64_t byte0 = bit >> 3;
    int     shift = static_cast<int>(bit & 7);
    // 2-bit value with shift in {0,2,4,6} never crosses a byte boundary.
    float qval = static_cast<float>((b_row[byte0] >> shift) & 0x3);

    acc += static_cast<float>(a_row[k]) * (qval - zp_val) * scale_val;
  }

  if (bias) {
    acc += static_cast<float>(bias[n]);
  }
  output[batch * M * N + m * N + n] = static_cast<__half>(acc);
}


template <int BLOCK_SIZE, int TILE_N>
__global__ void matmul_nbits_gemv_kernel_u2(
    const __half* __restrict__ A,
    const uint8_t* __restrict__ B,
    const __half* __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size)
{
    const int64_t n_base = static_cast<int64_t>(blockIdx.x) * TILE_N;
    const int64_t m      = blockIdx.y;
    const int64_t batch  = blockIdx.z;
    const int tid        = threadIdx.x;

    if (m >= M) return;

    const int64_t k_blocks  = (K + block_size - 1) / block_size;
    const int64_t row_bytes = (K * 2) / 8;  // K % 32 == 0 implies K % 4 == 0
    const int      bs_shift = __builtin_ctzll(
                                  static_cast<unsigned long long>(block_size));

    const __half* a_base = A + batch * M * K + m * K;

    const uint8_t* b_base_arr[TILE_N];
    const __half*  s_rows[TILE_N];
    int64_t        n_idx[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        int64_t n  = n_base + tn;
        int64_t sn = (n < N) ? n : 0;
        n_idx[tn]  = sn;
        b_base_arr[tn] = B + sn * row_bytes;
        s_rows[tn] = scales + sn * k_blocks;
    }

    float partial[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++)
        partial[tn] = 0.0f;

    const bool has_zp = (zp != nullptr);

    // ---- Phase 1: 32-lane (64-bit / uint2) vectorized main loop ----
    const int num_chunks = static_cast<int>(K / 32);
    for (int idx = tid; idx < num_chunks; idx += BLOCK_SIZE) {
        const int k32 = idx * 32;

        float a_vals[32];
        float a_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 32; i++) {
            a_vals[i] = static_cast<float>(a_base[k32 + i]);
            a_sum += a_vals[i];
        }

        const int grp = k32 >> bs_shift;
        const float a_sum_x_default_zp = a_sum * 2.0f;  // 2^(bits-1)

        uint2 b_pk[TILE_N];
        float sv[TILE_N];
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            b_pk[tn] = *reinterpret_cast<const uint2*>(
                            &b_base_arr[tn][idx * 8]);
            sv[tn]   = static_cast<float>(s_rows[tn][grp]);
        }

#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            const uint32_t w0 = b_pk[tn].x, w1 = b_pk[tn].y;
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; i++) {
                const int bit     = i * 2;
                const int wordIdx = bit >> 5;
                const int bitOff  = bit & 31;
                uint32_t lo = (wordIdx == 0) ? w0 : w1;
                uint32_t qv = (lo >> bitOff) & 0x3u;
                dot = __fmaf_rn(a_vals[i], static_cast<float>(qv), dot);
            }

            if (has_zp) {
                float zv = static_cast<float>(zp[n_idx[tn] * k_blocks + grp]);
                partial[tn] += (dot - a_sum * zv) * sv[tn];
            } else {
                partial[tn] += (dot - a_sum_x_default_zp) * sv[tn];
            }
        }
    }

    // ---- Phase 2: scalar tail (K % 32 remainder) ----
    {
        const int k_tail = num_chunks * 32;
        for (int k = k_tail + tid; k < static_cast<int>(K); k += BLOCK_SIZE) {
            const float   a_val = static_cast<float>(a_base[k]);
            const int     grp   = k >> bs_shift;
            const int64_t bit   = static_cast<int64_t>(k) * 2;
            const int64_t byte0 = bit >> 3;
            const int     shift = static_cast<int>(bit & 7);

#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float sv = static_cast<float>(s_rows[tn][grp]);
                float zv = has_zp
                              ? static_cast<float>(zp[n_idx[tn] * k_blocks + grp])
                              : 2.0f;
                float qval = static_cast<float>(
                                 (b_base_arr[tn][byte0] >> shift) & 0x3);
                partial[tn] += a_val * (qval - zv) * sv;
            }
        }
    }

    // --- Reduction: warp shuffle then shared memory (mirrors u3 path) ---
    // Compile-time wavefront size (32 on RDNA, 64 on CDNA/MI350). For a block
    // smaller than a hardware wave (e.g. BLOCK_SIZE=32 on wave64) only the first
    // BLOCK_SIZE lanes are live, so the single-wave shuffle tree below must start
    // at BLOCK_SIZE/2 -- starting at WARP_SIZE/2 would fold in inactive lanes.
    constexpr int WARP_SIZE = HIPDNN_WAVE_SIZE;
    const int warp_id   = tid / WARP_SIZE;
    const int lane_id   = tid % WARP_SIZE;
    constexpr int NUM_WARPS = (BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE;
    constexpr int RED_W = BLOCK_SIZE < WARP_SIZE ? BLOCK_SIZE : WARP_SIZE;

#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        for (int offset = RED_W >> 1; offset > 0; offset >>= 1)
            partial[tn] += __shfl_down(partial[tn], offset);
    }

    if constexpr (BLOCK_SIZE <= WARP_SIZE) {
        if (lane_id == 0) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                int64_t n = n_base + tn;
                if (n < N) {
                    float val = partial[tn];
                    if (bias)
                        val += static_cast<float>(bias[n]);
                    output[batch * M * N + m * N + n] =
                        static_cast<__half>(val);
                }
            }
        }
    } else {
        __shared__ float warp_sums[TILE_N * NUM_WARPS];

        if (lane_id == 0) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++)
                warp_sums[tn * NUM_WARPS + warp_id] = partial[tn];
        }
        __syncthreads();

        if (warp_id == 0) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float val = (lane_id < NUM_WARPS)
                                ? warp_sums[tn * NUM_WARPS + lane_id] : 0.0f;
                for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
                    val += __shfl_down(val, offset);

                if (lane_id == 0) {
                    int64_t n = n_base + tn;
                    if (n < N) {
                        if (bias)
                            val += static_cast<float>(bias[n]);
                        output[batch * M * N + m * N + n] =
                            static_cast<__half>(val);
                    }
                }
            }
        }
    }
}


__global__ void dequant_u4_to_fp16(
    const unsigned char* __restrict__ B_packed,
    const _Float16* __restrict__ scales,
    const _Float16* __restrict__ zeros,
    _Float16* __restrict__ B_fp16,
    int N, int K, int group_size, int num_groups_k)
{
    int n   = static_cast<int>(blockIdx.y);
    int k8  = (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (n >= N || k8 >= K) return;

    int grp = k8 / group_size;
    _Float16 scale = scales[n * num_groups_k + grp];
    _Float16 zp    = zeros ? zeros[n * num_groups_k + grp] : (_Float16)8;

    // Row stride is num_groups_k * (group_size/2) bytes (ONNX MatMulNBits
    // padding: the last group is padded to a full group_size even when
    // K % group_size != 0), not a plain K/2 -- see GemmFp16U4Impl's
    // b_byte_base comment. The intra-row offset (k8/2) is unaffected.
    uint32_t four = *reinterpret_cast<const uint32_t*>(
        &B_packed[n * (num_groups_k * (group_size / 2)) + k8 / 2]);

    _Float16 bv[8];
    bv[0] = (static_cast<_Float16>(static_cast<int>((four      ) & 0xF)) - zp) * scale;
    bv[1] = (static_cast<_Float16>(static_cast<int>((four >>  4) & 0xF)) - zp) * scale;
    bv[2] = (static_cast<_Float16>(static_cast<int>((four >>  8) & 0xF)) - zp) * scale;
    bv[3] = (static_cast<_Float16>(static_cast<int>((four >> 12) & 0xF)) - zp) * scale;
    bv[4] = (static_cast<_Float16>(static_cast<int>((four >> 16) & 0xF)) - zp) * scale;
    bv[5] = (static_cast<_Float16>(static_cast<int>((four >> 20) & 0xF)) - zp) * scale;
    bv[6] = (static_cast<_Float16>(static_cast<int>((four >> 24) & 0xF)) - zp) * scale;
    bv[7] = (static_cast<_Float16>(static_cast<int>((four >> 28)      )) - zp) * scale;

    _Float16* dst = &B_fp16[n * K + k8];
    *reinterpret_cast<uint2*>(dst)     = *reinterpret_cast<uint2*>(&bv[0]);
    *reinterpret_cast<uint2*>(dst + 4) = *reinterpret_cast<uint2*>(&bv[4]);
}


// Where neither WMMA builtin is available (non-RDNA3/3.5/4 arch), trap
// instead so this file still compiles and a WMMA launch on an unsupported
// arch aborts loudly rather than returning garbage.
#if !HIPDNN_HAS_WMMA
static __device__ __forceinline__ float8
hipdnn_wmma_f32_16x16x16_f16_unavailable(half16, half16, float8 c) {
  __builtin_trap();
  return c;
}
#undef HIPDNN_WMMA_F32_16X16X16_F16
#define HIPDNN_WMMA_F32_16X16X16_F16(a, b, c)                                  \
  hipdnn_wmma_f32_16x16x16_f16_unavailable((a), (b), (c))
#endif

#define WMMA_TILE 16

template <int BM_T, int BN_T, int WT_M, int WT_N, bool USE_ZEROS,
          bool BOUNDS_CHECK = false, bool FUSED_DQ = true>
__device__ __forceinline__
void GemmFp16U4Impl(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const void* __restrict__ B_data,
    const _Float16* __restrict__ scales,
    const _Float16* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    constexpr int WM_L   = BM_T / (WT_M * 16);
    constexpr int WN_L   = BN_T / (WT_N * 16);
    constexpr int THR_L  = WM_L * WN_L * 32;
    constexpr int BK_T   = BM_T / (WT_M * WT_N);

    constexpr int PAD_K   = 2;
    constexpr int K_STR   = BK_T + PAD_K;
    constexpr int A_VL    = (BM_T * BK_T) / (THR_L * 4);
    constexpr int A_GRP_K = BK_T / 2;
    constexpr int K_STEPS = BK_T / WMMA_TILE;

    static_assert(A_VL >= 1, "A_VL must be >= 1");
    static_assert(THR_L * 8 == BN_T * BK_T, "B loading balance: THR*8 must equal BN*BK");

    __shared__ _Float16 smA[2][BM_T][K_STR];
    __shared__ _Float16 smB[2][BN_T][K_STR];

    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lid  = tid % 32;
    const int lane = lid % 16;
    const int sub  = lid / 16;
    const int wrow = wid / WN_L;
    const int wcol = wid % WN_L;

    // Block-to-tile mapping with column swizzle for L2 locality.
    // Splits the grid into a "main" region of (n_tiles/sw)*sw cols where full
    // sw-wide groups apply, and a "tail" region of (n_tiles%sw) leftover cols
    // that uses plain row-major. The previous implementation fell back to
    // row-major only when bx >= n_tiles, which produced both collisions AND
    // missing tiles when n_tiles % sw != 0, leaving those tiles with stale
    // output data from prior kernel launches.
    const int n_tiles  = gridDim.x;
    const int m_tiles  = gridDim.y;
    const int block_id = blockIdx.y * n_tiles + blockIdx.x;
    const int sw          = (n_tiles >= swizzle_n) ? swizzle_n : n_tiles;
    const int main_cols   = (n_tiles / sw) * sw;
    const int main_blocks = main_cols * m_tiles;
    int by, bx;
    if (block_id < main_blocks) {
        const int super = block_id / (sw * m_tiles);
        const int rem   = block_id % (sw * m_tiles);
        by = rem / sw;
        bx = super * sw + rem % sw;
    } else {
        const int tail_id   = block_id - main_blocks;
        const int tail_cols = n_tiles - main_cols;
        by = tail_id / tail_cols;
        bx = main_cols + (tail_id % tail_cols);
    }
    const int row0 = by * BM_T;
    const int col0 = bx * BN_T;

    constexpr int B_COL_SHIFT = __builtin_ctz(BK_T / 8);
    const int b_col       = tid >> B_COL_SHIFT;
    const int b_sub       = tid & ((1 << B_COL_SHIFT) - 1);
    const int b_k8        = b_sub << 3;
    const int b_n_g_raw   = col0 + b_col;
    const bool b_valid    = BOUNDS_CHECK ? (b_n_g_raw < N) : true;
    const int b_n_g       = (BOUNDS_CHECK && !b_valid) ? 0 : b_n_g_raw;

    // Fused dequant state (compile-time eliminated when FUSED_DQ=false).
    // B_packed rows are padded to num_groups_k * (group_size/2) bytes (ONNX
    // MatMulNBits blob layout -- the last group is padded to a full
    // group_size even when K % group_size != 0), NOT a plain K/2 bytes/row;
    // see the row-stride comment on matmul_nbits_gemv_kernel's b_base_arr.
    // Within a row the nibbles are still packed compactly from k=0 (only the
    // tail beyond K is padding), so the intra-row byte offset (b_k8 >> 1)
    // below is unaffected -- only the per-row stride changes.
    [[maybe_unused]] const int b_byte_base =
        FUSED_DQ ? (b_n_g * (num_groups_k * (group_size >> 1)) + (b_k8 >> 1)) : 0;
    [[maybe_unused]] const int b_gid_base =
        FUSED_DQ ? (b_n_g * num_groups_k) : 0;
    // cached_k_grp is the quant-group index currently loaded into cached_s/
    // neg_zs; -1 forces a load on the first K-step. See the group derivation
    // in issueLoads (keyed on the thread's actual k offset, not just t).
    [[maybe_unused]] int      cached_k_grp = -1;
    [[maybe_unused]] _Float16 cached_s = 0;
    [[maybe_unused]] _Float16 neg_zs = 0;

    // Pre-dequant fp16 B pointer (compile-time eliminated when FUSED_DQ=true)
    [[maybe_unused]] const _Float16* B_fp16 =
        FUSED_DQ ? nullptr : static_cast<const _Float16*>(B_data);
    [[maybe_unused]] const unsigned char* B_packed =
        FUSED_DQ ? static_cast<const unsigned char*>(B_data) : nullptr;

    float8 acc[WT_M][WT_N];
#pragma unroll
    for(int i = 0; i < WT_M; i++)
#pragma unroll
        for(int j = 0; j < WT_N; j++)
#pragma unroll
            for(int e = 0; e < 8; e++)
                acc[i][j][e] = 0.0f;

    /* --- Register-prefetch pipeline: split load from dequant+store ---
     *
     * issueLoads(t): fire off global memory reads into registers (non-blocking).
     * storeToShared(buf): dequant B in registers, write A & B to smem.
     *
     * The K-loop interleaves: issueLoads → computeWMMA → storeToShared → sync
     * so that global loads overlap with Matrix Core WMMA execution.
     */

    uint32_t pf_a1[A_VL], pf_a2[A_VL];
    int      pf_am[A_VL], pf_ak[A_VL];
    [[maybe_unused]] uint32_t pf_b4 = 0;
    [[maybe_unused]] uint2    pf_bfp_lo = {0,0}, pf_bfp_hi = {0,0};

    auto issueLoads = [&](int t) __attribute__((always_inline))
    {
#pragma unroll
        for(int ld = 0; ld < A_VL; ld++)
        {
            int vid    = tid + ld * THR_L;
            pf_ak[ld]  = (vid % A_GRP_K) * 2;
            pf_am[ld]  = (vid / A_GRP_K) * 2;
            if constexpr (!BOUNDS_CHECK) {
                pf_a1[ld] = *reinterpret_cast<const uint32_t*>(
                                  &A[(row0 + pf_am[ld]) * lda + t + pf_ak[ld]]);
                pf_a2[ld] = *reinterpret_cast<const uint32_t*>(
                                  &A[(row0 + pf_am[ld] + 1) * lda + t + pf_ak[ld]]);
            } else {
                int row_a = row0 + pf_am[ld];
                if (row_a + 1 < M) {
                    pf_a1[ld] = *reinterpret_cast<const uint32_t*>(
                                      &A[row_a * lda + t + pf_ak[ld]]);
                    pf_a2[ld] = *reinterpret_cast<const uint32_t*>(
                                      &A[(row_a + 1) * lda + t + pf_ak[ld]]);
                } else if (row_a < M) {
                    pf_a1[ld] = *reinterpret_cast<const uint32_t*>(
                                      &A[row_a * lda + t + pf_ak[ld]]);
                    pf_a2[ld] = 0;
                } else {
                    pf_a1[ld] = 0;
                    pf_a2[ld] = 0;
                }
            }
        }

        if (b_valid) {
            if constexpr (FUSED_DQ) {
                pf_b4 = *reinterpret_cast<const uint32_t*>(
                    &B_packed[b_byte_base + (t >> 1)]);
                // This thread dequants 8 contiguous nibbles at k=[t+b_k8 ..
                // t+b_k8+7]. With block_size>=16 (kernel precondition) and b_k8
                // a multiple of 8, those 8 weights lie entirely in ONE quant
                // group: (t + b_k8) / group_size. Deriving the group from t
                // ALONE is only valid when BK_T <= group_size; the 128x32
                // WT2x1 tile has BK_T=64, so on a group_size<64 model (e.g. 32)
                // a single K-step straddles two groups and the upper half would
                // otherwise be dequantized with the previous group's scale.
                // Cache keyed on the group index so the common BK_T<=group_size
                // case still skips the reload across K-steps.
                const int grp = (t + b_k8) / group_size;
                if(__builtin_expect(grp != cached_k_grp, 0))
                {
                    cached_k_grp = grp;
                    cached_s     = scales[grp + b_gid_base];
                    if constexpr(USE_ZEROS)
                        neg_zs = -zeros[grp + b_gid_base] * cached_s;
                }
            } else {
                const _Float16* b_src = &B_fp16[b_n_g * K + t + b_k8];
                pf_bfp_lo = *reinterpret_cast<const uint2*>(&b_src[0]);
                pf_bfp_hi = *reinterpret_cast<const uint2*>(&b_src[4]);
            }
        }
    };

    auto storeToShared = [&](int buf) __attribute__((always_inline))
    {
        if (b_valid) {
            if constexpr (FUSED_DQ) {
                _Float16 s16 = cached_s;
                _Float16 bv[8];
                if constexpr (USE_ZEROS) {
                    _Float16 nzs = neg_zs;
                    bv[0] = static_cast<_Float16>((pf_b4      ) & 0xFu) * s16 + nzs;
                    bv[1] = static_cast<_Float16>((pf_b4 >>  4) & 0xFu) * s16 + nzs;
                    bv[2] = static_cast<_Float16>((pf_b4 >>  8) & 0xFu) * s16 + nzs;
                    bv[3] = static_cast<_Float16>((pf_b4 >> 12) & 0xFu) * s16 + nzs;
                    bv[4] = static_cast<_Float16>((pf_b4 >> 16) & 0xFu) * s16 + nzs;
                    bv[5] = static_cast<_Float16>((pf_b4 >> 20) & 0xFu) * s16 + nzs;
                    bv[6] = static_cast<_Float16>((pf_b4 >> 24) & 0xFu) * s16 + nzs;
                    bv[7] = static_cast<_Float16>((pf_b4 >> 28)       ) * s16 + nzs;
                } else {
                    _Float16 nzs = (_Float16)(-8) * s16;
                    bv[0] = static_cast<_Float16>((pf_b4      ) & 0xFu) * s16 + nzs;
                    bv[1] = static_cast<_Float16>((pf_b4 >>  4) & 0xFu) * s16 + nzs;
                    bv[2] = static_cast<_Float16>((pf_b4 >>  8) & 0xFu) * s16 + nzs;
                    bv[3] = static_cast<_Float16>((pf_b4 >> 12) & 0xFu) * s16 + nzs;
                    bv[4] = static_cast<_Float16>((pf_b4 >> 16) & 0xFu) * s16 + nzs;
                    bv[5] = static_cast<_Float16>((pf_b4 >> 20) & 0xFu) * s16 + nzs;
                    bv[6] = static_cast<_Float16>((pf_b4 >> 24) & 0xFu) * s16 + nzs;
                    bv[7] = static_cast<_Float16>((pf_b4 >> 28)       ) * s16 + nzs;
                }
                *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8])     = *reinterpret_cast<uint2*>(&bv[0]);
                *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8 + 4]) = *reinterpret_cast<uint2*>(&bv[4]);
            } else {
                *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8])     = pf_bfp_lo;
                *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8 + 4]) = pf_bfp_hi;
            }
        } else {
            uint2 zero2 = {0, 0};
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8])     = zero2;
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8 + 4]) = zero2;
        }

#pragma unroll
        for(int ld = 0; ld < A_VL; ld++)
        {
            *reinterpret_cast<uint32_t*>(&smA[buf][pf_am[ld]  ][pf_ak[ld]]) = pf_a1[ld];
            *reinterpret_cast<uint32_t*>(&smA[buf][pf_am[ld]+1][pf_ak[ld]]) = pf_a2[ld];
        }
    };

    auto computeWMMA = [&](int buf) __attribute__((always_inline))
    {
#pragma unroll
        for(int ks = 0; ks < K_STEPS; ks++)
        {
            half16 b_frag[WT_N];
            const int k_off = hipdnn_wmma_k_off(sub);
#pragma unroll
            for(int wn = 0; wn < WT_N; wn++)
            {
                int noff              = (wcol * WT_N + wn) * WMMA_TILE;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&b_frag[wn]);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smB[buf][noff + lane][ks * WMMA_TILE + k_off]);
#pragma unroll
                for(int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 2; i++)
                    dst[i] = src[i];
            }
#pragma unroll
            for(int wm = 0; wm < WT_M; wm++)
            {
                int moff              = (wrow * WT_M + wm) * WMMA_TILE;
                half16 a_frag;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&a_frag);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smA[buf][moff + lane][ks * WMMA_TILE + k_off]);
#pragma unroll
                for(int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 2; i++)
                    dst[i] = src[i];

#pragma unroll
                for(int wn = 0; wn < WT_N; wn++)
                    acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(
                        a_frag, b_frag[wn], acc[wm][wn]);
            }
        }
    };

    issueLoads(0);
    storeToShared(0);
    __syncthreads();

    int buf = 0;
    for(int t = BK_T; t < K; t += BK_T)
    {
        int nxt = 1 - buf;
        issueLoads(t);
        computeWMMA(buf);
        storeToShared(nxt);
        __syncthreads();
        buf = nxt;
    }
    computeWMMA(buf);

#pragma unroll
    for(int wm = 0; wm < WT_M; wm++)
    {
        int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
        for(int wn = 0; wn < WT_N; wn++)
        {
            int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
#pragma unroll
            for(int e = 0; e < 8; e++)
            {
                int r = hipdnn_wmma_acc_row(sub, e);
                if constexpr (BOUNDS_CHECK) {
                    if (mbase + r < M && nbase + lane < N)
                        C[(mbase + r) * ldc + nbase + lane] =
                            (_Float16)acc[wm][wn][e];
                } else {
                    C[(mbase + r) * ldc + nbase + lane] =
                        (_Float16)acc[wm][wn][e];
                }
            }
        }
    }
}


#define GEMM_WG_SZ(BM, BN, WM, WN) \
    ((BM / (WM * 16)) * (BN / (WN * 16)) * 32)

template <int BM_T = 128, int BN_T = 128, int WT_M = 2, int WT_N = 2,
          int WAVES_EU = 8, bool BC = false>
__global__
__attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N),
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(WAVES_EU)))
void MatMulNBitsWMMA_NoZP(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B_packed,
    const _Float16* __restrict__ scales,
    const _Float16* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16U4Impl<BM_T, BN_T, WT_M, WT_N, false, BC>(
        M, N, K, A, lda, B_packed, scales, zeros,
        group_size, num_groups_k, C, ldc, swizzle_n);
}


template <int BM_T = 128, int BN_T = 128, int WT_M = 2, int WT_N = 2,
          int WAVES_EU = 8, bool BC = false>
__global__
__attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N),
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(WAVES_EU)))
void MatMulNBitsWMMA_ZP(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B_packed,
    const _Float16* __restrict__ scales,
    const _Float16* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16U4Impl<BM_T, BN_T, WT_M, WT_N, true, BC>(
        M, N, K, A, lda, B_packed, scales, zeros,
        group_size, num_groups_k, C, ldc, swizzle_n);
}


template <int BM_T = 128, int BN_T = 128, int WT_M = 2, int WT_N = 2,
          int WAVES_EU = 8, bool BC = false>
__global__
__attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N),
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(WAVES_EU)))
void MatMulNBitsFp16GEMM(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const _Float16* __restrict__ B_fp16,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16U4Impl<BM_T, BN_T, WT_M, WT_N, false, BC, false>(
        M, N, K, A, lda, B_fp16, nullptr, nullptr,
        0, 0, C, ldc, swizzle_n);
}


__global__ void transpose2d_fp16_kernel(
    const _Float16* __restrict__ src,
    _Float16* __restrict__ dst,
    int rows, int cols)
{
    constexpr int TILE = 32;
    __shared__ _Float16 tile[TILE][TILE + 1];

    int bx = blockIdx.x * TILE;
    int by = blockIdx.y * TILE;

    int x = bx + threadIdx.x;
    int y = by + threadIdx.y;

    for (int j = 0; j < TILE; j += blockDim.y) {
        int yy = y + j;
        if (x < cols && yy < rows)
            tile[threadIdx.y + j][threadIdx.x] = src[yy * cols + x];
    }
    __syncthreads();

    x = by + threadIdx.x;
    y = bx + threadIdx.y;
    for (int j = 0; j < TILE; j += blockDim.y) {
        int yy = y + j;
        if (x < rows && yy < cols)
            dst[yy * rows + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}


template <int BM_T, int BN_T, int WT_M, int WT_N, bool USE_ZEROS,
          bool BOUNDS_CHECK = false>
__device__ __forceinline__
void GemmFp16I8Impl(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B,
    const _Float16* __restrict__ scales,
    const unsigned char* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    constexpr int WM_L   = BM_T / (WT_M * 16);
    constexpr int WN_L   = BN_T / (WT_N * 16);
    constexpr int THR_L  = WM_L * WN_L * 32;
    constexpr int BK_T   = BM_T / (WT_M * WT_N);

    constexpr int PAD_K   = 2;
    constexpr int K_STR   = BK_T + PAD_K;
    constexpr int A_VL    = (BM_T * BK_T) / (THR_L * 4);
    constexpr int A_GRP_K = BK_T / 2;
    constexpr int K_STEPS = BK_T / WMMA_TILE;

    static_assert(A_VL >= 1, "A_VL must be >= 1");
    static_assert(THR_L * 8 == BN_T * BK_T,
                  "B loading balance: THR*8 must equal BN*BK");

    __shared__ _Float16 smA[2][BM_T][K_STR];
    __shared__ _Float16 smB[2][BN_T][K_STR];

    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lid  = tid % 32;
    const int lane = lid % 16;
    const int sub  = lid / 16;
    const int wrow = wid / WN_L;
    const int wcol = wid % WN_L;

    // Block-to-tile mapping with column swizzle for L2 locality.
    // Splits the grid into a "main" region of (n_tiles/sw)*sw cols where full
    // sw-wide groups apply, and a "tail" region of (n_tiles%sw) leftover cols
    // that uses plain row-major. The previous implementation fell back to
    // row-major only when bx >= n_tiles, which produced both collisions AND
    // missing tiles when n_tiles % sw != 0, leaving those tiles with stale
    // output data from prior kernel launches.
    const int n_tiles  = gridDim.x;
    const int m_tiles  = gridDim.y;
    const int block_id = blockIdx.y * n_tiles + blockIdx.x;
    const int sw          = (n_tiles >= swizzle_n) ? swizzle_n : n_tiles;
    const int main_cols   = (n_tiles / sw) * sw;
    const int main_blocks = main_cols * m_tiles;
    int by, bx;
    if (block_id < main_blocks) {
        const int super = block_id / (sw * m_tiles);
        const int rem   = block_id % (sw * m_tiles);
        by = rem / sw;
        bx = super * sw + rem % sw;
    } else {
        const int tail_id   = block_id - main_blocks;
        const int tail_cols = n_tiles - main_cols;
        by = tail_id / tail_cols;
        bx = main_cols + (tail_id % tail_cols);
    }
    const int row0 = by * BM_T;
    const int col0 = bx * BN_T;

    // Each thread is responsible for an 8-element K-strip in one of the BN
    // columns. With THR*8 == BN*BK, threads divide neatly into
    // (BN columns) x (BK/8 K-strips per column).
    constexpr int B_COL_SHIFT = __builtin_ctz(BK_T / 8);
    const int b_col       = tid >> B_COL_SHIFT;
    const int b_sub       = tid & ((1 << B_COL_SHIFT) - 1);
    const int b_k8        = b_sub << 3;
    const int b_n_g_raw   = col0 + b_col;
    const bool b_valid    = BOUNDS_CHECK ? (b_n_g_raw < N) : true;
    const int b_n_g       = (BOUNDS_CHECK && !b_valid) ? 0 : b_n_g_raw;

    // Per-thread base offsets (constant across the K-loop).
    const int b_elem_base = b_n_g * K + b_k8;
    const int b_gid_base  = b_n_g * num_groups_k;
    // -1 forces a scale load on the first K-step; the group index is derived
    // from the thread's actual k offset (t+b_k8), see issueLoads below.
    int      cached_k_grp = -1;
    _Float16 cached_s     = 0;
    _Float16 neg_zs       = 0;

    float8 acc[WT_M][WT_N];
#pragma unroll
    for (int i = 0; i < WT_M; i++)
#pragma unroll
        for (int j = 0; j < WT_N; j++)
#pragma unroll
            for (int e = 0; e < 8; e++)
                acc[i][j][e] = 0.0f;

    // Register prefetch slots, mirroring the int4 path.
    uint32_t pf_a1[A_VL], pf_a2[A_VL];
    int      pf_am[A_VL], pf_ak[A_VL];
    uint2    pf_b8 = {0, 0};

    auto issueLoads = [&](int t) __attribute__((always_inline)) {
#pragma unroll
        for (int ld = 0; ld < A_VL; ld++) {
            int vid    = tid + ld * THR_L;
            pf_ak[ld]  = (vid % A_GRP_K) * 2;
            pf_am[ld]  = (vid / A_GRP_K) * 2;
            if constexpr (!BOUNDS_CHECK) {
                pf_a1[ld] = *reinterpret_cast<const uint32_t*>(
                                  &A[(row0 + pf_am[ld]) * lda + t + pf_ak[ld]]);
                pf_a2[ld] = *reinterpret_cast<const uint32_t*>(
                                  &A[(row0 + pf_am[ld] + 1) * lda + t + pf_ak[ld]]);
            } else {
                int row_a = row0 + pf_am[ld];
                if (row_a + 1 < M) {
                    pf_a1[ld] = *reinterpret_cast<const uint32_t*>(
                                      &A[row_a * lda + t + pf_ak[ld]]);
                    pf_a2[ld] = *reinterpret_cast<const uint32_t*>(
                                      &A[(row_a + 1) * lda + t + pf_ak[ld]]);
                } else if (row_a < M) {
                    pf_a1[ld] = *reinterpret_cast<const uint32_t*>(
                                      &A[row_a * lda + t + pf_ak[ld]]);
                    pf_a2[ld] = 0;
                } else {
                    pf_a1[ld] = 0;
                    pf_a2[ld] = 0;
                }
            }
        }

        if (b_valid) {
            // Load 8 bytes (= 8 int8 weights) per K-step. K-step is `t`,
            // and per-thread offset within the step is `b_k8`. The base
            // already includes b_k8, so the byte address is `b_elem_base + t`.
            pf_b8 = *reinterpret_cast<const uint2*>(&B[b_elem_base + t]);

            // Group/scale stepping: this thread loads 8 contiguous weights at
            // k=[t+b_k8 .. t+b_k8+7], which lie in one quant group
            // (t+b_k8)/group_size (block_size>=16, b_k8 multiple of 8). Derive
            // the group from the thread's actual k offset, NOT from t alone:
            // when BK_T > group_size a single K-step straddles two groups and
            // the upper half must use a later scale. Cache keyed on the group
            // index so BK_T<=group_size still reuses the load across K-steps.
            const int grp = (t + b_k8) / group_size;
            if (__builtin_expect(grp != cached_k_grp, 0)) {
                cached_k_grp = grp;
                cached_s = scales[grp + b_gid_base];
                if constexpr (USE_ZEROS) {
                    _Float16 zp_h = static_cast<_Float16>(
                                        zeros[grp + b_gid_base]);
                    neg_zs = -zp_h * cached_s;
                } else {
                    // ONNX default zp = 2^(bits-1) = 128 for bits=8.
                    neg_zs = (_Float16)(-128) * cached_s;
                }
            }
        }
    };

    auto storeToShared = [&](int buf) __attribute__((always_inline)) {
        if (b_valid) {
            _Float16 s16 = cached_s;
            _Float16 nzs = neg_zs;
            _Float16 bv[8];

            // Decode 8 uint8 weights from the two 32-bit lanes of pf_b8.
            // For each byte: bv[i] = (byte) * scale + (-zp * scale)
            //                       = (byte - zp) * scale
            // The factored form lets the FMA tree fuse mul+add into one op.
            bv[0] = static_cast<_Float16>((pf_b8.x      ) & 0xFFu) * s16 + nzs;
            bv[1] = static_cast<_Float16>((pf_b8.x >>  8) & 0xFFu) * s16 + nzs;
            bv[2] = static_cast<_Float16>((pf_b8.x >> 16) & 0xFFu) * s16 + nzs;
            bv[3] = static_cast<_Float16>((pf_b8.x >> 24)        ) * s16 + nzs;
            bv[4] = static_cast<_Float16>((pf_b8.y      ) & 0xFFu) * s16 + nzs;
            bv[5] = static_cast<_Float16>((pf_b8.y >>  8) & 0xFFu) * s16 + nzs;
            bv[6] = static_cast<_Float16>((pf_b8.y >> 16) & 0xFFu) * s16 + nzs;
            bv[7] = static_cast<_Float16>((pf_b8.y >> 24)        ) * s16 + nzs;

            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8])     =
                *reinterpret_cast<uint2*>(&bv[0]);
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8 + 4]) =
                *reinterpret_cast<uint2*>(&bv[4]);
        } else {
            uint2 zero2 = {0, 0};
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8])     = zero2;
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8 + 4]) = zero2;
        }

#pragma unroll
        for (int ld = 0; ld < A_VL; ld++) {
            *reinterpret_cast<uint32_t*>(&smA[buf][pf_am[ld]    ][pf_ak[ld]]) =
                pf_a1[ld];
            *reinterpret_cast<uint32_t*>(&smA[buf][pf_am[ld] + 1][pf_ak[ld]]) =
                pf_a2[ld];
        }
    };

    auto computeWMMA = [&](int buf) __attribute__((always_inline)) {
#pragma unroll
        for (int ks = 0; ks < K_STEPS; ks++) {
            half16 b_frag[WT_N];
            const int k_off = hipdnn_wmma_k_off(sub);
#pragma unroll
            for (int wn = 0; wn < WT_N; wn++) {
                int noff              = (wcol * WT_N + wn) * WMMA_TILE;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&b_frag[wn]);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smB[buf][noff + lane][ks * WMMA_TILE + k_off]);
#pragma unroll
                for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 2; i++)
                    dst[i] = src[i];
            }
#pragma unroll
            for (int wm = 0; wm < WT_M; wm++) {
                int moff              = (wrow * WT_M + wm) * WMMA_TILE;
                half16 a_frag;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&a_frag);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smA[buf][moff + lane][ks * WMMA_TILE + k_off]);
#pragma unroll
                for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 2; i++)
                    dst[i] = src[i];

#pragma unroll
                for (int wn = 0; wn < WT_N; wn++)
                    acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(
                        a_frag, b_frag[wn], acc[wm][wn]);
            }
        }
    };

    issueLoads(0);
    storeToShared(0);
    __syncthreads();

    int buf = 0;
    for (int t = BK_T; t < K; t += BK_T) {
        int nxt = 1 - buf;
        issueLoads(t);
        computeWMMA(buf);
        storeToShared(nxt);
        __syncthreads();
        buf = nxt;
    }
    computeWMMA(buf);

#pragma unroll
    for (int wm = 0; wm < WT_M; wm++) {
        int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
        for (int wn = 0; wn < WT_N; wn++) {
            int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                int r = hipdnn_wmma_acc_row(sub, e);
                if constexpr (BOUNDS_CHECK) {
                    if (mbase + r < M && nbase + lane < N)
                        C[(mbase + r) * ldc + nbase + lane] =
                            (_Float16)acc[wm][wn][e];
                } else {
                    C[(mbase + r) * ldc + nbase + lane] =
                        (_Float16)acc[wm][wn][e];
                }
            }
        }
    }
}


template <int BM_T = 128, int BN_T = 128, int WT_M = 2, int WT_N = 2,
          int WAVES_EU = 8, bool BC = false>
__global__
__attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N),
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(WAVES_EU)))
void MatMulNBitsWMMA_I8_NoZP(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B,
    const _Float16* __restrict__ scales,
    const unsigned char* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16I8Impl<BM_T, BN_T, WT_M, WT_N, false, BC>(
        M, N, K, A, lda, B, scales, zeros,
        group_size, num_groups_k, C, ldc, swizzle_n);
}


template <int BM_T = 128, int BN_T = 128, int WT_M = 2, int WT_N = 2,
          int WAVES_EU = 8, bool BC = false>
__global__
__attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N),
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(WAVES_EU)))
void MatMulNBitsWMMA_I8_ZP(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B,
    const _Float16* __restrict__ scales,
    const unsigned char* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16I8Impl<BM_T, BN_T, WT_M, WT_N, true, BC>(
        M, N, K, A, lda, B, scales, zeros,
        group_size, num_groups_k, C, ldc, swizzle_n);
}


template <int BM_T, int BN_T, int WT_M, int WT_N, bool USE_ZEROS,
          bool BOUNDS_CHECK = false>
__device__ __forceinline__
void GemmFp16U3Impl(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B,
    const _Float16* __restrict__ scales,
    const unsigned char* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    constexpr int WM_L   = BM_T / (WT_M * 16);
    constexpr int WN_L   = BN_T / (WT_N * 16);
    constexpr int THR_L  = WM_L * WN_L * 32;
    constexpr int BK_T   = BM_T / (WT_M * WT_N);

    constexpr int PAD_K   = 2;
    constexpr int K_STR   = BK_T + PAD_K;
    constexpr int A_VL    = (BM_T * BK_T) / (THR_L * 4);
    constexpr int A_GRP_K = BK_T / 2;
    constexpr int K_STEPS = BK_T / WMMA_TILE;

    static_assert(A_VL >= 1, "A_VL must be >= 1");
    static_assert(THR_L * 8 == BN_T * BK_T,
                  "B loading balance: THR*8 must equal BN*BK");
    static_assert(BK_T % 8 == 0,
                  "BK_T must be a multiple of 8 for byte-exact uint3 chunks");

    __shared__ _Float16 smA[2][BM_T][K_STR];
    __shared__ _Float16 smB[2][BN_T][K_STR];

    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lid  = tid % 32;
    const int lane = lid % 16;
    const int sub  = lid / 16;
    const int wrow = wid / WN_L;
    const int wcol = wid % WN_L;

    const int n_tiles  = gridDim.x;
    const int m_tiles  = gridDim.y;
    const int block_id = blockIdx.y * n_tiles + blockIdx.x;
    const int sw          = (n_tiles >= swizzle_n) ? swizzle_n : n_tiles;
    const int main_cols   = (n_tiles / sw) * sw;
    const int main_blocks = main_cols * m_tiles;
    int by, bx;
    if (block_id < main_blocks) {
        const int super = block_id / (sw * m_tiles);
        const int rem   = block_id % (sw * m_tiles);
        by = rem / sw;
        bx = super * sw + rem % sw;
    } else {
        const int tail_id   = block_id - main_blocks;
        const int tail_cols = n_tiles - main_cols;
        by = tail_id / tail_cols;
        bx = main_cols + (tail_id % tail_cols);
    }
    const int row0 = by * BM_T;
    const int col0 = bx * BN_T;

    // Each thread owns an 8-element K-strip in one of the BN columns, same
    // division as int4/int8 (THR*8 == BN*BK).
    constexpr int B_COL_SHIFT = __builtin_ctz(BK_T / 8);
    const int b_col       = tid >> B_COL_SHIFT;
    const int b_sub       = tid & ((1 << B_COL_SHIFT) - 1);
    const int b_k8        = b_sub << 3;
    const int b_n_g_raw   = col0 + b_col;
    const bool b_valid    = BOUNDS_CHECK ? (b_n_g_raw < N) : true;
    const int b_n_g       = (BOUNDS_CHECK && !b_valid) ? 0 : b_n_g_raw;

    // K % 32 == 0 (enforced at dispatch) implies K % 8 == 0, so this row
    // stride is always exact -- no ceil-division needed here.
    const int row_bytes    = (K * 3) >> 3;
    // Per-thread fixed byte offset (b_k8 is a multiple of 8, so *3 is
    // always a multiple of 8 -- see the alignment note above).
    const int b_byte_base  = b_n_g * row_bytes + ((b_k8 * 3) >> 3);
    const int b_gid_base   = b_n_g * num_groups_k;
    int      cached_k_grp  = -1;
    _Float16 cached_s      = 0;
    _Float16 neg_zs        = 0;

    float8 acc[WT_M][WT_N];
#pragma unroll
    for (int i = 0; i < WT_M; i++)
#pragma unroll
        for (int j = 0; j < WT_N; j++)
#pragma unroll
            for (int e = 0; e < 8; e++)
                acc[i][j][e] = 0.0f;

    uint32_t pf_a1[A_VL], pf_a2[A_VL];
    int      pf_am[A_VL], pf_ak[A_VL];
    uint32_t pf_b3 = 0;  // low 24 bits = 8 packed 3-bit values

    auto issueLoads = [&](int t) __attribute__((always_inline)) {
#pragma unroll
        for (int ld = 0; ld < A_VL; ld++) {
            int vid    = tid + ld * THR_L;
            pf_ak[ld]  = (vid % A_GRP_K) * 2;
            pf_am[ld]  = (vid / A_GRP_K) * 2;
            if constexpr (!BOUNDS_CHECK) {
                pf_a1[ld] = *reinterpret_cast<const uint32_t*>(
                                  &A[(row0 + pf_am[ld]) * lda + t + pf_ak[ld]]);
                pf_a2[ld] = *reinterpret_cast<const uint32_t*>(
                                  &A[(row0 + pf_am[ld] + 1) * lda + t + pf_ak[ld]]);
            } else {
                int row_a = row0 + pf_am[ld];
                if (row_a + 1 < M) {
                    pf_a1[ld] = *reinterpret_cast<const uint32_t*>(
                                      &A[row_a * lda + t + pf_ak[ld]]);
                    pf_a2[ld] = *reinterpret_cast<const uint32_t*>(
                                      &A[(row_a + 1) * lda + t + pf_ak[ld]]);
                } else if (row_a < M) {
                    pf_a1[ld] = *reinterpret_cast<const uint32_t*>(
                                      &A[row_a * lda + t + pf_ak[ld]]);
                    pf_a2[ld] = 0;
                } else {
                    pf_a1[ld] = 0;
                    pf_a2[ld] = 0;
                }
            }
        }

        if (b_valid) {
            // t is always a multiple of BK_T, which is always a multiple of
            // 8 (static_assert above), so (t*3)>>3 is exact -- no rounding.
            const int off  = b_byte_base + ((t * 3) >> 3);
            const uint32_t lo16 = *reinterpret_cast<const uint16_t*>(&B[off]);
            const uint32_t hi8  = B[off + 2];
            pf_b3 = lo16 | (hi8 << 16);

            // Same group-straddling handling as int4/int8: derive the group
            // from this thread's actual k offset (t+b_k8), not from t alone,
            // and cache across K-steps that share a group.
            const int grp = (t + b_k8) / group_size;
            if (__builtin_expect(grp != cached_k_grp, 0)) {
                cached_k_grp = grp;
                cached_s = scales[grp + b_gid_base];
                if constexpr (USE_ZEROS) {
                    _Float16 zp_h = static_cast<_Float16>(
                                        zeros[grp + b_gid_base]);
                    neg_zs = -zp_h * cached_s;
                } else {
                    // Default zp = 2^(bits-1) = 4 for bits=3.
                    neg_zs = (_Float16)(-4) * cached_s;
                }
            }
        }
    };

    auto storeToShared = [&](int buf) __attribute__((always_inline)) {
        if (b_valid) {
            _Float16 s16 = cached_s;
            _Float16 nzs = neg_zs;
            _Float16 bv[8];

            // 8 values x 3 bits = 24 bits, fully contained in pf_b3's low
            // 24 bits -- no cross-register boundary to handle (max shift
            // 3*7=21, value occupies bits [21,24)).
#pragma unroll
            for (int i = 0; i < 8; i++)
                bv[i] = static_cast<_Float16>((pf_b3 >> (3 * i)) & 0x7u) * s16 + nzs;

            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8])     =
                *reinterpret_cast<uint2*>(&bv[0]);
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8 + 4]) =
                *reinterpret_cast<uint2*>(&bv[4]);
        } else {
            uint2 zero2 = {0, 0};
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8])     = zero2;
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8 + 4]) = zero2;
        }

#pragma unroll
        for (int ld = 0; ld < A_VL; ld++) {
            *reinterpret_cast<uint32_t*>(&smA[buf][pf_am[ld]    ][pf_ak[ld]]) =
                pf_a1[ld];
            *reinterpret_cast<uint32_t*>(&smA[buf][pf_am[ld] + 1][pf_ak[ld]]) =
                pf_a2[ld];
        }
    };

    auto computeWMMA = [&](int buf) __attribute__((always_inline)) {
#pragma unroll
        for (int ks = 0; ks < K_STEPS; ks++) {
            half16 b_frag[WT_N];
            const int k_off = hipdnn_wmma_k_off(sub);
#pragma unroll
            for (int wn = 0; wn < WT_N; wn++) {
                int noff              = (wcol * WT_N + wn) * WMMA_TILE;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&b_frag[wn]);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smB[buf][noff + lane][ks * WMMA_TILE + k_off]);
#pragma unroll
                for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 2; i++)
                    dst[i] = src[i];
            }
#pragma unroll
            for (int wm = 0; wm < WT_M; wm++) {
                int moff              = (wrow * WT_M + wm) * WMMA_TILE;
                half16 a_frag;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&a_frag);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smA[buf][moff + lane][ks * WMMA_TILE + k_off]);
#pragma unroll
                for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 2; i++)
                    dst[i] = src[i];

#pragma unroll
                for (int wn = 0; wn < WT_N; wn++)
                    acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(
                        a_frag, b_frag[wn], acc[wm][wn]);
            }
        }
    };

    issueLoads(0);
    storeToShared(0);
    __syncthreads();

    int buf = 0;
    for (int t = BK_T; t < K; t += BK_T) {
        int nxt = 1 - buf;
        issueLoads(t);
        computeWMMA(buf);
        storeToShared(nxt);
        __syncthreads();
        buf = nxt;
    }
    computeWMMA(buf);

#pragma unroll
    for (int wm = 0; wm < WT_M; wm++) {
        int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
        for (int wn = 0; wn < WT_N; wn++) {
            int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                int r = hipdnn_wmma_acc_row(sub, e);
                if constexpr (BOUNDS_CHECK) {
                    if (mbase + r < M && nbase + lane < N)
                        C[(mbase + r) * ldc + nbase + lane] =
                            (_Float16)acc[wm][wn][e];
                } else {
                    C[(mbase + r) * ldc + nbase + lane] =
                        (_Float16)acc[wm][wn][e];
                }
            }
        }
    }
}


template <int BM_T = 128, int BN_T = 128, int WT_M = 2, int WT_N = 2,
          int WAVES_EU = 8, bool BC = false>
__global__
__attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N),
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(WAVES_EU)))
void MatMulNBitsWMMA_U3_NoZP(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B,
    const _Float16* __restrict__ scales,
    const unsigned char* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16U3Impl<BM_T, BN_T, WT_M, WT_N, false, BC>(
        M, N, K, A, lda, B, scales, zeros,
        group_size, num_groups_k, C, ldc, swizzle_n);
}


template <int BM_T = 128, int BN_T = 128, int WT_M = 2, int WT_N = 2,
          int WAVES_EU = 8, bool BC = false>
__global__
__attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N),
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(WAVES_EU)))
void MatMulNBitsWMMA_U3_ZP(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B,
    const _Float16* __restrict__ scales,
    const unsigned char* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16U3Impl<BM_T, BN_T, WT_M, WT_N, true, BC>(
        M, N, K, A, lda, B, scales, zeros,
        group_size, num_groups_k, C, ldc, swizzle_n);
}


/* -----------------------------------------------------------------------
 * uint3 -> fp16 dequant (for the separate-dequant "dq+gemm" path)
 *
 * Decodes the continuous per-row 3-bit bitstream into a dense fp16 [N, K]
 * buffer so the plain fp16 WMMA GEMM (MatMulNBitsFp16GEMM, shared with the
 * bits=4 path) can run at peak WMMA throughput with no in-loop decode.
 * Each thread handles one 8-value chunk: 8 values * 3 bits = 24 bits pack
 * into exactly 3 bytes, so the read never crosses into a 4th byte (same
 * alignment proof as the fused kernel). Mirrors dequant_u4_to_fp16.
 * ----------------------------------------------------------------------- */
__global__ void dequant_u3_to_fp16(
    const unsigned char* __restrict__ B,       // [N, (K*3)/8] bitstream
    const _Float16* __restrict__ scales,       // [N, num_groups_k]
    const unsigned char* __restrict__ zeros,   // [N, num_groups_k] or null
    _Float16* __restrict__ B_fp16,             // [N, K]
    int N, int K, int group_size, int num_groups_k)
{
    int n  = static_cast<int>(blockIdx.y);
    int k8 = (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (n >= N || k8 >= K) return;

    int grp = k8 / group_size;
    _Float16 scale = scales[n * num_groups_k + grp];
    // Default zp = 2^(bits-1) = 4 for bits=3 (matches the fused u3 path).
    _Float16 zp = zeros ? static_cast<_Float16>(zeros[n * num_groups_k + grp])
                        : (_Float16)4;
    _Float16 nzs = -zp * scale;

    // K % 32 == 0 (dispatch gate) => K % 8 == 0, so row stride is exact and
    // (k8 * 3) is a multiple of 8 -> byte offset needs no rounding.
    const int row_bytes = (K * 3) >> 3;
    const int off  = n * row_bytes + ((k8 * 3) >> 3);
    const uint32_t lo16 = *reinterpret_cast<const uint16_t*>(&B[off]);
    const uint32_t hi8  = B[off + 2];
    const uint32_t p    = lo16 | (hi8 << 16);  // low 24 bits = 8 packed vals

    _Float16 bv[8];
#pragma unroll
    for (int i = 0; i < 8; i++)
        bv[i] = static_cast<_Float16>((p >> (3 * i)) & 0x7u) * scale + nzs;

    _Float16* dst = &B_fp16[n * K + k8];
    *reinterpret_cast<uint2*>(dst)     = *reinterpret_cast<uint2*>(&bv[0]);
    *reinterpret_cast<uint2*>(dst + 4) = *reinterpret_cast<uint2*>(&bv[4]);
}


template <int BM_T, int BN_T, int WT_M, int WT_N, bool USE_ZEROS,
          bool BOUNDS_CHECK = false>
__device__ __forceinline__
void GemmFp16U2Impl(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B,
    const _Float16* __restrict__ scales,
    const unsigned char* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    constexpr int WM_L   = BM_T / (WT_M * 16);
    constexpr int WN_L   = BN_T / (WT_N * 16);
    constexpr int THR_L  = WM_L * WN_L * 32;
    constexpr int BK_T   = BM_T / (WT_M * WT_N);

    constexpr int PAD_K   = 2;
    constexpr int K_STR   = BK_T + PAD_K;
    constexpr int A_VL    = (BM_T * BK_T) / (THR_L * 4);
    constexpr int A_GRP_K = BK_T / 2;
    constexpr int K_STEPS = BK_T / WMMA_TILE;

    static_assert(A_VL >= 1, "A_VL must be >= 1");
    static_assert(THR_L * 8 == BN_T * BK_T,
                  "B loading balance: THR*8 must equal BN*BK");
    static_assert(BK_T % 8 == 0,
                  "BK_T must be a multiple of 8 for byte-exact uint2 chunks");

    __shared__ _Float16 smA[2][BM_T][K_STR];
    __shared__ _Float16 smB[2][BN_T][K_STR];

    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lid  = tid % 32;
    const int lane = lid % 16;
    const int sub  = lid / 16;
    const int wrow = wid / WN_L;
    const int wcol = wid % WN_L;

    const int n_tiles  = gridDim.x;
    const int m_tiles  = gridDim.y;
    const int block_id = blockIdx.y * n_tiles + blockIdx.x;
    const int sw          = (n_tiles >= swizzle_n) ? swizzle_n : n_tiles;
    const int main_cols   = (n_tiles / sw) * sw;
    const int main_blocks = main_cols * m_tiles;
    int by, bx;
    if (block_id < main_blocks) {
        const int super = block_id / (sw * m_tiles);
        const int rem   = block_id % (sw * m_tiles);
        by = rem / sw;
        bx = super * sw + rem % sw;
    } else {
        const int tail_id   = block_id - main_blocks;
        const int tail_cols = n_tiles - main_cols;
        by = tail_id / tail_cols;
        bx = main_cols + (tail_id % tail_cols);
    }
    const int row0 = by * BM_T;
    const int col0 = bx * BN_T;

    constexpr int B_COL_SHIFT = __builtin_ctz(BK_T / 8);
    const int b_col       = tid >> B_COL_SHIFT;
    const int b_sub       = tid & ((1 << B_COL_SHIFT) - 1);
    const int b_k8        = b_sub << 3;
    const int b_n_g_raw   = col0 + b_col;
    const bool b_valid    = BOUNDS_CHECK ? (b_n_g_raw < N) : true;
    const int b_n_g       = (BOUNDS_CHECK && !b_valid) ? 0 : b_n_g_raw;

    // K % 32 == 0 (enforced at dispatch) implies K % 4 == 0, so this row
    // stride is exact. b_k8 is a multiple of 8, so (b_k8*2)>>3 = b_k8/4 is
    // a multiple of 2 -> the per-thread uint16 load is always aligned.
    const int row_bytes    = (K * 2) >> 3;
    const int b_byte_base  = b_n_g * row_bytes + ((b_k8 * 2) >> 3);
    const int b_gid_base   = b_n_g * num_groups_k;
    int      cached_k_grp  = -1;
    _Float16 cached_s      = 0;
    _Float16 neg_zs        = 0;

    float8 acc[WT_M][WT_N];
#pragma unroll
    for (int i = 0; i < WT_M; i++)
#pragma unroll
        for (int j = 0; j < WT_N; j++)
#pragma unroll
            for (int e = 0; e < 8; e++)
                acc[i][j][e] = 0.0f;

    uint32_t pf_a1[A_VL], pf_a2[A_VL];
    int      pf_am[A_VL], pf_ak[A_VL];
    uint32_t pf_b2 = 0;  // low 16 bits = 8 packed 2-bit values

    auto issueLoads = [&](int t) __attribute__((always_inline)) {
#pragma unroll
        for (int ld = 0; ld < A_VL; ld++) {
            int vid    = tid + ld * THR_L;
            pf_ak[ld]  = (vid % A_GRP_K) * 2;
            pf_am[ld]  = (vid / A_GRP_K) * 2;
            if constexpr (!BOUNDS_CHECK) {
                pf_a1[ld] = *reinterpret_cast<const uint32_t*>(
                                  &A[(row0 + pf_am[ld]) * lda + t + pf_ak[ld]]);
                pf_a2[ld] = *reinterpret_cast<const uint32_t*>(
                                  &A[(row0 + pf_am[ld] + 1) * lda + t + pf_ak[ld]]);
            } else {
                int row_a = row0 + pf_am[ld];
                if (row_a + 1 < M) {
                    pf_a1[ld] = *reinterpret_cast<const uint32_t*>(
                                      &A[row_a * lda + t + pf_ak[ld]]);
                    pf_a2[ld] = *reinterpret_cast<const uint32_t*>(
                                      &A[(row_a + 1) * lda + t + pf_ak[ld]]);
                } else if (row_a < M) {
                    pf_a1[ld] = *reinterpret_cast<const uint32_t*>(
                                      &A[row_a * lda + t + pf_ak[ld]]);
                    pf_a2[ld] = 0;
                } else {
                    pf_a1[ld] = 0;
                    pf_a2[ld] = 0;
                }
            }
        }

        if (b_valid) {
            // t is a multiple of BK_T (multiple of 8), so (t*2)>>3 = t/4 is
            // even -> the 2-byte read is exact and aligned.
            const int off  = b_byte_base + ((t * 2) >> 3);
            pf_b2 = *reinterpret_cast<const uint16_t*>(&B[off]);

            const int grp = (t + b_k8) / group_size;
            if (__builtin_expect(grp != cached_k_grp, 0)) {
                cached_k_grp = grp;
                cached_s = scales[grp + b_gid_base];
                if constexpr (USE_ZEROS) {
                    _Float16 zp_h = static_cast<_Float16>(
                                        zeros[grp + b_gid_base]);
                    neg_zs = -zp_h * cached_s;
                } else {
                    // Default zp = 2^(bits-1) = 2 for bits=2.
                    neg_zs = (_Float16)(-2) * cached_s;
                }
            }
        }
    };

    auto storeToShared = [&](int buf) __attribute__((always_inline)) {
        if (b_valid) {
            _Float16 s16 = cached_s;
            _Float16 nzs = neg_zs;
            _Float16 bv[8];

            // 8 values x 2 bits = 16 bits, fully contained in pf_b2's low
            // 16 bits (max shift 2*7=14, value occupies bits [14,16)).
#pragma unroll
            for (int i = 0; i < 8; i++)
                bv[i] = static_cast<_Float16>((pf_b2 >> (2 * i)) & 0x3u) * s16 + nzs;

            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8])     =
                *reinterpret_cast<uint2*>(&bv[0]);
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8 + 4]) =
                *reinterpret_cast<uint2*>(&bv[4]);
        } else {
            uint2 zero2 = {0, 0};
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8])     = zero2;
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8 + 4]) = zero2;
        }

#pragma unroll
        for (int ld = 0; ld < A_VL; ld++) {
            *reinterpret_cast<uint32_t*>(&smA[buf][pf_am[ld]    ][pf_ak[ld]]) =
                pf_a1[ld];
            *reinterpret_cast<uint32_t*>(&smA[buf][pf_am[ld] + 1][pf_ak[ld]]) =
                pf_a2[ld];
        }
    };

    auto computeWMMA = [&](int buf) __attribute__((always_inline)) {
#pragma unroll
        for (int ks = 0; ks < K_STEPS; ks++) {
            half16 b_frag[WT_N];
            const int k_off = hipdnn_wmma_k_off(sub);
#pragma unroll
            for (int wn = 0; wn < WT_N; wn++) {
                int noff              = (wcol * WT_N + wn) * WMMA_TILE;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&b_frag[wn]);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smB[buf][noff + lane][ks * WMMA_TILE + k_off]);
#pragma unroll
                for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 2; i++)
                    dst[i] = src[i];
            }
#pragma unroll
            for (int wm = 0; wm < WT_M; wm++) {
                int moff              = (wrow * WT_M + wm) * WMMA_TILE;
                half16 a_frag;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&a_frag);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smA[buf][moff + lane][ks * WMMA_TILE + k_off]);
#pragma unroll
                for (int i = 0; i < HIPDNN_WMMA_FRAG_ELEMS / 2; i++)
                    dst[i] = src[i];

#pragma unroll
                for (int wn = 0; wn < WT_N; wn++)
                    acc[wm][wn] = HIPDNN_WMMA_F32_16X16X16_F16(
                        a_frag, b_frag[wn], acc[wm][wn]);
            }
        }
    };

    issueLoads(0);
    storeToShared(0);
    __syncthreads();

    int buf = 0;
    for (int t = BK_T; t < K; t += BK_T) {
        int nxt = 1 - buf;
        issueLoads(t);
        computeWMMA(buf);
        storeToShared(nxt);
        __syncthreads();
        buf = nxt;
    }
    computeWMMA(buf);

#pragma unroll
    for (int wm = 0; wm < WT_M; wm++) {
        int mbase = row0 + (wrow * WT_M + wm) * WMMA_TILE;
#pragma unroll
        for (int wn = 0; wn < WT_N; wn++) {
            int nbase = col0 + (wcol * WT_N + wn) * WMMA_TILE;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                int r = hipdnn_wmma_acc_row(sub, e);
                if constexpr (BOUNDS_CHECK) {
                    if (mbase + r < M && nbase + lane < N)
                        C[(mbase + r) * ldc + nbase + lane] =
                            (_Float16)acc[wm][wn][e];
                } else {
                    C[(mbase + r) * ldc + nbase + lane] =
                        (_Float16)acc[wm][wn][e];
                }
            }
        }
    }
}


template <int BM_T = 128, int BN_T = 128, int WT_M = 2, int WT_N = 2,
          int WAVES_EU = 8, bool BC = false>
__global__
__attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N),
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(WAVES_EU)))
void MatMulNBitsWMMA_U2_NoZP(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B,
    const _Float16* __restrict__ scales,
    const unsigned char* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16U2Impl<BM_T, BN_T, WT_M, WT_N, false, BC>(
        M, N, K, A, lda, B, scales, zeros,
        group_size, num_groups_k, C, ldc, swizzle_n);
}


template <int BM_T = 128, int BN_T = 128, int WT_M = 2, int WT_N = 2,
          int WAVES_EU = 8, bool BC = false>
__global__
__attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N),
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(WAVES_EU)))
void MatMulNBitsWMMA_U2_ZP(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B,
    const _Float16* __restrict__ scales,
    const unsigned char* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16U2Impl<BM_T, BN_T, WT_M, WT_N, true, BC>(
        M, N, K, A, lda, B, scales, zeros,
        group_size, num_groups_k, C, ldc, swizzle_n);
}


/* -----------------------------------------------------------------------
 * uint2 -> fp16 dequant (for the separate-dequant "dq+gemm" path).
 * Each thread handles one 8-value chunk: 8 values * 2 bits = 16 bits = a
 * single uint16, so the read never crosses a 3rd byte. Mirrors
 * dequant_u3_to_fp16.
 * ----------------------------------------------------------------------- */
__global__ void dequant_u2_to_fp16(
    const unsigned char* __restrict__ B,       // [N, (K*2)/8] stream
    const _Float16* __restrict__ scales,       // [N, num_groups_k]
    const unsigned char* __restrict__ zeros,   // [N, num_groups_k] or null
    _Float16* __restrict__ B_fp16,             // [N, K]
    int N, int K, int group_size, int num_groups_k)
{
    int n  = static_cast<int>(blockIdx.y);
    int k8 = (static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (n >= N || k8 >= K) return;

    int grp = k8 / group_size;
    _Float16 scale = scales[n * num_groups_k + grp];
    // Default zp = 2^(bits-1) = 2 for bits=2 (matches the fused u2 path).
    _Float16 zp = zeros ? static_cast<_Float16>(zeros[n * num_groups_k + grp])
                        : (_Float16)2;
    _Float16 nzs = -zp * scale;

    const int row_bytes = (K * 2) >> 3;
    const int off  = n * row_bytes + ((k8 * 2) >> 3);
    const uint32_t p = *reinterpret_cast<const uint16_t*>(&B[off]);

    _Float16 bv[8];
#pragma unroll
    for (int i = 0; i < 8; i++)
        bv[i] = static_cast<_Float16>((p >> (2 * i)) & 0x3u) * scale + nzs;

    _Float16* dst = &B_fp16[n * K + k8];
    *reinterpret_cast<uint2*>(dst)     = *reinterpret_cast<uint2*>(&bv[0]);
    *reinterpret_cast<uint2*>(dst + 4) = *reinterpret_cast<uint2*>(&bv[4]);
}


__global__ void matmul_nbits_add_bias_rowmajor(
    _Float16* __restrict__ C,
    const _Float16* __restrict__ bias,
    int M, int N, int ldc)
{
    int n = static_cast<int>(blockIdx.x) * 256 + threadIdx.x;
    int m = static_cast<int>(blockIdx.y);
    if (m < M && n < N)
        C[m * ldc + n] += bias[n];
}


__global__ void unpack_zp_u8_to_fp16_kernel(
    const unsigned char* __restrict__ zp_packed,
    _Float16* __restrict__ zp_fp16,
    int N, int groups_k, int packed_cols)
{
    int n  = static_cast<int>(blockIdx.y);
    int g2 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N || g2 >= packed_cols) return;

    unsigned char packed = zp_packed[n * packed_cols + g2];
    int g_lo = g2 * 2;
    int g_hi = g_lo + 1;

    zp_fp16[n * groups_k + g_lo] = static_cast<_Float16>(packed & 0xF);
    if (g_hi < groups_k)
        zp_fp16[n * groups_k + g_hi] = static_cast<_Float16>(packed >> 4);
}


/* Unpack packed nibble ZPs [N, ceil(k_blocks/2)] → [N, k_blocks] uint8,
 * one byte per zero_point (0-15). Needed because the GEMV row-major and
 * naive kernels index zp[n * k_blocks + grp], expecting one value per byte. */
__global__ void unpack_zp_nibbles_to_u8_kernel(
    const uint8_t* __restrict__ packed,
    uint8_t* __restrict__ out,
    int N, int k_blocks, int packed_cols)
{
    int n  = static_cast<int>(blockIdx.y);
    int g2 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N || g2 >= packed_cols) return;

    uint8_t p = packed[n * packed_cols + g2];
    int g_lo = g2 * 2;
    out[n * k_blocks + g_lo] = p & 0xF;
    if (g_lo + 1 < k_blocks)
        out[n * k_blocks + g_lo + 1] = p >> 4;
}


/* Unpack packed 2-bit ZPs [N, ceil(k_blocks/4)] → [N, k_blocks] uint8, one
 * byte per zero_point (0-3). ONNX MatMulNBits packs zero_points at `bits`
 * bits; for bits=2 that is 4 group-zero_points per byte, LSB-first:
 *   byte = zp[4i] | zp[4i+1]<<2 | zp[4i+2]<<4 | zp[4i+3]<<6
 * The u2 GEMV/WMMA/naive kernels index zp[n * k_blocks + grp], expecting one
 * value per byte, so we spread the packed byte out here (mirrors the nibble
 * unpack used by the bits=4 path). */
__global__ void unpack_zp_2bit_to_u8_kernel(
    const uint8_t* __restrict__ packed,
    uint8_t* __restrict__ out,
    int N, int k_blocks, int packed_cols)
{
    int n  = static_cast<int>(blockIdx.y);
    int g4 = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N || g4 >= packed_cols) return;

    uint8_t p = packed[n * packed_cols + g4];
    int g0 = g4 * 4;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        int g = g0 + j;
        if (g < k_blocks)
            out[n * k_blocks + g] = (p >> (2 * j)) & 0x3;
    }
}


/* Unpack packed 3-bit ZPs [N, ceil(k_blocks*3/8)] → [N, k_blocks] uint8, one
 * byte per zero_point (0-7). Mirrors the u3 *weight* packing (Section 1c/1d):
 * a continuous per-row 3-bit stream where zero_point g occupies bits
 * [3g, 3g+3), LSB-first, with each row padded to a byte boundary. A 3-bit
 * field spans at most two bytes, so reading the byte pair around the bit
 * offset and shifting recovers the value. One thread per group (unlike the
 * 2-/4-bit kernels' one-thread-per-byte, which can't be used here because
 * fields straddle byte boundaries). */
__global__ void unpack_zp_3bit_to_u8_kernel(
    const uint8_t* __restrict__ packed,
    uint8_t* __restrict__ out,
    int N, int k_blocks, int packed_cols)
{
    int n = static_cast<int>(blockIdx.y);
    int g = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (n >= N || g >= k_blocks) return;

    int bit_pos = g * 3;
    int bp = bit_pos >> 3;
    int sh = bit_pos & 7;
    const uint8_t* row = packed + static_cast<size_t>(n) * packed_cols;
    uint32_t lo = row[bp];
    uint32_t hi = (bp + 1 < packed_cols) ? row[bp + 1] : 0u;
    uint32_t combined = lo | (hi << 8);
    out[n * k_blocks + g] = static_cast<uint8_t>((combined >> sh) & 0x7);
}

#endif  // HIPDNN_EP_RTC_MATMUL_NBITS_DEVICE_H
