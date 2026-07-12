/*
 * Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * MatMulNBits — Fused Dequant + MatMul for int4 packed weights (FP16)
 *
 * Public API (hip_matmul_nbits) — all tensors are ROW-MAJOR per ONNX convention:
 *        A      : FP16 row-major [batch, M, K]
 *        B      : uint4 packed [N, K/2]
 *        scales : FP16 [N, num_groups_k]
 *        zeros  : FP16 [N, num_groups_k]  (or NULL, default zp=8)
 *        output : FP16 row-major [batch, M, N]
 *
 * Internally auto-dispatched to one of four execution paths:
 *
 *   1. RDNA3 WMMA fast path (batch==1, K%32==0, M>=16):
 *      Transposes A/C to col-major internally, then uses WMMA intrinsics
 *      with double-buffered shared memory, grid swizzling, and inline-asm
 *      batched dequant. Requires gfx11xx/gfx12xx.
 *
 *   2. GEMV col-major path (batch==1, K%32==0, 1 < M < 16):
 *      Transposes A/C internally. K-parallel reduction GEMV with
 *      COL_MAJOR=true, autotuned (BLOCK_SIZE, TILE_N).
 *
 *   3. GEMV row-major path (batch==1, K%32==0, M=1 or fallthrough):
 *      Direct row-major GEMV with uint8 zero points (no ZP conversion).
 *      M=1 decode skips path #2 entirely — row/col layouts are identical
 *      for a single row, avoiding 225 unnecessary ZP conversion kernels
 *      per Llama 8B inference.
 *
 *   4. Naive fallback (arbitrary alignment, batched):
 *      Per-element row-major kernel, uint8 zero points.
 *
 * When zero_points is NULL, dequantization uses the ONNX MatMulNBits
 * default zp = 2^(bits-1) (= 8 for bits=4) to match the CPU EP (MLAS).
 */

#include "hip_custom_kernels.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>

/* =======================================================================
 * Section 1: Naive fallback kernel (row-major, any alignment, batched)
 * ======================================================================= */

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
  const uint8_t* b_row = B + n * (K / 2);
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

// fp32 activation variant of matmul_nbits_kernel. Same logic but A and output
// are float32. Used when the model runs fp32 activations (e.g. ORCA W2A8
// where DequantizeLinear outputs fp32 feeding into the lm_head MatMulNBits).
__global__ void matmul_nbits_kernel_f32(
    const float* __restrict__ A,
    const uint8_t* __restrict__ B,
    const float* __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size) {
  int64_t n = static_cast<int64_t>(blockIdx.x) * kBlockDim + threadIdx.x;
  int64_t m = static_cast<int64_t>(blockIdx.y) * kBlockDim + threadIdx.y;
  int64_t batch = blockIdx.z;

  if (m >= M || n >= N) return;

  int64_t k_blocks = (K + block_size - 1) / block_size;
  int64_t blob_size = block_size / 2;

  float acc = 0.0f;
  const float* a_row = A + batch * M * K + m * K;
  const uint8_t* b_row = B + n * (K / 2);
  const float* s_row = scales + n * k_blocks;

  for (int64_t blk = 0; blk < k_blocks; blk++) {
    float scale_val = s_row[blk];
    float zp_val = (zp != nullptr)
                       ? static_cast<float>(zp[n * k_blocks + blk])
                       : 8.0f;

    int64_t k_start = blk * block_size;
    int64_t k_end = k_start + block_size;
    if (k_end > K) k_end = K;
    const uint8_t* b_block = b_row + blk * blob_size;

    for (int64_t k = k_start; k < k_end; k++) {
      int64_t in_blk  = k - k_start;
      int64_t byte_off = in_blk / 2;
      int nibble = static_cast<int>(in_blk & 1);
      uint8_t packed = b_block[byte_off];
      float qval = (nibble == 0) ? static_cast<float>(packed & 0xF)
                                 : static_cast<float>(packed >> 4);
      acc += a_row[k] * (qval - zp_val) * scale_val;
    }
  }

  if (bias) acc += bias[n];
  output[batch * M * N + m * N + n] = acc;
}

/* ---------------------------------------------------------------------------
 * Section 1a-f32: Optimized bits=4 GEMV for M=1 with fp32 activations (w4a32).
 *
 * The tuned bits=4 GEMV (getOrTuneGemvConfig/launchGemvConfig) only supports
 * fp16 activations, so w4a32 shapes (e.g. the ORCA lm_head: M=1, N=100352,
 * K=5120) fell through to the naive thread-per-output f32 kernel — which for
 * M=1 uses ONE thread per output column, walking all of K serially with no
 * coalescing (~55 ms/token, 2% of BW).
 *
 * This kernel keeps the exact fp32 math (no precision change) but parallelizes
 * properly: one threadblock per output column n; BT threads split the K-blocks,
 * read weights 16B-coalesced (uint4 = 32 packed 4-bit == one block_size=32
 * group), and reduce in shared memory. Memory-bound instead of latency-bound.
 * ------------------------------------------------------------------------- */
// One WAVEFRONT per output column: lanes split the K-blocks, weights read
// 16B-coalesced (uint4 == 32 packed 4-bit), reduced via warp shuffle (no
// shared memory, no __syncthreads). WARPS_PER_BLOCK columns per block keeps
// occupancy high (few, large blocks instead of N tiny ones). The activation
// A[K] is tiny and reused by every column, so it stays L2-resident.
__global__ void matmul_nbits_gemv_f32_m1_kernel(
    const float*   __restrict__ A,       // [K]  (M == 1)
    const uint8_t* __restrict__ B,       // [N, K/2]  packed 4-bit, row-major
    const float*   __restrict__ scales,  // [N, k_blocks]
    const uint8_t* __restrict__ zp,      // [N, k_blocks] or null (default 8)
    const float*   __restrict__ bias,    // [N] or null
    float*         __restrict__ output,  // [N]
    int64_t N, int64_t K, int64_t block_size) {
  const int lane      = threadIdx.x & (warpSize - 1);
  const int warp_id   = threadIdx.x / warpSize;
  const int warps_pb  = blockDim.x  / warpSize;
  const int64_t n = static_cast<int64_t>(blockIdx.x) * warps_pb + warp_id;
  if (n >= N) return;

  const int64_t k_blocks  = (K + block_size - 1) / block_size;
  const int64_t blob_size = block_size / 2;
  const uint8_t* b_row = B + n * (K / 2);
  const float*   s_row = scales + n * k_blocks;
  const uint8_t* z_row = zp ? (zp + n * k_blocks) : nullptr;

  float acc = 0.0f;
  for (int64_t blk = lane; blk < k_blocks; blk += warpSize) {
    const float   scale_val = s_row[blk];
    const float   zp_val    = z_row ? static_cast<float>(z_row[blk]) : 8.0f;
    const int64_t k_start   = blk * block_size;
    int64_t       k_end     = k_start + block_size;
    if (k_end > K) k_end = K;
    const uint8_t* b_block  = b_row + blk * blob_size;
    const int64_t  cnt      = k_end - k_start;    // weights in this block

    int64_t i = 0;
    // Vectorized: 16 bytes (uint4) == 32 packed 4-bit weights per iteration.
    // (b_row and each block are 16B-aligned since K/2 and blob_size are.)
    for (; i + 32 <= cnt; i += 32) {
      uint4 w = *reinterpret_cast<const uint4*>(b_block + (i >> 1));
      const unsigned words[4] = {w.x, w.y, w.z, w.w};
      const float* a_ptr = A + k_start + i;
      #pragma unroll
      for (int wd = 0; wd < 4; ++wd) {
        const unsigned W = words[wd];
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
          const float q = static_cast<float>((W >> (4 * p)) & 0xFu);
          acc += a_ptr[wd * 8 + p] * (q - zp_val) * scale_val;
        }
      }
    }
    // Scalar tail (block_size not a multiple of 32, or ragged final block).
    for (; i < cnt; ++i) {
      const uint8_t packed = b_block[i >> 1];
      const float q = (i & 1) ? static_cast<float>(packed >> 4)
                              : static_cast<float>(packed & 0xF);
      acc += A[k_start + i] * (q - zp_val) * scale_val;
    }
  }

  // Warp-shuffle reduction across the wavefront (wave32 or wave64).
  for (int o = warpSize >> 1; o > 0; o >>= 1)
    acc += __shfl_down(acc, o);

  if (lane == 0) {
    if (bias) acc += bias[n];
    output[n] = acc;
  }
}

static void launch_gemv_f32_m1(
    hipStream_t stream,
    const float* A, const uint8_t* B, const float* scales,
    const uint8_t* zp, const float* bias, float* output,
    int64_t N, int64_t K, int64_t block_size) {
  int dev = 0; hipGetDevice(&dev);
  int warp = 64; hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, dev);
  const int BT = 256;
  const int warps_pb = BT / warp;
  dim3 grid(static_cast<unsigned>((N + warps_pb - 1) / warps_pb));
  dim3 block(BT);

  hipLaunchKernelGGL(matmul_nbits_gemv_f32_m1_kernel, grid, block, 0,
                     stream, A, B, scales, zp, bias, output, N, K, block_size);
}

/* =======================================================================
 * Section 1b-i2: Naive bits=2 fallback kernel (row-major, any alignment)
 *
 * Before:  bits=2 was rejected with an error message
 * After:
 *   matmul_nbits_kernel_i2 -- thread-per-output naive kernel
 *   matmul_nbits_gemv_i2_kernel<BS,TN> -- vectorized GEMV for M<=16
 *   launch_gemv_i2() -- fixed-config launcher (BS=64, TN=8)
 *
 * Packing: four 2-bit weights per byte, little-endian crumb order.
 *   weight[k] = (byte[k/4] >> (2*(k%4))) & 0x3
 * Default ZP: 2 = 2^(bits-1) for bits=2
 * ======================================================================= */

__global__ void matmul_nbits_kernel_i2(
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
  int64_t blob_size = block_size / 4;

  float acc = 0.0f;
  const __half*  a_row = A + batch * M * K + m * K;
  const uint8_t* b_row = B + n * (K / 4);
  const __half*  s_row = scales + n * k_blocks;

  for (int64_t blk = 0; blk < k_blocks; blk++) {
    float scale_val = static_cast<float>(s_row[blk]);
    float zp_val    = (zp != nullptr)
                          ? static_cast<float>(zp[n * k_blocks + blk])
                          : 2.0f;

    int64_t k_start = blk * block_size;
    int64_t k_end   = k_start + block_size;
    if (k_end > K) k_end = K;
    const uint8_t* b_block = b_row + blk * blob_size;

    for (int64_t k = k_start; k < k_end; k++) {
      int64_t in_blk   = k - k_start;
      int64_t byte_off = in_blk / 4;
      int     crumb    = static_cast<int>(in_blk & 3);
      uint8_t packed   = b_block[byte_off];
      float   qval     = static_cast<float>((packed >> (crumb * 2)) & 0x3);
      acc += static_cast<float>(a_row[k]) * (qval - zp_val) * scale_val;
    }
  }

  if (bias) {
    acc += static_cast<float>(bias[n]);
  }
  output[batch * M * N + m * N + n] = static_cast<__half>(acc);
}

/* =======================================================================
 * Section 1b-i2-gemv: Vectorized GEMV for bits=2 (M<=16, decode path)
 *
 * A 16-byte uint4 load covers 64 two-bit weights == one block_size=64 group.
 * Factored dequant: (dot - a_sum*zp) * scale.
 * Launch: grid(ceil(N/TN), M, batch), block(BS)
 * ======================================================================= */

// Template on activation/scale/output type so both fp16 and fp32 paths
// use the same vectorized bits=2 GEMV logic without code duplication.
// T = float for ORCA (fp32 activations + fp32 scales + fp32 output)
// T = __half for standard fp16 models
template <int BLOCK_SIZE, int TILE_N, typename T = __half>
__global__ void matmul_nbits_gemv_i2_kernel(
    const T* __restrict__ A,
    const uint8_t* __restrict__ B,
    const T* __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const T* __restrict__ bias,
    T* __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size)
{
    static_assert(BLOCK_SIZE >= 32, "BLOCK_SIZE must be >= 32");
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    const int64_t n_base = static_cast<int64_t>(blockIdx.x) * TILE_N;
    const int64_t m      = blockIdx.y;
    const int64_t batch  = blockIdx.z;
    const int     tid    = threadIdx.x;

    if (m >= M) return;

    const int64_t k_blocks = (K + block_size - 1) / block_size;
    const int     bs_shift = __builtin_ctzll(
                                 static_cast<unsigned long long>(block_size));

    const T* a_base = A + batch * M * K + m * K;
    const bool has_zp = (zp != nullptr);
    // bits=2 zero_points are packed 4-per-byte along the k_blocks axis
    // (row stride = ceil(k_blocks/4) bytes). Unpack the 2-bit field for
    // block `gblk` of output row `nrow`.
    const int64_t zp_row_bytes = (k_blocks + 3) / 4;
    auto read_zp2 = [&](int64_t nrow, int gblk) -> float {
        uint8_t b = zp[nrow * zp_row_bytes + (gblk >> 2)];
        return static_cast<float>((b >> (2 * (gblk & 3))) & 0x3);
    };

    const uint8_t* b_base_arr[TILE_N];
    const T*       s_rows[TILE_N];
    int64_t        n_idx[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        int64_t n  = n_base + tn;
        int64_t sn = (n < N) ? n : 0;
        n_idx[tn]      = sn;
        b_base_arr[tn] = B + sn * (K / 4);
        s_rows[tn]     = scales + sn * k_blocks;
    }

    float partial[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) partial[tn] = 0.0f;

    // Phase 1: 64-weight (16-byte / uint4) vectorized main loop
    const int num_u128 = static_cast<int>(K / 64);
    for (int idx = tid; idx < num_u128; idx += BLOCK_SIZE) {
        const int k64 = idx * 64;

        float a_vals[64];
        float a_sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 64; i++) {
            a_vals[i]  = static_cast<float>(a_base[k64 + i]);
            a_sum     += a_vals[i];
        }

        const int   grp            = k64 >> bs_shift;
        const float a_sum_x_def_zp = a_sum * 2.0f;

        uint4  b_pk[TILE_N];
        float  sv[TILE_N];
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            b_pk[tn] = *reinterpret_cast<const uint4*>(&b_base_arr[tn][idx * 16]);
            sv[tn]   = static_cast<float>(s_rows[tn][grp]);
        }

#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; i++)
                dot = __fmaf_rn(a_vals[i],
                       static_cast<float>((b_pk[tn].x >> (i * 2)) & 0x3), dot);
#pragma unroll
            for (int i = 0; i < 16; i++)
                dot = __fmaf_rn(a_vals[16 + i],
                       static_cast<float>((b_pk[tn].y >> (i * 2)) & 0x3), dot);
#pragma unroll
            for (int i = 0; i < 16; i++)
                dot = __fmaf_rn(a_vals[32 + i],
                       static_cast<float>((b_pk[tn].z >> (i * 2)) & 0x3), dot);
#pragma unroll
            for (int i = 0; i < 16; i++)
                dot = __fmaf_rn(a_vals[48 + i],
                       static_cast<float>((b_pk[tn].w >> (i * 2)) & 0x3), dot);

            if (has_zp) {
                float zv = read_zp2(n_idx[tn], grp);
                partial[tn] += (dot - a_sum * zv) * sv[tn];
            } else {
                partial[tn] += (dot - a_sum_x_def_zp) * sv[tn];
            }
        }
    }

    // Scalar tail: K%64 remainder
    {
        const int k_tail_start = num_u128 * 64;
        for (int k = k_tail_start + tid; k < static_cast<int>(K); k += BLOCK_SIZE) {
            float av  = static_cast<float>(a_base[k]);
            int   grp = k >> bs_shift;
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                int64_t sn       = n_idx[tn];
                int     byte_off = k / 4;
                int     crumb    = k & 3;
                float   qval     = static_cast<float>(
                                       (b_base_arr[tn][byte_off] >> (crumb * 2)) & 0x3);
                float   zv       = has_zp ? read_zp2(sn, grp) : 2.0f;
                float   sv_val   = static_cast<float>(s_rows[tn][grp]);
                partial[tn]     += av * (qval - zv) * sv_val;
            }
        }
    }

    // Warp-shuffle + shared-mem reduction
    __shared__ float warp_sums[TILE_N * NUM_WARPS];
    const int lane = tid % WARP_SIZE;
    const int wid  = tid / WARP_SIZE;

#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        float val = partial[tn];
        for (int off = WARP_SIZE / 2; off >= 1; off >>= 1)
            val += __shfl_down(val, off);
        if (lane == 0)
            warp_sums[tn * NUM_WARPS + wid] = val;
    }
    __syncthreads();

    if (tid < TILE_N) {
        float sum = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            sum += warp_sums[tid * NUM_WARPS + w];

        int64_t n = n_base + tid;
        if (n < N) {
            if (bias) sum += static_cast<float>(bias[n]);
            output[batch * M * N + m * N + n] = static_cast<T>(sum);
        }
    }
}

/* =======================================================================
 * Section 1c-i2-fused: Fused DequantizeLinear + bits=2 MatMulNBits +
 *                       QuantizeLinear GEMV kernel
 *
 * Eliminates two global memory roundtrips per layer:
 *   Before: uint16 --[DQ]--> f32 --[MatMul]--> f32 --[Q]--> uint16
 *   After:  uint16 --[fused kernel]--> uint16
 *
 * Activation quantization (per-tensor scalars):
 *   dq: a_fp32 = (uint16_in - dq_zp) * dq_scale
 *   q:  out_u16 = clamp(round(result / q_scale) + q_zp, 0, 65535)
 *
 * Weight dequant is identical to the unfused bits=2 GEMV:
 *   w_fp32 = (packed_crumb - w_zp) * w_scale  (per block)
 *
 * Launch: grid(ceil(N/TN), M, batch), block(BS)
 * ======================================================================= */

/* -----------------------------------------------------------------------
 * Lean fused kernel: TILE_N=1, minimal registers, for M=1 decode path.
 * One BLOCK_SIZE-thread block computes one output element.
 * Register-lean: only scalar accumulators (no TILE_N arrays).
 * grid(N, M, 1), block(BLOCK_SIZE).
 * ----------------------------------------------------------------------- */
/* -----------------------------------------------------------------------
 * M=1 fused scalar kernel: correct per-block scale, zero register pressure.
 *
 * One block per output column n. BLOCK_SIZE threads stride over K.
 * Each thread independently accumulates per-block (dot, sum_A) then
 * applies the correct w_scale[n,grp] after the K loop.
 *
 * Key correctness fix vs previous version: we accumulate separate
 * (dot_acc, sum_acc) per quant group and apply per-group scale inline.
 * This matches the unfused path exactly — no approximations.
 *
 * Register pressure: 2 scalars (dot_dq, sum_A_dq) per thread → no spill.
 * -----------------------------------------------------------------------*/
template <int BLOCK_SIZE>
__global__ void matmul_nbits_gemv_i2_fused_lean_kernel(
    const uint16_t* __restrict__ A_u16,
    const uint8_t*  __restrict__ B,
    const float*    __restrict__ w_scales,
    const uint8_t*  __restrict__ w_zp,
    const float*    __restrict__ dq_scale,
    const uint16_t* __restrict__ dq_zp,
    const float*    __restrict__ q_scale,
    const uint16_t* __restrict__ q_zp,
    uint16_t*       __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size)
{
    static_assert(BLOCK_SIZE >= 32, "BLOCK_SIZE must be >= 32");
    const int64_t n   = blockIdx.x;
    const int64_t m   = blockIdx.y;
    const int     tid = threadIdx.x;

    if (n >= N || m >= M) return;

    const float dq_s = dq_scale[0];
    const float dq_z = dq_zp ? static_cast<float>(dq_zp[0]) : 0.0f;
    const float q_s  = q_scale[0];
    const float q_z  = q_zp  ? static_cast<float>(q_zp[0])  : 0.0f;

    const int64_t k_blocks = (K + block_size - 1) / block_size;
    const int     bs_shift = __builtin_ctzll(static_cast<unsigned long long>(block_size));

    const uint16_t* a_base = A_u16 + m * K;
    const uint8_t*  b_row  = B + n * (K / 4);
    const float*    s_row  = w_scales + n * k_blocks;

    // Accumulate full result in a single pass.
    // For each k: a_dq = (a_raw - dq_z) * dq_s
    //             w_dq = (w_crumb - w_zp) * w_scale[grp]
    //             result += a_dq * w_dq
    // = sum_k [ (a_raw - dq_z) * dq_s * (w - w_zp) * w_scale ]
    // Using factored form per group:
    //   result_grp = (dot_raw - dq_z*sum_w - w_zp*(sum_a_raw - bs*dq_z)) * dq_s * w_scale
    // To keep registers minimal: just compute the direct product per element.
    // No per-group arrays needed — just running sum in float.
    float acc = 0.0f;

    for (int k = tid; k < static_cast<int>(K); k += BLOCK_SIZE) {
        float a_dq  = (static_cast<float>(a_base[k]) - dq_z) * dq_s;
        int   grp   = k >> bs_shift;
        int   boff  = k / 4;
        int   crumb = k & 3;
        float w_raw = static_cast<float>((b_row[boff] >> (crumb*2)) & 0x3);

        // Weight dequant: w_zp packed 2-bit (4 per byte)
        float wz;
        if (w_zp) {
            int64_t zi  = n * k_blocks + grp;
            wz = static_cast<float>((w_zp[zi / 4] >> ((zi % 4) * 2)) & 0x3);
        } else {
            wz = 2.0f;  // default ZP=2 for bits=2
        }
        float w_scale_val = s_row[grp];
        acc += a_dq * (w_raw - wz) * w_scale_val;
    }

    // Warp-shuffle + shared-mem reduction
    constexpr int WARP = 32;
    constexpr int NW   = BLOCK_SIZE / WARP;
    __shared__ float smem[NW];

    int lane = tid % WARP, wid = tid / WARP;
    for (int off = WARP/2; off >= 1; off >>= 1)
        acc += __shfl_down(acc, off);
    if (lane == 0) smem[wid] = acc;
    __syncthreads();

    if (tid < NW) {
        float val = smem[tid];
        for (int off = NW/2; off >= 1; off >>= 1)
            val += __shfl_down(val, off);
        if (tid == 0) {
            float qval = __builtin_rintf(val / q_s) + q_z;
            qval = (qval < 0.f) ? 0.f : (qval > 65535.f) ? 65535.f : qval;
            output[m * N + n] = static_cast<uint16_t>(qval);
        }
    }
}

/* -----------------------------------------------------------------------
 * Scalar fused kernel for M=1 decode: processes ONE k-element per
 * iteration, covering TILE_N output columns. Avoids float a_vals[64]
 * register spill by reading A values scalar inline.
 * Same warp-shuffle + shared-mem reduction as unfused GEMV.
 * grid(ceil(N/TILE_N), 1, 1), block(BLOCK_SIZE).
 * ----------------------------------------------------------------------- */
template <int BLOCK_SIZE, int TILE_N>
__global__ void matmul_nbits_gemv_i2_fused_scalar_kernel(
    const uint16_t* __restrict__ A_u16,
    const uint8_t*  __restrict__ B,
    const float*    __restrict__ w_scales,
    const uint8_t*  __restrict__ w_zp,
    const float*    __restrict__ dq_scale,
    const uint16_t* __restrict__ dq_zp,
    const float*    __restrict__ q_scale,
    const uint16_t* __restrict__ q_zp,
    uint16_t*       __restrict__ output,
    int64_t M, int64_t N, int64_t K,
    int64_t block_size)
{
    static_assert(BLOCK_SIZE >= 32, "BLOCK_SIZE >= 32");
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    const int64_t n_base = static_cast<int64_t>(blockIdx.x) * TILE_N;
    const int64_t m      = blockIdx.y;
    const int     tid    = threadIdx.x;

    if (m >= M) return;

    // Load activation quantization scalars once
    const float dq_s = dq_scale[0];
    const float dq_z = dq_zp ? static_cast<float>(dq_zp[0]) : 0.0f;
    const float q_s  = q_scale[0];
    const float q_z  = q_zp  ? static_cast<float>(q_zp[0])  : 0.0f;
    const float inv_qs = 1.0f / q_s;

    const int64_t k_blocks = (K + block_size - 1) / block_size;
    const int     bs_shift = __builtin_ctzll(static_cast<unsigned long long>(block_size));

    const uint16_t* a_base = A_u16 + m * K;

    // Setup per-tile-N column pointers
    const uint8_t* b_base_arr[TILE_N];
    const float*   s_rows[TILE_N];
    int64_t        n_idx[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        int64_t n  = n_base + tn;
        int64_t sn = (n < N) ? n : 0;
        n_idx[tn]      = sn;
        b_base_arr[tn] = B + sn * (K / 4);
        s_rows[tn]     = w_scales + sn * k_blocks;
    }

    float partial[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) partial[tn] = 0.0f;

    // Pre-load per-group scales and ZP into shared mem to avoid per-k global loads.
    // For K=5120, block_size=64: k_blocks=80 groups, 20 ZP bytes per N row.
    // Shared: scales[TILE_N][k_blocks], zp_vals[TILE_N][k_blocks] (float)
    // This trades smem for random global ZP reads.
    extern __shared__ float smem_base[];
    float* smem_scales = smem_base;                     // [TILE_N][k_blocks]
    float* smem_wzp    = smem_base + TILE_N * k_blocks; // [TILE_N][k_blocks]

    // Cooperative fill: BLOCK_SIZE threads fill TILE_N * k_blocks entries
    int total_fill = TILE_N * static_cast<int>(k_blocks);
    for (int idx = tid; idx < total_fill; idx += BLOCK_SIZE) {
        int tn  = idx / static_cast<int>(k_blocks);
        int grp = idx % static_cast<int>(k_blocks);
        smem_scales[idx] = s_rows[tn][grp];
        if (w_zp) {
            int64_t zi = n_idx[tn] * k_blocks + grp;
            smem_wzp[idx] = static_cast<float>((w_zp[zi / 4] >> ((zi % 4) * 2)) & 0x3);
        } else {
            smem_wzp[idx] = 2.0f;
        }
    }
    __syncthreads();

    // Scalar K loop: read scales/ZP from smem (fast), B and A from global
    for (int k = tid; k < static_cast<int>(K); k += BLOCK_SIZE) {
        float a_dq = (static_cast<float>(a_base[k]) - dq_z) * dq_s;
        int   grp  = k >> bs_shift;
        int   boff = k / 4;
        int   crumb= k & 3;

#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            float w_raw   = static_cast<float>(
                                (b_base_arr[tn][boff] >> (crumb*2)) & 0x3);
            float wz      = smem_wzp   [tn * k_blocks + grp];
            float ws_val  = smem_scales[tn * k_blocks + grp];
            partial[tn]  += a_dq * (w_raw - wz) * ws_val;
        }
    }

    // Warp-shuffle + shared-mem reduction. Reuse smem after the fill is done.
    // warp_sums sits at smem_base + 2*TILE_N*k_blocks (after scales + ZP).
    float* warp_sums = smem_base + 2 * TILE_N * static_cast<int>(k_blocks);
    const int lane = tid % WARP_SIZE;
    const int wid  = tid / WARP_SIZE;

#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        float val = partial[tn];
        for (int off = WARP_SIZE/2; off >= 1; off >>= 1)
            val += __shfl_down(val, off);
        if (lane == 0)
            warp_sums[tn * NUM_WARPS + wid] = val;
    }
    __syncthreads();

    if (tid < TILE_N) {
        float sum = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            sum += warp_sums[tid * NUM_WARPS + w];

        int64_t n = n_base + tid;
        if (n < N) {
            // Quantize output
            float qval = __builtin_rintf(sum * inv_qs) + q_z;
            qval = (qval < 0.f) ? 0.f : (qval > 65535.f) ? 65535.f : qval;
            output[m * N + n] = static_cast<uint16_t>(qval);
        }
    }
}

// launchScalarFusedGemvI2Config is defined after kGemvConfigs below
static void launchScalarFusedGemvI2Config(
    int cid, hipStream_t stream,
    const uint16_t* A_u16, const uint8_t* B,
    const float* w_scales, const uint8_t* w_zp,
    const float* dq_scale, const uint16_t* dq_zp,
    const float* q_scale,  const uint16_t* q_zp,
    uint16_t* output,
    int64_t M, int64_t N, int64_t K, int64_t block_size);

template <int BLOCK_SIZE, int TILE_N>
__global__ void matmul_nbits_gemv_i2_fused_kernel(
    const uint16_t* __restrict__ A_u16,  // quantized activation [batch,M,K]
    const uint8_t*  __restrict__ B,      // packed 2-bit weights [N, K/4]
    const float*    __restrict__ w_scales, // weight scales [N, k_blocks]
    const uint8_t*  __restrict__ w_zp,   // weight zero points (nullable)
    const float*    __restrict__ dq_scale, // activation DQ scale (scalar)
    const uint16_t* __restrict__ dq_zp,  // activation DQ zero point (scalar)
    const float*    __restrict__ q_scale, // output Q scale (scalar)
    const uint16_t* __restrict__ q_zp,   // output Q zero point (scalar)
    uint16_t*       __restrict__ output, // quantized output [batch,M,N]
    int64_t M, int64_t N, int64_t K,
    int64_t block_size)
{
    static_assert(BLOCK_SIZE >= 32, "BLOCK_SIZE must be >= 32");
    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

    const int64_t n_base = static_cast<int64_t>(blockIdx.x) * TILE_N;
    const int64_t m      = blockIdx.y;
    const int64_t batch  = blockIdx.z;
    const int     tid    = threadIdx.x;

    if (m >= M) return;

    // Load per-tensor activation DQ/Q params from scalar buffers
    const float dq_s  = dq_scale[0];
    const float dq_z  = dq_zp  ? static_cast<float>(dq_zp[0])  : 0.0f;
    const float q_s   = q_scale[0];
    const float q_z   = q_zp   ? static_cast<float>(q_zp[0])   : 0.0f;
    const float inv_qs = 1.0f / q_s;

    const int64_t k_blocks = (K + block_size - 1) / block_size;
    const int     bs_shift = __builtin_ctzll(
                                 static_cast<unsigned long long>(block_size));

    const uint16_t* a_base = A_u16 + batch * M * K + m * K;
    const bool has_wzp = (w_zp != nullptr);

    const uint8_t* b_base_arr[TILE_N];
    const float*   s_rows[TILE_N];
    int64_t        n_idx[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        int64_t n  = n_base + tn;
        int64_t sn = (n < N) ? n : 0;
        n_idx[tn]      = sn;
        b_base_arr[tn] = B + sn * (K / 4);
        s_rows[tn]     = w_scales + sn * k_blocks;
    }

    float partial[TILE_N];
#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) partial[tn] = 0.0f;

    // Phase 1: 64-weight (16-byte) vectorized main loop.
    //
    // Factored dequant (avoids float a_vals[64] register spill):
    //   fused result = Q(dq_s * w_scale * (dot_AB - dq_z*sum_B - w_zp*(sum_A - dq_z*bs)))
    // where:
    //   dot_AB = sum_k(A_raw[k] * W[k])   (raw uint16 * 2-bit weight dot)
    //   sum_B  = sum_k(W[k])              (sum of 2-bit weights in block)
    //   sum_A  = sum_k(A_raw[k])          (sum of uint16 activations in block)
    //   bs     = block_size (64)
    // This is equivalent to (dot_dq - a_sum_dq * w_zp) * w_scale after algebra.
    //
    // Vectorized main loop: 64 elements per outer iteration using 16-byte loads.
    // Uses factored dequant to avoid float a_vals[64] register spill:
    //   result = (dot_dq - a_sum_dq*w_zp) * w_scale
    //   dot_dq = (dot_AB - dq_z*sum_B) * dq_s  [dot_AB=sum(A_raw*W)]
    //   a_sum_dq = (sum_A - 64*dq_z) * dq_s    [sum_A=sum(A_raw)]
    // With TILE_N small (1-2), register pressure stays manageable.
    const int num_u128 = static_cast<int>(K / 64);
    for (int idx = tid; idx < num_u128; idx += BLOCK_SIZE) {
        const int k64 = idx * 64;
        const int grp = k64 >> bs_shift;

        // Load 64 uint16 A values via 8 × uint4 loads (128 bits each = 8 uint16)
        // Process as 4 groups of 16 (matching the 4 uint32 words of b_pk).
        // Only one pass over A is needed since we compute both dot_AB and sum_A.

        // Load weight packs
        uint4  b_pk[TILE_N];
        float  wscale[TILE_N];
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            b_pk[tn]   = *reinterpret_cast<const uint4*>(&b_base_arr[tn][idx * 16]);
            wscale[tn] = s_rows[tn][grp];
        }

        float dot_AB[TILE_N];
        float sum_B[TILE_N];
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) { dot_AB[tn] = 0.0f; sum_B[tn] = 0.0f; }
        float sum_A_raw = 0.0f;

        // Process 64 elements as 4 groups of 16 (one uint32 of B per group)
        // Each uint32 of B holds 16 two-bit weights.
#pragma unroll
        for (int chunk = 0; chunk < 4; chunk++) {
            // Load 16 uint16 A values for this chunk
            const uint16_t* a_ptr = a_base + k64 + chunk * 16;
            // Use uint4 load for 16 × uint16 = 256 bits (two uint4 loads)
            uint4 a0 = *reinterpret_cast<const uint4*>(a_ptr);      // first 8 uint16
            uint4 a1 = *reinterpret_cast<const uint4*>(a_ptr + 8);  // next  8 uint16

            const uint16_t* a0p = reinterpret_cast<const uint16_t*>(&a0);
            const uint16_t* a1p = reinterpret_cast<const uint16_t*>(&a1);

            // Accumulate sum_A_raw
#pragma unroll
            for (int i = 0; i < 8; i++) { sum_A_raw += a0p[i]; sum_A_raw += a1p[i]; }

#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                uint32_t bw = (chunk==0)?b_pk[tn].x:
                              (chunk==1)?b_pk[tn].y:
                              (chunk==2)?b_pk[tn].z:b_pk[tn].w;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    float a = static_cast<float>(a0p[i]);
                    float w = static_cast<float>((bw >> (i*2)) & 0x3);
                    dot_AB[tn] = __fmaf_rn(a, w, dot_AB[tn]);
                    sum_B[tn] += w;
                }
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    float a = static_cast<float>(a1p[i]);
                    float w = static_cast<float>((bw >> ((i+8)*2)) & 0x3);
                    dot_AB[tn] = __fmaf_rn(a, w, dot_AB[tn]);
                    sum_B[tn] += w;
                }
            }
        }

        const float sum_A_dq = (sum_A_raw - 64.0f * dq_z) * dq_s;

#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            float dot_dq = (dot_AB[tn] - dq_z * sum_B[tn]) * dq_s;
            if (has_wzp) {
                int64_t zp_byte = n_idx[tn] * (k_blocks / 4) + grp / 4;
                int     zp_cr   = grp & 3;
                float   wz = static_cast<float>((w_zp[zp_byte] >> (zp_cr * 2)) & 0x3);
                partial[tn] += (dot_dq - wz * sum_A_dq) * wscale[tn];
            } else {
                partial[tn] += (dot_dq - 2.0f * sum_A_dq) * wscale[tn];
            }
        }
    }

    // Scalar tail: K%64 remainder
    {
        const int k_tail_start = num_u128 * 64;
        for (int k = k_tail_start + tid; k < static_cast<int>(K); k += BLOCK_SIZE) {
            float av  = (static_cast<float>(a_base[k]) - dq_z) * dq_s;
            int   grp = k >> bs_shift;
            int   boff = k / 4, crumb = k & 3;
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float w = static_cast<float>((b_base_arr[tn][boff] >> (crumb*2)) & 0x3);
                float wz;
                if (has_wzp) {
                    int64_t zb = n_idx[tn] * (k_blocks / 4) + grp / 4;
                    wz = static_cast<float>((w_zp[zb] >> ((grp&3)*2)) & 0x3);
                } else { wz = 2.0f; }
                partial[tn] += av * (w - wz) * s_rows[tn][grp];
            }
        }
    }

    // Warp-shuffle + shared-mem reduction
    __shared__ float warp_sums[TILE_N * NUM_WARPS];
    const int lane = tid % WARP_SIZE;
    const int wid  = tid / WARP_SIZE;

#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        float val = partial[tn];
        for (int off = WARP_SIZE/2; off >= 1; off >>= 1)
            val += __shfl_down(val, off);
        if (lane == 0)
            warp_sums[tn * NUM_WARPS + wid] = val;
    }
    __syncthreads();

    if (tid < TILE_N) {
        float sum = 0.0f;
        for (int w = 0; w < NUM_WARPS; w++)
            sum += warp_sums[tid * NUM_WARPS + w];

        int64_t n = n_base + tid;
        if (n < N) {
            // Quantize output on the fly
            float q = __builtin_rintf(sum * inv_qs) + q_z;
            q = (q < 0.0f) ? 0.0f : (q > 65535.0f) ? 65535.0f : q;
            output[batch * M * N + m * N + n] = static_cast<uint16_t>(q);
        }
    }
}

// Forward-declare templated autotuner
template <typename T>
static void launch_gemv_i2_autotuned(hipStream_t stream,
    const T* A, const uint8_t* B, const T* scales, const uint8_t* zp, const T* bias,
    T* output, int64_t M, int64_t N, int64_t K, int64_t batch_count, int64_t block_size);

template <typename T = __half>
static void launch_gemv_i2(
    hipStream_t stream,
    const T* A, const uint8_t* B,
    const T* scales, const uint8_t* zp, const T* bias,
    T* output,
    int64_t M, int64_t N, int64_t K, int64_t batch_count, int64_t block_size)
{
    launch_gemv_i2_autotuned<T>(stream, A, B, scales, zp, bias, output,
                                M, N, K, batch_count, block_size);
}

/* =======================================================================
 * Section 1a: Naive int8 fallback kernel (row-major, any alignment, batched)
 *
 * ONNX MatMulNBits with bits=8 layout (no nibble packing):
 *   B      : uint8 [N, K]                     (1 byte per weight)
 *   scales : fp16  [N, num_groups_k]
 *   zeros  : uint8 [N, num_groups_k]          (or NULL, default zp=128)
 *
 * Dequant per element:
 *   w_fp = (uint8_w - zp) * scale
 *
 * Each thread computes one C[m, n] across all batches via blockIdx.z.
 * Mirrors matmul_nbits_kernel (int4) so behaviour and debug shape are
 * symmetric; chosen as the first int8 path because it has no alignment
 * or shape constraints and immediately unblocks Qwen3.5-style models
 * that mix bits=4 and bits=8 in the same graph.
 * ======================================================================= */

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

/* =======================================================================
 * Section 1a-gemv: GEMV-style kernel for int8 weights (autotuned)
 *
 * Mirrors the int4 GEMV kernel below (Section 1b) but specialized for the
 * MatMulNBits bits=8 layout:
 *   B is uint8 [N, K] (one byte per weight, no nibble packing)
 *   default zp = 128 (= 2^(bits-1) per the ONNX MatMulNBits spec)
 *   no col-major / WMMA-compat path: int8 always runs row-major.
 *
 * Each threadblock computes TILE_N output elements at row m, sharing the
 * same A row across the N dimension. BLOCK_SIZE threads cooperate on the
 * K reduction via warp shuffle + shared mem (identical to int4 path).
 *
 * Key optimizations carried over from int4:
 *   - 16-byte main loop (uint4 = 128 bits): 16 weights per global txn
 *   - Load-compute separation: B loads issued before FMA compute
 *   - Factored dequant:  dot*scale - a_sum*zp*scale
 *     (for w=int8: same identity as nibble; saves the per-element zp
 *     subtract by hoisting it out of the K-inner loop -- ~30% fewer
 *     fp32 ops in the FMA tree)
 *   - Bit-shift group index (block_size is power-of-2; gated at dispatch)
 *
 * Constraint: requires block_size >= 16 (so a 16-element vector lies in one
 * group). The Qwen3.5 int8 ops use block_size=128, which satisfies this.
 * Smaller block_size falls back to the naive int8 kernel above.
 *
 * Launch: grid(ceil(N/TILE_N), M, batch), block(BLOCK_SIZE)
 * ======================================================================= */

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

/* =======================================================================
 * Section 1b: GEMV-style kernel for small M (autotuned, vectorized)
 *
 * Template parameters:
 *   BLOCK_SIZE - threads per block (K-reduction parallelism)
 *   TILE_N     - N columns per block (1..64, amortizes A loads)
 *
 * When M is small (e.g. M=1 decode token), each threadblock computes
 * TILE_N output elements sharing the same A row load, with BLOCK_SIZE
 * threads cooperating on the K reduction via warp shuffle + shared mem.
 *
 * Key optimizations:
 *   - 32-nibble main loop: uint4 (128-bit) B load → 32 nibbles per txn,
 *     with uint2 (16-nibble) and uint32 (8-nibble) remainder phases
 *   - Load-compute separation: all B loads issued before FMA compute
 *   - Explicit __fmaf_rn intrinsics for dot product accumulation
 *   - Factored dequant:  (dot - a_sum*zp) * scale  (saves ~40% FLOPs)
 *   - Bit-shift group index (block_size is always power-of-2)
 *   - TILE_N up to 64 for large-N shapes (N>=2048)
 *   - Compile-time WARP_SIZE=32 for RDNA 3 dead-code elimination
 *     (runtime __builtin_amdgcn_wavefrontsize() caused 13% regression)
 *   - Single-warp fast path: BLOCK_SIZE=32 uses shuffle-only reduction
 *     (no LDS sync needed)
 *
 * At runtime, an autotune pass benchmarks all 35 (BLOCK_SIZE, TILE_N)
 * combinations on the first call for each problem shape and caches the
 * fastest configuration for subsequent calls.
 * BLOCK_SIZE ∈ {32,64,128,256,512,1024}, TILE_N ∈ {1,2,4,8,16,32,64}.
 * BS=32 configs use single-warp shuffle-only reduction (no LDS).
 * TILE_N=32 tested only when N>=1024; TILE_N=64 only when N>=2048.
 *
 * Launch: grid(ceil(N/TILE_N), M, batch), block(BLOCK_SIZE)
 * ======================================================================= */

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
        b_base_arr[tn] = B + sn * (K / 2);
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
    constexpr int WARP_SIZE = 32;
    const int warp_id   = tid / WARP_SIZE;
    const int lane_id   = tid % WARP_SIZE;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

#pragma unroll
    for (int tn = 0; tn < TILE_N; tn++) {
        for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
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

/* -----------------------------------------------------------------------
 * GEMV Autotune: config table, dispatch, benchmark, cache
 * ----------------------------------------------------------------------- */

struct GemvConfig {
    int threads;
    int tile_n;
};

static constexpr GemvConfig kGemvConfigs[] = {
    // BLOCK_SIZE=32: single warp (wave32) — shuffle-only reduction, no LDS
    { 32, 1}, { 32, 2}, { 32, 4}, { 32, 8},
    { 64, 1}, { 64, 2}, { 64, 4}, { 64, 8}, { 64, 16},
    {128, 1}, {128, 2}, {128, 4}, {128, 8}, {128, 16},
    {256, 1}, {256, 2}, {256, 4}, {256, 8}, {256, 16},
    {512, 1}, {512, 2}, {512, 4}, {512, 8}, {512, 16},
    {1024, 1}, {1024, 2}, {1024, 4}, {1024, 8}, {1024, 16},
    { 64, 32}, {128, 32}, {256, 32},
    { 64, 64}, {128, 64}, {256, 64},
};
static constexpr int kNumGemvConfigs =
    static_cast<int>(sizeof(kGemvConfigs) / sizeof(kGemvConfigs[0]));

static void launchGemvConfig(
    int config_id, hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs,
    bool col_major = false)
{
    auto& c = kGemvConfigs[config_id];
    dim3 grid(static_cast<unsigned>((N + c.tile_n - 1) / c.tile_n),
              static_cast<unsigned>(M),
              static_cast<unsigned>(col_major ? 1 : batch));
    dim3 block(c.threads);

#define GEMV_CASE(ID, BS, TN) \
    case ID: \
        if (col_major) \
            hipLaunchKernelGGL( \
                HIP_KERNEL_NAME(matmul_nbits_gemv_kernel<BS, TN, true>), \
                grid, block, 0, stream, A, B, sc, zp, bi, out, M, N, K, bs); \
        else \
            hipLaunchKernelGGL( \
                HIP_KERNEL_NAME(matmul_nbits_gemv_kernel<BS, TN, false>), \
                grid, block, 0, stream, A, B, sc, zp, bi, out, M, N, K, bs); \
        break

    switch (config_id) {
        GEMV_CASE( 0,  32,  1);  GEMV_CASE( 1,  32,  2);
        GEMV_CASE( 2,  32,  4);  GEMV_CASE( 3,  32,  8);
        GEMV_CASE( 4,  64,  1);  GEMV_CASE( 5,  64,  2);
        GEMV_CASE( 6,  64,  4);  GEMV_CASE( 7,  64,  8);
        GEMV_CASE( 8,  64, 16);
        GEMV_CASE( 9, 128,  1);  GEMV_CASE(10, 128,  2);
        GEMV_CASE(11, 128,  4);  GEMV_CASE(12, 128,  8);
        GEMV_CASE(13, 128, 16);
        GEMV_CASE(14, 256,  1);  GEMV_CASE(15, 256,  2);
        GEMV_CASE(16, 256,  4);  GEMV_CASE(17, 256,  8);
        GEMV_CASE(18, 256, 16);
        GEMV_CASE(19, 512,  1);  GEMV_CASE(20, 512,  2);
        GEMV_CASE(21, 512,  4);  GEMV_CASE(22, 512,  8);
        GEMV_CASE(23, 512, 16);
        GEMV_CASE(24, 1024,  1); GEMV_CASE(25, 1024,  2);
        GEMV_CASE(26, 1024,  4); GEMV_CASE(27, 1024,  8);
        GEMV_CASE(28, 1024, 16);
        GEMV_CASE(29,  64, 32);  GEMV_CASE(30, 128, 32);
        GEMV_CASE(31, 256, 32);
        GEMV_CASE(32,  64, 64);  GEMV_CASE(33, 128, 64);
        GEMV_CASE(34, 256, 64);
        default: break;
    }
#undef GEMV_CASE
}

// Scalar fused launcher definition (uses kGemvConfigs for tile_n/threads)
#define I2_SCALAR_FUSED_CASE(ID, BS, TN) \
    case ID: \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(matmul_nbits_gemv_i2_fused_scalar_kernel<BS, TN>), \
            grd, blk, smem_bytes, stream, \
            A_u16, B, w_scales, w_zp, dq_scale, dq_zp, q_scale, q_zp, \
            output, M, N, K, block_size); \
        break

static void launchScalarFusedGemvI2Config(
    int cid, hipStream_t stream,
    const uint16_t* A_u16, const uint8_t* B,
    const float* w_scales, const uint8_t* w_zp,
    const float* dq_scale, const uint16_t* dq_zp,
    const float* q_scale,  const uint16_t* q_zp,
    uint16_t* output,
    int64_t M, int64_t N, int64_t K, int64_t block_size)
{
    auto& c = kGemvConfigs[cid];
    int64_t k_blocks = (K + block_size - 1) / block_size;
    // Dynamic smem: 2*TILE_N*k_blocks (scales+ZP) + TILE_N*NUM_WARPS (reduction)
    int num_warps = c.threads / 32;
    size_t smem_bytes = static_cast<size_t>(
        (2 * c.tile_n * k_blocks + c.tile_n * num_warps) * sizeof(float));
    dim3 grd(static_cast<unsigned>((N + c.tile_n - 1) / c.tile_n),
             static_cast<unsigned>(M), 1u);
    dim3 blk(c.threads);
    (void)hipGetLastError();
    switch (cid) {
        I2_SCALAR_FUSED_CASE( 0,  32,  1);  I2_SCALAR_FUSED_CASE( 1,  32,  2);
        I2_SCALAR_FUSED_CASE( 2,  32,  4);  I2_SCALAR_FUSED_CASE( 3,  32,  8);
        I2_SCALAR_FUSED_CASE( 4,  64,  1);  I2_SCALAR_FUSED_CASE( 5,  64,  2);
        I2_SCALAR_FUSED_CASE( 6,  64,  4);  I2_SCALAR_FUSED_CASE( 7,  64,  8);
        I2_SCALAR_FUSED_CASE( 8,  64, 16);
        I2_SCALAR_FUSED_CASE( 9, 128,  1);  I2_SCALAR_FUSED_CASE(10, 128,  2);
        I2_SCALAR_FUSED_CASE(11, 128,  4);  I2_SCALAR_FUSED_CASE(12, 128,  8);
        I2_SCALAR_FUSED_CASE(13, 128, 16);
        I2_SCALAR_FUSED_CASE(14, 256,  1);  I2_SCALAR_FUSED_CASE(15, 256,  2);
        I2_SCALAR_FUSED_CASE(16, 256,  4);  I2_SCALAR_FUSED_CASE(17, 256,  8);
        I2_SCALAR_FUSED_CASE(18, 256, 16);
        I2_SCALAR_FUSED_CASE(19, 512,  1);  I2_SCALAR_FUSED_CASE(20, 512,  2);
        I2_SCALAR_FUSED_CASE(21, 512,  4);  I2_SCALAR_FUSED_CASE(22, 512,  8);
        I2_SCALAR_FUSED_CASE(23, 512, 16);
        I2_SCALAR_FUSED_CASE(24, 1024,  1); I2_SCALAR_FUSED_CASE(25, 1024,  2);
        I2_SCALAR_FUSED_CASE(26, 1024,  4); I2_SCALAR_FUSED_CASE(27, 1024,  8);
        I2_SCALAR_FUSED_CASE(28, 1024, 16);
        I2_SCALAR_FUSED_CASE(29,  64, 32);  I2_SCALAR_FUSED_CASE(30, 128, 32);
        I2_SCALAR_FUSED_CASE(31, 256, 32);
        I2_SCALAR_FUSED_CASE(32,  64, 64);  I2_SCALAR_FUSED_CASE(33, 128, 64);
        I2_SCALAR_FUSED_CASE(34, 256, 64);
        default: break;
    }
}
#undef I2_SCALAR_FUSED_CASE

static int tuneGemvConfig(
    hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs,
    bool col_major = false)
{
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);

    float best_ms = 1e30f;
    int   best_id = 8;
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    for (int cid = 0; cid < kNumGemvConfigs; cid++) {
        auto& cfg = kGemvConfigs[cid];
        if (cfg.tile_n >= 64 && N < 2048) continue;
        if (cfg.tile_n >= 32 && cfg.tile_n < 64 && N < 1024) continue;

        for (int w = 0; w < WARMUP; w++)
            launchGemvConfig(cid, stream, A, B, sc, zp, bi, out,
                             M, N, K, batch, bs, col_major);

        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchGemvConfig(cid, stream, A, B, sc, zp, bi, out,
                             M, N, K, batch, bs, col_major);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   config[%2d] threads=%3d tile_n=%d : %.4f ms\n",
            cid, kGemvConfigs[cid].threads, kGemvConfigs[cid].tile_n,
            ms / ITERS);

        if (ms < best_ms) {
            best_ms = ms;
            best_id = cid;
        }
    }

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] GEMV autotune: M=%lld N=%lld K=%lld bs=%lld col_major=%d -> "
        "best config[%d] threads=%d tile_n=%d (%.4f ms/iter)\n",
        (long long)M, (long long)N, (long long)K, (long long)bs, col_major,
        best_id, kGemvConfigs[best_id].threads,
        kGemvConfigs[best_id].tile_n, best_ms / ITERS);

    return best_id;
}

static std::unordered_map<std::string, int> gemv_cache;
static std::mutex gemv_tune_mutex;

static std::string gemvCacheKey(int64_t M, int64_t N, int64_t K, int64_t bs,
                                bool col_major, bool has_zp) {
    // has_zp must be in the key: the asym path executes one extra read_zp +
    // fp32 mul per K-tile per N-tile, so its optimal (threads, tile_n) can
    // differ from the sym tune. Without it, a process that loads both sym and
    // asym variants would silently reuse the wrong autotune result for one of
    // them. (WMMA path's autotune key already includes has_zp — keep parity.)
    return std::to_string(M) + "_" + std::to_string(N) + "_" +
           std::to_string(K) + "_" + std::to_string(bs) + "_" +
           (col_major ? "1" : "0") + "_" + (has_zp ? "1" : "0");
}

static int getOrTuneGemvConfig(
    hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs,
    bool col_major = false)
{
    // M=1: col_major and row_major layouts are identical (single row) —
    // reuse the row_major autotune result to avoid a redundant tuning pass.
    bool effective_col_major = (M == 1) ? false : col_major;
    std::string key = gemvCacheKey(M, N, K, bs, effective_col_major,
                                   /*has_zp=*/zp != nullptr);

    std::lock_guard<std::mutex> lock(gemv_tune_mutex);

    auto it = gemv_cache.find(key);
    if (it != gemv_cache.end())
        return it->second;

    int best = tuneGemvConfig(stream, A, B, sc, zp, bi, out,
                               M, N, K, batch, bs, col_major);
    gemv_cache[key] = best;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] GEMV cache: stored key=\"%s\" -> config[%d] "
        "(cache size=%zu)\n",
        key.c_str(), best, gemv_cache.size());

    return best;
}

/* -----------------------------------------------------------------------
 * GEMV (bits=2) Autotune: dispatch, benchmark, cache
 * ----------------------------------------------------------------------- */

// T=__half for fp16 activations, T=float for fp32 (ORCA W2A8)
#define I2_GEMV_CASE(ID, BS, TN) \
    case ID: \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(matmul_nbits_gemv_i2_kernel<BS, TN, T>), \
            grd, blk, 0, stream, \
            A, B, scales, zp, bias, output, M, N, K, block_size); \
        break

template <typename T = __half>
static void launchGemvI2Config(int cid, hipStream_t stream,
    const T* A, const uint8_t* B, const T* scales, const uint8_t* zp,
    const T* bias, T* output,
    int64_t M, int64_t N, int64_t K, int64_t block_size) {
    auto& c = kGemvConfigs[cid];
    dim3 grd(static_cast<unsigned>((N + c.tile_n - 1) / c.tile_n),
             static_cast<unsigned>(M), 1u);
    dim3 blk(c.threads);
    (void)hipGetLastError();
    switch (cid) {
        I2_GEMV_CASE( 0,  32,  1);  I2_GEMV_CASE( 1,  32,  2);
        I2_GEMV_CASE( 2,  32,  4);  I2_GEMV_CASE( 3,  32,  8);
        I2_GEMV_CASE( 4,  64,  1);  I2_GEMV_CASE( 5,  64,  2);
        I2_GEMV_CASE( 6,  64,  4);  I2_GEMV_CASE( 7,  64,  8);
        I2_GEMV_CASE( 8,  64, 16);
        I2_GEMV_CASE( 9, 128,  1);  I2_GEMV_CASE(10, 128,  2);
        I2_GEMV_CASE(11, 128,  4);  I2_GEMV_CASE(12, 128,  8);
        I2_GEMV_CASE(13, 128, 16);
        I2_GEMV_CASE(14, 256,  1);  I2_GEMV_CASE(15, 256,  2);
        I2_GEMV_CASE(16, 256,  4);  I2_GEMV_CASE(17, 256,  8);
        I2_GEMV_CASE(18, 256, 16);
        I2_GEMV_CASE(19, 512,  1);  I2_GEMV_CASE(20, 512,  2);
        I2_GEMV_CASE(21, 512,  4);  I2_GEMV_CASE(22, 512,  8);
        I2_GEMV_CASE(23, 512, 16);
        I2_GEMV_CASE(24, 1024,  1); I2_GEMV_CASE(25, 1024,  2);
        I2_GEMV_CASE(26, 1024,  4); I2_GEMV_CASE(27, 1024,  8);
        I2_GEMV_CASE(28, 1024, 16);
        I2_GEMV_CASE(29,  64, 32);  I2_GEMV_CASE(30, 128, 32);
        I2_GEMV_CASE(31, 256, 32);
        I2_GEMV_CASE(32,  64, 64);  I2_GEMV_CASE(33, 128, 64);
        I2_GEMV_CASE(34, 256, 64);
        default: break;
    }
}
#undef I2_GEMV_CASE

static std::unordered_map<std::string, int> g_gemv_i2_cache;
static std::mutex g_gemv_i2_mutex;

template <typename T = __half>
static void launch_gemv_i2_autotuned(
    hipStream_t stream,
    const T* A, const uint8_t* B,
    const T* scales, const uint8_t* zp, const T* bias,
    T* output,
    int64_t M, int64_t N, int64_t K, int64_t batch_count, int64_t block_size)
{
    (void)batch_count;
    std::string key = gemvCacheKey(M, N, K, block_size, false, zp != nullptr);
    key += "_i2";
    if (sizeof(T) == 4) key += "_f32";  // separate cache for fp32

    {
        std::lock_guard<std::mutex> lk(g_gemv_i2_mutex);
        auto it = g_gemv_i2_cache.find(key);
        if (it != g_gemv_i2_cache.end()) {
            launchGemvI2Config<T>(it->second, stream, A, B, scales, zp,
                                  bias, output, M, N, K, block_size);
            return;
        }
    }

    // Autotune
    constexpr int ITERS = 5;
    int best_id = 6; // BS=64, TN=4 default
    float best_ms = 1e9f;

    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0); hipEventCreate(&ev1);

    for (int cid = 0; cid < kNumGemvConfigs; cid++) {
        auto& c = kGemvConfigs[cid];
        if (c.tile_n > N || c.threads > 1024) continue;

        launchGemvI2Config<T>(cid, stream, A, B, scales, zp, bias,
                              output, M, N, K, block_size);
        hipStreamSynchronize(stream);
        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchGemvI2Config<T>(cid, stream, A, B, scales, zp, bias,
                                  output, M, N, K, block_size);
        hipEventRecord(ev1, stream);
        hipStreamSynchronize(stream);

        float ms = 0;
        hipEventElapsedTime(&ms, ev0, ev1);
        ms /= ITERS;
        if (ms < best_ms) { best_ms = ms; best_id = cid; }
    }
    hipEventDestroy(ev0); hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] gemv_i2 autotune N=%lld K=%lld -> "
        "config[%d] BS=%d TN=%d %.3fms\n",
        (long long)N, (long long)K, best_id,
        kGemvConfigs[best_id].threads, kGemvConfigs[best_id].tile_n,
        best_ms);

    {
        std::lock_guard<std::mutex> lk(g_gemv_i2_mutex);
        g_gemv_i2_cache[key] = best_id;
    }
    launchGemvI2Config<T>(best_id, stream, A, B, scales, zp, bias,
                          output, M, N, K, block_size);
}

/* -----------------------------------------------------------------------
 * Fused bits=2 GEMV launcher: DQ + MatMul + Q in one kernel
 * ----------------------------------------------------------------------- */

#define I2_FUSED_GEMV_CASE(ID, BS, TN) \
    case ID: \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(matmul_nbits_gemv_i2_fused_kernel<BS, TN>), \
            grd, blk, 0, stream, \
            A_u16, B, w_scales, w_zp, dq_scale, dq_zp, q_scale, q_zp, \
            output, M, N, K, block_size); \
        break

static void launchFusedGemvI2Config(
    int cid, hipStream_t stream,
    const uint16_t* A_u16, const uint8_t* B,
    const float* w_scales, const uint8_t* w_zp,
    const float* dq_scale, const uint16_t* dq_zp,
    const float* q_scale,  const uint16_t* q_zp,
    uint16_t* output,
    int64_t M, int64_t N, int64_t K, int64_t block_size)
{
    auto& c = kGemvConfigs[cid];
    dim3 grd(static_cast<unsigned>((N + c.tile_n - 1) / c.tile_n),
             static_cast<unsigned>(M), 1u);
    dim3 blk(c.threads);
    (void)hipGetLastError();
    switch (cid) {
        I2_FUSED_GEMV_CASE( 0,  32,  1);  I2_FUSED_GEMV_CASE( 1,  32,  2);
        I2_FUSED_GEMV_CASE( 2,  32,  4);  I2_FUSED_GEMV_CASE( 3,  32,  8);
        I2_FUSED_GEMV_CASE( 4,  64,  1);  I2_FUSED_GEMV_CASE( 5,  64,  2);
        I2_FUSED_GEMV_CASE( 6,  64,  4);  I2_FUSED_GEMV_CASE( 7,  64,  8);
        I2_FUSED_GEMV_CASE( 8,  64, 16);
        I2_FUSED_GEMV_CASE( 9, 128,  1);  I2_FUSED_GEMV_CASE(10, 128,  2);
        I2_FUSED_GEMV_CASE(11, 128,  4);  I2_FUSED_GEMV_CASE(12, 128,  8);
        I2_FUSED_GEMV_CASE(13, 128, 16);
        I2_FUSED_GEMV_CASE(14, 256,  1);  I2_FUSED_GEMV_CASE(15, 256,  2);
        I2_FUSED_GEMV_CASE(16, 256,  4);  I2_FUSED_GEMV_CASE(17, 256,  8);
        I2_FUSED_GEMV_CASE(18, 256, 16);
        I2_FUSED_GEMV_CASE(19, 512,  1);  I2_FUSED_GEMV_CASE(20, 512,  2);
        I2_FUSED_GEMV_CASE(21, 512,  4);  I2_FUSED_GEMV_CASE(22, 512,  8);
        I2_FUSED_GEMV_CASE(23, 512, 16);
        I2_FUSED_GEMV_CASE(24, 1024,  1); I2_FUSED_GEMV_CASE(25, 1024,  2);
        I2_FUSED_GEMV_CASE(26, 1024,  4); I2_FUSED_GEMV_CASE(27, 1024,  8);
        I2_FUSED_GEMV_CASE(28, 1024, 16);
        I2_FUSED_GEMV_CASE(29,  64, 32);  I2_FUSED_GEMV_CASE(30, 128, 32);
        I2_FUSED_GEMV_CASE(31, 256, 32);
        I2_FUSED_GEMV_CASE(32,  64, 64);  I2_FUSED_GEMV_CASE(33, 128, 64);
        I2_FUSED_GEMV_CASE(34, 256, 64);
        default: break;
    }
}
#undef I2_FUSED_GEMV_CASE

static std::unordered_map<std::string, int> g_gemv_i2_fused_cache;
static std::mutex g_gemv_i2_fused_mutex;

extern "C" int hip_matmul_nbits_i2_fused(
    void* stream,
    const void*  A_u16,    // quantized input uint16 [M, K]
    const void*  B,        // packed 2-bit weights uint8 [N, K/4]
    const void*  w_scales, // weight scales float32 [N, k_blocks]
    const void*  w_zp,     // weight zero points uint8 [N, k_blocks/2] (nullable)
    const void*  dq_scale, // activation DQ scale float32 scalar
    const void*  dq_zp,    // activation DQ zero point uint16 scalar (nullable)
    const void*  q_scale,  // output Q scale float32 scalar
    const void*  q_zp,     // output Q zero point uint16 scalar (nullable)
    void*        output,   // quantized output uint16 [M, N]
    int64_t M, int64_t N, int64_t K,
    int64_t block_size)
{
    if (M <= 0 || N <= 0 || K <= 0) return 0;
    if (!A_u16 || !B || !w_scales || !dq_scale || !q_scale || !output)
        return -1;

    hipStream_t s = static_cast<hipStream_t>(stream);
    const auto* au   = static_cast<const uint16_t*>(A_u16);
    const auto* bpu  = static_cast<const uint8_t*>(B);
    const auto* ws   = static_cast<const float*>(w_scales);
    const auto* wzp  = static_cast<const uint8_t*>(w_zp);
    const auto* dqs  = static_cast<const float*>(dq_scale);
    const auto* dqzp = static_cast<const uint16_t*>(dq_zp);
    const auto* qs   = static_cast<const float*>(q_scale);
    const auto* qzp  = static_cast<const uint16_t*>(q_zp);
    auto*        out  = static_cast<uint16_t*>(output);

    // M=1 decode: vectorized fused kernel with small TILE_N to avoid
    // register spill. TILE_N=1 gives coalesced B reads + minimal reg pressure.
    if (M == 1) {
        // For M=1 decode: autotune the vectorized fused kernel.
        std::string fused_key = gemvCacheKey(M, N, K, block_size, false, wzp != nullptr);
        fused_key += "_i2f";
        int fused_cfg = 7; // BS=64, TN=8 default
        {
            std::lock_guard<std::mutex> lk(g_gemv_i2_fused_mutex);
            auto it = g_gemv_i2_fused_cache.find(fused_key);
            if (it != g_gemv_i2_fused_cache.end()) {
                fused_cfg = it->second;
                launchFusedGemvI2Config(fused_cfg, s, au, bpu, ws, wzp,
                                        dqs, dqzp, qs, qzp, out,
                                        M, N, K, block_size);
                hipError_t err = hipGetLastError();
                if (err != hipSuccess)
                    fprintf(stderr, "[custom_kernels] i2_fused M=1 failed: %s\n",
                            hipGetErrorString(err));
                return static_cast<int>(err);
            }
        }

        // Autotune
        constexpr int ITERS = 5;
        float best_ms = 1e9f;
        hipEvent_t ev0, ev1;
        hipEventCreate(&ev0); hipEventCreate(&ev1);
        for (int cid = 0; cid < kNumGemvConfigs; cid++) {
            auto& c = kGemvConfigs[cid];
            if (c.tile_n > N || c.threads > 1024) continue;
            launchFusedGemvI2Config(cid, s, au, bpu, ws, wzp,
                                    dqs, dqzp, qs, qzp, out,
                                    M, N, K, block_size);
            hipStreamSynchronize(s);
            hipEventRecord(ev0, s);
            for (int i = 0; i < ITERS; i++)
                launchFusedGemvI2Config(cid, s, au, bpu, ws, wzp,
                                        dqs, dqzp, qs, qzp, out,
                                        M, N, K, block_size);
            hipEventRecord(ev1, s);
            hipStreamSynchronize(s);
            float ms = 0; hipEventElapsedTime(&ms, ev0, ev1); ms /= ITERS;
            if (ms < best_ms) { best_ms = ms; fused_cfg = cid; }
        }
        hipEventDestroy(ev0); hipEventDestroy(ev1);
        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels] i2_fused M=1 autotune N=%lld K=%lld -> "
            "config[%d] BS=%d TN=%d %.3fms\n",
            (long long)N, (long long)K, fused_cfg,
            kGemvConfigs[fused_cfg].threads, kGemvConfigs[fused_cfg].tile_n,
            best_ms);
        {
            std::lock_guard<std::mutex> lk(g_gemv_i2_fused_mutex);
            g_gemv_i2_fused_cache[fused_key] = fused_cfg;
        }
        launchFusedGemvI2Config(fused_cfg, s, au, bpu, ws, wzp,
                                dqs, dqzp, qs, qzp, out,
                                M, N, K, block_size);
        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
            fprintf(stderr, "[custom_kernels] i2_fused M=1 failed: %s\n",
                    hipGetErrorString(err));
        return static_cast<int>(err);
    }

    // M>1 prefill path: vectorized TILE_N template kernel.
    std::string key = gemvCacheKey(M, N, K, block_size, false, wzp != nullptr);
    key += "_i2f";

    int best_id = 2; // BS=32, TN=4 — best from previous autotune

    {
        std::lock_guard<std::mutex> lk(g_gemv_i2_fused_mutex);
        auto it = g_gemv_i2_fused_cache.find(key);
        if (it != g_gemv_i2_fused_cache.end())
            best_id = it->second;
    }

    (void)hipGetLastError();
    launchFusedGemvI2Config(best_id, s, au, bpu, ws, wzp,
                            dqs, dqzp, qs, qzp, out,
                            M, N, K, block_size);
    {
        hipError_t err = hipGetLastError();
        if (err != hipSuccess)
            fprintf(stderr, "[custom_kernels] hip_matmul_nbits_i2_fused failed: %s\n",
                    hipGetErrorString(err));
        return static_cast<int>(err);
    }

    // Autotune on first call (disabled for now — use fixed configs above)
    constexpr int ITERS = 5;
    float best_ms = 1e9f;

    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0); hipEventCreate(&ev1);

    for (int cid = 0; cid < kNumGemvConfigs; cid++) {
        auto& c = kGemvConfigs[cid];
        if (c.tile_n > N || c.threads > 1024) continue;

        launchFusedGemvI2Config(cid, s, au, bpu, ws, wzp,
                                dqs, dqzp, qs, qzp, out,
                                M, N, K, block_size);
        hipStreamSynchronize(s);
        hipEventRecord(ev0, s);
        for (int i = 0; i < ITERS; i++)
            launchFusedGemvI2Config(cid, s, au, bpu, ws, wzp,
                                    dqs, dqzp, qs, qzp, out,
                                    M, N, K, block_size);
        hipEventRecord(ev1, s);
        hipStreamSynchronize(s);
        float ms = 0; hipEventElapsedTime(&ms, ev0, ev1); ms /= ITERS;
        if (ms < best_ms) { best_ms = ms; best_id = cid; }
    }
    hipEventDestroy(ev0); hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] i2_fused autotune N=%lld K=%lld -> "
        "config[%d] BS=%d TN=%d %.3fms\n",
        (long long)N, (long long)K, best_id,
        kGemvConfigs[best_id].threads, kGemvConfigs[best_id].tile_n,
        best_ms);

    {
        std::lock_guard<std::mutex> lk(g_gemv_i2_fused_mutex);
        g_gemv_i2_fused_cache[key] = best_id;
    }

    launchFusedGemvI2Config(best_id, s, au, bpu, ws, wzp,
                            dqs, dqzp, qs, qzp, out,
                            M, N, K, block_size);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "[custom_kernels] hip_matmul_nbits_i2_fused failed: %s\n",
                hipGetErrorString(err));
    }
    return static_cast<int>(err);
}

/* -----------------------------------------------------------------------
 * GEMV (int8) Autotune: dispatch, benchmark, cache
 *
 * Reuses kGemvConfigs (28 configs of BLOCK_SIZE x TILE_N) and the same
 * autotune mechanics as the int4 path, but launches the int8 GEMV kernel
 * matmul_nbits_gemv_kernel_i8 instead. A separate cache is used because
 * the int8 kernel's per-config performance ranking differs from int4 (B
 * is 2x larger at fp16-equivalent footprint, the inner loop has half as
 * many decode shifts, and zp is read directly as uint8 rather than
 * unpacked first).
 * ----------------------------------------------------------------------- */

static void launchGemvConfigI8(
    int config_id, hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs)
{
    auto& c = kGemvConfigs[config_id];
    dim3 grid(static_cast<unsigned>((N + c.tile_n - 1) / c.tile_n),
              static_cast<unsigned>(M),
              static_cast<unsigned>(batch));
    dim3 block(c.threads);

    // The (BS, TN) at every case ID below MUST match kGemvConfigs[ID].
    // grid/block above are computed from kGemvConfigs[config_id], while the
    // kernel template parameters BLOCK_SIZE/TILE_N are baked in here at
    // compile time. If the two ever drift, the launched grid/block won't
    // match the template the kernel was compiled for: cross-warp reduction
    // reads uninitialized warp_sums slots (NaN/Inf in output), the K
    // accumulation stride (BLOCK_SIZE) won't match the launched thread
    // count, and only a fraction of the N outputs get written -- the rest
    // keep whatever was in the (often reused) output buffer beforehand.
    // This kept layers.0 in_proj_a inheriting in_proj_b's residual values
    // and was masked for in_proj_b only because autotune's prior launches
    // happened to leave the slot in a "looks correct" state.
#define GEMV_I8_CASE(ID, BS, TN) \
    case ID: \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(matmul_nbits_gemv_kernel_i8<BS, TN>), \
            grid, block, 0, stream, A, B, sc, zp, bi, out, M, N, K, bs); \
        break

    switch (config_id) {
        GEMV_I8_CASE( 0,  32,  1);  GEMV_I8_CASE( 1,  32,  2);
        GEMV_I8_CASE( 2,  32,  4);  GEMV_I8_CASE( 3,  32,  8);
        GEMV_I8_CASE( 4,  64,  1);  GEMV_I8_CASE( 5,  64,  2);
        GEMV_I8_CASE( 6,  64,  4);  GEMV_I8_CASE( 7,  64,  8);
        GEMV_I8_CASE( 8,  64, 16);
        GEMV_I8_CASE( 9, 128,  1);  GEMV_I8_CASE(10, 128,  2);
        GEMV_I8_CASE(11, 128,  4);  GEMV_I8_CASE(12, 128,  8);
        GEMV_I8_CASE(13, 128, 16);
        GEMV_I8_CASE(14, 256,  1);  GEMV_I8_CASE(15, 256,  2);
        GEMV_I8_CASE(16, 256,  4);  GEMV_I8_CASE(17, 256,  8);
        GEMV_I8_CASE(18, 256, 16);
        GEMV_I8_CASE(19, 512,  1);  GEMV_I8_CASE(20, 512,  2);
        GEMV_I8_CASE(21, 512,  4);  GEMV_I8_CASE(22, 512,  8);
        GEMV_I8_CASE(23, 512, 16);
        GEMV_I8_CASE(24, 1024,  1); GEMV_I8_CASE(25, 1024,  2);
        GEMV_I8_CASE(26, 1024,  4); GEMV_I8_CASE(27, 1024,  8);
        GEMV_I8_CASE(28, 1024, 16);
        GEMV_I8_CASE(29,  64, 32);  GEMV_I8_CASE(30, 128, 32);
        GEMV_I8_CASE(31, 256, 32);
        GEMV_I8_CASE(32,  64, 64);  GEMV_I8_CASE(33, 128, 64);
        GEMV_I8_CASE(34, 256, 64);
        default: break;
    }
#undef GEMV_I8_CASE
}

static int tuneGemvConfigI8(
    hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs)
{
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);

    float best_ms = 1e30f;
    int   best_id = 8; // 128 threads, TILE_N=8 -- safe default for typical N
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    for (int cid = 0; cid < kNumGemvConfigs; cid++) {
        auto& cfg = kGemvConfigs[cid];
        if (cfg.tile_n >= 32 && N < 1024) continue;

        for (int w = 0; w < WARMUP; w++)
            launchGemvConfigI8(cid, stream, A, B, sc, zp, bi, out,
                               M, N, K, batch, bs);

        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchGemvConfigI8(cid, stream, A, B, sc, zp, bi, out,
                               M, N, K, batch, bs);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   i8 config[%2d] threads=%3d tile_n=%d : %.4f ms\n",
            cid, kGemvConfigs[cid].threads, kGemvConfigs[cid].tile_n,
            ms / ITERS);

        if (ms < best_ms) {
            best_ms = ms;
            best_id = cid;
        }
    }

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] GEMV-i8 autotune: M=%lld N=%lld K=%lld bs=%lld -> "
        "best config[%d] threads=%d tile_n=%d (%.4f ms/iter)\n",
        (long long)M, (long long)N, (long long)K, (long long)bs,
        best_id, kGemvConfigs[best_id].threads,
        kGemvConfigs[best_id].tile_n, best_ms / ITERS);

    return best_id;
}

static std::unordered_map<std::string, int> gemv_i8_cache;
static std::mutex gemv_i8_tune_mutex;

static std::string gemvI8CacheKey(int64_t M, int64_t N, int64_t K,
                                   int64_t bs, bool has_zp) {
    // has_zp affects the inner loop (one global ZP read per iter), so
    // configs may rank differently with vs without -- key on it.
    return std::to_string(M) + "_" + std::to_string(N) + "_" +
           std::to_string(K) + "_" + std::to_string(bs) + "_" +
           (has_zp ? "1" : "0");
}

static int getOrTuneGemvConfigI8(
    hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs)
{
    const bool has_zp = (zp != nullptr);
    std::string key = gemvI8CacheKey(M, N, K, bs, has_zp);

    std::lock_guard<std::mutex> lock(gemv_i8_tune_mutex);

    auto it = gemv_i8_cache.find(key);
    if (it != gemv_i8_cache.end())
        return it->second;

    int best = tuneGemvConfigI8(stream, A, B, sc, zp, bi, out,
                                 M, N, K, batch, bs);
    gemv_i8_cache[key] = best;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] GEMV-i8 cache: stored key=\"%s\" -> config[%d] "
        "(cache size=%zu)\n",
        key.c_str(), best, gemv_i8_cache.size());

    return best;
}

/* =======================================================================
 * Section 2: RDNA3 WMMA Fused GEMM + Dequantization Kernel
 *
 * Computes C = A x dequant(B_packed)^T  where:
 *   A        : FP16 row-major (M x K), stride lda = K
 *   B_packed : uint4 packed into bytes (N x K/2), row-major per column
 *   scales   : FP16 (N x num_groups_k), per-column per-group scale
 *   zeros    : FP16 (N x num_groups_k), per-column per-group zero point
 *   C        : FP16 row-major (M x N), stride ldc = N
 *
 * Requires RDNA3+ (gfx11xx/gfx12xx) for WMMA intrinsics.
 * K must be multiple of 32. M and N can be any positive value (boundary-safe).
 * ======================================================================= */

/* =======================================================================
 * Section 2a: Standalone dequantization kernel (uint4 → fp16)
 *
 * For the separate dequant + fp16 GEMM path.
 * Each thread processes 8 nibbles (= 4 bytes) from one N row.
 * ======================================================================= */

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

    uint32_t four = *reinterpret_cast<const uint32_t*>(
        &B_packed[n * (K / 2) + k8 / 2]);

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

/* =======================================================================
 * Section 2a-2: bits=2 WMMA Decode Kernel (M=1 specialization)
 *
 * Dedicated WMMA kernel for single-token decode (M=1) with 2-bit weights.
 * Uses RDNA3 __builtin_amdgcn_wmma_f32_16x16x16_f16_w32 intrinsic.
 *
 * Strategy for M=1: pad A to M=16 rows in shared memory (rows 1..15 = 0),
 * run standard WMMA, output only row 0. Gives full matrix-core throughput.
 *
 * Tile: one warp (32 threads) computes BN output columns where BN=N_TILE.
 * K is processed in WMMA_TILE=16 steps.
 * Grid: (ceil(N/N_TILE), 1). Block: (32 threads = 1 warp).
 *
 * Fragment layout (RDNA3 WMMA 16x16x16 f16/f32):
 *   A frag: each thread holds 16 fp16 values = one row of the 16x16 A tile
 *           mapped as smA[lane][k_step*16 + 0..15]
 *   B frag: each thread holds 16 fp16 values = one row of the 16x16 B tile
 *           mapped as smB[noff+lane][k_step*16 + 0..15]
 *   acc: float8 per (wm, wn) pair
 *
 * T = float for ORCA (fp32 A/scales/C), __half for standard fp16 models.
 * ======================================================================= */

#ifndef WMMA_TILE
#define WMMA_TILE 16
#endif

// N_TILE must be a multiple of 16 (one WMMA column tile per N-step).
// BLK=32: one warp. K must be multiple of 16.
template <int N_TILE = 64, typename T = float>
__global__
__attribute__((amdgpu_flat_work_group_size(32, 32)))
void matmul_nbits_u2_wmma_decode(
    const T*       __restrict__ A,        // [K]  — M=1, batch collapsed
    const uint8_t* __restrict__ B,        // [N, K/4] packed 2-bit
    const T*       __restrict__ scales,   // [N, num_groups_k] fp32 or fp16
    const uint8_t* __restrict__ zeros,    // [N, ceil(num_groups_k/4)] nullable
    T*             __restrict__ C,        // [N] output
    int N, int K, int group_size, int num_groups_k)
{
    constexpr int N_STEPS = N_TILE / WMMA_TILE;  // WMMA column steps per block
    // Use uint32 smem so 8-uint32 loads align naturally.
    // Each row: WMMA_TILE fp16 = 8 uint32.
    constexpr int KSTR_U32 = WMMA_TILE / 2;  // 8 uint32 per row (16 fp16)
    __shared__ uint32_t smA_u32[WMMA_TILE][KSTR_U32];
    __shared__ uint32_t smB_u32[N_TILE][KSTR_U32];

    const int n0  = static_cast<int>(blockIdx.x) * N_TILE;
    const int tid = threadIdx.x;   // 0..31
    const int lane = tid & 15;     // 0..15
    const int sub  = tid >> 4;     // 0 or 1

    // Accumulator array: N_STEPS × float8, one per WMMA N-step
    float8 acc[N_STEPS];
#pragma unroll
    for (int ns = 0; ns < N_STEPS; ns++)
#pragma unroll
        for (int e = 0; e < 8; e++) acc[ns][e] = 0.0f;

    // Main K loop: process in WMMA_TILE=16 steps
    for (int k0 = 0; k0 < K; k0 += WMMA_TILE) {
        const int grp = k0 / group_size;

        // ---- Fill smA: row 0 = A[k0..k0+15], rows 1..15 = 0 ----
        // 32 threads: tid 0..15 fill row 0, tid 16..31 fill row 1 (zeros)
        // Load smA: 32 threads, each handles 1 uint32 slot.
        // tid 0..7 → smA_u32[0][0..7] (row 0, all 16 fp16 as 8 uint32)
        // tid 8..31 → smA_u32[1..3][0..7] zeros (rows 1..3)
        // Then zero out rows 4..15 with the remaining threads.
        {
            // Each thread fills smA_u32[row][col] where row=tid/8, col=tid%8
            int row = tid / KSTR_U32;
            int col = tid % KSTR_U32;
            if (row == 0) {
                // Real A row: two fp16 values at k0 + col*2 and k0 + col*2+1
                int ki0 = k0 + col * 2;
                int ki1 = ki0 + 1;
                _Float16 v0 = (ki0 < K) ? (_Float16)static_cast<float>(A[ki0]) : (_Float16)0;
                _Float16 v1 = (ki1 < K) ? (_Float16)static_cast<float>(A[ki1]) : (_Float16)0;
                smA_u32[0][col] = *reinterpret_cast<const uint16_t*>(&v0) |
                                  (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&v1)) << 16);
            } else if (row < WMMA_TILE) {
                smA_u32[row][col] = 0u;
            }
            // Any remaining rows
            for (int r = 32 / KSTR_U32; r < WMMA_TILE; r++) {
                if (tid < KSTR_U32) smA_u32[r][tid] = 0u;
            }
        }

        // ---- Fill smB: dequantize N_TILE B columns for K-tile k0..k0+15 ----
        // 32 threads, each handles N_TILE/32 = 2 columns (for N_TILE=64).
        // For each column n: load 16 two-bit weights = 4 bytes at k0/4.
        for (int pass = 0; pass < N_TILE; pass += 32) {
            int n = n0 + pass + tid;
            if (n < N) {
                // 4 bytes = 16 two-bit weights at k0 (k0 is multiple of 16)
                uint32_t bw = *reinterpret_cast<const uint32_t*>(
                    &B[static_cast<int64_t>(n) * (K / 4) + k0 / 4]);
                _Float16 sc = (_Float16)static_cast<float>(scales[n * num_groups_k + grp]);
                float wz_val = 2.0f;  // default ZP=2 for bits=2
                if (zeros) {
                    int zi = n * num_groups_k + grp;
                    wz_val = static_cast<float>((zeros[zi / 4] >> ((zi % 4) * 2)) & 0x3);
                }
                _Float16 neg_zs = (_Float16)(-wz_val * (float)sc);
                // Pack pairs of dequantized weights into uint32 for smB_u32
#pragma unroll
                for (int ki = 0; ki < KSTR_U32; ki++) {
                    float wf0 = static_cast<float>((bw >> ((ki*2  ) * 2)) & 0x3);
                    float wf1 = static_cast<float>((bw >> ((ki*2+1) * 2)) & 0x3);
                    _Float16 b0 = (_Float16)wf0 * sc + neg_zs;
                    _Float16 b1 = (_Float16)wf1 * sc + neg_zs;
                    smB_u32[pass + tid][ki] =
                        *reinterpret_cast<const uint16_t*>(&b0) |
                        (static_cast<uint32_t>(*reinterpret_cast<const uint16_t*>(&b1)) << 16);
                }
            } else {
#pragma unroll
                for (int ki = 0; ki < KSTR_U32; ki++)
                    smB_u32[pass + tid][ki] = 0u;
            }
        }

        __syncthreads();

        // ---- WMMA compute ----
        // Load A fragment from smA_u32[lane][0..7]: 8 uint32 = 16 fp16
        half16 a_frag;
        {
            uint32_t* dst = reinterpret_cast<uint32_t*>(&a_frag);
#pragma unroll
            for (int i = 0; i < KSTR_U32; i++) dst[i] = smA_u32[lane][i];
        }

#pragma unroll
        for (int ns = 0; ns < N_STEPS; ns++) {
            int noff = ns * WMMA_TILE;
            half16 b_frag;
            {
                uint32_t* dst = reinterpret_cast<uint32_t*>(&b_frag);
#pragma unroll
                for (int i = 0; i < KSTR_U32; i++) dst[i] = smB_u32[noff + lane][i];
            }
            acc[ns] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
                a_frag, b_frag, acc[ns]);
        }

        __syncthreads();
    }

    // ---- Write output: only row 0 is valid for M=1 ----
    // WMMA output layout: thread writes rows (e*2 + sub) for e=0..7.
    // Row 0: e=0, sub=0 → tid where sub=0 (tid < 16) writes acc[ns][0].
    if (sub == 0) {
#pragma unroll
        for (int ns = 0; ns < N_STEPS; ns++) {
            int n = n0 + ns * WMMA_TILE + lane;
            if (n < N)
                C[n] = static_cast<T>(acc[ns][0]);
        }
    }
}

// Launcher for the M=1 WMMA decode kernel
template <int N_TILE = 64, typename T = float>
static void launch_u2_wmma_decode(
    hipStream_t stream,
    const T* A, const uint8_t* B,
    const T* scales, const uint8_t* zeros,
    T* C, int N, int K, int gs, int ngk)
{
    dim3 grd(static_cast<unsigned>((N + N_TILE - 1) / N_TILE));
    dim3 blk(32);
    (void)hipGetLastError();
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(matmul_nbits_u2_wmma_decode<N_TILE, T>),
        grd, blk, 0, stream,
        A, B, scales, zeros, C, N, K, gs, ngk);
}

/* =======================================================================
 * Section 2b: WMMA kernel (supports both fused dequant and fp16 B)
 * ======================================================================= */

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float    float8 __attribute__((ext_vector_type(8)));

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

    // Fused dequant state (compile-time eliminated when FUSED_DQ=false)
    [[maybe_unused]] const int b_byte_base =
        FUSED_DQ ? (b_n_g * (K >> 1) + (b_k8 >> 1)) : 0;
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
#pragma unroll
            for(int wn = 0; wn < WT_N; wn++)
            {
                int noff              = (wcol * WT_N + wn) * WMMA_TILE;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&b_frag[wn]);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smB[buf][noff + lane][ks * WMMA_TILE]);
#pragma unroll
                for(int i = 0; i < 8; i++)
                    dst[i] = src[i];
            }
#pragma unroll
            for(int wm = 0; wm < WT_M; wm++)
            {
                int moff              = (wrow * WT_M + wm) * WMMA_TILE;
                half16 a_frag;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&a_frag);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smA[buf][moff + lane][ks * WMMA_TILE]);
#pragma unroll
                for(int i = 0; i < 8; i++)
                    dst[i] = src[i];

#pragma unroll
                for(int wn = 0; wn < WT_N; wn++)
                    acc[wm][wn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
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
                int r = e * 2 + sub;
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

/* =======================================================================
 * Section 2c: bits=2 WMMA fused dequant GEMM
 *
 * Same structure as GemmFp16U4Impl but for 2-bit packed weights.
 * B_packed layout: [N, K/4] uint8 — four 2-bit crumbs per byte.
 * Each thread loads 8 weights (= 2 bytes = uint16_t load, not uint32_t).
 * Default ZP = 2 (= 2^(bits-1)); scales are fp32 for ORCA.
 *
 * Key differences from 4-bit:
 *   - b_byte_base = b_n_g * (K/4) + (b_k8/4)  [K/4 bytes per row]
 *   - Load 2 bytes (uint16_t) to get 8 two-bit weights
 *   - Dequant: crumb = (pf_b2 >> (i*2)) & 0x3
 *   - Default ZP = 2 (symmetric, 2^(bits-1))
 *   - scales are float32 (ORCA), NOT float16 like bits=4
 * ======================================================================= */

template <int BM_T, int BN_T, int WT_M, int WT_N, bool USE_ZEROS,
          bool BOUNDS_CHECK = false>
__device__ __forceinline__
void GemmFp16U2Impl(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B_packed,  // [N, K/4] bits=2
    const float* __restrict__ scales_fp32,        // [N, num_groups_k] float32
    const unsigned char* __restrict__ zeros_u8,   // [N, ceil(num_groups_k/4)] nullable
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
    // For bits=2: each thread handles b_k8=8 weights = 2 bytes.
    // Thread count per B column: THR_L / (BK_T/8) = THR_L*8/BK_T
    // B loading: THR_L * 8 weights total = BN_T * BK_T weights.
    static_assert(THR_L * 8 == BN_T * BK_T, "B loading balance");

    __shared__ _Float16 smA[2][BM_T][K_STR];
    __shared__ _Float16 smB[2][BN_T][K_STR];

    const int tid  = threadIdx.x;
    const int wid  = tid / 32;
    const int lid  = tid % 32;
    const int lane = lid % 16;
    const int sub  = lid / 16;
    const int wrow = wid / WN_L;
    const int wcol = wid % WN_L;

    // Block-to-tile mapping (same swizzle as bits=4)
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

    // B loading: same column structure as bits=4 but 2 bytes per 8 weights
    constexpr int B_COL_SHIFT = __builtin_ctz(BK_T / 8);
    const int b_col       = tid >> B_COL_SHIFT;
    const int b_sub       = tid & ((1 << B_COL_SHIFT) - 1);
    const int b_k8        = b_sub << 3;  // start K index for this thread's 8 weights
    const int b_n_g_raw   = col0 + b_col;
    const bool b_valid    = BOUNDS_CHECK ? (b_n_g_raw < N) : true;
    const int b_n_g       = (BOUNDS_CHECK && !b_valid) ? 0 : b_n_g_raw;

    // bits=2: K/4 bytes per row; 8 weights = 2 bytes at offset b_k8/4
    const int b_byte_base = b_n_g * (K / 4) + (b_k8 / 4);
    const int b_gid_base  = b_n_g * num_groups_k;

    int      cached_k_grp = -1;
    _Float16 cached_s = 0;
    _Float16 cached_nzs = 0;

    float8 acc[WT_M][WT_N];
#pragma unroll
    for(int i = 0; i < WT_M; i++)
#pragma unroll
        for(int j = 0; j < WT_N; j++)
#pragma unroll
            for(int e = 0; e < 8; e++)
                acc[i][j][e] = 0.0f;

    uint32_t pf_a1[A_VL], pf_a2[A_VL];
    int      pf_am[A_VL], pf_ak[A_VL];
    uint16_t pf_b2 = 0;  // 2 bytes = 8 two-bit weights (vs 4 bytes for bits=4)

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
                    pf_a1[ld] = 0; pf_a2[ld] = 0;
                }
            }
        }
        if (b_valid) {
            // Load 2 bytes = 8 two-bit weights
            pf_b2 = *reinterpret_cast<const uint16_t*>(
                &B_packed[b_byte_base + (t / 4)]);
            const int grp = (t + b_k8) / group_size;
            if(__builtin_expect(grp != cached_k_grp, 0))
            {
                cached_k_grp = grp;
                // fp32 scale → fp16 for WMMA (WMMA operates on fp16)
                cached_s     = (_Float16)scales_fp32[grp + b_gid_base];
                if constexpr(USE_ZEROS) {
                    // ZP is 2-bit packed: 4 ZPs per byte
                    int zp_byte  = (grp + b_gid_base) / 4;
                    int zp_crumb = (grp + b_gid_base) % 4;
                    float zp_val = static_cast<float>(
                                       (zeros_u8[zp_byte] >> (zp_crumb * 2)) & 0x3);
                    cached_nzs   = (_Float16)(-zp_val * (float)cached_s);
                } else {
                    cached_nzs = (_Float16)((_Float16)(-2) * cached_s); // default ZP=2
                }
            }
        }
    };

    auto storeToShared = [&](int buf) __attribute__((always_inline))
    {
        if (b_valid) {
            _Float16 s16  = cached_s;
            _Float16 nzs  = cached_nzs;
            _Float16 bv[8];
            // Extract 8 two-bit crumbs from pf_b2 (uint16_t = 16 bits = 8 crumbs)
            bv[0] = static_cast<_Float16>((pf_b2      ) & 0x3u) * s16 + nzs;
            bv[1] = static_cast<_Float16>((pf_b2 >>  2) & 0x3u) * s16 + nzs;
            bv[2] = static_cast<_Float16>((pf_b2 >>  4) & 0x3u) * s16 + nzs;
            bv[3] = static_cast<_Float16>((pf_b2 >>  6) & 0x3u) * s16 + nzs;
            bv[4] = static_cast<_Float16>((pf_b2 >>  8) & 0x3u) * s16 + nzs;
            bv[5] = static_cast<_Float16>((pf_b2 >> 10) & 0x3u) * s16 + nzs;
            bv[6] = static_cast<_Float16>((pf_b2 >> 12) & 0x3u) * s16 + nzs;
            bv[7] = static_cast<_Float16>((pf_b2 >> 14)        ) * s16 + nzs;
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8])     = *reinterpret_cast<uint2*>(&bv[0]);
            *reinterpret_cast<uint2*>(&smB[buf][b_col][b_k8 + 4]) = *reinterpret_cast<uint2*>(&bv[4]);
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

    // computeWMMA is identical to the 4-bit version — operates on fp16 smB
    auto computeWMMA = [&](int buf) __attribute__((always_inline))
    {
#pragma unroll
        for(int ks = 0; ks < K_STEPS; ks++)
        {
            half16 b_frag[WT_N];
#pragma unroll
            for(int wn = 0; wn < WT_N; wn++)
            {
                int noff              = (wcol * WT_N + wn) * WMMA_TILE;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&b_frag[wn]);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smB[buf][noff + lane][ks * WMMA_TILE]);
#pragma unroll
                for(int i = 0; i < 8; i++) dst[i] = src[i];
            }
#pragma unroll
            for(int wm = 0; wm < WT_M; wm++)
            {
                int moff = (wrow * WT_M + wm) * WMMA_TILE;
                half16 a_frag;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&a_frag);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smA[buf][moff + lane][ks * WMMA_TILE]);
#pragma unroll
                for(int i = 0; i < 8; i++) dst[i] = src[i];
#pragma unroll
                for(int wn = 0; wn < WT_N; wn++)
                    acc[wm][wn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
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

    // Write outputs (same as bits=4)
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
                int r = e * 2 + sub;
                if constexpr (BOUNDS_CHECK) {
                    if (mbase + r < M && nbase + lane < N)
                        C[(mbase + r) * ldc + nbase + lane] = (_Float16)acc[wm][wn][e];
                } else {
                    C[(mbase + r) * ldc + nbase + lane] = (_Float16)acc[wm][wn][e];
                }
            }
        }
    }
}

// bits=2 WMMA entry points
// GEMM_WG_SZ is defined later in the file; replicate here for the U2 kernels.
#ifndef GEMM_WG_SZ
#define GEMM_WG_SZ(BM, BN, WM, WN) \
    ((BM / (WM * 16)) * (BN / (WN * 16)) * 32)
#endif

template <int BM_T = 128, int BN_T = 128, int WT_M = 2, int WT_N = 2,
          int WAVES_EU = 8, bool BC = false>
__global__
__attribute__((amdgpu_flat_work_group_size(
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N),
    GEMM_WG_SZ(BM_T, BN_T, WT_M, WT_N))))
__attribute__((amdgpu_waves_per_eu(WAVES_EU)))
void MatMulNBitsU2WMMA_NoZP(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B_packed,
    const float* __restrict__ scales,
    const unsigned char* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16U2Impl<BM_T, BN_T, WT_M, WT_N, false, BC>(
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
void MatMulNBitsU2WMMA_ZP(
    int M, int N, int K,
    const _Float16* __restrict__ A, int lda,
    const unsigned char* __restrict__ B_packed,
    const float* __restrict__ scales,
    const unsigned char* __restrict__ zeros,
    int group_size, int num_groups_k,
    _Float16* __restrict__ C, int ldc,
    int swizzle_n = 4)
{
    GemmFp16U2Impl<BM_T, BN_T, WT_M, WT_N, true, BC>(
        M, N, K, A, lda, B_packed, scales, zeros,
        group_size, num_groups_k, C, ldc, swizzle_n);
}

/* WMMA kernel entry points with per-variant GPU attributes */

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

/* fp16 GEMM kernel entry (for pre-dequantized B, FUSED_DQ=false) */

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

/* -----------------------------------------------------------------------
 * WMMA Autotune: config table, dispatch, benchmark, cache
 *
 * Tile shapes: 64x64, 64x128, 128x64, 128x128, 128x256, 256x128
 *              + 64x32, 128x32 (WT2x1, small-N only)
 * Swizzle values: 2, 4, 8, 16, 32, 64 (not all combos)
 * Warp-tile variants: WT2x2 (default), WT4x2 (256x128), WT2x1 (64x32/128x32)
 * Each tile × swizzle combo appears in fused and separate-dequant variants.
 * Autotune benchmarks applicable configs and picks the fastest.
 * BN<64 tiles (WT2x1) are skipped when N > 2*BN (i.e. N>64).
 * ----------------------------------------------------------------------- */

struct WmmaConfig {
    int  bm, bn;
    int  swizzle_n;
    bool fused;
    int  wt_m = 2, wt_n = 2;
};

static constexpr WmmaConfig kWmmaConfigs[] = {
    // --- fused dequant + GEMM ---
    { 64,  64,  2, true}, { 64,  64,  4, true}, { 64,  64,  8, true}, { 64,  64, 16, true},
    { 64, 128,  2, true}, { 64, 128,  4, true}, { 64, 128,  8, true}, { 64, 128, 16, true},
    {128,  64,  2, true}, {128,  64,  4, true}, {128,  64,  8, true}, {128,  64, 16, true},
    {128, 128,  2, true}, {128, 128,  4, true}, {128, 128,  8, true}, {128, 128, 16, true}, {128, 128, 32, true},
    {128, 256,  4, true}, {128, 256,  8, true}, {128, 256, 16, true},
    {128, 128, 64, true},
    {128, 256, 32, true},
    // --- WT_M=4 fused (256x128 tile, higher compute-to-smem ratio) ---
    {256, 128,  4, true, 4, 2}, {256, 128,  8, true, 4, 2},
    {256, 128, 16, true, 4, 2}, {256, 128, 32, true, 4, 2},
    // --- WT_M=2, WT_N=1 fused (BN=32 — optimized for small N) ---
    { 64,  32,  4, true, 2, 1}, { 64,  32,  8, true, 2, 1},
    {128,  32,  4, true, 2, 1}, {128,  32,  8, true, 2, 1},
    // --- separate dequant + fp16 GEMM ---
    { 64,  64,  2, false}, { 64,  64,  4, false}, { 64,  64,  8, false}, { 64,  64, 16, false},
    { 64, 128,  2, false}, { 64, 128,  4, false}, { 64, 128,  8, false}, { 64, 128, 16, false},
    {128,  64,  2, false}, {128,  64,  4, false}, {128,  64,  8, false}, {128,  64, 16, false},
    {128, 128,  2, false}, {128, 128,  4, false}, {128, 128,  8, false}, {128, 128, 16, false}, {128, 128, 32, false},
    {128, 256,  4, false}, {128, 256,  8, false}, {128, 256, 16, false},
    {128, 128, 64, false},
    {128, 256, 32, false},
    // --- WT_M=4 separate dequant (256x128 tile) ---
    {256, 128,  4, false, 4, 2}, {256, 128,  8, false, 4, 2},
    {256, 128, 16, false, 4, 2}, {256, 128, 32, false, 4, 2},
    // --- WT_M=2, WT_N=1 separate dequant (BN=32 — for small N) ---
    { 64,  32,  4, false, 2, 1}, { 64,  32,  8, false, 2, 1},
    {128,  32,  4, false, 2, 1}, {128,  32,  8, false, 2, 1},
};
static constexpr int kNumWmmaConfigs =
    static_cast<int>(sizeof(kWmmaConfigs) / sizeof(kWmmaConfigs[0]));

static void launchDequant(
    hipStream_t stream,
    const unsigned char* B_packed, const _Float16* scales,
    const _Float16* zeros, _Float16* B_fp16,
    int N, int K, int gs, int ngk)
{
    constexpr int DQ_THR = 256;
    int k_groups = (K + DQ_THR * 8 - 1) / (DQ_THR * 8);
    dim3 dq_grid(static_cast<unsigned>(k_groups), static_cast<unsigned>(N));
    dim3 dq_block(DQ_THR);
    hipLaunchKernelGGL(dequant_u4_to_fp16, dq_grid, dq_block, 0, stream,
                       B_packed, scales, zeros, B_fp16, N, K, gs, ngk);
}

static void launchWmmaConfig(
    int config_id, hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const _Float16* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp,
    _Float16* B_fp16_buf = nullptr)
{
    auto& c = kWmmaConfigs[config_id];
    bool need_bc = (M % c.bm != 0) || (N % c.bn != 0);

    if (!c.fused) {
        launchDequant(stream, B, scales, zeros, B_fp16_buf, N, K, gs, ngk);

#define FP16_GEMM_LAUNCH(BM_V, BN_V, BC_V) do { \
    constexpr int WM_ = BM_V / (2 * 16); \
    constexpr int WN_ = BN_V / (2 * 16); \
    constexpr int THR_ = WM_ * WN_ * 32; \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V), \
               static_cast<unsigned>((M + BM_V - 1) / BM_V)); \
    dim3 block_(THR_); \
    hipLaunchKernelGGL( \
        HIP_KERNEL_NAME(MatMulNBitsFp16GEMM<BM_V, BN_V, 2, 2, 8, BC_V>), \
        grid_, block_, 0, stream, \
        M, N, K, A, lda, B_fp16_buf, C, ldc, c.swizzle_n); \
} while(0)

#define FP16_TILE_CASE(BM_V, BN_V) \
    if (need_bc) FP16_GEMM_LAUNCH(BM_V, BN_V, true); \
    else         FP16_GEMM_LAUNCH(BM_V, BN_V, false)

#define FP16_GEMM_LAUNCH_WT4(BM_V, BN_V, BC_V) do { \
    constexpr int WM_ = BM_V / (4 * 16); \
    constexpr int WN_ = BN_V / (2 * 16); \
    constexpr int THR_ = WM_ * WN_ * 32; \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V), \
               static_cast<unsigned>((M + BM_V - 1) / BM_V)); \
    dim3 block_(THR_); \
    hipLaunchKernelGGL( \
        HIP_KERNEL_NAME(MatMulNBitsFp16GEMM<BM_V, BN_V, 4, 2, 6, BC_V>), \
        grid_, block_, 0, stream, \
        M, N, K, A, lda, B_fp16_buf, C, ldc, c.swizzle_n); \
} while(0)

#define FP16_TILE_CASE_WT4(BM_V, BN_V) \
    if (need_bc) FP16_GEMM_LAUNCH_WT4(BM_V, BN_V, true); \
    else         FP16_GEMM_LAUNCH_WT4(BM_V, BN_V, false)

#define FP16_GEMM_LAUNCH_WT21(BM_V, BN_V, BC_V) do { \
    constexpr int WM_ = BM_V / (2 * 16); \
    constexpr int WN_ = BN_V / (1 * 16); \
    constexpr int THR_ = WM_ * WN_ * 32; \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V), \
               static_cast<unsigned>((M + BM_V - 1) / BM_V)); \
    dim3 block_(THR_); \
    hipLaunchKernelGGL( \
        HIP_KERNEL_NAME(MatMulNBitsFp16GEMM<BM_V, BN_V, 2, 1, 8, BC_V>), \
        grid_, block_, 0, stream, \
        M, N, K, A, lda, B_fp16_buf, C, ldc, c.swizzle_n); \
} while(0)

#define FP16_TILE_CASE_WT21(BM_V, BN_V) \
    if (need_bc) FP16_GEMM_LAUNCH_WT21(BM_V, BN_V, true); \
    else         FP16_GEMM_LAUNCH_WT21(BM_V, BN_V, false)

        if      (c.wt_m == 4 && c.bm == 256 && c.bn == 128)
                                                  FP16_TILE_CASE_WT4(256, 128);
        else if (c.wt_n == 1 && c.bm == 128 && c.bn == 32)
                                                  FP16_TILE_CASE_WT21(128, 32);
        else if (c.wt_n == 1 && c.bm ==  64 && c.bn == 32)
                                                  FP16_TILE_CASE_WT21( 64, 32);
        else if (c.bm ==  64 && c.bn ==  64) FP16_TILE_CASE( 64,  64);
        else if (c.bm ==  64 && c.bn == 128) FP16_TILE_CASE( 64, 128);
        else if (c.bm == 128 && c.bn ==  64) FP16_TILE_CASE(128,  64);
        else if (c.bm == 128 && c.bn == 256) FP16_TILE_CASE(128, 256);
        else                                  FP16_TILE_CASE(128, 128);

#undef FP16_TILE_CASE_WT21
#undef FP16_GEMM_LAUNCH_WT21
#undef FP16_TILE_CASE_WT4
#undef FP16_GEMM_LAUNCH_WT4
#undef FP16_TILE_CASE
#undef FP16_GEMM_LAUNCH
        return;
    }

#define WMMA_LAUNCH(BM_V, BN_V, BC_V) do { \
    constexpr int WM_ = BM_V / (2 * 16); \
    constexpr int WN_ = BN_V / (2 * 16); \
    constexpr int THR_ = WM_ * WN_ * 32; \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V), \
               static_cast<unsigned>((M + BM_V - 1) / BM_V)); \
    dim3 block_(THR_); \
    if (has_zp) { \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(MatMulNBitsWMMA_ZP<BM_V, BN_V, 2, 2, 8, BC_V>), \
            grid_, block_, 0, stream, \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    } else { \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(MatMulNBitsWMMA_NoZP<BM_V, BN_V, 2, 2, 8, BC_V>), \
            grid_, block_, 0, stream, \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    } \
} while(0)

#define WMMA_TILE_CASE(BM_V, BN_V) \
    if (need_bc) WMMA_LAUNCH(BM_V, BN_V, true); \
    else         WMMA_LAUNCH(BM_V, BN_V, false)

#define WMMA_LAUNCH_WT4(BM_V, BN_V, BC_V) do { \
    constexpr int WM_ = BM_V / (4 * 16); \
    constexpr int WN_ = BN_V / (2 * 16); \
    constexpr int THR_ = WM_ * WN_ * 32; \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V), \
               static_cast<unsigned>((M + BM_V - 1) / BM_V)); \
    dim3 block_(THR_); \
    if (has_zp) { \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(MatMulNBitsWMMA_ZP<BM_V, BN_V, 4, 2, 6, BC_V>), \
            grid_, block_, 0, stream, \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    } else { \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(MatMulNBitsWMMA_NoZP<BM_V, BN_V, 4, 2, 6, BC_V>), \
            grid_, block_, 0, stream, \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    } \
} while(0)

#define WMMA_TILE_CASE_WT4(BM_V, BN_V) \
    if (need_bc) WMMA_LAUNCH_WT4(BM_V, BN_V, true); \
    else         WMMA_LAUNCH_WT4(BM_V, BN_V, false)

#define WMMA_LAUNCH_WT21(BM_V, BN_V, BC_V) do { \
    constexpr int WM_ = BM_V / (2 * 16); \
    constexpr int WN_ = BN_V / (1 * 16); \
    constexpr int THR_ = WM_ * WN_ * 32; \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V), \
               static_cast<unsigned>((M + BM_V - 1) / BM_V)); \
    dim3 block_(THR_); \
    if (has_zp) { \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(MatMulNBitsWMMA_ZP<BM_V, BN_V, 2, 1, 8, BC_V>), \
            grid_, block_, 0, stream, \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    } else { \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(MatMulNBitsWMMA_NoZP<BM_V, BN_V, 2, 1, 8, BC_V>), \
            grid_, block_, 0, stream, \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    } \
} while(0)

#define WMMA_TILE_CASE_WT21(BM_V, BN_V) \
    if (need_bc) WMMA_LAUNCH_WT21(BM_V, BN_V, true); \
    else         WMMA_LAUNCH_WT21(BM_V, BN_V, false)

    if      (c.wt_m == 4 && c.bm == 256 && c.bn == 128)
                                              WMMA_TILE_CASE_WT4(256, 128);
    else if (c.wt_n == 1 && c.bm == 128 && c.bn == 32)
                                              WMMA_TILE_CASE_WT21(128, 32);
    else if (c.wt_n == 1 && c.bm ==  64 && c.bn == 32)
                                              WMMA_TILE_CASE_WT21( 64, 32);
    else if (c.bm ==  64 && c.bn ==  64) WMMA_TILE_CASE( 64,  64);
    else if (c.bm ==  64 && c.bn == 128) WMMA_TILE_CASE( 64, 128);
    else if (c.bm == 128 && c.bn ==  64) WMMA_TILE_CASE(128,  64);
    else if (c.bm == 128 && c.bn == 256) WMMA_TILE_CASE(128, 256);
    else                                  WMMA_TILE_CASE(128, 128);

#undef WMMA_TILE_CASE_WT21
#undef WMMA_LAUNCH_WT21
#undef WMMA_TILE_CASE_WT4
#undef WMMA_LAUNCH_WT4
#undef WMMA_TILE_CASE
#undef WMMA_LAUNCH
}

// Cached temp buffer for dequant+GEMM path
static _Float16* g_dq_buf = nullptr;
static size_t    g_dq_buf_elems = 0;

static _Float16* ensureDqBuffer(int N, int K) {
    size_t need = static_cast<size_t>(N) * K;
    if (g_dq_buf && g_dq_buf_elems >= need) return g_dq_buf;
    if (g_dq_buf) hipFree(g_dq_buf);
    hipMalloc(&g_dq_buf, need * sizeof(_Float16));
    g_dq_buf_elems = need;
    return g_dq_buf;
}

/* -----------------------------------------------------------------------
 * bits=2 WMMA launcher: fused dequant + WMMA GEMM for fp16 A, bits=2 B.
 * A is fp16 col-major (lda=M), C is fp16 col-major (ldc=M).
 * Requires K%32==0, M>=16. Uses the same tile configs as bits=4 WMMA.
 * ----------------------------------------------------------------------- */

#define U2_WMMA_LAUNCH(BM_V, BN_V, WM_V, WN_V, WEU, BC_V, HAS_ZP) do { \
    constexpr int THR_ = (BM_V / (WM_V * 16)) * (BN_V / (WN_V * 16)) * 32; \
    dim3 grid_u2(static_cast<unsigned>((N + BN_V - 1) / BN_V), \
                 static_cast<unsigned>((M + BM_V - 1) / BM_V)); \
    dim3 block_u2(THR_); \
    if (HAS_ZP) { \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(MatMulNBitsU2WMMA_ZP<BM_V, BN_V, WM_V, WN_V, WEU, BC_V>), \
            grid_u2, block_u2, 0, stream, \
            M, N, K, A, lda, B_packed, scales_fp32, zeros_u8, gs, ngk, C, ldc, sw); \
    } else { \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(MatMulNBitsU2WMMA_NoZP<BM_V, BN_V, WM_V, WN_V, WEU, BC_V>), \
            grid_u2, block_u2, 0, stream, \
            M, N, K, A, lda, B_packed, scales_fp32, zeros_u8, gs, ngk, C, ldc, sw); \
    } \
} while(0)

static void launchU2WmmaConfig(
    int config_id, hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B_packed,
    const float* scales_fp32,
    const unsigned char* zeros_u8,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp)
{
    auto& c = kWmmaConfigs[config_id];
    bool need_bc = (M % c.bm != 0) || (N % c.bn != 0);
    int  sw      = c.swizzle_n;

    // Dispatch over the same tile configs as bits=4 WMMA.
    // Only fused path (no separate dequant needed for bits=2).
#define U2_TILE_CASE(BM_V, BN_V) \
    if (need_bc) U2_WMMA_LAUNCH(BM_V, BN_V, 2, 2, 8, true,  has_zp); \
    else         U2_WMMA_LAUNCH(BM_V, BN_V, 2, 2, 8, false, has_zp)

#define U2_TILE_CASE_WT4(BM_V, BN_V) \
    if (need_bc) U2_WMMA_LAUNCH(BM_V, BN_V, 4, 2, 6, true,  has_zp); \
    else         U2_WMMA_LAUNCH(BM_V, BN_V, 4, 2, 6, false, has_zp)

#define U2_TILE_CASE_WN1(BM_V, BN_V) \
    if (need_bc) U2_WMMA_LAUNCH(BM_V, BN_V, 2, 1, 8, true,  has_zp); \
    else         U2_WMMA_LAUNCH(BM_V, BN_V, 2, 1, 8, false, has_zp)

    if      (c.bm ==  64 && c.bn ==  64) U2_TILE_CASE( 64,  64);
    else if (c.bm ==  64 && c.bn == 128) U2_TILE_CASE( 64, 128);
    else if (c.bm == 128 && c.bn ==  64) U2_TILE_CASE(128,  64);
    else if (c.bm == 128 && c.bn == 128) U2_TILE_CASE(128, 128);
    else if (c.bm == 128 && c.bn == 256) U2_TILE_CASE(128, 256);
    else if (c.bm == 256 && c.bn == 128 && c.wt_m == 4) U2_TILE_CASE_WT4(256, 128);
    else if (c.bm ==  64 && c.bn ==  32 && c.wt_n == 1) U2_TILE_CASE_WN1( 64,  32);
    else if (c.bm == 128 && c.bn ==  32 && c.wt_n == 1) U2_TILE_CASE_WN1(128,  32);
    else U2_TILE_CASE(128, 128); // fallback

#undef U2_TILE_CASE
#undef U2_TILE_CASE_WT4
#undef U2_TILE_CASE_WN1
}
#undef U2_WMMA_LAUNCH

// Autotune cache for bits=2 WMMA
static std::unordered_map<std::string, int> g_wmma_u2_cache;
static std::mutex g_wmma_u2_mutex;

static int getOrTuneWmmaU2Config(
    hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B_packed,
    const float* scales_fp32,
    const unsigned char* zeros_u8,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp)
{
    // Key: same as bits=4 WMMA autotune but with _u2 suffix
    std::string key = std::to_string(M) + "_" + std::to_string(N) + "_" +
                      std::to_string(K) + "_" + std::to_string(gs) + "_" +
                      (has_zp ? "1" : "0") + "_u2";
    {
        std::lock_guard<std::mutex> lk(g_wmma_u2_mutex);
        auto it = g_wmma_u2_cache.find(key);
        if (it != g_wmma_u2_cache.end()) return it->second;
    }

    // Use only fused configs (indices 0..kNumWmmaConfigs/2-1 roughly)
    constexpr int ITERS = 5;
    int best_id = 12; // default: 128x128 sw=2 fused
    float best_ms = 1e9f;

    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0); hipEventCreate(&ev1);

    for (int cid = 0; cid < kNumWmmaConfigs; cid++) {
        auto& c = kWmmaConfigs[cid];
        if (!c.fused) continue; // bits=2 always uses fused path
        // K must be multiple of BK_T = bm/(wt_m*wt_n); check feasibility
        int BK_T = c.bm / (c.wt_m * c.wt_n);
        if (K % BK_T != 0) continue;
        if (K % 32  != 0) continue;

        launchU2WmmaConfig(cid, stream, M, N, K, A, lda, B_packed,
                           scales_fp32, zeros_u8, gs, ngk, C, ldc, has_zp);
        hipStreamSynchronize(stream);
        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchU2WmmaConfig(cid, stream, M, N, K, A, lda, B_packed,
                               scales_fp32, zeros_u8, gs, ngk, C, ldc, has_zp);
        hipEventRecord(ev1, stream);
        hipStreamSynchronize(stream);
        float ms = 0; hipEventElapsedTime(&ms, ev0, ev1); ms /= ITERS;
        if (ms < best_ms) { best_ms = ms; best_id = cid; }
    }
    hipEventDestroy(ev0); hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] bits=2 WMMA autotune M=%d N=%d K=%d -> "
        "config[%d] bm=%d bn=%d sw=%d %.3fms\n",
        M, N, K, best_id,
        kWmmaConfigs[best_id].bm, kWmmaConfigs[best_id].bn,
        kWmmaConfigs[best_id].swizzle_n, best_ms);

    {
        std::lock_guard<std::mutex> lk(g_wmma_u2_mutex);
        g_wmma_u2_cache[key] = best_id;
    }
    return best_id;
}

/* -----------------------------------------------------------------------
 * GPU 2-D FP16 transpose + scratch buffer helpers
 *
 * transpose2d_fp16_kernel: transposes a [rows x cols] row-major matrix
 * into a [cols x rows] row-major matrix (equivalent to col-major view).
 * Used to convert row-major ONNX tensors to the col-major layout that
 * WMMA / GEMV kernels expect, and to convert results back.
 * ----------------------------------------------------------------------- */

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

static void launchTranspose2DFp16(hipStream_t stream,
                                  const void* src, void* dst,
                                  int rows, int cols)
{
    constexpr int TILE = 32;
    dim3 block(TILE, 8);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    hipLaunchKernelGGL(transpose2d_fp16_kernel, grid, block, 0, stream,
                       static_cast<const _Float16*>(src),
                       static_cast<_Float16*>(dst),
                       rows, cols);
}

static _Float16* g_a_col_buf = nullptr;
static size_t    g_a_col_buf_elems = 0;

static _Float16* ensureAColBuffer(int M, int K) {
    size_t need = static_cast<size_t>(M) * K;
    if (g_a_col_buf && g_a_col_buf_elems >= need) return g_a_col_buf;
    if (g_a_col_buf) hipFree(g_a_col_buf);
    hipMalloc(&g_a_col_buf, need * sizeof(_Float16));
    g_a_col_buf_elems = need;
    return g_a_col_buf;
}

static _Float16* g_out_col_buf = nullptr;
static size_t    g_out_col_buf_elems = 0;

static _Float16* ensureOutColBuffer(int M, int N) {
    size_t need = static_cast<size_t>(M) * N;
    if (g_out_col_buf && g_out_col_buf_elems >= need) return g_out_col_buf;
    if (g_out_col_buf) hipFree(g_out_col_buf);
    hipMalloc(&g_out_col_buf, need * sizeof(_Float16));
    g_out_col_buf_elems = need;
    return g_out_col_buf;
}

// ---- fp32 <-> fp16 conversion scratch buffers for the fp32 activation path ----
// When element_size_bytes==4, we cast A fp32->fp16 on the GPU before calling
// the autotuned fp16 GEMV/WMMA kernels, then cast output fp16->fp32.
static _Float16* g_a_fp32cast_buf = nullptr;
static size_t    g_a_fp32cast_elems = 0;
static _Float16* g_out_fp32cast_buf = nullptr;
static size_t    g_out_fp32cast_elems = 0;

static _Float16* ensureAFp32CastBuf(int64_t n) {
    size_t need = static_cast<size_t>(n);
    if (g_a_fp32cast_buf && g_a_fp32cast_elems >= need) return g_a_fp32cast_buf;
    if (g_a_fp32cast_buf) hipFree(g_a_fp32cast_buf);
    hipMalloc(&g_a_fp32cast_buf, need * sizeof(_Float16));
    g_a_fp32cast_elems = need;
    return g_a_fp32cast_buf;
}
static _Float16* ensureOutFp32CastBuf(int64_t n) {
    size_t need = static_cast<size_t>(n);
    if (g_out_fp32cast_buf && g_out_fp32cast_elems >= need) return g_out_fp32cast_buf;
    if (g_out_fp32cast_buf) hipFree(g_out_fp32cast_buf);
    hipMalloc(&g_out_fp32cast_buf, need * sizeof(_Float16));
    g_out_fp32cast_elems = need;
    return g_out_fp32cast_buf;
}

// cast_f32_to_f16_kernel and cast_f16_to_f32_kernel are defined in cast_kernel.hip
extern __global__ void cast_f32_to_f16_kernel(const float* __restrict__ in,
                                               __half* __restrict__ out, int64_t n);
extern __global__ void cast_f16_to_f32_kernel(const __half* __restrict__ in,
                                               float* __restrict__ out, int64_t n);

static void gpuCastF32ToF16(hipStream_t s, const void* src, _Float16* dst, int64_t n) {
    dim3 blk(256);
    dim3 grd(static_cast<unsigned>((n + 255) / 256));
    (void)hipGetLastError();
    hipLaunchKernelGGL(cast_f32_to_f16_kernel, grd, blk, 0, s,
                       static_cast<const float*>(src),
                       reinterpret_cast<__half*>(dst), n);
}
static void gpuCastF16ToF32(hipStream_t s, const _Float16* src, void* dst, int64_t n) {
    dim3 blk(256);
    dim3 grd(static_cast<unsigned>((n + 255) / 256));
    (void)hipGetLastError();
    hipLaunchKernelGGL(cast_f16_to_f32_kernel, grd, blk, 0, s,
                       reinterpret_cast<const __half*>(src),
                       static_cast<float*>(dst), n);
}

static int tuneWmmaConfig(
    hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const _Float16* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp)
{
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);

    _Float16* dq_buf = ensureDqBuffer(N, K);

    float best_ms = 1e30f;
    int   best_id = 9;
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    for (int cid = 0; cid < kNumWmmaConfigs; cid++) {
        auto& cfg = kWmmaConfigs[cid];
        if (cfg.bn < 64 && N > 2 * cfg.bn) continue;

        for (int w = 0; w < WARMUP; w++)
            launchWmmaConfig(cid, stream, M, N, K, A, lda, B,
                             scales, zeros, gs, ngk, C, ldc, has_zp, dq_buf);

        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchWmmaConfig(cid, stream, M, N, K, A, lda, B,
                             scales, zeros, gs, ngk, C, ldc, has_zp, dq_buf);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   wmma config[%2d] %s bm=%3d bn=%3d sw=%d : %.4f ms\n",
            cid, kWmmaConfigs[cid].fused ? "fused " : "dq+gem",
            kWmmaConfigs[cid].bm, kWmmaConfigs[cid].bn,
            kWmmaConfigs[cid].swizzle_n, ms / ITERS);

        if (ms < best_ms) {
            best_ms = ms;
            best_id = cid;
        }
    }

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] WMMA autotune: M=%d N=%d K=%d gs=%d zp=%s -> "
        "best config[%d] %s bm=%d bn=%d sw=%d (%.4f ms/iter)\n",
        M, N, K, gs, has_zp ? "yes" : "no",
        best_id, kWmmaConfigs[best_id].fused ? "fused" : "dq+gemm",
        kWmmaConfigs[best_id].bm, kWmmaConfigs[best_id].bn,
        kWmmaConfigs[best_id].swizzle_n, best_ms / ITERS);

    return best_id;
}

static std::unordered_map<std::string, int> wmma_cache;
static std::mutex wmma_tune_mutex;
static std::once_flag wmma_cache_init_flag;
// Rebuild invalidates cache: __DATE__/__TIME__ change on every recompilation.
static constexpr const char kWmmaBuildTimestamp[] = __DATE__ " " __TIME__;

static std::string wmmaCacheKey(int M, int N, int K, int gs, bool has_zp) {
    return std::to_string(M) + "_" + std::to_string(N) + "_" +
           std::to_string(K) + "_" + std::to_string(gs) + "_" +
           (has_zp ? "1" : "0");
}

// Disk-persistent WMMA autotune cache: avoids re-tuning ~1500 kernel launches
// on every process start. File is per-GPU-arch in %TEMP%.
static const std::string& getWmmaCachePath() {
    static std::string path = [] {
        std::string tmp_dir;
#ifdef _WIN32
        char* v = nullptr;
        size_t len = 0;
        if (_dupenv_s(&v, &len, "TEMP") == 0 && v != nullptr) {
            tmp_dir = v;
            free(v);
        }
#else
        const char* v = getenv("TEMP");
        if (!v) v = getenv("TMPDIR");
        if (!v) v = "/tmp";
        tmp_dir = v;
#endif
        if (tmp_dir.empty()) return std::string();

        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, 0) != hipSuccess || prop.gcnArchName[0] == '\0')
            return std::string();

        return tmp_dir + "/morphizen_wmma_cache_" + prop.gcnArchName + ".txt";
    }();
    return path;
}

static void loadWmmaCache() {
    const std::string& path = getWmmaCachePath();
    if (path.empty()) return;

    FILE* f = fopen(path.c_str(), "r");
    if (!f) return;

    // First line is the build timestamp — discard cache if binary was rebuilt.
    char ts_line[64];
    if (!fgets(ts_line, sizeof(ts_line), f)) { fclose(f); return; }
    size_t ts_len = strlen(ts_line);
    if (ts_len > 0 && ts_line[ts_len - 1] == '\n') ts_line[ts_len - 1] = '\0';
    if (strcmp(ts_line, kWmmaBuildTimestamp) != 0) {
        fclose(f);
        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels] WMMA cache: build mismatch (file=\"%s\", "
            "binary=\"%s\"), will re-tune\n", ts_line, kWmmaBuildTimestamp);
        return;
    }

    char key[128];
    int config_id;
    int loaded = 0;
    while (fscanf(f, "%127s %d", key, &config_id) == 2) {
        if (config_id >= 0 && config_id < kNumWmmaConfigs) {
            wmma_cache[key] = config_id;
            loaded++;
        }
    }
    fclose(f);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] WMMA cache: loaded %d entries from %s\n",
        loaded, path.c_str());
}

static void saveWmmaCache() {
    const std::string& path = getWmmaCachePath();
    if (path.empty()) return;

    FILE* f = fopen(path.c_str(), "w");
    if (!f) {
        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels] WMMA cache: failed to write %s\n", path.c_str());
        return;
    }

    fprintf(f, "%s\n", kWmmaBuildTimestamp);
    for (const auto& entry : wmma_cache)
        fprintf(f, "%s %d\n", entry.first.c_str(), entry.second);
    fclose(f);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] WMMA cache: saved %zu entries to %s\n",
        wmma_cache.size(), path.c_str());
}

static int getOrTuneWmmaConfig(
    hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const _Float16* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp)
{
    std::call_once(wmma_cache_init_flag, loadWmmaCache);

    std::string key = wmmaCacheKey(M, N, K, gs, has_zp);

    std::lock_guard<std::mutex> lock(wmma_tune_mutex);

    auto it = wmma_cache.find(key);
    if (it != wmma_cache.end())
        return it->second;

    int best = tuneWmmaConfig(stream, M, N, K, A, lda, B,
                               scales, zeros, gs, ngk, C, ldc, has_zp);
    wmma_cache[key] = best;
    saveWmmaCache();

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] WMMA cache: stored key=\"%s\" -> config[%d] "
        "(cache size=%zu)\n",
        key.c_str(), best, wmma_cache.size());

    return best;
}

/* =======================================================================
 * Section 2c: WMMA kernel for int8 weights (fused dequant)
 *
 * Mirrors GemmFp16U4Impl above, but specialized for the bits=8 layout:
 *   B      : uint8 [N, K]                       (one byte per weight)
 *   scales : fp16  [N, num_groups_k]
 *   zeros  : uint8 [N, num_groups_k]            (or NULL, default zp=128)
 *
 * The shared-memory tile pipeline, double buffering, register prefetch,
 * grid swizzle, and writeback are all identical to the int4 path. Two
 * differences:
 *
 *   1. B byte indexing: per-thread base is `b_n_g * K + b_k8` (each weight
 *      is a full byte, no nibble packing), and the K-step advance is
 *      `t` bytes (not `t/2`).
 *
 *   2. Per-thread B load is 8 bytes (uint2 = 64-bit) for 8 elements,
 *      versus 4 bytes (uint32_t) for 8 nibbles in int4. The
 *      `THR * 8 == BN * BK` invariant -- 8 elements per thread per
 *      K-step -- is preserved, so the tile/warp shape math is unchanged.
 *
 *   3. ZP is read as uint8 directly (not pre-converted to fp16) since the
 *      nibble unpack of int4 zeros isn't needed. Zero is applied via the
 *      same `neg_zs = -zp * scale` precomputation, just with a uint8
 *      source.
 *
 * Default zp follows the ONNX MatMulNBits spec: 2^(bits-1) = 128 for
 * bits=8, baked into the USE_ZEROS=false branch as a constexpr.
 * ======================================================================= */

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
#pragma unroll
            for (int wn = 0; wn < WT_N; wn++) {
                int noff              = (wcol * WT_N + wn) * WMMA_TILE;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&b_frag[wn]);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smB[buf][noff + lane][ks * WMMA_TILE]);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    dst[i] = src[i];
            }
#pragma unroll
            for (int wm = 0; wm < WT_M; wm++) {
                int moff              = (wrow * WT_M + wm) * WMMA_TILE;
                half16 a_frag;
                uint32_t* dst         = reinterpret_cast<uint32_t*>(&a_frag);
                const uint32_t* src   = reinterpret_cast<const uint32_t*>(
                    &smA[buf][moff + lane][ks * WMMA_TILE]);
#pragma unroll
                for (int i = 0; i < 8; i++)
                    dst[i] = src[i];

#pragma unroll
                for (int wn = 0; wn < WT_N; wn++)
                    acc[wm][wn] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
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
                int r = e * 2 + sub;
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

/* int8 WMMA entry kernels: fused dequant only (no separate-DQ variant in v1) */

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

/* -----------------------------------------------------------------------
 * WMMA-i8 Autotune: subset of the int4 fused tile shapes
 *
 * v1 ships a fused-only path -- no separate-dequant int8 variant -- so
 * the table is the fused subset of kWmmaConfigs. The WT_M=4 (256x128) and
 * WT_N=1 (BN=32) special tile shapes are omitted in v1; the basic
 * 64x64..128x256 sweep covers Qwen3.5-MoE int8 shapes (N in {32, 2048,
 * 4096, 8192}, K=2048) without them.
 * ----------------------------------------------------------------------- */

struct WmmaConfigI8 {
    int bm, bn;
    int swizzle_n;
};

static constexpr WmmaConfigI8 kWmmaI8Configs[] = {
    { 64,  64,  2}, { 64,  64,  4}, { 64,  64,  8}, { 64,  64, 16},
    { 64, 128,  2}, { 64, 128,  4}, { 64, 128,  8}, { 64, 128, 16},
    {128,  64,  2}, {128,  64,  4}, {128,  64,  8}, {128,  64, 16},
    {128, 128,  2}, {128, 128,  4}, {128, 128,  8}, {128, 128, 16},
    {128, 128, 32}, {128, 128, 64},
    {128, 256,  4}, {128, 256,  8}, {128, 256, 16}, {128, 256, 32},
};
static constexpr int kNumWmmaI8Configs =
    static_cast<int>(sizeof(kWmmaI8Configs) / sizeof(kWmmaI8Configs[0]));

static void launchWmmaConfigI8(
    int config_id, hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const unsigned char* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp)
{
    auto& c = kWmmaI8Configs[config_id];
    bool need_bc = (M % c.bm != 0) || (N % c.bn != 0);

#define WMMA_I8_LAUNCH(BM_V, BN_V, BC_V) do {                                 \
    constexpr int WM_  = BM_V / (2 * 16);                                     \
    constexpr int WN_  = BN_V / (2 * 16);                                     \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    if (has_zp) {                                                             \
        hipLaunchKernelGGL(                                                   \
            HIP_KERNEL_NAME(                                                  \
                MatMulNBitsWMMA_I8_ZP<BM_V, BN_V, 2, 2, 8, BC_V>),            \
            grid_, block_, 0, stream,                                         \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    } else {                                                                  \
        hipLaunchKernelGGL(                                                   \
            HIP_KERNEL_NAME(                                                  \
                MatMulNBitsWMMA_I8_NoZP<BM_V, BN_V, 2, 2, 8, BC_V>),          \
            grid_, block_, 0, stream,                                         \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    }                                                                         \
} while (0)

#define WMMA_I8_TILE_CASE(BM_V, BN_V) \
    if (need_bc) WMMA_I8_LAUNCH(BM_V, BN_V, true);  \
    else         WMMA_I8_LAUNCH(BM_V, BN_V, false)

         if (c.bm ==  64 && c.bn ==  64) WMMA_I8_TILE_CASE( 64,  64);
    else if (c.bm ==  64 && c.bn == 128) WMMA_I8_TILE_CASE( 64, 128);
    else if (c.bm == 128 && c.bn ==  64) WMMA_I8_TILE_CASE(128,  64);
    else if (c.bm == 128 && c.bn == 256) WMMA_I8_TILE_CASE(128, 256);
    else                                  WMMA_I8_TILE_CASE(128, 128);

#undef WMMA_I8_TILE_CASE
#undef WMMA_I8_LAUNCH
}

static int tuneWmmaConfigI8(
    hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const unsigned char* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp)
{
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);

    float best_ms = 1e30f;
    int   best_id = 12; // 128x128 sw=2: a safe default for typical N>=64
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    for (int cid = 0; cid < kNumWmmaI8Configs; cid++) {
        auto& cfg = kWmmaI8Configs[cid];
        // Skip large BN tiles when N is too small to benefit.
        if (cfg.bn > N * 2) continue;

        for (int w = 0; w < WARMUP; w++)
            launchWmmaConfigI8(cid, stream, M, N, K, A, lda, B,
                               scales, zeros, gs, ngk, C, ldc, has_zp);

        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchWmmaConfigI8(cid, stream, M, N, K, A, lda, B,
                               scales, zeros, gs, ngk, C, ldc, has_zp);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   wmma-i8 config[%2d] bm=%3d bn=%3d sw=%d : %.4f ms\n",
            cid, cfg.bm, cfg.bn, cfg.swizzle_n, ms / ITERS);

        if (ms < best_ms) {
            best_ms = ms;
            best_id = cid;
        }
    }

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] WMMA-i8 autotune: M=%d N=%d K=%d gs=%d zp=%s -> "
        "best config[%d] bm=%d bn=%d sw=%d (%.4f ms/iter)\n",
        M, N, K, gs, has_zp ? "yes" : "no",
        best_id, kWmmaI8Configs[best_id].bm, kWmmaI8Configs[best_id].bn,
        kWmmaI8Configs[best_id].swizzle_n, best_ms / ITERS);

    return best_id;
}

static std::unordered_map<std::string, int> wmma_i8_cache;
static std::mutex wmma_i8_tune_mutex;

static std::string wmmaI8CacheKey(int M, int N, int K, int gs, bool has_zp) {
    return std::to_string(M) + "_" + std::to_string(N) + "_" +
           std::to_string(K) + "_" + std::to_string(gs) + "_" +
           (has_zp ? "1" : "0");
}

static int getOrTuneWmmaConfigI8(
    hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const unsigned char* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp)
{
    std::string key = wmmaI8CacheKey(M, N, K, gs, has_zp);

    std::lock_guard<std::mutex> lock(wmma_i8_tune_mutex);

    auto it = wmma_i8_cache.find(key);
    if (it != wmma_i8_cache.end())
        return it->second;

    int best = tuneWmmaConfigI8(stream, M, N, K, A, lda, B,
                                 scales, zeros, gs, ngk, C, ldc, has_zp);
    wmma_i8_cache[key] = best;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] WMMA-i8 cache: stored key=\"%s\" -> config[%d] "
        "(cache size=%zu)\n",
        key.c_str(), best, wmma_i8_cache.size());

    return best;
}

/* =======================================================================
 * Section 3: Post-WMMA bias addition (row-major output)
 *
 * C is row-major [M x N] with ldc = N.
 * bias is [N]. Adds bias[n] to every element in column n.
 * Threads sweep N (coalesced along row), blocks sweep M.
 * ======================================================================= */

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

/* =======================================================================
 * Section 4: Host-side launcher (extern "C")
 * ======================================================================= */

/* =======================================================================
 * Section 4a: uint8 packed nibble zero_points -> FP16 conversion
 *
 * ONNX MatMulNBits uint8 packed format:  [N, ceil(groups_k/2)]
 *   Each byte holds two 4-bit zero_points:  low nibble = zp[2*i],
 *                                           high nibble = zp[2*i+1]
 * Output FP16 format: [N, groups_k]
 * ======================================================================= */

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

// Stand-alone launchers (declared in hip_custom_kernels.h). Used by the
// pointer-keyed cache in lib/Runtime/real/matmul_nbits.cpp so that the
// unpack/convert kernels run at most once per (zero_points pointer) over
// the lifetime of a RuntimeState — instead of once per matmul_nbits call,
// which on 8B asym decode is ~225 redundant launches per Compute().
extern "C" void hip_matmul_nbits_unpack_zp_u8(
    void* stream, const void* zp_packed, void* dst_u8, int N, int groups_k)
{
    int packed_cols = (groups_k + 1) / 2;
    constexpr int THR = 256;
    dim3 grid(static_cast<unsigned>((packed_cols + THR - 1) / THR),
              static_cast<unsigned>(N));
    dim3 block(THR);
    hipLaunchKernelGGL(unpack_zp_nibbles_to_u8_kernel,
                       grid, block, 0, static_cast<hipStream_t>(stream),
                       static_cast<const uint8_t*>(zp_packed),
                       static_cast<uint8_t*>(dst_u8),
                       N, groups_k, packed_cols);
}

extern "C" void hip_matmul_nbits_convert_zp_fp16(
    void* stream, const void* zp_packed, void* dst_fp16, int N, int groups_k)
{
    int packed_cols = (groups_k + 1) / 2;
    constexpr int THR = 256;
    dim3 grid(static_cast<unsigned>((packed_cols + THR - 1) / THR),
              static_cast<unsigned>(N));
    dim3 block(THR);
    hipLaunchKernelGGL(unpack_zp_u8_to_fp16_kernel,
                       grid, block, 0, static_cast<hipStream_t>(stream),
                       static_cast<const unsigned char*>(zp_packed),
                       static_cast<_Float16*>(dst_fp16),
                       N, groups_k, packed_cols);
}

extern "C" int hip_matmul_nbits(
    void* stream,
    const void* A,
    const void* B,
    const void* scales,
    const void* zero_points,
    const void* bias,
    void* output,
    int64_t M, int64_t N, int64_t K,
    int64_t batch_count,
    int64_t bits,
    int64_t block_size,
    int64_t element_size_bytes,
    int64_t zp_elem_size,
    const void* pre_unpacked_zp_u8,
    const void* pre_unpacked_zp_fp16) {
  if (bits != 2 && bits != 4 && bits != 8) {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits: only bits=2, bits=4 or bits=8 "
            "supported, got %lld\n",
            (long long)bits);
    return -1;
  }

  /* -------------------------------------------------------------------
   * bits=2 path: ONNX MatMulNBits with 2-bit weights uses a packed
   * [N, K/4] uint8 layout (4 crumbs per byte, little-endian).
   * Default zp = 2 (= 2^(bits-1)). No WMMA or ZP conversion needed.
   * ------------------------------------------------------------------- */
  if (bits == 2) {
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);

    // Zero points for bits=2 are packed 4-per-byte (matching weight packing).
    // The GemmFp16U2Impl kernel handles the packed ZP correctly.
    const uint8_t* zp_for_i2_u8 = static_cast<const uint8_t*>(zero_points);

    // M=1 WMMA decode: disabled — for M=1, WMMA has too much smem-fill
    // overhead relative to the GEMV kernel. GEMV is better for M=1.
    const bool wmma_decode_eligible = false;

    const bool wmma_eligible =
        (batch_count == 1) && (K % 32 == 0) && (M >= kBlockDim) &&
        (element_size_bytes == 2);  // fp16 activations only for large-M WMMA

    if (wmma_decode_eligible) {
      // M=1 WMMA decode: pad to 16×K in smem, run WMMA, output row 0.
      // This achieves matrix-core throughput even for single-token decode.
      int gs   = static_cast<int>(block_size);
      int ngk  = static_cast<int>((K + block_size - 1) / block_size);
      int iN   = static_cast<int>(N);
      int iK   = static_cast<int>(K);

      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: bits=2 WMMA-decode M=1 N=%d K=%d "
          "bs=%d elem=%lld has_zp=%d\n",
          iN, iK, gs, (long long)element_size_bytes,
          (int)(zero_points != nullptr));

      (void)hipGetLastError();
      if (element_size_bytes == 4) {
        launch_u2_wmma_decode<64, float>(hip_stream,
            static_cast<const float*>(A),
            static_cast<const uint8_t*>(B),
            static_cast<const float*>(scales),
            zp_for_i2_u8,
            static_cast<float*>(output),
            iN, iK, gs, ngk);
      } else {
        launch_u2_wmma_decode<64, __half>(hip_stream,
            static_cast<const __half*>(A),
            static_cast<const uint8_t*>(B),
            static_cast<const __half*>(scales),
            zp_for_i2_u8,
            static_cast<__half*>(output),
            iN, iK, gs, ngk);
      }

    } else if (wmma_eligible) {
      // Large-M WMMA path: fused dequant + matrix-core GEMM.
      // Scales are fp32 for ORCA bits=2; ZP is packed uint8.
      int iM  = static_cast<int>(M);
      int iN  = static_cast<int>(N);
      int iK  = static_cast<int>(K);
      int gs  = static_cast<int>(block_size);
      int ngk = static_cast<int>((K + block_size - 1) / block_size);
      int lda = iK;  // row-major A
      int ldc = iN;  // row-major C

      const auto* a_ptr = static_cast<const _Float16*>(A);
      const auto* b_ptr = static_cast<const unsigned char*>(B);
      const auto* s_fp32 = static_cast<const float*>(scales);
      auto* out_ptr = static_cast<_Float16*>(output);
      bool has_zp = (zp_for_i2_u8 != nullptr);

      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: bits=2 WMMA M=%d N=%d K=%d "
          "bs=%d has_zp=%d\n", iM, iN, iK, gs, (int)has_zp);

      (void)hipGetLastError();
      int cfg = getOrTuneWmmaU2Config(
          hip_stream, iM, iN, iK, a_ptr, lda, b_ptr,
          s_fp32, zp_for_i2_u8, gs, ngk, out_ptr, ldc, has_zp);
      launchU2WmmaConfig(cfg, hip_stream, iM, iN, iK, a_ptr, lda, b_ptr,
                         s_fp32, zp_for_i2_u8, gs, ngk, out_ptr, ldc, has_zp);

    } else if (M <= kBlockDim) {
      // GEMV path for small M (decode).
      // Use fp32 path for ORCA (elem_size=4), fp16 otherwise.
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: bits=2 GEMV M=%lld N=%lld "
          "K=%lld bs=%lld elem=%lld\n", (long long)M, (long long)N,
          (long long)K, (long long)block_size, (long long)element_size_bytes);

      // ZP is the model's packed 2-bit zero_points (4-per-byte); the GEMV
      // kernel unpacks it per block (read_zp2). nullptr => symmetric default 2.
      if (element_size_bytes == 4) {
        // fp32 activations + fp32 scales → fp32 output (ORCA W2A8).
        // For M=1 decode: use the LDS-cooperative kernel that loads A into
        // shared memory once per block (eliminates per-thread LDS spill of
        // a_vals[64] from the TILE_N kernel).
        launch_gemv_i2<float>(hip_stream,
                     static_cast<const float*>(A),
                     static_cast<const uint8_t*>(B),
                     static_cast<const float*>(scales),
                     zp_for_i2_u8,
                     static_cast<const float*>(bias),
                     static_cast<float*>(output),
                     M, N, K, batch_count, block_size);
      } else {
        // fp16 path
        launch_gemv_i2<__half>(hip_stream,
                     static_cast<const __half*>(A),
                     static_cast<const uint8_t*>(B),
                     static_cast<const __half*>(scales),
                     zp_for_i2_u8,
                     static_cast<const __half*>(bias),
                     static_cast<__half*>(output),
                     M, N, K, batch_count, block_size);
      }
    } else {
      // Larger-M fallback (e.g. prefill). Route through the correct 2-bit GEMV
      // kernel (handles any M and the packed-2-bit ZP via read_zp2). The old
      // naive fp32 launcher used the 4-bit kernel (matmul_nbits_kernel_f32),
      // which is wrong for bits=2; and both branches dropped ZP (nullptr).
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: bits=2 fallback->gemv M=%lld "
          "N=%lld K=%lld bs=%lld\n", (long long)M, (long long)N, (long long)K,
          (long long)block_size);
      (void)hipGetLastError();
      if (element_size_bytes == 4) {
        launch_gemv_i2<float>(hip_stream,
            static_cast<const float*>(A), static_cast<const uint8_t*>(B),
            static_cast<const float*>(scales), zp_for_i2_u8,
            static_cast<const float*>(bias), static_cast<float*>(output),
            M, N, K, batch_count, block_size);
      } else {
        launch_gemv_i2<__half>(hip_stream,
            static_cast<const __half*>(A), static_cast<const uint8_t*>(B),
            static_cast<const __half*>(scales), zp_for_i2_u8,
            static_cast<const __half*>(bias), static_cast<__half*>(output),
            M, N, K, batch_count, block_size);
      }
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits bits=2 launch failed: %s\n",
              hipGetErrorString(err));
    }
    return static_cast<int>(err);
  }

  if (M <= 0 || N <= 0 || K <= 0 || batch_count <= 0) return 0;

  if (element_size_bytes != 2 && element_size_bytes != 4) {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits: only fp16/fp32 supported, "
            "got element_size %lld\n",
            (long long)element_size_bytes);
    return -1;
  }

  hipStream_t hip_stream = static_cast<hipStream_t>(stream);

  /* -------------------------------------------------------------------
   * bits=8 path: ONNX MatMulNBits with int8 weights uses an unpacked
   * [N, K] uint8 layout (no nibble packing) and default zp=128. None of
   * the bit-twiddling int4 fast paths apply, so dispatch directly to
   * the dedicated int8 naive kernel. Models that mix bits=4 and bits=8
   * in the same graph (e.g. Qwen3.5-MoE int8 lm_head with int4 expert
   * weights) hit this branch only for the int8 ops; the int4 ops
   * continue through the WMMA / GEMV fast paths below.
   * ------------------------------------------------------------------- */
  if (bits == 8) {
    if (zero_points && zp_elem_size != 1) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits bits=8: zero_points must "
              "be uint8 (zp_elem_size=1), got zp_elem_size=%lld\n",
              (long long)zp_elem_size);
      return -1;
    }

    // Three int8 paths, in priority order:
    //
    //   1. WMMA-i8 fast path (M >= 16, batch=1, K%32==0, group_size % 32 == 0
    //      and >= 32): autotuned 16x16x16 fp16 WMMA tiles with fused
    //      dequant. Used for prefill-shaped GEMMs (Qwen3.5-MoE int8
    //      lm_head, expert up/gate projections at seq_len > 1).
    //   2. GEMV-i8 path (block_size >= 16): autotuned BLOCK_SIZE x TILE_N
    //      reduction, vectorized B loads. Used for decode (M=1) and small-M
    //      shapes that don't meet WMMA's M>=16 threshold.
    //   3. Naive fallback: thread-per-element kernel for unusual shapes
    //      that fail both gates above.
    //
    // The WMMA path's group_size gate enforces group_size >= BK_T and
    // group_size % BK_T == 0 for all configs in kWmmaI8Configs (max BK_T =
    // 32, min = 16), so groups always align with K-step boundaries inside
    // the kernel. Common quantization (gs in {32, 64, 128}) satisfies it.
    constexpr int kI8GemvMinBs = 16;
    const bool wmma_data_format = (batch_count == 1) && (K % 32 == 0);
    const bool use_wmma_i8 =
        wmma_data_format && (M >= kBlockDim) &&
        (block_size >= 32) && (block_size % 32 == 0);
    const bool use_gemv_i8 = !use_wmma_i8 && (block_size >= kI8GemvMinBs);

    if (use_wmma_i8) {
      int iM  = static_cast<int>(M);
      int iN  = static_cast<int>(N);
      int iK  = static_cast<int>(K);
      int gs  = static_cast<int>(block_size);
      int ngk = static_cast<int>((K + block_size - 1) / block_size);
      int lda = iK;
      int ldc = iN;

      const auto* a_ptr  = static_cast<const _Float16*>(A);
      const auto* b_ptr  = static_cast<const unsigned char*>(B);
      const auto* s_ptr  = static_cast<const _Float16*>(scales);
      const auto* z_ptr  = static_cast<const unsigned char*>(zero_points);
      auto*       o_ptr  = static_cast<_Float16*>(output);

      bool has_zp = (z_ptr != nullptr);
      int cfg_id = getOrTuneWmmaConfigI8(
          hip_stream, iM, iN, iK, a_ptr, lda, b_ptr,
          s_ptr, z_ptr, gs, ngk, o_ptr, ldc, has_zp);

      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: int8 WMMA M=%d N=%d K=%d "
          "bs=%d zp=%s bias=%s cfg=%d\n",
          iM, iN, iK, gs, has_zp ? "yes" : "null",
          bias ? "yes" : "null", cfg_id);

      launchWmmaConfigI8(cfg_id, hip_stream, iM, iN, iK, a_ptr, lda, b_ptr,
                         s_ptr, z_ptr, gs, ngk, o_ptr, ldc, has_zp);

      // Bias is added as a separate row-major pass (same kernel as int4 WMMA).
      if (bias) {
        dim3 bias_grid(static_cast<unsigned>((iN + 255) / 256),
                       static_cast<unsigned>(iM));
        dim3 bias_block(256);
        hipLaunchKernelGGL(
            matmul_nbits_add_bias_rowmajor, bias_grid, bias_block,
            0, hip_stream,
            o_ptr, static_cast<const _Float16*>(bias), iM, iN, ldc);
      }
    } else if (use_gemv_i8) {
      const auto* a_ptr  = static_cast<const __half*>(A);
      const auto* b_ptr  = static_cast<const uint8_t*>(B);
      const auto* sc_ptr = static_cast<const __half*>(scales);
      const auto* zp_ptr = static_cast<const uint8_t*>(zero_points);
      const auto* bi_ptr = static_cast<const __half*>(bias);
      auto*       o_ptr  = static_cast<__half*>(output);

      int cfg_id = getOrTuneGemvConfigI8(
          hip_stream, a_ptr, b_ptr, sc_ptr, zp_ptr, bi_ptr, o_ptr,
          M, N, K, batch_count, block_size);

      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: int8 GEMV M=%lld N=%lld "
          "K=%lld batch=%lld bs=%lld zp=%s bias=%s cfg=%d\n",
          (long long)M, (long long)N, (long long)K, (long long)batch_count,
          (long long)block_size, zero_points ? "yes" : "null",
          bias ? "yes" : "null", cfg_id);

      launchGemvConfigI8(cfg_id, hip_stream, a_ptr, b_ptr, sc_ptr, zp_ptr,
                         bi_ptr, o_ptr, M, N, K, batch_count, block_size);
    } else {
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: int8 naive M=%lld N=%lld K=%lld "
          "batch=%lld bs=%lld zp=%s bias=%s\n",
          (long long)M, (long long)N, (long long)K, (long long)batch_count,
          (long long)block_size, zero_points ? "yes" : "null",
          bias ? "yes" : "null");

      dim3 blk(kBlockDim, kBlockDim);
      dim3 grd(static_cast<unsigned>((N + kBlockDim - 1) / kBlockDim),
               static_cast<unsigned>((M + kBlockDim - 1) / kBlockDim),
               static_cast<unsigned>(batch_count));

      hipLaunchKernelGGL(
          matmul_nbits_kernel_i8, grd, blk, 0, hip_stream,
          static_cast<const __half*>(A), static_cast<const uint8_t*>(B),
          static_cast<const __half*>(scales),
          static_cast<const uint8_t*>(zero_points),
          static_cast<const __half*>(bias), static_cast<__half*>(output),
          M, N, K, block_size);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits int8 launch failed: %s\n",
              hipGetErrorString(err));
    }
    return static_cast<int>(err);
  }

  // Packed nibble ZPs [N, ceil(k_blocks/2)] must be unpacked before use:
  //   - FP16 for WMMA and col-major GEMV paths (only when M > 1)
  //   - Individual uint8 for row-major GEMV and naive fallback paths
  // When zp_elem_size == 2 (FP16) or ZPs are NULL, no conversion needed.
  // The runtime wrapper (lib/Runtime/real/matmul_nbits.cpp) caches the
  // unpacked buffers per zero_points pointer and passes them via
  // pre_unpacked_zp_{u8,fp16} so we skip the kernel launches here.
  const void* zp_for_fp16_paths = zero_points;
  const void* zp_for_u8_paths = zero_points;
  if (zero_points && zp_elem_size == 1) {
    if (!pre_unpacked_zp_u8) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits: pre_unpacked_zp_u8 is "
              "required for asym (zp_elem_size==1) path\n");
      return -1;
    }
    zp_for_u8_paths = pre_unpacked_zp_u8;
    // pre_unpacked_zp_fp16 may legitimately be null (the wrapper only fills
    // it for the WMMA / col-major-GEMV-with-M>1 paths). The dispatch below
    // only consumes zp_for_fp16_paths in those branches.
    if (pre_unpacked_zp_fp16)
      zp_for_fp16_paths = pre_unpacked_zp_fp16;
  }

  /* -------------------------------------------------------------------
   * WMMA data format: batch==1 && K%32==0
   *
   * Caller passes row-major A [M, K] and receives row-major C [M, N].
   * WMMA kernel loads A row-major directly (K-first thread mapping for
   * coalesced reads) and writes C row-major (no transposes needed).
   * zero_points (if not NULL) must be FP16 [N, num_groups_k].
   *
   * When M >= kBlockDim (16): WMMA intrinsics (RDNA3 fast path)
   * When M <  kBlockDim:      GEMV kernel with COL_MAJOR=true
   * ------------------------------------------------------------------- */
  bool wmma_data_format = (batch_count == 1) && (K % 32 == 0);

  if (wmma_data_format && M >= kBlockDim) {
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_matmul_nbits: WMMA fast path M=%lld N=%lld "
        "K=%lld bs=%lld zp=%s bias=%s\n",
        (long long)M, (long long)N, (long long)K, (long long)block_size,
        zero_points ? "yes" : "null", bias ? "yes" : "null");

    int iM  = static_cast<int>(M);
    int iN  = static_cast<int>(N);
    int iK  = static_cast<int>(K);
    int gs  = static_cast<int>(block_size);
    int ngk = static_cast<int>((K + block_size - 1) / block_size);
    int lda = iK;
    int ldc = iN;

    const auto* a_ptr = static_cast<const _Float16*>(A);
    const auto* b_ptr = static_cast<const unsigned char*>(B);
    const auto* s_ptr = static_cast<const _Float16*>(scales);
    const auto* z_ptr = static_cast<const _Float16*>(zp_for_fp16_paths);

    // WMMA writes row-major C directly to caller's output buffer (ldc = N)
    auto* out_ptr = static_cast<_Float16*>(output);

    bool has_zp = (z_ptr != nullptr);
    int wmma_cfg = getOrTuneWmmaConfig(
        hip_stream, iM, iN, iK, a_ptr, lda, b_ptr,
        s_ptr, z_ptr, gs, ngk, out_ptr, ldc, has_zp);

    _Float16* dq_buf = kWmmaConfigs[wmma_cfg].fused
                           ? nullptr : ensureDqBuffer(iN, iK);
    launchWmmaConfig(wmma_cfg, hip_stream,
                     iM, iN, iK, a_ptr, lda, b_ptr,
                     s_ptr, z_ptr, gs, ngk, out_ptr, ldc, has_zp, dq_buf);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits WMMA launch failed: %s\n",
              hipGetErrorString(err));
      return static_cast<int>(err);
    }

    if (bias) {
      dim3 bias_grid(static_cast<unsigned>((iN + 255) / 256),
                     static_cast<unsigned>(iM));
      dim3 bias_block(256);
      hipLaunchKernelGGL(
          matmul_nbits_add_bias_rowmajor, bias_grid, bias_block,
          0, hip_stream,
          out_ptr, static_cast<const _Float16*>(bias), iM, iN, ldc);

      err = hipGetLastError();
      if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_matmul_nbits bias launch failed: %s\n",
                hipGetErrorString(err));
        return static_cast<int>(err);
      }
    }

    return static_cast<int>(hipGetLastError());
  }

  /* -------------------------------------------------------------------
   * GEMV path with col-major data — small M (>1), WMMA data format
   *
   * When batch==1, K%32==0, but 1 < M < kBlockDim: internally transpose
   * A row->col and output col->row so the caller sees row-major I/O.
   * GEMV kernel runs with COL_MAJOR=true for K-parallel reduction.
   *
   * M=1 skips this path entirely — row/col layouts are identical for a
   * single row, and the row-major GEMV avoids the ZP conversion kernel.
   * ------------------------------------------------------------------- */
  if (wmma_data_format && M > 1 && M < kBlockDim) {
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_matmul_nbits: GEMV col-major path M=%lld "
        "N=%lld K=%lld bs=%lld zp=%s bias=%s\n",
        (long long)M, (long long)N, (long long)K,
        (long long)block_size,
        zero_points ? "yes" : "null", bias ? "yes" : "null");

    int iM = static_cast<int>(M);
    int iN = static_cast<int>(N);
    int iK = static_cast<int>(K);

    // Row-major A [M, K] -> col-major (skip if M==1: row/col identical)
    const __half* a_col;
    if (M > 1) {
      _Float16* a_buf = ensureAColBuffer(iM, iK);
      launchTranspose2DFp16(hip_stream, A, a_buf, iM, iK);
      a_col = reinterpret_cast<const __half*>(a_buf);
    } else {
      a_col = static_cast<const __half*>(A);
    }

    // GEMV output to col-major scratch (skip if M==1)
    __half* o_col;
    if (M > 1) {
      o_col = reinterpret_cast<__half*>(ensureOutColBuffer(iM, iN));
    } else {
      o_col = static_cast<__half*>(output);
    }

    const auto* b_u = static_cast<const uint8_t*>(B);
    const auto* s_h = static_cast<const __half*>(scales);
    const auto* z_u = static_cast<const uint8_t*>(zp_for_fp16_paths);
    const auto* b_h = static_cast<const __half*>(bias);

    int config_id = getOrTuneGemvConfig(
        hip_stream, a_col, b_u, s_h, z_u, b_h, o_col,
        M, N, K, 1, block_size, /*col_major=*/true);

    launchGemvConfig(config_id, hip_stream,
                     a_col, b_u, s_h, z_u, b_h, o_col,
                     M, N, K, 1, block_size, /*col_major=*/true);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits GEMV col-major launch "
              "failed: %s\n",
              hipGetErrorString(err));
      return static_cast<int>(err);
    }

    // Col-major output -> row-major (skip if M==1)
    if (M > 1) {
      launchTranspose2DFp16(hip_stream, o_col, output, iN, iM);
      err = hipGetLastError();
      if (err != hipSuccess) {
        fprintf(stderr,
                "[custom_kernels] hip_matmul_nbits GEMV out-transpose "
                "failed: %s\n",
                hipGetErrorString(err));
      }
    }
    return static_cast<int>(err);
  }

  /* -------------------------------------------------------------------
   * GEMV path — small M, row-major data (non-WMMA format)
   *
   * For small M (≤ kBlockDim) with row-major A and uint8 zero points.
   * On the first call for each (M,N,K,block_size) shape, all configs
   * are benchmarked and the fastest is cached for subsequent calls.
   * ------------------------------------------------------------------- */
  if (M <= kBlockDim && element_size_bytes == 2) {
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_matmul_nbits: GEMV path M=%lld N=%lld K=%lld "
        "batch=%lld bs=%lld zp=%s bias=%s\n",
        (long long)M, (long long)N, (long long)K, (long long)batch_count,
        (long long)block_size,
        zero_points ? "yes" : "null", bias ? "yes" : "null");

    const auto* a_h = static_cast<const __half*>(A);
    const auto* b_u = static_cast<const uint8_t*>(B);
    const auto* s_h = static_cast<const __half*>(scales);
    const auto* z_u = static_cast<const uint8_t*>(zp_for_u8_paths);
    const auto* b_h = static_cast<const __half*>(bias);
    auto*       o_h = static_cast<__half*>(output);

    int config_id = getOrTuneGemvConfig(
        hip_stream, a_h, b_u, s_h, z_u, b_h, o_h,
        M, N, K, batch_count, block_size);

    launchGemvConfig(config_id, hip_stream,
                     a_h, b_u, s_h, z_u, b_h, o_h,
                     M, N, K, batch_count, block_size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits GEMV launch failed: %s\n",
              hipGetErrorString(err));
    }
    return static_cast<int>(err);
  }

  /* -------------------------------------------------------------------
   * Optimized bits=4 GEMV for M=1 with fp32 activations (w4a32, e.g. lm_head).
   * One block per output column, K-blocks split across BT threads, 16B
   * coalesced weight loads. Replaces the naive f32 path for this shape.
   * ------------------------------------------------------------------- */
  if (bits == 4 && batch_count == 1 && M == 1 && element_size_bytes == 4) {
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_matmul_nbits: fp32 GEMV M=1 N=%lld K=%lld "
        "bs=%lld zp=%s bias=%s\n",
        (long long)N, (long long)K, (long long)block_size,
        zero_points ? "yes" : "null", bias ? "yes" : "null");

    (void)hipGetLastError();
    launch_gemv_f32_m1(
        hip_stream,
        static_cast<const float*>(A), static_cast<const uint8_t*>(B),
        static_cast<const float*>(scales),
        static_cast<const uint8_t*>(zp_for_u8_paths),
        static_cast<const float*>(bias), static_cast<float*>(output),
        N, K, block_size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits fp32 GEMV launch failed: %s\n",
              hipGetErrorString(err));
    }
    return static_cast<int>(err);
  }

  /* -------------------------------------------------------------------
   * Naive fallback (row-major, batched, uint8 zero points)
   * ------------------------------------------------------------------- */
  CUSTOM_KERNELS_DEBUG_LOG(
      "[custom_kernels] hip_matmul_nbits: naive fallback M=%lld N=%lld K=%lld "
      "batch=%lld bits=%lld bs=%lld elem=%lld zp=%s bias=%s\n",
      (long long)M, (long long)N, (long long)K, (long long)batch_count,
      (long long)bits, (long long)block_size, (long long)element_size_bytes,
      zero_points ? "yes" : "null", bias ? "yes" : "null");

  dim3 blk(kBlockDim, kBlockDim);
  dim3 grd(static_cast<unsigned>((N + kBlockDim - 1) / kBlockDim),
           static_cast<unsigned>((M + kBlockDim - 1) / kBlockDim),
           static_cast<unsigned>(batch_count));

  (void)hipGetLastError();
  if (element_size_bytes == 4) {
    hipLaunchKernelGGL(
        matmul_nbits_kernel_f32, grd, blk, 0, hip_stream,
        static_cast<const float*>(A), static_cast<const uint8_t*>(B),
        static_cast<const float*>(scales),
        static_cast<const uint8_t*>(zp_for_u8_paths),
        static_cast<const float*>(bias), static_cast<float*>(output),
        M, N, K, block_size);
  } else {
    hipLaunchKernelGGL(
        matmul_nbits_kernel, grd, blk, 0, hip_stream,
        static_cast<const __half*>(A), static_cast<const uint8_t*>(B),
        static_cast<const __half*>(scales),
        static_cast<const uint8_t*>(zp_for_u8_paths),
        static_cast<const __half*>(bias), static_cast<__half*>(output),
        M, N, K, block_size);
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits naive launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
