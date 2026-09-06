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
 *   2. GEMV col-major path (batch==1, K%32==0, 1 < M < kWmmaMinM):
 *      Transposes A/C internally. K-parallel reduction GEMV with
 *      COL_MAJOR=true, autotuned (BLOCK_SIZE, TILE_N). One block per row,
 *      so an M-row call streams the weight M times -- see the GEMV-M
 *      autotune section for why that is the main cost in this range.
 *
 *   2b. M-blocked variant of 2, on by default, HIPDNN_EP_MATMUL_SMALLM=0 to
 *      disable: as 2 but with TILE_M>1, so one block covers several output
 *      rows and reads the weight once per tile of rows. The bits=8 GEMV has
 *      the same treatment behind HIPDNN_EP_MATMUL_SMALLM_I8.
 *
 *      For bits=4 this is only the fallback in the verify band: M in
 *      [2, HIPDNN_EP_MATMUL_DP4A_MAX_M] prefers the W4A8 integer-dot kernel
 *      (Section 1b-dp4a), which is M-blocked too and both reads the weight
 *      once and dequantizes it once per tile of rows. The float path here
 *      repeats the dequant per row, so it stays ALU-bound where the dp4a one
 *      is not.
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
 *
 * bits=3 (uint3) is a separate, custom exploratory path (see Sections
 * 1c/1d and 2d below) — NOT an ONNX MatMulNBits convention. Weights are
 * packed as a continuous per-row 3-bit bitstream (not nibble/byte aligned
 * like bits=4/8), giving ~25% less weight traffic than bits=4.
 * zero_points for bits=3 are plain (non-packed) uint8; default zp =
 * 2^(bits-1) = 4. Three paths, mirroring the bits=8 dispatch:
 *   1. WMMA fast path (M>=16, K%32==0, group_size%32==0 and >=32): the
 *      3-bit bitstream is decoded inline in the K-loop (no separate
 *      dequant buffer) — 8 values always pack into exactly 3 bytes
 *      (24 bits, no remainder), so every thread's per-K-step decode is a
 *      single 32-bit register with no cross-byte-boundary shifting. See
 *      Section 2d for the full alignment proof.
 *   2. GEMV (decode/small-M, block_size>=32 power of two, K%32==0):
 *      vectorized 96-bit (uint3) loads, 32 weights per transaction.
 *   3. Naive fallback: shapes meeting neither gate above.
 */

#include "hip_custom_kernels.h"
#include "hip_arch_compat.h"
#include "debug_log.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

/* -----------------------------------------------------------------------
 * Shared disk-persistent autotune cache (GEMV int4 / int8 paths)
 *
 * WHY: the GEMV autotune (kGemvConfigs) picks the fastest (threads, tile_n)
 * by on-device timing. Kept purely in-process, every process re-runs that
 * timing and -- because several configs are within measurement noise of each
 * other -- can settle on a DIFFERENT winner run-to-run, so single-token
 * decode TPS swings run-to-run between otherwise-identical processes.
 * Persisting the winning config to disk (keyed by problem shape + a build
 * stamp) makes the first process prime the file and every later process load
 * the SAME config -> deterministic decode. Mirrors the WMMA disk cache
 * (getWmmaCachePath / loadWmmaCache / saveWmmaCache) further down this file.
 *
 * File format: first line = build stamp (invalidates the file on recompile),
 * then "<key> <config_id>" per line. One file per GPU arch in %TEMP%.
 * ----------------------------------------------------------------------- */
static constexpr const char kTuneCacheBuildStamp[] = __DATE__ " " __TIME__;

static std::string getTuneCachePath(const char *name) {
    std::string tmp_dir;
#ifdef _WIN32
    char *v = nullptr;
    size_t len = 0;
    if (_dupenv_s(&v, &len, "TEMP") == 0 && v != nullptr) {
        tmp_dir = v;
        free(v);
    }
#else
    const char *v = getenv("TEMP");
    if (!v) v = getenv("TMPDIR");
    if (!v) v = "/tmp";
    tmp_dir = v;
#endif
    if (tmp_dir.empty()) return std::string();

    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess ||
        prop.gcnArchName[0] == '\0')
        return std::string();

    return tmp_dir + "/morphizen_" + name + "_cache_" + prop.gcnArchName +
           ".txt";
}

static void loadTuneCacheFile(const std::string &path,
                              std::unordered_map<std::string, int> &cache,
                              int maxConfigId) {
    if (path.empty()) return;
    FILE *f = fopen(path.c_str(), "r");
    if (!f) return;
    // First line is the build stamp -- discard the cache if the binary was
    // rebuilt (config ids / timings may have changed).
    char ts_line[64];
    if (!fgets(ts_line, sizeof(ts_line), f)) {
        fclose(f);
        return;
    }
    size_t ts_len = strlen(ts_line);
    if (ts_len > 0 && ts_line[ts_len - 1] == '\n') ts_line[ts_len - 1] = '\0';
    if (strcmp(ts_line, kTuneCacheBuildStamp) != 0) {
        fclose(f);
        return;
    }
    char key[128];
    int cfg;
    while (fscanf(f, "%127s %d", key, &cfg) == 2)
        if (cfg >= 0 && cfg < maxConfigId) cache[key] = cfg;
    fclose(f);
}

static void saveTuneCacheFile(const std::string &path,
                              const std::unordered_map<std::string, int> &cache) {
    if (path.empty()) return;
    FILE *f = fopen(path.c_str(), "w");
    if (!f) return;
    fprintf(f, "%s\n", kTuneCacheBuildStamp);
    for (const auto &e : cache) fprintf(f, "%s %d\n", e.first.c_str(), e.second);
    fclose(f);
}

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
 * Each threadblock computes TILE_N output elements for TILE_M rows, sharing
 * the loaded weight bytes across those rows. BLOCK_SIZE threads cooperate on
 * the K reduction via warp shuffle + shared mem (identical to int4 path).
 *
 * Key optimizations carried over from int4:
 *   - 16-byte main loop (uint4 = 128 bits): 16 weights per global txn
 *   - Load-compute separation: B loads issued before FMA compute
 *   - Factored dequant:  dot*scale - a_sum*zp*scale
 *     (for w=int8: same identity as nibble; saves the per-element zp
 *     subtract by hoisting it out of the K-inner loop -- ~30% fewer
 *     fp32 ops in the FMA tree)
 *   - Bit-shift group index (block_size is power-of-2; gated at dispatch)
 *   - TILE_M row blocking, so the weight is read once per tile of rows
 *     rather than once per row (see the GEMV-M-i8 autotune section)
 *
 * WHY TILE_M matters more here than it does for int4. A speculative verify
 * pass calls this with M = k+1 rows, and at TILE_M == 1 grid.y = M gives each
 * row its own threadblock streaming the whole weight: M times the traffic for
 * the same answer. Measured on a 622 MB int8 lm_head, M=1 -> M=8 costs
 * 4.10 -> 31.54 ms with M*W/t pinned at 152-158 GB/s at every M, which is the
 * signature of a kernel that is purely traffic-bound and reading per row.
 * int8 also has more headroom for the trade than int4 does: one byte-extract
 * plus one FMA per weight per row against int4's shift+mask+convert+FMA over
 * half the bytes, so blocking rows costs a quarter of the ALU per byte that
 * the same change costs on the nibble path.
 *
 * Constraint: requires block_size >= 16 (so a 16-element vector lies in one
 * group). The Qwen3.5 int8 ops use block_size=128, which satisfies this.
 * Smaller block_size falls back to the naive int8 kernel above.
 *
 * Launch: grid(ceil(N/TILE_N), ceil(M/TILE_M), batch), block(BLOCK_SIZE)
 * ======================================================================= */

// Occupancy target, which is really a register-budget request: the VGPR file is
// 1536 per SIMD in wave32, so N waves/SIMD caps a kernel at 1536/N registers.
//
// Left at the compiler's preferred 9 for TILE_M == 1, which fits in 166 with no
// scratch. For TILE_M > 1 it must be lowered, because the default heuristic
// picks 8 waves (a 192-register cap) and then spills the remainder: measured
// 1024-2264 bytes of scratch per lane, reloaded inside the K loop, which is what
// made every multi-row config 6-10x slower than the single-row one it was meant
// to beat. 6 waves gives 256, the wave32 architectural maximum, and is the
// correct trade for a kernel whose limit is memory latency rather than
// arithmetic: fewer, fatter waves that never touch scratch.
// A minimum alone does nothing here: asking for "at least 6" leaves the
// compiler free to reach 8, which it prefers, so the cap has to be a maximum.
//
// Shared by the int8 GEMV below and the int4 GEMV in section 1b, which have the
// same accumulator tile and the same reason to want it in registers.
#define GEMV_WAVES_MIN(TM) ((TM) > 1 ? 1 : 9)
#define GEMV_WAVES_MAX(TM) ((TM) > 1 ? 4 : 9)

template <int BLOCK_SIZE, int TILE_N, int TILE_M = 1>
__global__
__attribute__((amdgpu_waves_per_eu(GEMV_WAVES_MIN(TILE_M),
                                   GEMV_WAVES_MAX(TILE_M))))
void matmul_nbits_gemv_kernel_i8(
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
    const int64_t m0     = static_cast<int64_t>(blockIdx.y) * TILE_M;
    const int64_t batch  = blockIdx.z;
    const int tid        = threadIdx.x;

    if (m0 >= M) return;

    // Rows owned by this block. TILE_M == 1 collapses every TILE_M-dependent
    // construct below to exactly the pre-existing single-row code.
    const int rows = (TILE_M == 1)
                       ? 1
                       : static_cast<int>((M - m0) < TILE_M ? (M - m0) : TILE_M);

    const int64_t k_blocks = (K + block_size - 1) / block_size;
    const int     bs_shift = __builtin_ctzll(
                                 static_cast<unsigned long long>(block_size));

    // Row-major A only (no COL_MAJOR variant -- int8 has no WMMA path).
    const __half* a_base   = A + batch * M * K + m0 * K;

    // Per-tile-N column pointers (clamped to N-1 for OOB lanes; the
    // out-of-bounds tile guards in the writeback skip those results).
    //
    // Held in registers at TILE_M == 1 exactly as they always were. At
    // TILE_M > 1 they cost 6 dwords per column that partial[][] needs more, so
    // they are recomputed from the tile base instead -- the same trade the int4
    // M-blocked path makes, and for the same reason: every entry is affine in
    // tn, so it is two multiplies against a tile that would otherwise spill.
    constexpr int NADDR = (TILE_M == 1) ? TILE_N : 1;
    [[maybe_unused]] const uint8_t* b_base_arr[NADDR];
    [[maybe_unused]] const __half*  s_rows[NADDR];
    [[maybe_unused]] int64_t        n_idx[NADDR];
    if constexpr (TILE_M == 1) {
#pragma unroll
        for (int tn = 0; tn < NADDR; tn++) {
            int64_t n  = n_base + tn;
            int64_t sn = (n < N) ? n : 0;
            n_idx[tn]  = sn;
            b_base_arr[tn] = B + sn * K;
            s_rows[tn] = scales + sn * k_blocks;
        }
    }

    // Column offset inside the tile, clamped so an out-of-range column aliases
    // the tile's first column. Which in-bounds column it reads is irrelevant:
    // the writeback drops those lanes.
    auto col_off = [&](int tn) __attribute__((always_inline)) -> int64_t {
        return (n_base + tn < N) ? tn : 0;
    };
    auto b_ptr = [&](int tn) __attribute__((always_inline)) -> const uint8_t* {
        if constexpr (TILE_M == 1) return b_base_arr[tn];
        else return B + (n_base + col_off(tn)) * K;
    };
    auto s_ptr = [&](int tn) __attribute__((always_inline)) -> const __half* {
        if constexpr (TILE_M == 1) return s_rows[tn];
        else return scales + (n_base + col_off(tn)) * k_blocks;
    };
    auto zp_row = [&](int tn) __attribute__((always_inline)) -> int64_t {
        if constexpr (TILE_M == 1) return n_idx[tn];
        else return n_base + col_off(tn);
    };
    auto read_a = [&](int tm, int64_t k) __attribute__((always_inline)) -> float {
        return static_cast<float>(a_base[static_cast<int64_t>(tm) * K + k]);
    };

    float partial[TILE_M][TILE_N];
#pragma unroll
    for (int tm = 0; tm < TILE_M; tm++)
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++)
            partial[tm][tn] = 0.0f;

    const bool has_zp = (zp != nullptr);

    // ---- Phase 1: 16-byte (128-bit / uint4) vectorized main loop ----
    // 16 weights per global memory transaction; matches the 16-nibble loop
    // in the int4 kernel but at twice the bytes-per-iter (since each
    // weight is now a full byte).
    const int64_t num_u128 = K / 16;
    for (int64_t idx = static_cast<int64_t>(tid); idx < num_u128;
         idx += static_cast<int64_t>(BLOCK_SIZE)) {
        const int64_t k16 = idx * 16;

        // 16-byte chunks lie in one group iff block_size >= 16 (precondition).
        const int64_t grp = k16 >> bs_shift;

        // 16-byte (uint4) loads from each of TILE_N B columns, consumed by
        // every row this block owns. That reuse is the point of TILE_M > 1.
        uint4 b_pk[TILE_N];
        float sv[TILE_N];
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            b_pk[tn] = *reinterpret_cast<const uint4*>(&b_ptr(tn)[k16]);
            sv[tn]   = static_cast<float>(s_ptr(tn)[grp]);
        }
        // Hoisting the zero point costs TILE_N registers, which is only worth
        // it when there are rows to amortise it over; at TILE_M == 1 it stays
        // inside the tn loop so the decode path's register pressure is
        // unchanged.
        float zv_h[TILE_M > 1 ? TILE_N : 1];
        if (TILE_M > 1 && has_zp) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++)
                zv_h[tn] = static_cast<float>(zp[zp_row(tn) * k_blocks + grp]);
        }

        if constexpr (TILE_M == 1) {
            // Single row: the 16-wide A window is live once, so keep the
            // original shape. This is the decode path and its codegen is tuned.
            float a_vals[16];
            float a_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                a_vals[i] = read_a(0, k16 + i);
                a_sum += a_vals[i];
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
                    float zv = static_cast<float>(
                                   zp[n_idx[tn] * k_blocks + grp]);
                    partial[0][tn] += dot * sv[tn] - a_sum * zv * sv[tn];
                } else {
                    // ONNX MatMulNBits default zp for bits=8 is 2^(bits-1).
                    partial[0][tn] += dot * sv[tn] - a_sum * 128.0f * sv[tn];
                }
            }
        } else {
            // Several rows: quarter the 16-wide group so the unrolled row loop
            // holds one 4-wide A window rather than TILE_M copies of a 16-wide
            // one, and so each quarter lines up with one uint32 lane of b_pk.
            //
            // The row loop must stay a compile-time unroll -- a data-dependent
            // exit on `rows` makes the trip count dynamic, partial[tm][tn] gets
            // a runtime index, and the whole accumulator tile lands in scratch.
            // That, not a register shortfall, is what made the int4 version of
            // this path 6-10x slower before it was predicated instead. Rows
            // beyond `rows` are computed against a clamped A row and dropped at
            // the store: wasted FMAs against an already-loaded weight, which is
            // the cheap half of this kernel.
#pragma unroll
            for (int tm = 0; tm < TILE_M; tm++) {
                const int tm_r = (tm < rows) ? tm : 0;

                float acc[TILE_N];
#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++) acc[tn] = 0.0f;
                float a_sum = 0.0f;

#pragma unroll
                for (int q = 0; q < 4; q++) {
                    float a4[4];
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        a4[i] = read_a(tm_r, k16 + q * 4 + i);
                        a_sum += a4[i];
                    }
#pragma unroll
                    for (int tn = 0; tn < TILE_N; tn++) {
                        const uint32_t w = (q == 0)   ? b_pk[tn].x
                                           : (q == 1) ? b_pk[tn].y
                                           : (q == 2) ? b_pk[tn].z
                                                      : b_pk[tn].w;
#pragma unroll
                        for (int i = 0; i < 4; i++)
                            acc[tn] = __fmaf_rn(
                                a4[i],
                                static_cast<float>((w >> (i * 8)) & 0xFFu),
                                acc[tn]);
                    }
                }

#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++) {
                    const float zv = has_zp ? zv_h[tn] : 128.0f;
                    partial[tm][tn] += (acc[tn] - a_sum * zv) * sv[tn];
                }
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
            const int64_t grp = k8 >> bs_shift;

            // Only runs when K % 16 != 0, so it shares the function's register
            // budget but can never be its peak: an 8-wide A window per row is
            // narrow enough to leave the phase-1 tile shape unconstrained.
#pragma unroll
            for (int tm = 0; tm < TILE_M; tm++) {
                const int tm_r = (TILE_M == 1 || tm < rows) ? tm : 0;

                float a_vals[8];
                float a_sum = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    a_vals[i] = read_a(tm_r, k8 + i);
                    a_sum += a_vals[i];
                }

#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++) {
                    float sv = static_cast<float>(s_ptr(tn)[grp]);
                    uint2 packed = *reinterpret_cast<const uint2*>(
                                        &b_ptr(tn)[k8]);

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
                                       zp[zp_row(tn) * k_blocks + grp]);
                        partial[tm][tn] += dot * sv - a_sum * zv * sv;
                    } else {
                        partial[tm][tn] += dot * sv - a_sum * 128.0f * sv;
                    }
                }
            }
        }
    }

    // ---- Phase 3: scalar tail (K%8 remainder) ----
    {
        const int64_t k_tail = (K / 8) * 8;
        for (int64_t k = k_tail + static_cast<int64_t>(tid);
             k < K; k += static_cast<int64_t>(BLOCK_SIZE)) {
            const int64_t grp = k >> bs_shift;

#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float sv = static_cast<float>(s_ptr(tn)[grp]);
                float zv = has_zp
                              ? static_cast<float>(
                                    zp[zp_row(tn) * k_blocks + grp])
                              : 128.0f;
                float qval = static_cast<float>(b_ptr(tn)[k]);
#pragma unroll
                for (int tm = 0; tm < TILE_M; tm++) {
                    const int tm_r = (TILE_M == 1 || tm < rows) ? tm : 0;
                    partial[tm][tn] += read_a(tm_r, k) * (qval - zv) * sv;
                }
            }
        }
    }

    // --- Reduction: warp shuffle then shared memory (mirrors int4 path) ---
    const int wave_size = __builtin_amdgcn_wavefrontsize();
    const int warp_id   = tid / wave_size;
    const int lane_id   = tid % wave_size;
    const int num_warps = BLOCK_SIZE / wave_size;

#pragma unroll
    for (int tm = 0; tm < TILE_M; tm++) {
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            for (int offset = wave_size >> 1; offset > 0; offset >>= 1)
                partial[tm][tn] += __shfl_down(partial[tm][tn], offset);
        }
    }

    constexpr int MAX_WARPS = BLOCK_SIZE / 32;
    __shared__ float warp_sums[TILE_M * TILE_N * MAX_WARPS];

    if (lane_id == 0) {
#pragma unroll
        for (int tm = 0; tm < TILE_M; tm++)
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++)
                warp_sums[(tm * TILE_N + tn) * MAX_WARPS + warp_id] =
                    partial[tm][tn];
    }
    __syncthreads();

    if (warp_id == 0) {
#pragma unroll
        for (int tm = 0; tm < TILE_M; tm++) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float val =
                    (lane_id < num_warps)
                        ? warp_sums[(tm * TILE_N + tn) * MAX_WARPS + lane_id]
                        : 0.0f;
                for (int offset = wave_size >> 1; offset > 0; offset >>= 1)
                    val += __shfl_down(val, offset);

                if (lane_id == 0) {
                    if (TILE_M > 1 && tm >= rows) continue;
                    int64_t n = n_base + tn;
                    if (n < N) {
                        if (bias)
                            val += static_cast<float>(bias[n]);
                        output[batch * M * N + (m0 + tm) * N + n] =
                            static_cast<__half>(val);
                    }
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

// GEMV_WAVES_MIN/MAX are defined next to the int8 GEMV in section 1a-gemv;
// both kernels hold the same partial[TILE_M][TILE_N] accumulator and want the
// same register budget for it.
template <int BLOCK_SIZE, int TILE_N, bool COL_MAJOR = false, int TILE_M = 1>
__global__
__attribute__((amdgpu_waves_per_eu(GEMV_WAVES_MIN(TILE_M),
                                   GEMV_WAVES_MAX(TILE_M))))
void matmul_nbits_gemv_kernel(
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
    const int64_t m0     = static_cast<int64_t>(blockIdx.y) * TILE_M;
    const int64_t batch  = COL_MAJOR ? 0 : blockIdx.z;
    const int tid        = threadIdx.x;

    if (m0 >= M) return;

    // Rows owned by this block. TILE_M == 1 collapses every TILE_M-dependent
    // construct below to exactly the pre-existing single-row code.
    const int rows = (TILE_M == 1)
                       ? 1
                       : static_cast<int>((M - m0) < TILE_M ? (M - m0) : TILE_M);

    const int64_t k_blocks = (K + block_size - 1) / block_size;
    const int     bs_shift = __builtin_ctzll(
                                 static_cast<unsigned long long>(block_size));

    const __half* a_base;
    int64_t       a_stride;  // step between consecutive k
    int64_t       a_mstep;   // step between consecutive rows
    if constexpr (COL_MAJOR) {
        a_base   = A + m0;
        a_stride = M;
        a_mstep  = 1;
    } else {
        a_base   = A + batch * M * K + m0 * K;
        a_stride = 1;
        a_mstep  = K;
    }

    [[maybe_unused]] const __half* zp_fp16 =
        COL_MAJOR ? reinterpret_cast<const __half*>(zp) : nullptr;

    // Packed 4-bit weight rows are [N, k_blocks, block_size/2] bytes. The row
    // stride must use k_blocks (= ceil(K/block_size)), NOT K/2: when K is not a
    // multiple of block_size the last block is padded, so the true row is
    // k_blocks*(block_size/2) bytes, which exceeds K/2. Using K/2 misaligns
    // every row n>0 by the padding delta and corrupts the output. When
    // K % block_size == 0 this equals K/2 (backward compatible).
    const int64_t b_row_stride = k_blocks * (block_size / 2);

    // Per-column addressing. At TILE_M == 1 it is held in registers exactly as
    // it always was; at TILE_M > 1 it is recomputed from the tile base instead.
    //
    // WHY the split: the three arrays below cost 6 dwords per column (96 at
    // TILE_N=16, over a third of the 256-VGPR file) and every entry is affine in
    // tn, so holding them is a pure trade of registers for two multiplies. At
    // TILE_M == 1 that trade is right and the decode configs are tuned around
    // it. At TILE_M > 1 the registers are needed for partial[][] and the tile
    // spills without them, which is what made the first attempt at this path
    // 5-12x slower than the code it replaced.
    constexpr int NADDR = (TILE_M == 1) ? TILE_N : 1;
    [[maybe_unused]] const uint8_t* b_base_arr[NADDR];
    [[maybe_unused]] const __half*  s_rows[NADDR];
    [[maybe_unused]] int64_t        n_idx[NADDR];
    if constexpr (TILE_M == 1) {
#pragma unroll
        for (int tn = 0; tn < NADDR; tn++) {
            int64_t n  = n_base + tn;
            int64_t sn = (n < N) ? n : 0;
            n_idx[tn]  = sn;
            b_base_arr[tn] = B + sn * b_row_stride;
            s_rows[tn] = scales + sn * k_blocks;
        }
    }

    // Column offset inside the tile, clamped so an out-of-range column aliases
    // the tile's first column (n_base < N always, since grid.x covers N). Which
    // in-bounds column it reads is irrelevant: `store` drops those lanes.
    auto col_off = [&](int tn) __attribute__((always_inline)) -> int64_t {
        return (n_base + tn < N) ? tn : 0;
    };
    auto b_ptr = [&](int tn) __attribute__((always_inline)) -> const uint8_t* {
        if constexpr (TILE_M == 1) return b_base_arr[tn];
        else return B + (n_base + col_off(tn)) * b_row_stride;
    };
    auto s_ptr = [&](int tn) __attribute__((always_inline)) -> const __half* {
        if constexpr (TILE_M == 1) return s_rows[tn];
        else return scales + (n_base + col_off(tn)) * k_blocks;
    };
    auto zp_row = [&](int tn) __attribute__((always_inline)) -> int64_t {
        if constexpr (TILE_M == 1) return n_idx[tn];
        else return n_base + col_off(tn);
    };

    float partial[TILE_M][TILE_N];
#pragma unroll
    for (int tm = 0; tm < TILE_M; tm++)
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++)
            partial[tm][tn] = 0.0f;

    const bool has_zp = (zp != nullptr);

    // Helper: read zero-point for (n_idx, grp) — FP16 when COL_MAJOR
    auto read_zp = [&](int64_t nidx, int64_t grp) __attribute__((always_inline)) -> float {
        if constexpr (COL_MAJOR)
            return static_cast<float>(zp_fp16[nidx * k_blocks + grp]);
        else
            return static_cast<float>(zp[nidx * k_blocks + grp]);
    };

    // Helper: read A element for row `tm` of this block at K-offset
    auto read_a = [&](int tm, int64_t k_off) __attribute__((always_inline)) -> float {
        return static_cast<float>(a_base[tm * a_mstep + k_off * a_stride]);
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
        const int grp = k32 >> bs_shift;

        // B, its scales and its zero points are loaded once here and consumed by
        // every row this block owns. That reuse is the entire purpose of
        // TILE_M > 1: the packed weight is by far the dominant traffic, and one
        // threadblock per row streams all of it once per row.
        uint4   b_pk[TILE_N];
        float   sv[TILE_N];
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            b_pk[tn] = *reinterpret_cast<const uint4*>(
                            &b_ptr(tn)[idx * 16]);
            sv[tn]   = static_cast<float>(s_ptr(tn)[grp]);
        }
        // Hoisting the zero point costs TILE_N registers, which is only worth it
        // when there are rows to amortise it over; at TILE_M == 1 it stays where
        // it was, inside the tn loop, so register pressure is unchanged for the
        // wide-TILE_N decode configs.
        float zv_h[TILE_M > 1 ? TILE_N : 1];
        if (TILE_M > 1 && has_zp) {
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++)
                zv_h[tn] = read_zp(zp_row(tn), grp);
        }

        if constexpr (TILE_M == 1) {
            // Single row: the 32-wide A window is live once, so keep the
            // original shape. This is the decode path and its codegen is tuned.
            float a_vals[32];
            float a_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 32; i++) {
                a_vals[i] = read_a(0, k32 + i);
                a_sum += a_vals[i];
            }
            const float a_sum_x_default_zp = a_sum * 8.0f;

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
                    partial[0][tn] += (dot - a_sum * read_zp(zp_row(tn), grp))
                                      * sv[tn];
                } else {
                    partial[0][tn] += (dot - a_sum_x_default_zp) * sv[tn];
                }
            }
        } else {
            // Several rows: quarter the 32-wide group, dequantize each quarter of
            // B once, then apply it to every row.
            //
            // Two things force this shape. First, the row loop must stay a
            // compile-time unroll: a data-dependent `break` on `rows` makes the
            // trip count dynamic, and then raw[tm][tn] has a runtime index and
            // cannot live in registers at all -- the whole tile lands in scratch.
            // That, not a register shortfall, is what made this path 6-10x slower
            // than the single-row one. Out-of-range rows are computed against a
            // clamped A row and dropped by `store`: wasted arithmetic, no spill.
            //
            // Second, quartering caps the A window at 8 values and ends its live
            // range inside each quarter, so the unrolled row loop stops holding
            // TILE_M copies of it -- 128 dwords at TILE_M=4, half the register
            // file. The per-row accumulator acc[] is reused the same way.
            //
            // Hoisting the nibble unpack out of the row loop was tried here, so
            // each weight is dequantized once per tile instead of once per row
            // (bq[TILE_N][8] holding the unpacked quarter). It is the obvious win
            // on paper -- unpack is ~3 ops against the 1 FMA that consumes it --
            // and it did help the isolated benchmark slightly (3.84 -> 3.79 ms on
            // a 311 MB lm_head at M=4). In the full model it lost, on every shape:
            // 37.1 -> 41.1 ms for the 36 gate/up projections, 13.2 -> 15.4 for
            // q/o, 92 -> 100 ms on the whole verify call. The extra live state
            // costs more in latency hiding on cold weights than it saves in ALU,
            // and in-model these weights are always cold. Left out deliberately.
            //
            // Accumulation order changes (four chained 8-FMA runs per group rather
            // than one 32-FMA run), which moves the last bits; measured worst-case
            // relative error stays ~2e-4 against a 2e-2 tolerance.
#pragma unroll
            for (int tm = 0; tm < TILE_M; tm++) {
                const int tm_r = (tm < rows) ? tm : 0;

                float acc[TILE_N];
#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++) acc[tn] = 0.0f;
                float a_sum = 0.0f;

#pragma unroll
                for (int q = 0; q < 4; q++) {
                    float a8[8];
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        a8[i] = read_a(tm_r, k32 + q * 8 + i);
                        a_sum += a8[i];
                    }
#pragma unroll
                    for (int tn = 0; tn < TILE_N; tn++) {
                        const uint32_t w = (q == 0)   ? b_pk[tn].x
                                           : (q == 1) ? b_pk[tn].y
                                           : (q == 2) ? b_pk[tn].z
                                                      : b_pk[tn].w;
#pragma unroll
                        for (int i = 0; i < 8; i++)
                            acc[tn] = __fmaf_rn(
                                a8[i],
                                static_cast<float>((w >> (i * 4)) & 0xF),
                                acc[tn]);
                    }
                }

#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++) {
                    const float zv = has_zp ? zv_h[tn] : 8.0f;
                    partial[tm][tn] += (acc[tn] - a_sum * zv) * sv[tn];
                }
            }
        }
    }

    // ---- Phase 1b: 16-nibble (64-bit) remainder (K%32 ≥ 16) ----
    {
        const int k16_base = num_u128 * 32;
        const int num_u64 = (static_cast<int>(K) - k16_base) / 16;
        for (int idx = tid; idx < num_u64; idx += BLOCK_SIZE) {
            const int k16 = k16_base + idx * 16;
            const int grp = k16 >> bs_shift;

            uint2   b_pk[TILE_N];
            float   sv[TILE_N];
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                b_pk[tn] = *reinterpret_cast<const uint2*>(
                                &b_ptr(tn)[(k16_base / 2) + idx * 8]);
                sv[tn]   = static_cast<float>(s_ptr(tn)[grp]);
            }
            float zv_h[TILE_M > 1 ? TILE_N : 1];
            if (TILE_M > 1 && has_zp) {
#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++)
                    zv_h[tn] = read_zp(zp_row(tn), grp);
            }

            if constexpr (TILE_M == 1) {
                float a_vals[16];
                float a_sum = 0.0f;
#pragma unroll
                for (int i = 0; i < 16; i++) {
                    a_vals[i] = read_a(0, k16 + i);
                    a_sum += a_vals[i];
                }
                const float a_sum_x_default_zp = a_sum * 8.0f;

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
                        partial[0][tn] += (dot - a_sum * read_zp(zp_row(tn), grp))
                                          * sv[tn];
                    } else {
                        partial[0][tn] += (dot - a_sum_x_default_zp) * sv[tn];
                    }
                }
            } else {
                // Halved for the same reason phase 1 is quartered: an unrolled
                // row loop would otherwise hold TILE_M copies of a 16-wide A
                // window. Only runs when K % 32 != 0, but it shares the
                // function's register budget, so it cannot be the peak.
#pragma unroll
                for (int tm = 0; tm < TILE_M; tm++) {
                    const int tm_r = (tm < rows) ? tm : 0;  // see phase 1

                    float acc[TILE_N];
#pragma unroll
                    for (int tn = 0; tn < TILE_N; tn++) acc[tn] = 0.0f;
                    float a_sum = 0.0f;

#pragma unroll
                    for (int q = 0; q < 2; q++) {
                        float a8[8];
#pragma unroll
                        for (int i = 0; i < 8; i++) {
                            a8[i] = read_a(tm_r, k16 + q * 8 + i);
                            a_sum += a8[i];
                        }
#pragma unroll
                        for (int tn = 0; tn < TILE_N; tn++) {
                            const uint32_t w = (q == 0) ? b_pk[tn].x : b_pk[tn].y;
#pragma unroll
                            for (int i = 0; i < 8; i++)
                                acc[tn] = __fmaf_rn(
                                    a8[i],
                                    static_cast<float>((w >> (i * 4)) & 0xF),
                                    acc[tn]);
                        }
                    }

#pragma unroll
                    for (int tn = 0; tn < TILE_N; tn++) {
                        const float zv = has_zp ? zv_h[tn] : 8.0f;
                        partial[tm][tn] += (acc[tn] - a_sum * zv) * sv[tn];
                    }
                }
            }
        }
    }

    // ---- Phase 2: 8-nibble (32-bit) middle (K%8 remainder after uint4+uint2) ----
    {
        const int k8_base   = (static_cast<int>(K) / 16) * 16;
        const int num_u32_m = (static_cast<int>(K) - k8_base) / 8;
        for (int idx2 = tid; idx2 < num_u32_m; idx2 += BLOCK_SIZE) {
            const int k8  = k8_base + idx2 * 8;
            const int grp = k8 >> bs_shift;

            uint32_t b_pk[TILE_N];
            float    sv[TILE_N];
#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                b_pk[tn] = *reinterpret_cast<const uint32_t*>(
                                &b_ptr(tn)[k8 / 2]);
                sv[tn]   = static_cast<float>(s_ptr(tn)[grp]);
            }
            float zv_h[TILE_M > 1 ? TILE_N : 1];
            if (TILE_M > 1 && has_zp) {
#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++)
                    zv_h[tn] = read_zp(zp_row(tn), grp);
            }

#pragma unroll
            for (int tm = 0; tm < TILE_M; tm++) {
                const int tm_r = (TILE_M == 1 || tm < rows) ? tm : 0;

                float a_vals[8];
                float a_sum = 0.0f;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    a_vals[i] = read_a(tm_r, k8 + i);
                    a_sum += a_vals[i];
                }
                const float a_sum_x_default_zp = a_sum * 8.0f;

#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++) {
                    float dot = 0.0f;
#pragma unroll
                    for (int i = 0; i < 8; i++)
                        dot = __fmaf_rn(a_vals[i],
                               static_cast<float>((b_pk[tn] >> (i * 4)) & 0xF), dot);

                    if (has_zp) {
                        float zv = (TILE_M > 1) ? zv_h[tn] : read_zp(zp_row(tn), grp);
                        partial[tm][tn] += (dot - a_sum * zv) * sv[tn];
                    } else {
                        partial[tm][tn] += (dot - a_sum_x_default_zp) * sv[tn];
                    }
                }
            }
        }
    }

    // ---- Phase 3: scalar tail (K%8 remainder) ----
    {
        const int k_tail = (static_cast<int>(K) / 8) * 8;
        for (int k = k_tail + tid; k < static_cast<int>(K);
             k += BLOCK_SIZE) {
            const int   grp    = k >> bs_shift;
            const int   nibble = k & 1;

#pragma unroll
            for (int tn = 0; tn < TILE_N; tn++) {
                float sv = static_cast<float>(s_ptr(tn)[grp]);
                float zv = has_zp ? read_zp(zp_row(tn), grp) : 8.0f;
                uint8_t packed = b_ptr(tn)[k / 2];
                float   qval   = (nibble == 0)
                                   ? static_cast<float>(packed & 0xF)
                                   : static_cast<float>(packed >> 4);
#pragma unroll
                for (int tm = 0; tm < TILE_M; tm++) {
                    const int tm_r = (TILE_M == 1 || tm < rows) ? tm : 0;
                    partial[tm][tn] += read_a(tm_r, k) * (qval - zv) * sv;
                }
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
    for (int tm = 0; tm < TILE_M; tm++) {
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            for (int offset = RED_W >> 1; offset > 0; offset >>= 1)
                partial[tm][tn] += __shfl_down(partial[tm][tn], offset);
        }
    }

    // NOTE: nontemporal stores were tried here but regressed perf by ~3% on
    // gfx1150 LPDDR5X — the output is too small (28KB) to pollute cache, and NT
    // writes compete with read traffic on the shared memory bus.
    auto store = [&](int tm, int tn, float val)
                     __attribute__((always_inline)) -> void {
        if (TILE_M > 1 && tm >= rows) return;
        int64_t n = n_base + tn;
        if (n >= N) return;
        if (bias)
            val += static_cast<float>(bias[n]);
        if constexpr (COL_MAJOR)
            output[n * M + m0 + tm] = static_cast<__half>(val);
        else
            output[batch * M * N + (m0 + tm) * N + n] = static_cast<__half>(val);
    };

    // Single warp (BLOCK_SIZE==32 in wave32): shuffle has the final result.
    if constexpr (BLOCK_SIZE <= WARP_SIZE) {
        if (lane_id == 0) {
#pragma unroll
            for (int tm = 0; tm < TILE_M; tm++)
#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++)
                    store(tm, tn, partial[tm][tn]);
        }
    } else {
        __shared__ float warp_sums[TILE_M * TILE_N * NUM_WARPS];

        if (lane_id == 0) {
#pragma unroll
            for (int tm = 0; tm < TILE_M; tm++)
#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++)
                    warp_sums[(tm * TILE_N + tn) * NUM_WARPS + warp_id] =
                        partial[tm][tn];
        }
        __syncthreads();

        if (warp_id == 0) {
#pragma unroll
            for (int tm = 0; tm < TILE_M; tm++) {
#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++) {
                    float val =
                        (lane_id < NUM_WARPS)
                            ? warp_sums[(tm * TILE_N + tn) * NUM_WARPS + lane_id]
                            : 0.0f;
                    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
                        val += __shfl_down(val, offset);

                    if (lane_id == 0)
                        store(tm, tn, val);
                }
            }
        }
    }
}

/* =======================================================================
 * Section 1c: W4A8 integer-dot-product (dp4a) GEMV path  [decode + verify]
 *
 * WHY: the int4 GEMV kernel above is dequant-ALU-bound, not bandwidth-bound
 * (external RDNA3 W4A16 roofline: ~4-6 ALU instructions per weight byte --
 * shift + mask + int->float convert + fp32 FMA -- before a weight enters the
 * dot). This path replaces that chain with the gfx11 signed/unsigned integer
 * dot builtin `v_dot4_i32_iu8` (`__builtin_amdgcn_sudot4`): each instruction
 * folds 4 MACs, so 8 nibbles cost 2 sudot4 + a few mask/shift ops instead of
 * 8 float-converts + 8 fp32 FMAs.
 *
 * Precondition: the activation row is dynamically quantized to per-group int8
 * once per token by `matmul_nbits_quant_act_i8_kernel` below (amortized across
 * all N output columns), and packed nibble-DEINTERLEAVED so the weight's
 * even/odd nibble split (`x & 0x0F0F0F0F` / `(x>>4) & 0x0F0F0F0F`) pairs
 * byte-for-byte with the activation without any per-column reshuffle.
 *
 * Factored dequant (integer): with a[k] ~= a_scale[grp] * aq[k] (int8),
 *   out = sum_k a[k]*(w[k]-zp)*scale[grp]
 *       = a_scale[grp] * scale[grp] * ( dot_int(aq, w) - zp * sum(aq) )
 * where dot_int and sum(aq) are int32 (sum via sudot4 against 0x01010101),
 * and a_scale/scale/zp are per-group constants.
 *
 * Only the K%32==0 main path is implemented (all target decode shapes have
 * K a multiple of 32); the dispatcher falls back to the fp GEMV otherwise.
 *
 * TILE_M extends the same idea to the speculative verify pass. That pass runs
 * M = k+1 rows through the identical weights, and on the fp path those weights
 * are the binding cost twice over: read once per row-tile at best, and
 * dequantized once per row always, since the nibble unpack sits inside the row
 * loop. The integer-dot form removes the second cost outright -- the two mask
 * operations that split a uint32 of packed nibbles into even/odd bytes are
 * per-weight, not per-row, and every row after the first adds only two sudot4.
 * Against the fp M-blocked path's shift + mask + convert + FMA per weight per
 * row, that is ~16x less row-dependent arithmetic, which is what it takes to
 * turn a read-once kernel from ALU-bound back into traffic-bound.
 * ======================================================================= */

// Deinterleaved-int8 activation layout, per 8-element chunk c (k = 8c + p):
//   bytes[8c+0..3] = {aq[8c+0], aq[8c+2], aq[8c+4], aq[8c+6]}   (a_lo, even p)
//   bytes[8c+4..7] = {aq[8c+1], aq[8c+3], aq[8c+5], aq[8c+7]}   (a_hi, odd p)
// This matches the byte order produced by masking a packed-nibble uint32 with
// 0x0F0F0F0F (even nibbles) and (>>4)&0x0F0F0F0F (odd nibbles). One block per
// (K-group, row); group boundaries are 8-aligned (block_size is a power of two
// >= 32), so no chunk straddles a group. Rows are quantized independently --
// each gets its own per-group scale -- because they are different tokens and a
// shared scale would let one row's outliers cost every other row precision.
template <int BLOCK_SIZE>
__global__ void matmul_nbits_quant_act_i8_kernel(
    const __half* __restrict__ A_all,
    int8_t*       __restrict__ a_qb_all,
    float*        __restrict__ a_scale_all,
    int64_t K, int64_t block_size, int64_t k_blocks)
{
    const int64_t row = blockIdx.y;
    const __half* __restrict__ A     = A_all + row * K;
    int8_t*       __restrict__ a_qb  = a_qb_all + row * K;
    float*        __restrict__ a_scale = a_scale_all + row * k_blocks;

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
// Row-major only; requires K % 32 == 0. Reads the per-group int8 activation
// (a_qb) + scales (a_scale) prepared ONCE per call by
// matmul_nbits_quant_act_i8_kernel above -- the quant is amortized across all
// N/TILE_N column blocks (fusing it into the GEMV instead re-quantizes A in
// every block, which is a net loss: measured ~1.3-2x slower on decode shapes
// because the redundant fp16 read + convert per block dwarfs the one launch it
// would save).
//
// TILE_M rows share one pass over B. Unlike the fp kernel this does not need
// the A window quartered to stay off scratch: a row's contribution to a 32-wide
// K step is 8 ints of deinterleaved int8 (a_lo/a_hi), not 32 floats, so at
// TILE_M=4 the whole activation tile is 32 registers rather than 128.
//
// The occupancy request differs from the fp kernel's and must: this kernel had
// no waves_per_eu attribute at all before TILE_M existed, and giving the
// single-row instantiation the fp path's (9, 9) cost 2.2x on a 311 MB lm_head
// at M=1 (2.24 -> 5.02 ms) -- a 170-register budget the integer path does not
// fit. (1, 16) is the wave32 architectural range, i.e. no constraint, which
// restores the codegen the decode path was tuned with. TILE_M > 1 still needs
// the cap for the same reason the fp kernel does: the accumulator tile has to
// stay in registers.
#define DP4A_WAVES_MIN(TM) 1
#define DP4A_WAVES_MAX(TM) ((TM) > 1 ? 4 : 16)

template <int BLOCK_SIZE, int TILE_N, int TILE_M = 1>
__global__
__attribute__((amdgpu_waves_per_eu(DP4A_WAVES_MIN(TILE_M),
                                   DP4A_WAVES_MAX(TILE_M))))
void matmul_nbits_gemv_dp4a_kernel(
    const int8_t*  __restrict__ a_qb,
    const float*   __restrict__ a_scale,
    const uint8_t* __restrict__ B,
    const __half*  __restrict__ scales,
    const uint8_t* __restrict__ zp,
    const __half*  __restrict__ bias,
    __half*        __restrict__ output,
    int64_t M, int64_t N, int64_t K, int64_t block_size)
{
    const int64_t n_base = static_cast<int64_t>(blockIdx.x) * TILE_N;
    const int64_t m0     = static_cast<int64_t>(blockIdx.y) * TILE_M;
    const int     tid    = threadIdx.x;
    const int64_t k_blocks = (K + block_size - 1) / block_size;
    const int     bs_shift =
        __builtin_ctzll(static_cast<unsigned long long>(block_size));
    const bool has_zp = (zp != nullptr);

    if (m0 >= M) return;
    const int rows = (TILE_M == 1)
                       ? 1
                       : static_cast<int>((M - m0) < TILE_M ? (M - m0) : TILE_M);

    // Held in registers at TILE_M == 1, recomputed at TILE_M > 1 -- the same
    // trade the fp M-blocked kernel makes, for the same reason.
    constexpr int NADDR = (TILE_M == 1) ? TILE_N : 1;
    [[maybe_unused]] const uint8_t* b_base_arr[NADDR];
    [[maybe_unused]] const __half*  s_rows[NADDR];
    [[maybe_unused]] int64_t        n_idx[NADDR];
    if constexpr (TILE_M == 1) {
#pragma unroll
        for (int tn = 0; tn < NADDR; tn++) {
            int64_t n  = n_base + tn;
            int64_t sn = (n < N) ? n : 0;
            n_idx[tn]  = sn;
            b_base_arr[tn] = B + sn * (K / 2);
            s_rows[tn] = scales + sn * k_blocks;
        }
    }
    auto col_idx = [&](int tn) __attribute__((always_inline)) -> int64_t {
        if constexpr (TILE_M == 1) return n_idx[tn];
        else return n_base + ((n_base + tn < N) ? tn : 0);
    };
    auto b_ptr = [&](int tn) __attribute__((always_inline)) -> const uint8_t* {
        if constexpr (TILE_M == 1) return b_base_arr[tn];
        else return B + col_idx(tn) * (K / 2);
    };
    auto s_ptr = [&](int tn) __attribute__((always_inline)) -> const __half* {
        if constexpr (TILE_M == 1) return s_rows[tn];
        else return scales + col_idx(tn) * k_blocks;
    };

    float partial[TILE_M][TILE_N];
#pragma unroll
    for (int tm = 0; tm < TILE_M; tm++)
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++)
            partial[tm][tn] = 0.0f;

    const int num_u128 = static_cast<int>(K / 32);
    for (int idx = tid; idx < num_u128; idx += BLOCK_SIZE) {
        const int k32 = idx * 32;
        const int grp = k32 >> bs_shift;

        // 4 deinterleaved chunks (a_lo/a_hi) covering the 32 activations, for
        // each row this block owns. Out-of-range rows alias row 0 and are
        // dropped at the store: the row loop has to stay a compile-time unroll
        // or partial[][] gets a runtime index and spills to scratch.
        int   a_lo[TILE_M][4], a_hi[TILE_M][4];
        int   a_sum_q[TILE_M];
        float a_sc[TILE_M];
#pragma unroll
        for (int tm = 0; tm < TILE_M; tm++) {
            const int tm_r = (TILE_M == 1 || tm < rows) ? tm : 0;
            const int8_t* aq = a_qb + static_cast<int64_t>(tm_r) * K +
                               static_cast<int64_t>(idx) * 32;
            a_sc[tm] = a_scale[static_cast<int64_t>(tm_r) * k_blocks + grp];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                a_lo[tm][j] = *reinterpret_cast<const int*>(aq + j * 8);
                a_hi[tm][j] = *reinterpret_cast<const int*>(aq + j * 8 + 4);
            }
            // sum(aq) over the 32 elements: signed-int8 dot against all-ones.
            int s = 0;
#pragma unroll
            for (int j = 0; j < 4; j++) {
                s = hipdnn_sudot4(a_lo[tm][j], 0x01010101, s);
                s = hipdnn_sudot4(a_hi[tm][j], 0x01010101, s);
            }
            a_sum_q[tm] = s;
        }

#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            uint4 b = *reinterpret_cast<const uint4*>(&b_ptr(tn)[idx * 16]);
            const unsigned comp[4] = {b.x, b.y, b.z, b.w};
            int dot[TILE_M];
#pragma unroll
            for (int tm = 0; tm < TILE_M; tm++) dot[tm] = 0;
            // The two masks below are what TILE_M amortises: they split the
            // packed nibbles once per weight, and every extra row costs only
            // the two sudot4 that consume the result.
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int lo_w = static_cast<int>(comp[j] & 0x0F0F0F0Fu);
                const int hi_w = static_cast<int>((comp[j] >> 4) & 0x0F0F0F0Fu);
#pragma unroll
                for (int tm = 0; tm < TILE_M; tm++) {
                    dot[tm] = hipdnn_sudot4(a_lo[tm][j], lo_w, dot[tm]);
                    dot[tm] = hipdnn_sudot4(a_hi[tm][j], hi_w, dot[tm]);
                }
            }
            const float sv = static_cast<float>(s_ptr(tn)[grp]);
            const int   zv = has_zp
                                 ? static_cast<int>(zp[col_idx(tn) * k_blocks + grp])
                                 : 8;
#pragma unroll
            for (int tm = 0; tm < TILE_M; tm++)
                partial[tm][tn] += a_sc[tm] * sv *
                                   static_cast<float>(dot[tm] - a_sum_q[tm] * zv);
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
    for (int tm = 0; tm < TILE_M; tm++) {
#pragma unroll
        for (int tn = 0; tn < TILE_N; tn++) {
            for (int offset = RED_W >> 1; offset > 0; offset >>= 1)
                partial[tm][tn] += __shfl_down(partial[tm][tn], offset);
        }
    }

    auto store = [&](int tm, int tn, float val)
                     __attribute__((always_inline)) -> void {
        if (TILE_M > 1 && tm >= rows) return;
        int64_t n = n_base + tn;
        if (n >= N) return;
        if (bias)
            val += static_cast<float>(bias[n]);
        output[(m0 + tm) * N + n] = static_cast<__half>(val);
    };

    if constexpr (BLOCK_SIZE <= WARP_SIZE) {
        if (lane_id == 0) {
#pragma unroll
            for (int tm = 0; tm < TILE_M; tm++)
#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++)
                    store(tm, tn, partial[tm][tn]);
        }
    } else {
        __shared__ float warp_sums[TILE_M * TILE_N * NUM_WARPS];
        if (lane_id == 0) {
#pragma unroll
            for (int tm = 0; tm < TILE_M; tm++)
#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++)
                    warp_sums[(tm * TILE_N + tn) * NUM_WARPS + warp_id] =
                        partial[tm][tn];
        }
        __syncthreads();
        if (warp_id == 0) {
#pragma unroll
            for (int tm = 0; tm < TILE_M; tm++) {
#pragma unroll
                for (int tn = 0; tn < TILE_N; tn++) {
                    float val =
                        (lane_id < NUM_WARPS)
                            ? warp_sums[(tm * TILE_N + tn) * NUM_WARPS + lane_id]
                            : 0.0f;
                    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
                        val += __shfl_down(val, offset);
                    if (lane_id == 0)
                        store(tm, tn, val);
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

// ---------------------------------------------------------------------------
// Cold probing for the M=1 GEMV autotune.
//
// Timing launches back to back on the real weight leaves that weight in the
// 32 MB Infinity Cache after the first one, so the tuner ranks configs on cache
// behaviour. A decode step never sees it that way: 5.7 GB of weights stream past
// between one use of a tensor and the next, so every read is a DRAM read. The
// bias is systematic rather than random -- with no memory latency to hide, the
// extra waves of a wide config have nothing to do, so warm probing prefers
// narrow ones. On Qwen3-8B it picked 32- and 128-thread configs where 256
// threads is 6.8% faster over the whole decode step (36.4 -> 34.1 ms).
//
// Streaming a buffer larger than the cache before each timed launch restores the
// conditions the kernel actually runs in. HIPDNN_EP_MATMUL_TUNE_COLD=0 restores
// warm probing, which is the A/B control for the numbers above.
// ---------------------------------------------------------------------------

// Overshoots gfx1151's 32 MB Infinity Cache so no part of B survives a pass.
static constexpr size_t kGemvEvictBytes = 64u << 20;

static bool gemvTuneColdEnabled() {
    static const bool on = [] {
#ifdef _WIN32
        char buf[8];
        unsigned long n = GetEnvironmentVariableA(
            "HIPDNN_EP_MATMUL_TUNE_COLD", buf, sizeof(buf));
        return !(n > 0 && buf[0] == '0');
#else
        const char* v = getenv("HIPDNN_EP_MATMUL_TUNE_COLD");
        return !(v && v[0] == '0');
#endif
    }();
    return on;
}

// Scratch for the eviction stream. Null means probe warm -- a tuning pass that
// cannot get 64 MB should still produce a config rather than fail.
static void* gemvEvictAlloc() {
    if (!gemvTuneColdEnabled()) return nullptr;
    void* p = nullptr;
    if (hipMalloc(&p, kGemvEvictBytes) != hipSuccess) return nullptr;
    return p;
}

static void gemvEvict(void* scratch, hipStream_t stream) {
    if (scratch) (void)hipMemsetAsync(scratch, 0, kGemvEvictBytes, stream);
}

// Times ITERS launches of one config, each preceded by a cache eviction and
// measured on its own event pair so the eviction is not counted. Returns the
// summed kernel time; callers only compare it across configs.
template <int ITERS, typename LaunchFn>
static float gemvTimeConfigCold(hipStream_t stream, void* evict,
                                LaunchFn&& launch) {
    hipEvent_t evs[2 * ITERS];
    for (auto& e : evs) hipEventCreate(&e);

    for (int i = 0; i < ITERS; i++) {
        gemvEvict(evict, stream);
        hipEventRecord(evs[2 * i], stream);
        launch();
        hipEventRecord(evs[2 * i + 1], stream);
    }
    hipEventSynchronize(evs[2 * ITERS - 1]);

    float total = 0.0f;
    for (int i = 0; i < ITERS; i++) {
        float t = 0.0f;
        hipEventElapsedTime(&t, evs[2 * i], evs[2 * i + 1]);
        total += t;
    }
    for (auto& e : evs) hipEventDestroy(e);
    return total;
}

static int tuneGemvConfig(
    hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs,
    bool col_major = false)
{
    float best_ms = 1e30f;
    int   best_id = 8;
    constexpr int WARMUP = 1;
    // Fewer than the warm path used: every iteration now costs a 64 MB evict,
    // and cold measurements vary less because they do not depend on what the
    // previous config left behind.
    constexpr int ITERS  = 3;

    void* evict = gemvEvictAlloc();

    for (int cid = 0; cid < kNumGemvConfigs; cid++) {
        auto& cfg = kGemvConfigs[cid];
        if (cfg.tile_n >= 64 && N < 2048) continue;
        if (cfg.tile_n >= 32 && cfg.tile_n < 64 && N < 1024) continue;

        auto launch = [&] {
            launchGemvConfig(cid, stream, A, B, sc, zp, bi, out,
                             M, N, K, batch, bs, col_major);
        };
        for (int w = 0; w < WARMUP; w++) launch();

        float ms = gemvTimeConfigCold<ITERS>(stream, evict, launch);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   config[%2d] threads=%3d tile_n=%d : %.4f ms\n",
            cid, kGemvConfigs[cid].threads, kGemvConfigs[cid].tile_n,
            ms / ITERS);

        if (ms < best_ms) {
            best_ms = ms;
            best_id = cid;
        }
    }

    if (evict) (void)hipFree(evict);

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
static std::once_flag gemv_cache_init_flag;
static std::string gemv_cache_path;

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

    // Load the disk-persisted cache once per process so repeated runs settle on
    // the same config -> deterministic decode (see kTuneCacheBuildStamp above).
    std::call_once(gemv_cache_init_flag, [] {
        gemv_cache_path = getTuneCachePath("gemv");
        loadTuneCacheFile(gemv_cache_path, gemv_cache, kNumGemvConfigs);
    });

    std::lock_guard<std::mutex> lock(gemv_tune_mutex);

    auto it = gemv_cache.find(key);
    if (it != gemv_cache.end())
        return it->second;

    int best = tuneGemvConfig(stream, A, B, sc, zp, bi, out,
                               M, N, K, batch, bs, col_major);
    gemv_cache[key] = best;
    saveTuneCacheFile(gemv_cache_path, gemv_cache);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] GEMV cache: stored key=\"%s\" -> config[%d] "
        "(cache size=%zu)\n",
        key.c_str(), best, gemv_cache.size());

    return best;
}

/* -----------------------------------------------------------------------
 * M-blocked GEMV autotune + dispatch (small M: 1 < M < kWmmaMinM)
 *
 * WHY: the plain GEMV launches grid(N/TILE_N, M, batch), i.e. one threadblock
 * per output row, so an M-row call streams the whole packed weight M times.
 * Measured on gfx1151 with a 311 MB lm_head (4096x151936, int4, block 32), the
 * per-token cost is flat at 1.52-1.56 ms for every M in 2..7 while aggregate
 * traffic sits at ~201 GB/s of a ~256 GB/s bus: the kernel is bandwidth-bound
 * on a weight it re-reads once per row. Weights smaller than the 32 MB
 * Infinity Cache (e.g. a 25 MB MLP projection) hide most of it, which is why
 * this only shows up on the widest matmuls -- and in real decode, where the
 * cache is being swept by the rest of the layer, it shows up on all of them.
 *
 * TILE_M > 1 gives one block several rows so the B load is amortised: traffic
 * drops from M x weight to 1 x weight, and A (M*K*2 bytes, tens of KB) is
 * re-read instead, which is cache-resident. This is the same reuse a GEMM gets
 * from its M tile, without WMMA's requirement that M fill a 16-row tile.
 *
 * TILE_M is autotuned rather than fixed, because the traffic-optimal choice is
 * not the fastest one: a larger TILE_M cuts weight traffic but costs registers
 * (partial[TILE_M][TILE_N]) and parallelism (grid.y shrinks), and rows past M
 * are computed and discarded, so tile_m=8 at M=4 wastes half its arithmetic.
 * TILE_M*TILE_N is capped at 32, and TILE_M=1 is kept in the table so the tuner
 * can always fall back to per-row behaviour.
 *
 * Measured on a 4096x151936 lm_head at M=4 (isolated, 512 threads): tile_m=1 is
 * 6.54 ms, and every spill-free tile_m=8 config lands at 3.77-3.88 ms whatever
 * the thread count -- a flat plateau at ~82 GB/s of weight read once, against
 * ~48 GB/s read four times. The configs that still spill (tile_n>=8 with
 * tile_m=2, tile_n=16 with tile_m=2) remain 6-10x slower and the tuner avoids
 * them; check with -Rpass-analysis=kernel-resource-usage before adding shapes.
 * ----------------------------------------------------------------------- */

// Default ON; set HIPDNN_EP_MATMUL_SMALLM=0 to fall back to the per-row path.
//
// This was default-off for as long as tile_m > 1 was slower than the tile_m = 1
// path it was meant to beat -- by 5-12x, which looked like register pressure but
// was not. The cause was `if (tm >= rows) break;` in the row loop: a
// data-dependent exit makes the trip count dynamic, so the loop is no longer a
// compile-time unroll, partial[tm][tn] gets a runtime index, and the whole
// accumulator tile is forced to scratch. The compiler reported it plainly once
// asked (-Rpass-analysis=kernel-resource-usage): 192 VGPRs and 1024-2264 bytes
// of scratch per lane for every tile_m > 1 config, against 165 and zero for
// tile_m = 1. Predicating the row bound instead of branching on it, and
// quartering the A window so the unrolled row loop stops holding TILE_M copies
// of it, removes the scratch entirely for the tile shapes that matter.
//
// What it buys, measured on a 4096x151936 lm_head at M=4: 6.54 ms -> 3.79 ms
// isolated (1.73x), and on a full Qwen3-8B verify call 99.2 -> 94.4 ms with
// v/step 2.72 -> 2.60. The end-to-end gain is much smaller than the isolated one
// because the int4 kernel only carries about half this model's matmul traffic;
// the rest is int8, which has no M-blocking (see matmul_nbits_gemv_kernel_i8).
static bool gemvMBlockedEnabled() {
    static const bool enabled = [] {
#ifdef _WIN32
        // Static-CRT DLLs cannot see the host process environment through
        // getenv; go to Win32 directly (same reason as debug_log.h).
        char buf[8];
        unsigned long n = GetEnvironmentVariableA(
            "HIPDNN_EP_MATMUL_SMALLM", buf, sizeof(buf));
        return !(n > 0 && buf[0] == '0');
#else
        const char* v = getenv("HIPDNN_EP_MATMUL_SMALLM");
        return !(v && v[0] == '0');
#endif
    }();
    return enabled;
}

// M at or above which int4 dispatch prefers WMMA over the col-major GEMV, used
// when the tier is not tuned (see preferWmmaTier). HIPDNN_EP_MATMUL_WMMA_MIN_M
// pins it to a fixed value and disables tuning, which is how the per-shape
// crossovers quoted at preferWmmaTier were measured in the first place.
// Clamped to [1, kTierTuneMaxM] so a bad value degrades to a valid tier.
static constexpr int kWmmaMinMDefault = 8;

// Widest M whose tier is decided by measurement. Above this WMMA always wins
// and is not worth timing against: GEMV's cost grows linearly in M without
// bound (it re-reads the weight per row) while WMMA's is flat per row-tile, so
// the gap only widens. The cap also keeps tuning affordable -- one GEMV launch
// on a 311 MB lm_head costs ~1.55 ms per row, so timing it at a prefill M of
// 16k would take ~25 s per launch.
static constexpr int kTierTuneMaxM = 64;

// -1 when unset, meaning "tune the tier instead of using a fixed cutoff".
static int wmmaMinMOverride() {
    static const int m = [] {
#ifdef _WIN32
        // Static-CRT DLLs cannot see the host process environment through
        // getenv; go to Win32 directly (same reason as debug_log.h).
        char buf[16];
        unsigned long n = GetEnvironmentVariableA(
            "HIPDNN_EP_MATMUL_WMMA_MIN_M", buf, sizeof(buf));
        if (n == 0 || n >= sizeof(buf)) return -1;
        int v = atoi(buf);
#else
        const char* s = getenv("HIPDNN_EP_MATMUL_WMMA_MIN_M");
        if (!s) return -1;
        int v = atoi(s);
#endif
        if (v < 1 || v > kTierTuneMaxM) return -1;
        return v;
    }();
    return m;
}

static int wmmaMinM() {
    const int ov = wmmaMinMOverride();
    return ov > 0 ? ov : kWmmaMinMDefault;
}

// Tier tuning is on unless a fixed cutoff is pinned or it is switched off with
// HIPDNN_EP_MATMUL_TIER_TUNE=0.
static bool wmmaTierTuningEnabled() {
    if (wmmaMinMOverride() > 0) return false;
    static const bool on = [] {
#ifdef _WIN32
        char buf[8];
        unsigned long n = GetEnvironmentVariableA(
            "HIPDNN_EP_MATMUL_TIER_TUNE", buf, sizeof(buf));
        return !(n > 0 && buf[0] == '0');
#else
        const char* v = getenv("HIPDNN_EP_MATMUL_TIER_TUNE");
        return !(v && v[0] == '0');
#endif
    }();
    return on;
}

struct GemvMConfig {
    int threads;
    int tile_n;
    int tile_m;
};

// Three parameters, and the one that turned out to dominate is not tile shape:
//
//   threads * tile_n  outstanding B loads per block. This kernel is memory-
//                     LATENCY bound, not occupancy bound, so what matters is
//                     how many loads are in flight. The pre-existing col-major
//                     tune picks threads=512 tile_n=16 for a 4096x151936
//                     lm_head even though K/32 = 128 leaves 3 of 4 threads with
//                     no K work at all; a table capped at 256 threads measured
//                     17-23 ms against that config's 6.3 ms.
//   tile_m            amortises the weight read across rows (traffic ~ 1/tile_m)
//   tile_n            amortises the activation window across columns: each
//                     k-step issues 32 scalar fp16 A loads per row against
//                     tile_m * tile_n * 32 FMAs
//
// So the table sweeps threads widely and keeps tile_m * tile_n in the 16-32
// band, which bounds partial[][]. tile_m = 1 is retained so the tuner can
// always fall back to per-row behaviour.
static constexpr GemvMConfig kGemvMConfigs[] = {
    // tile_m=2 is deliberately narrow (tile_n 2 and 4, not 8 and 16). The wide
    // pairings are the ones that still spill -- 476 bytes/lane at tile_n=8, 1124
    // at tile_n=16 -- so the tuner could never use them, and without a spill-free
    // tile_m=2 its only read-once options at M=2 were tile_m=4 and 8, which
    // compute two to four times the rows they need. That cost 5% at M=2.
    {  64,  8, 1}, {  64, 16, 1}, {  64,  2, 2}, {  64,  4, 2},
    {  64,  4, 4}, {  64,  8, 4}, {  64,  2, 8}, {  64,  4, 8},
    { 128,  8, 1}, { 128, 16, 1}, { 128,  2, 2}, { 128,  4, 2},
    { 128,  4, 4}, { 128,  8, 4}, { 128,  2, 8}, { 128,  4, 8},
    { 256,  8, 1}, { 256, 16, 1}, { 256,  2, 2}, { 256,  4, 2},
    { 256,  4, 4}, { 256,  8, 4}, { 256,  2, 8}, { 256,  4, 8},
    { 512,  8, 1}, { 512, 16, 1}, { 512,  2, 2}, { 512,  4, 2},
    { 512,  4, 4}, { 512,  8, 4}, { 512,  2, 8}, { 512,  4, 8},
    {1024,  8, 1}, {1024, 16, 1}, {1024,  2, 2}, {1024,  4, 2},
    {1024,  4, 4}, {1024,  8, 4}, {1024,  2, 8}, {1024,  4, 8},
    // Wide-tile_n M-blocked variants, appended so the ids above keep meaning
    // (the on-disk tune cache stores ids). Rationale: once the tuner is
    // restricted to one traffic class, what separates candidates is how many B
    // loads are in flight -- threads * tile_n -- and the widest M-blocked tile
    // the table offered was tile_n=8. A read-once tile_m=4 config reaches only
    // ~55 GB/s on a 12288x4096 layer where the tile_m=1 config reaches ~138,
    // which is the signature of too few outstanding loads, not of traffic.
    // These may exceed the register file and spill; that is safe to offer,
    // because within a single traffic class the back-to-back timing below is a
    // valid latency comparison and will simply not pick them.
    {  64, 16, 4}, { 128, 16, 4}, { 256, 16, 4}, { 512, 16, 4},
    {  64,  8, 8}, { 128,  8, 8}, { 256,  8, 8}, { 512,  8, 8},
};
static constexpr int kNumGemvMConfigs =
    static_cast<int>(sizeof(kGemvMConfigs) / sizeof(kGemvMConfigs[0]));

static void launchGemvMConfig(
    int config_id, hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs)
{
    auto& c = kGemvMConfigs[config_id];
    dim3 grid(static_cast<unsigned>((N + c.tile_n - 1) / c.tile_n),
              static_cast<unsigned>((M + c.tile_m - 1) / c.tile_m),
              1);
    dim3 block(c.threads);

    // COL_MAJOR, not row-major. Measured at 512 threads / tile_n 16 / tile_m 1
    // on a 4096x151936 lm_head at M=4, the row-major instantiation of this same
    // kernel costs 11.6 ms against the col-major one's 6.3 ms -- 1.85x, for
    // identical tiling and B indexing. Row-major looked preferable on paper
    // (no A/C transposes, uint8 zero points instead of a nibble->fp16 pass) but
    // it loses far more than those cost, so M-blocking is layered on top of the
    // layout that is actually fast.
#define GEMV_M_CASE(ID, BS, TN, TM)                                            \
    case ID:                                                                   \
        hipLaunchKernelGGL(                                                    \
            HIP_KERNEL_NAME(matmul_nbits_gemv_kernel<BS, TN, true, TM>),        \
            grid, block, 0, stream, A, B, sc, zp, bi, out, M, N, K, bs);        \
        break

#define GEMV_M_BLOCK(BASE, BS)                                                  \
    GEMV_M_CASE(BASE + 0, BS,  8, 1); GEMV_M_CASE(BASE + 1, BS, 16, 1);         \
    GEMV_M_CASE(BASE + 2, BS,  2, 2); GEMV_M_CASE(BASE + 3, BS,  4, 2);         \
    GEMV_M_CASE(BASE + 4, BS,  4, 4); GEMV_M_CASE(BASE + 5, BS,  8, 4);         \
    GEMV_M_CASE(BASE + 6, BS,  2, 8); GEMV_M_CASE(BASE + 7, BS,  4, 8)

    switch (config_id) {
        GEMV_M_BLOCK( 0,   64);
        GEMV_M_BLOCK( 8,  128);
        GEMV_M_BLOCK(16,  256);
        GEMV_M_BLOCK(24,  512);
        GEMV_M_BLOCK(32, 1024);
        GEMV_M_CASE(40,   64, 16, 4); GEMV_M_CASE(41,  128, 16, 4);
        GEMV_M_CASE(42,  256, 16, 4); GEMV_M_CASE(43,  512, 16, 4);
        GEMV_M_CASE(44,   64,  8, 8); GEMV_M_CASE(45,  128,  8, 8);
        GEMV_M_CASE(46,  256,  8, 8); GEMV_M_CASE(47,  512,  8, 8);
        default: break;
    }
#undef GEMV_M_BLOCK
#undef GEMV_M_CASE
}

static int tuneGemvMConfig(
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
    int   best_id = 25;  // 512 threads, tile_n 16, tile_m 1: safe default
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    // Weight reads per call: grid.y = ceil(M/tile_m) threadblocks each stream the
    // whole packed weight, so this is the traffic multiplier and nothing else in
    // the config changes it. Only configs that achieve the minimum are timed.
    //
    // WHY restrict instead of letting the measurement decide: the loop below
    // times 5 back-to-back launches on one weight, so from the second launch on
    // it is resident in the 32 MB Infinity Cache, where re-reading it per row is
    // nearly free. That makes the measurement blind to precisely the quantity
    // that dominates a real forward pass, where every layer's weight arrives
    // cold from DRAM. Measured in-model at M=4 on a 12288x4096 layer: the
    // tuner's own pick costs 41.0 ms against 36.3 ms for a minimum-traffic
    // config, and 58.5 ms for the tile_m=1 config the warm ranking prefers.
    // Comparisons *within* one traffic class are still latency comparisons,
    // which back-to-back timing does handle, so the measurement keeps its job of
    // choosing threads and tile_n -- it just no longer gets to choose traffic.
    int min_reads = 1 << 30;
    for (int cid = 0; cid < kNumGemvMConfigs; cid++) {
        if (kGemvMConfigs[cid].tile_n > N) continue;
        const int reads =
            static_cast<int>((M + kGemvMConfigs[cid].tile_m - 1) /
                             kGemvMConfigs[cid].tile_m);
        if (reads < min_reads) min_reads = reads;
    }

    for (int cid = 0; cid < kNumGemvMConfigs; cid++) {
        // A tile wider than N only adds bounds-checked lanes.
        if (kGemvMConfigs[cid].tile_n > N) continue;
        // A tile taller than M is *not* excluded: at M=3 only tile_m=4 reads the
        // weight once, and at M=5..7 only tile_m=8 does. The rows beyond M are
        // clamped and discarded at the store, so the cost is wasted FMAs against
        // an already-loaded weight -- which is the cheap half of this kernel.
        if ((M + kGemvMConfigs[cid].tile_m - 1) / kGemvMConfigs[cid].tile_m >
            min_reads)
            continue;

        for (int w = 0; w < WARMUP; w++)
            launchGemvMConfig(cid, stream, A, B, sc, zp, bi, out,
                              M, N, K, batch, bs);

        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchGemvMConfig(cid, stream, A, B, sc, zp, bi, out,
                              M, N, K, batch, bs);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   configM[%2d] threads=%3d tile_n=%d tile_m=%d "
            ": %.4f ms\n",
            cid, kGemvMConfigs[cid].threads, kGemvMConfigs[cid].tile_n,
            kGemvMConfigs[cid].tile_m, ms / ITERS);

        if (ms < best_ms) {
            best_ms = ms;
            best_id = cid;
        }
    }

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] GEMV-M autotune: M=%lld N=%lld K=%lld bs=%lld -> "
        "best configM[%d] threads=%d tile_n=%d tile_m=%d (%.4f ms/iter)\n",
        (long long)M, (long long)N, (long long)K, (long long)bs,
        best_id, kGemvMConfigs[best_id].threads,
        kGemvMConfigs[best_id].tile_n, kGemvMConfigs[best_id].tile_m,
        best_ms / ITERS);

    return best_id;
}

static std::unordered_map<std::string, int> gemv_m_cache;
static std::mutex gemv_m_tune_mutex;
static std::once_flag gemv_m_cache_init_flag;
static std::string gemv_m_cache_path;

// HIPDNN_EP_MATMUL_SMALLM_CFG=<id> pins the config, bypassing the autotune.
// The autotune measures 5 back-to-back launches on the first call, which ranks
// configs under conditions a real decode loop does not reproduce -- its winner
// for lm_head at M=4 measured 5.5 ms while the same config measured 14.7 ms in
// steady state. Pinning lets a config be evaluated the way it will be used.
static int gemvMConfigOverride() {
    static const int cfg = [] {
#ifdef _WIN32
        char buf[16];
        unsigned long n = GetEnvironmentVariableA(
            "HIPDNN_EP_MATMUL_SMALLM_CFG", buf, sizeof(buf));
        if (n == 0 || n >= sizeof(buf)) return -1;
        return atoi(buf);
#else
        const char* v = getenv("HIPDNN_EP_MATMUL_SMALLM_CFG");
        return v ? atoi(v) : -1;
#endif
    }();
    return cfg;
}

static int getOrTuneGemvMConfig(
    hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs)
{
    const int forced = gemvMConfigOverride();
    if (forced >= 0 && forced < kNumGemvMConfigs &&
        kGemvMConfigs[forced].tile_m <= M)
        return forced;

    std::string key = std::to_string(M) + "_" + std::to_string(N) + "_" +
                      std::to_string(K) + "_" + std::to_string(bs) + "_" +
                      (zp != nullptr ? "1" : "0");

    std::call_once(gemv_m_cache_init_flag, [] {
        gemv_m_cache_path = getTuneCachePath("gemv_m");
        loadTuneCacheFile(gemv_m_cache_path, gemv_m_cache, kNumGemvMConfigs);
    });

    std::lock_guard<std::mutex> lock(gemv_m_tune_mutex);

    auto it = gemv_m_cache.find(key);
    if (it != gemv_m_cache.end())
        return it->second;

    int best = tuneGemvMConfig(stream, A, B, sc, zp, bi, out,
                               M, N, K, batch, bs);
    gemv_m_cache[key] = best;
    saveTuneCacheFile(gemv_m_cache_path, gemv_m_cache);

    return best;
}

/* -----------------------------------------------------------------------
 * W4A8 dp4a GEMV Autotune + dispatch (M >= 1, K%32==0)
 *
 * Reuses kGemvConfigs + the tune-cache mechanics; a separate cache is kept
 * because the dp4a kernel's per-config ranking differs from the fp path
 * (integer dot vs float FMA changes the occupancy/latency tradeoff).
 * ----------------------------------------------------------------------- */

static void launchDp4aConfig(
    int config_id, hipStream_t stream,
    const int8_t* a_qb, const float* a_scale,
    const uint8_t* B, const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t N, int64_t K, int64_t bs)
{
    auto& c = kGemvConfigs[config_id];
    dim3 grid(static_cast<unsigned>((N + c.tile_n - 1) / c.tile_n), 1u, 1u);
    dim3 block(c.threads);

#define DP4A_CASE(ID, BS, TN) \
    case ID: \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(matmul_nbits_gemv_dp4a_kernel<BS, TN, 1>), \
            grid, block, 0, stream, a_qb, a_scale, B, sc, zp, bi, out, \
            1, N, K, bs); \
        break

    switch (config_id) {
        DP4A_CASE( 0,  32,  1);  DP4A_CASE( 1,  32,  2);
        DP4A_CASE( 2,  32,  4);  DP4A_CASE( 3,  32,  8);
        DP4A_CASE( 4,  64,  1);  DP4A_CASE( 5,  64,  2);
        DP4A_CASE( 6,  64,  4);  DP4A_CASE( 7,  64,  8);
        DP4A_CASE( 8,  64, 16);
        DP4A_CASE( 9, 128,  1);  DP4A_CASE(10, 128,  2);
        DP4A_CASE(11, 128,  4);  DP4A_CASE(12, 128,  8);
        DP4A_CASE(13, 128, 16);
        DP4A_CASE(14, 256,  1);  DP4A_CASE(15, 256,  2);
        DP4A_CASE(16, 256,  4);  DP4A_CASE(17, 256,  8);
        DP4A_CASE(18, 256, 16);
        DP4A_CASE(19, 512,  1);  DP4A_CASE(20, 512,  2);
        DP4A_CASE(21, 512,  4);  DP4A_CASE(22, 512,  8);
        DP4A_CASE(23, 512, 16);
        DP4A_CASE(24, 1024,  1); DP4A_CASE(25, 1024,  2);
        DP4A_CASE(26, 1024,  4); DP4A_CASE(27, 1024,  8);
        DP4A_CASE(28, 1024, 16);
        DP4A_CASE(29,  64, 32);  DP4A_CASE(30, 128, 32);
        DP4A_CASE(31, 256, 32);
        DP4A_CASE(32,  64, 64);  DP4A_CASE(33, 128, 64);
        DP4A_CASE(34, 256, 64);
        default: break;
    }
#undef DP4A_CASE
}

static int tuneDp4aConfig(
    hipStream_t stream,
    const int8_t* a_qb, const float* a_scale,
    const uint8_t* B, const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t N, int64_t K, int64_t bs)
{
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);
    float best_ms = 1e30f;
    int   best_id = 9;  // 128x1 default
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    for (int cid = 0; cid < kNumGemvConfigs; cid++) {
        auto& cfg = kGemvConfigs[cid];
        if (cfg.tile_n >= 64 && N < 2048) continue;
        if (cfg.tile_n >= 32 && cfg.tile_n < 64 && N < 1024) continue;

        for (int w = 0; w < WARMUP; w++)
            launchDp4aConfig(cid, stream, a_qb, a_scale, B, sc, zp, bi, out,
                             N, K, bs);

        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchDp4aConfig(cid, stream, a_qb, a_scale, B, sc, zp, bi, out,
                             N, K, bs);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);
        if (ms < best_ms) { best_ms = ms; best_id = cid; }
    }
    hipEventDestroy(ev0);
    hipEventDestroy(ev1);
    return best_id;
}

static std::unordered_map<std::string, int> dp4a_cache;
static std::mutex dp4a_tune_mutex;
static std::once_flag dp4a_cache_init_flag;
static std::string dp4a_cache_path;

static int getOrTuneDp4aConfig(
    hipStream_t stream,
    const int8_t* a_qb, const float* a_scale,
    const uint8_t* B, const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t N, int64_t K, int64_t bs)
{
    std::string key = std::to_string(N) + "_" + std::to_string(K) + "_" +
                      std::to_string(bs) + "_" + (zp ? "1" : "0");

    std::call_once(dp4a_cache_init_flag, [] {
        dp4a_cache_path = getTuneCachePath("dp4a");
        loadTuneCacheFile(dp4a_cache_path, dp4a_cache, kNumGemvConfigs);
    });

    std::lock_guard<std::mutex> lock(dp4a_tune_mutex);
    auto it = dp4a_cache.find(key);
    if (it != dp4a_cache.end())
        return it->second;

    int best = tuneDp4aConfig(stream, a_qb, a_scale, B, sc, zp, bi, out,
                              N, K, bs);
    dp4a_cache[key] = best;
    saveTuneCacheFile(dp4a_cache_path, dp4a_cache);
    return best;
}

/* -----------------------------------------------------------------------
 * W4A8 dp4a GEMV-M: the same integer-dot GEMV with rows blocked.
 *
 * Why this and not the fp M-blocked path for the verify pass: at M > 1 the fp
 * kernel is already read-once, and Phase H established that it is nonetheless
 * ~2.5x less efficient per byte than the single-row path -- because the nibble
 * unpack sits inside the row loop, so a tile_m=4 block dequantizes every weight
 * four times, and hoisting it costs more live state than it saves. The integer
 * form does not have that trade to make: the unpack is two mask instructions
 * per uint32 and they are already outside the row loop, so a fourth row costs
 * two sudot4 and nothing else.
 *
 * The config table mirrors kGemvMConfigs so the tile shapes are ones already
 * known not to spill, minus the 1024-thread row (K/32 <= 384 here).
 * ----------------------------------------------------------------------- */

static constexpr GemvMConfig kGemvMDp4aConfigs[] = {
    {  64,  8, 1}, {  64, 16, 1}, {  64,  2, 2}, {  64,  4, 2},
    {  64,  4, 4}, {  64,  8, 4}, {  64,  2, 8}, {  64,  4, 8},
    { 128,  8, 1}, { 128, 16, 1}, { 128,  2, 2}, { 128,  4, 2},
    { 128,  4, 4}, { 128,  8, 4}, { 128,  2, 8}, { 128,  4, 8},
    { 256,  8, 1}, { 256, 16, 1}, { 256,  2, 2}, { 256,  4, 2},
    { 256,  4, 4}, { 256,  8, 4}, { 256,  2, 8}, { 256,  4, 8},
    { 512,  8, 1}, { 512, 16, 1}, { 512,  2, 2}, { 512,  4, 2},
    { 512,  4, 4}, { 512,  8, 4}, { 512,  2, 8}, { 512,  4, 8},
    {  64, 16, 4}, { 128, 16, 4}, { 256, 16, 4}, { 512, 16, 4},
    {  64,  8, 8}, { 128,  8, 8}, { 256,  8, 8}, { 512,  8, 8},
};
static constexpr int kNumGemvMDp4aConfigs =
    static_cast<int>(sizeof(kGemvMDp4aConfigs) / sizeof(kGemvMDp4aConfigs[0]));

static void launchDp4aMConfig(
    int config_id, hipStream_t stream,
    const int8_t* a_qb, const float* a_scale,
    const uint8_t* B, const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K, int64_t bs)
{
    auto& c = kGemvMDp4aConfigs[config_id];
    dim3 grid(static_cast<unsigned>((N + c.tile_n - 1) / c.tile_n),
              static_cast<unsigned>((M + c.tile_m - 1) / c.tile_m), 1u);
    dim3 block(c.threads);

    // The (BS, TN, TM) at every case ID below MUST match
    // kGemvMDp4aConfigs[ID]; see launchGemvConfigI8 for what drift costs.
#define DP4A_M_CASE(ID, BS, TN, TM)                                            \
    case ID:                                                                   \
        hipLaunchKernelGGL(                                                    \
            HIP_KERNEL_NAME(matmul_nbits_gemv_dp4a_kernel<BS, TN, TM>),        \
            grid, block, 0, stream, a_qb, a_scale, B, sc, zp, bi, out,         \
            M, N, K, bs);                                                      \
        break

#define DP4A_M_BLOCK(BASE, BS)                                                 \
    DP4A_M_CASE(BASE + 0, BS,  8, 1); DP4A_M_CASE(BASE + 1, BS, 16, 1);        \
    DP4A_M_CASE(BASE + 2, BS,  2, 2); DP4A_M_CASE(BASE + 3, BS,  4, 2);        \
    DP4A_M_CASE(BASE + 4, BS,  4, 4); DP4A_M_CASE(BASE + 5, BS,  8, 4);        \
    DP4A_M_CASE(BASE + 6, BS,  2, 8); DP4A_M_CASE(BASE + 7, BS,  4, 8)

    switch (config_id) {
        DP4A_M_BLOCK( 0,   64);
        DP4A_M_BLOCK( 8,  128);
        DP4A_M_BLOCK(16,  256);
        DP4A_M_BLOCK(24,  512);
        DP4A_M_CASE(32,   64, 16, 4); DP4A_M_CASE(33,  128, 16, 4);
        DP4A_M_CASE(34,  256, 16, 4); DP4A_M_CASE(35,  512, 16, 4);
        DP4A_M_CASE(36,   64,  8, 8); DP4A_M_CASE(37,  128,  8, 8);
        DP4A_M_CASE(38,  256,  8, 8); DP4A_M_CASE(39,  512,  8, 8);
        default: break;
    }
#undef DP4A_M_BLOCK
#undef DP4A_M_CASE
}

static std::unordered_map<std::string, int> dp4a_m_cache;
static std::mutex dp4a_m_tune_mutex;
static std::once_flag dp4a_m_cache_init_flag;
static std::string dp4a_m_cache_path;

static int getOrTuneDp4aMConfig(
    hipStream_t stream,
    const int8_t* a_qb, const float* a_scale,
    const uint8_t* B, const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K, int64_t bs)
{
    std::string key = std::to_string(M) + "_" + std::to_string(N) + "_" +
                      std::to_string(K) + "_" + std::to_string(bs) + "_" +
                      (zp ? "1" : "0");

    std::call_once(dp4a_m_cache_init_flag, [] {
        dp4a_m_cache_path = getTuneCachePath("dp4a_m");
        loadTuneCacheFile(dp4a_m_cache_path, dp4a_m_cache,
                          kNumGemvMDp4aConfigs);
    });

    std::lock_guard<std::mutex> lock(dp4a_m_tune_mutex);
    auto it = dp4a_m_cache.find(key);
    if (it != dp4a_m_cache.end())
        return it->second;

    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);
    float best_ms = 1e30f;
    int   best_id = 29;  // 512 threads, tile_n 8, tile_m 4: read-once default
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    // Only minimum-traffic configs are timed, for the reason spelled out in
    // tuneGemvMConfig: five back-to-back launches leave the weight in the
    // Infinity Cache, so this measurement cannot see traffic, only latency.
    int min_reads = 1 << 30;
    for (int cid = 0; cid < kNumGemvMDp4aConfigs; cid++) {
        if (kGemvMDp4aConfigs[cid].tile_n > N) continue;
        const int reads =
            static_cast<int>((M + kGemvMDp4aConfigs[cid].tile_m - 1) /
                             kGemvMDp4aConfigs[cid].tile_m);
        if (reads < min_reads) min_reads = reads;
    }

    for (int cid = 0; cid < kNumGemvMDp4aConfigs; cid++) {
        if (kGemvMDp4aConfigs[cid].tile_n > N) continue;
        if ((M + kGemvMDp4aConfigs[cid].tile_m - 1) /
                kGemvMDp4aConfigs[cid].tile_m > min_reads)
            continue;

        for (int w = 0; w < WARMUP; w++)
            launchDp4aMConfig(cid, stream, a_qb, a_scale, B, sc, zp, bi, out,
                              M, N, K, bs);
        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchDp4aMConfig(cid, stream, a_qb, a_scale, B, sc, zp, bi, out,
                              M, N, K, bs);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   dp4a configM[%2d] threads=%3d tile_n=%d "
            "tile_m=%d : %.4f ms\n",
            cid, kGemvMDp4aConfigs[cid].threads, kGemvMDp4aConfigs[cid].tile_n,
            kGemvMDp4aConfigs[cid].tile_m, ms / ITERS);

        if (ms < best_ms) { best_ms = ms; best_id = cid; }
    }
    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] dp4a-M autotune: M=%lld N=%lld K=%lld bs=%lld -> "
        "best configM[%d] threads=%d tile_n=%d tile_m=%d (%.4f ms/iter)\n",
        (long long)M, (long long)N, (long long)K, (long long)bs, best_id,
        kGemvMDp4aConfigs[best_id].threads, kGemvMDp4aConfigs[best_id].tile_n,
        kGemvMDp4aConfigs[best_id].tile_m, best_ms / ITERS);

    dp4a_m_cache[key] = best_id;
    saveTuneCacheFile(dp4a_m_cache_path, dp4a_m_cache);
    return best_id;
}

// Public entry: W4A8 dp4a GEMV for M rows (M == 1 is the decode path).
//   A          : fp16 activations [M, K] (row-major, batch==1)
//   B          : packed int4 weights [N, K/2]
//   scales     : fp16 [N, k_blocks]
//   zp_u8      : pre-unpacked uint8 zero points [N, k_blocks] or nullptr (sym)
//   out        : fp16 [M, N]
//   a_qb_scratch/a_scale_scratch : per-session scratch, caller-sized to
//       >= M*K bytes and >= M*k_blocks floats respectively.
// Returns hip status (hipSuccess on success). K must be a multiple of 32.
extern "C" int hip_matmul_nbits_dp4a(
    void* stream_v,
    const void* A, const void* B, const void* scales, const void* zp_u8,
    const void* bias, void* out,
    int64_t M, int64_t N, int64_t K, int64_t block_size,
    void* a_qb_scratch, void* a_scale_scratch)
{
    if (K % 32 != 0) return hipErrorInvalidValue;
    if (M < 1) return hipErrorInvalidValue;
    hipStream_t stream = static_cast<hipStream_t>(stream_v);

    const __half*  Ah   = static_cast<const __half*>(A);
    const uint8_t* Bp   = static_cast<const uint8_t*>(B);
    const __half*  sc   = static_cast<const __half*>(scales);
    const uint8_t* zp   = static_cast<const uint8_t*>(zp_u8);
    const __half*  bi   = static_cast<const __half*>(bias);
    __half*        outp = static_cast<__half*>(out);
    int8_t*        a_qb = static_cast<int8_t*>(a_qb_scratch);
    float*         a_sc = static_cast<float*>(a_scale_scratch);

    const int64_t k_blocks = (K + block_size - 1) / block_size;

    // Step 1: dynamic per-group int8 quantization of the activations.
    // One block per (K-group, row), fixed 128 threads (bs in {32,64,128,256}).
    (void)hipGetLastError();
    hipLaunchKernelGGL(HIP_KERNEL_NAME(matmul_nbits_quant_act_i8_kernel<128>),
                       dim3(static_cast<unsigned>(k_blocks),
                            static_cast<unsigned>(M)),
                       dim3(128), 0,
                       stream, Ah, a_qb, a_sc, K, block_size, k_blocks);

    // Step 2: dp4a GEMV (autotuned config). M == 1 keeps the single-row
    // launcher and its own tune cache, which has more history behind it.
    if (M == 1) {
        int cfg = getOrTuneDp4aConfig(stream, a_qb, a_sc, Bp, sc, zp, bi, outp,
                                      N, K, block_size);
        launchDp4aConfig(cfg, stream, a_qb, a_sc, Bp, sc, zp, bi, outp,
                         N, K, block_size);
    } else {
        int cfg = getOrTuneDp4aMConfig(stream, a_qb, a_sc, Bp, sc, zp, bi, outp,
                                       M, N, K, block_size);
        launchDp4aMConfig(cfg, stream, a_qb, a_sc, Bp, sc, zp, bi, outp,
                          M, N, K, block_size);
    }
    return hipGetLastError();
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
    float best_ms = 1e30f;
    int   best_id = 8; // 128 threads, TILE_N=8 -- safe default for typical N
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 3; // cold iterations; see the note above tuneGemvConfig

    void* evict = gemvEvictAlloc();

    for (int cid = 0; cid < kNumGemvConfigs; cid++) {
        auto& cfg = kGemvConfigs[cid];
        if (cfg.tile_n >= 32 && N < 1024) continue;

        auto launch = [&] {
            launchGemvConfigI8(cid, stream, A, B, sc, zp, bi, out,
                               M, N, K, batch, bs);
        };
        for (int w = 0; w < WARMUP; w++) launch();

        float ms = gemvTimeConfigCold<ITERS>(stream, evict, launch);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   i8 config[%2d] threads=%3d tile_n=%d : %.4f ms\n",
            cid, kGemvConfigs[cid].threads, kGemvConfigs[cid].tile_n,
            ms / ITERS);

        if (ms < best_ms) {
            best_ms = ms;
            best_id = cid;
        }
    }

    if (evict) (void)hipFree(evict);

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
static std::once_flag gemv_i8_cache_init_flag;
static std::string gemv_i8_cache_path;

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

    std::call_once(gemv_i8_cache_init_flag, [] {
        gemv_i8_cache_path = getTuneCachePath("gemv_i8");
        loadTuneCacheFile(gemv_i8_cache_path, gemv_i8_cache, kNumGemvConfigs);
    });

    std::lock_guard<std::mutex> lock(gemv_i8_tune_mutex);

    auto it = gemv_i8_cache.find(key);
    if (it != gemv_i8_cache.end())
        return it->second;

    int best = tuneGemvConfigI8(stream, A, B, sc, zp, bi, out,
                                 M, N, K, batch, bs);
    gemv_i8_cache[key] = best;
    saveTuneCacheFile(gemv_i8_cache_path, gemv_i8_cache);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] GEMV-i8 cache: stored key=\"%s\" -> config[%d] "
        "(cache size=%zu)\n",
        key.c_str(), best, gemv_i8_cache.size());

    return best;
}

/* -----------------------------------------------------------------------
 * GEMV-M (int8) Autotune: config table, dispatch, benchmark, cache
 *
 * The int8 sibling of the GEMV-M section further down, and the reason it
 * exists: on a mixed int4/int8 checkpoint like Qwen3-8B int4-KLD, the int8 ops
 * carry roughly half the weight traffic -- every k/v projection, a third of the
 * attention and MLP projections, and the whole 622 MB lm_head -- and until now
 * none of it was M-blocked. A width-4 verify pass therefore streamed those
 * ~2.6 GB four times while the int4 half streamed its ~2.5 GB once.
 *
 * Same three parameters as the int4 table and the same reasoning about them
 * (threads * tile_n sets outstanding B loads, tile_m sets traffic, tile_n
 * amortises the A window), so the shapes are kept identical. The 1024-thread
 * row is dropped: int8 walks K in 16-byte steps, so K/16 <= 768 for every shape
 * in this model and a 1024-thread block leaves most of itself with no K work.
 * ----------------------------------------------------------------------- */

static constexpr GemvMConfig kGemvMI8Configs[] = {
    {  64,  8, 1}, {  64, 16, 1}, {  64,  2, 2}, {  64,  4, 2},
    {  64,  4, 4}, {  64,  8, 4}, {  64,  2, 8}, {  64,  4, 8},
    { 128,  8, 1}, { 128, 16, 1}, { 128,  2, 2}, { 128,  4, 2},
    { 128,  4, 4}, { 128,  8, 4}, { 128,  2, 8}, { 128,  4, 8},
    { 256,  8, 1}, { 256, 16, 1}, { 256,  2, 2}, { 256,  4, 2},
    { 256,  4, 4}, { 256,  8, 4}, { 256,  2, 8}, { 256,  4, 8},
    { 512,  8, 1}, { 512, 16, 1}, { 512,  2, 2}, { 512,  4, 2},
    { 512,  4, 4}, { 512,  8, 4}, { 512,  2, 8}, { 512,  4, 8},
    // Wide-tile_n M-blocked variants, appended so the ids above keep meaning
    // (the on-disk tune cache stores ids).
    {  64, 16, 4}, { 128, 16, 4}, { 256, 16, 4}, { 512, 16, 4},
    {  64,  8, 8}, { 128,  8, 8}, { 256,  8, 8}, { 512,  8, 8},
};
static constexpr int kNumGemvMI8Configs =
    static_cast<int>(sizeof(kGemvMI8Configs) / sizeof(kGemvMI8Configs[0]));

static void launchGemvMConfigI8(
    int config_id, hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs)
{
    auto& c = kGemvMI8Configs[config_id];
    dim3 grid(static_cast<unsigned>((N + c.tile_n - 1) / c.tile_n),
              static_cast<unsigned>((M + c.tile_m - 1) / c.tile_m),
              static_cast<unsigned>(batch));
    dim3 block(c.threads);

    // The (BS, TN, TM) at every case ID below MUST match kGemvMI8Configs[ID];
    // see the warning on launchGemvConfigI8 for what drift costs.
#define GEMV_MI8_CASE(ID, BS, TN, TM)                                          \
    case ID:                                                                   \
        hipLaunchKernelGGL(                                                    \
            HIP_KERNEL_NAME(matmul_nbits_gemv_kernel_i8<BS, TN, TM>),          \
            grid, block, 0, stream, A, B, sc, zp, bi, out, M, N, K, bs);       \
        break

#define GEMV_MI8_BLOCK(BASE, BS)                                               \
    GEMV_MI8_CASE(BASE + 0, BS,  8, 1); GEMV_MI8_CASE(BASE + 1, BS, 16, 1);    \
    GEMV_MI8_CASE(BASE + 2, BS,  2, 2); GEMV_MI8_CASE(BASE + 3, BS,  4, 2);    \
    GEMV_MI8_CASE(BASE + 4, BS,  4, 4); GEMV_MI8_CASE(BASE + 5, BS,  8, 4);    \
    GEMV_MI8_CASE(BASE + 6, BS,  2, 8); GEMV_MI8_CASE(BASE + 7, BS,  4, 8)

    switch (config_id) {
        GEMV_MI8_BLOCK( 0,   64);
        GEMV_MI8_BLOCK( 8,  128);
        GEMV_MI8_BLOCK(16,  256);
        GEMV_MI8_BLOCK(24,  512);
        GEMV_MI8_CASE(32,   64, 16, 4); GEMV_MI8_CASE(33,  128, 16, 4);
        GEMV_MI8_CASE(34,  256, 16, 4); GEMV_MI8_CASE(35,  512, 16, 4);
        GEMV_MI8_CASE(36,   64,  8, 8); GEMV_MI8_CASE(37,  128,  8, 8);
        GEMV_MI8_CASE(38,  256,  8, 8); GEMV_MI8_CASE(39,  512,  8, 8);
        default: break;
    }
#undef GEMV_MI8_BLOCK
#undef GEMV_MI8_CASE
}

static int tuneGemvMConfigI8(
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
    int   best_id = 29;  // 512 threads, tile_n 8, tile_m 4: read-once default
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    // Weight reads per call: grid.y = ceil(M/tile_m) threadblocks each stream
    // the whole weight, so this is the traffic multiplier and nothing else in
    // the config changes it. Only configs that achieve the minimum are timed --
    // the loop below runs 5 back-to-back launches on one weight, which is
    // resident in the 32 MB Infinity Cache from the second launch on, so it
    // cannot see traffic at all. Comparisons within one traffic class are still
    // valid latency comparisons, which is the job the measurement keeps.
    int min_reads = 1 << 30;
    for (int cid = 0; cid < kNumGemvMI8Configs; cid++) {
        if (kGemvMI8Configs[cid].tile_n > N) continue;
        const int reads =
            static_cast<int>((M + kGemvMI8Configs[cid].tile_m - 1) /
                             kGemvMI8Configs[cid].tile_m);
        if (reads < min_reads) min_reads = reads;
    }

    for (int cid = 0; cid < kNumGemvMI8Configs; cid++) {
        if (kGemvMI8Configs[cid].tile_n > N) continue;
        // A tile taller than M is not excluded: at M=3 only tile_m=4 reads the
        // weight once, and at M=5..7 only tile_m=8 does.
        if ((M + kGemvMI8Configs[cid].tile_m - 1) / kGemvMI8Configs[cid].tile_m >
            min_reads)
            continue;

        for (int w = 0; w < WARMUP; w++)
            launchGemvMConfigI8(cid, stream, A, B, sc, zp, bi, out,
                                M, N, K, batch, bs);

        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchGemvMConfigI8(cid, stream, A, B, sc, zp, bi, out,
                                M, N, K, batch, bs);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   i8 configM[%2d] threads=%3d tile_n=%d "
            "tile_m=%d : %.4f ms\n",
            cid, kGemvMI8Configs[cid].threads, kGemvMI8Configs[cid].tile_n,
            kGemvMI8Configs[cid].tile_m, ms / ITERS);

        if (ms < best_ms) {
            best_ms = ms;
            best_id = cid;
        }
    }

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] GEMV-M-i8 autotune: M=%lld N=%lld K=%lld bs=%lld -> "
        "best configM[%d] threads=%d tile_n=%d tile_m=%d (%.4f ms/iter)\n",
        (long long)M, (long long)N, (long long)K, (long long)bs,
        best_id, kGemvMI8Configs[best_id].threads,
        kGemvMI8Configs[best_id].tile_n, kGemvMI8Configs[best_id].tile_m,
        best_ms / ITERS);

    return best_id;
}

static std::unordered_map<std::string, int> gemv_m_i8_cache;
static std::mutex gemv_m_i8_tune_mutex;
static std::once_flag gemv_m_i8_cache_init_flag;
static std::string gemv_m_i8_cache_path;

// HIPDNN_EP_MATMUL_SMALLM_I8=0 falls back to the per-row int8 path. Separate
// from HIPDNN_EP_MATMUL_SMALLM so the int4 and int8 M-blocked paths can be
// attributed independently; both are on by default.
static bool gemvMBlockedEnabledI8() {
    static const bool enabled = [] {
#ifdef _WIN32
        char buf[8];
        unsigned long n = GetEnvironmentVariableA(
            "HIPDNN_EP_MATMUL_SMALLM_I8", buf, sizeof(buf));
        return !(n > 0 && buf[0] == '0');
#else
        const char* v = getenv("HIPDNN_EP_MATMUL_SMALLM_I8");
        return !(v && v[0] == '0');
#endif
    }();
    return enabled;
}

// HIPDNN_EP_MATMUL_SMALLM_I8_CFG=<id> pins the config, bypassing the autotune,
// so a candidate can be evaluated in a real forward pass rather than under the
// cache-warm conditions the tuner measures in.
static int gemvMI8ConfigOverride() {
    static const int cfg = [] {
#ifdef _WIN32
        char buf[16];
        unsigned long n = GetEnvironmentVariableA(
            "HIPDNN_EP_MATMUL_SMALLM_I8_CFG", buf, sizeof(buf));
        if (n == 0 || n >= sizeof(buf)) return -1;
        return atoi(buf);
#else
        const char* v = getenv("HIPDNN_EP_MATMUL_SMALLM_I8_CFG");
        return v ? atoi(v) : -1;
#endif
    }();
    return cfg;
}

static int getOrTuneGemvMConfigI8(
    hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs)
{
    const int forced = gemvMI8ConfigOverride();
    if (forced >= 0 && forced < kNumGemvMI8Configs)
        return forced;

    std::string key = std::to_string(M) + "_" + std::to_string(N) + "_" +
                      std::to_string(K) + "_" + std::to_string(bs) + "_" +
                      (zp != nullptr ? "1" : "0");

    std::call_once(gemv_m_i8_cache_init_flag, [] {
        gemv_m_i8_cache_path = getTuneCachePath("gemv_m_i8");
        loadTuneCacheFile(gemv_m_i8_cache_path, gemv_m_i8_cache,
                          kNumGemvMI8Configs);
    });

    std::lock_guard<std::mutex> lock(gemv_m_i8_tune_mutex);

    auto it = gemv_m_i8_cache.find(key);
    if (it != gemv_m_i8_cache.end())
        return it->second;

    int best = tuneGemvMConfigI8(stream, A, B, sc, zp, bi, out,
                                 M, N, K, batch, bs);
    gemv_m_i8_cache[key] = best;
    saveTuneCacheFile(gemv_m_i8_cache_path, gemv_m_i8_cache);

    return best;
}

/* =======================================================================
 * Section 1c: Naive uint3 fallback kernel (row-major, any alignment, batched)
 *
 * Custom bits=3 packing (NOT part of the ONNX MatMulNBits nibble/byte
 * convention used by bits=4/8 above): weights are packed as a *continuous*
 * per-row 3-bit bitstream — value k occupies bits [3k, 3k+3) of row n's
 * bitstream, LSB-first. Row byte stride = ceil(K*3/8) bytes. This is the
 * natural bit-exact packing for a bit-width that doesn't divide 8 evenly,
 * and keeps B ~25% smaller than the uint4 layout (3/8 byte vs 4/8 byte per
 * weight) — the whole point of going below 4 bits is bandwidth, so we don't
 * waste a byte per weight the way the bits=8 path does.
 *
 * Zero points (if present) are plain uint8, one byte per group (NOT
 * bit-packed like bits=4's nibble zp) — 3-bit-packing the zp stream too
 * would only save a few dozen bytes total and isn't worth a dedicated
 * unpack kernel for this exploratory path. Default zp = 2^(bits-1) = 4.
 * ======================================================================= */

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

/* =======================================================================
 * Section 1d: GEMV-style kernel for uint3 weights (autotuned, vectorized)
 *
 * Fast-path precondition: K % 32 == 0 AND block_size is a power of two
 * >= 32. block_size >= 32 matters because one scale/zp is read per 32-lane
 * chunk (grp = k32 >> bs_shift computed once for the whole chunk) — the
 * same implicit precondition the int4 GEMV kernel above has for its own
 * 32-nibble chunks. Smaller block_size falls back to the naive kernel.
 *
 * 32 elements = 96 bits = 12 bytes pack exactly into 3 x uint32 words,
 * loaded as one 96-bit `uint3` vector load per column. Every chunk's byte
 * offset (= chunk_idx * 12) is a multiple of 4, so the load is always
 * naturally aligned despite the "odd" 12-byte stride.
 *
 * Per-lane extraction: value i (0..31) occupies bits [3i, 3i+3) of the
 * 96-bit {w0,w1,w2} stream. Only lanes i=10 (straddles w0/w1) and i=21
 * (straddles w1/w2) cross a word boundary; every other lane is a single
 * shift+mask. The inner loop is fully unrolled over a compile-time index,
 * so the boundary-check branch folds away at compile time for the 30
 * non-crossing lanes.
 *
 * Reduction mirrors the int4 GEMV kernel's compile-time WARP_SIZE=32 path
 * (measured faster than a runtime __builtin_amdgcn_wavefrontsize() query
 * on RDNA3, per that kernel's notes).
 * ======================================================================= */

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

/* -----------------------------------------------------------------------
 * GEMV (uint3) Autotune: dispatch, benchmark, cache
 *
 * Reuses kGemvConfigs / GemvConfig from the int4 autotune above (same
 * (BLOCK_SIZE, TILE_N) search space), but launches matmul_nbits_gemv_kernel_u3
 * and caches into a dedicated map — the u3 kernel's per-config performance
 * ranking is unrelated to int4/int8 (different bytes/weight, different
 * unpack cost), so sharing a cache key space would silently reuse a
 * mistuned config.
 * ----------------------------------------------------------------------- */

static void launchGemvConfigU3(
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

#define GEMV_U3_CASE(ID, BS, TN) \
    case ID: \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(matmul_nbits_gemv_kernel_u3<BS, TN>), \
            grid, block, 0, stream, A, B, sc, zp, bi, out, M, N, K, bs); \
        break

    switch (config_id) {
        GEMV_U3_CASE( 0,  32,  1);  GEMV_U3_CASE( 1,  32,  2);
        GEMV_U3_CASE( 2,  32,  4);  GEMV_U3_CASE( 3,  32,  8);
        GEMV_U3_CASE( 4,  64,  1);  GEMV_U3_CASE( 5,  64,  2);
        GEMV_U3_CASE( 6,  64,  4);  GEMV_U3_CASE( 7,  64,  8);
        GEMV_U3_CASE( 8,  64, 16);
        GEMV_U3_CASE( 9, 128,  1);  GEMV_U3_CASE(10, 128,  2);
        GEMV_U3_CASE(11, 128,  4);  GEMV_U3_CASE(12, 128,  8);
        GEMV_U3_CASE(13, 128, 16);
        GEMV_U3_CASE(14, 256,  1);  GEMV_U3_CASE(15, 256,  2);
        GEMV_U3_CASE(16, 256,  4);  GEMV_U3_CASE(17, 256,  8);
        GEMV_U3_CASE(18, 256, 16);
        GEMV_U3_CASE(19, 512,  1);  GEMV_U3_CASE(20, 512,  2);
        GEMV_U3_CASE(21, 512,  4);  GEMV_U3_CASE(22, 512,  8);
        GEMV_U3_CASE(23, 512, 16);
        GEMV_U3_CASE(24, 1024,  1); GEMV_U3_CASE(25, 1024,  2);
        GEMV_U3_CASE(26, 1024,  4); GEMV_U3_CASE(27, 1024,  8);
        GEMV_U3_CASE(28, 1024, 16);
        GEMV_U3_CASE(29,  64, 32);  GEMV_U3_CASE(30, 128, 32);
        GEMV_U3_CASE(31, 256, 32);
        GEMV_U3_CASE(32,  64, 64);  GEMV_U3_CASE(33, 128, 64);
        GEMV_U3_CASE(34, 256, 64);
        default: break;
    }
#undef GEMV_U3_CASE
}

static int tuneGemvConfigU3(
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
    int   best_id = 8;
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    for (int cid = 0; cid < kNumGemvConfigs; cid++) {
        auto& cfg = kGemvConfigs[cid];
        if (cfg.tile_n >= 32 && N < 1024) continue;

        for (int w = 0; w < WARMUP; w++)
            launchGemvConfigU3(cid, stream, A, B, sc, zp, bi, out,
                               M, N, K, batch, bs);

        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchGemvConfigU3(cid, stream, A, B, sc, zp, bi, out,
                               M, N, K, batch, bs);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   u3 config[%2d] threads=%3d tile_n=%d : %.4f ms\n",
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
        "[custom_kernels] GEMV-u3 autotune: M=%lld N=%lld K=%lld bs=%lld -> "
        "best config[%d] threads=%d tile_n=%d (%.4f ms/iter)\n",
        (long long)M, (long long)N, (long long)K, (long long)bs,
        best_id, kGemvConfigs[best_id].threads,
        kGemvConfigs[best_id].tile_n, best_ms / ITERS);

    return best_id;
}

static std::unordered_map<std::string, int> gemv_u3_cache;
static std::mutex gemv_u3_tune_mutex;

static std::string gemvU3CacheKey(int64_t M, int64_t N, int64_t K,
                                  int64_t bs, bool has_zp) {
    return std::to_string(M) + "_" + std::to_string(N) + "_" +
           std::to_string(K) + "_" + std::to_string(bs) + "_" +
           (has_zp ? "1" : "0");
}

static int getOrTuneGemvConfigU3(
    hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs)
{
    const bool has_zp = (zp != nullptr);
    std::string key = gemvU3CacheKey(M, N, K, bs, has_zp);

    std::lock_guard<std::mutex> lock(gemv_u3_tune_mutex);

    auto it = gemv_u3_cache.find(key);
    if (it != gemv_u3_cache.end())
        return it->second;

    int best = tuneGemvConfigU3(stream, A, B, sc, zp, bi, out,
                                M, N, K, batch, bs);
    gemv_u3_cache[key] = best;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] GEMV-u3 cache: stored key=\"%s\" -> config[%d] "
        "(cache size=%zu)\n",
        key.c_str(), best, gemv_u3_cache.size());

    return best;
}

/* =======================================================================
 * Section 1e: Naive uint2 fallback kernel (row-major, any alignment, batched)
 *
 * bits=2 packing: 4 values per byte, LSB-first (value k occupies bits
 * [2k, 2k+2) of row n's byte stream). Because 2 divides 8 evenly, this is
 * simultaneously the ONNX MatMulNBits blockwise convention (blob_size =
 * block_size*2/8 = block_size/4 bytes per block) AND a plain continuous
 * per-row bitstream -- when block_size % 4 == 0 the two layouts are
 * byte-identical, so no cross-byte straddling ever occurs (max shift 6 ->
 * value bits [6,8), always inside one byte). Row byte stride = ceil(K/4).
 *
 * Zero points (if present) are plain uint8, one byte per group (mirroring
 * the u3 path). Default zp = 2^(bits-1) = 2.
 * ======================================================================= */

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

/* =======================================================================
 * Section 1f: GEMV-style kernel for uint2 weights (autotuned, vectorized)
 *
 * Fast-path precondition: K % 32 == 0 AND block_size is a power of two
 * >= 32 (mirrors u3/u4). 32 elements = 64 bits = 8 bytes pack exactly into
 * 2 x uint32, loaded as one 64-bit `uint2` vector load per column. Every
 * chunk's byte offset (= chunk_idx * 8) is a multiple of 8, so the load is
 * always naturally aligned.
 *
 * Per-lane extraction: value i (0..31) occupies bits [2i, 2i+2) of the
 * 64-bit {w0,w1} stream. Because 2 divides 32 evenly, value 16 starts
 * exactly at bit 32 (the word boundary) -- NO value ever straddles two
 * words, so every lane is a single shift+mask (simpler than u3's uint3
 * loop, which had to special-case lanes 10 and 21).
 * ======================================================================= */

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

/* -----------------------------------------------------------------------
 * GEMV (uint2) Autotune: dispatch, benchmark, cache. Reuses kGemvConfigs
 * from the int4 autotune (same (BLOCK_SIZE, TILE_N) search space), caches
 * into a dedicated map -- the u2 decode cost differs from u3/u4.
 * ----------------------------------------------------------------------- */

static void launchGemvConfigU2(
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

#define GEMV_U2_CASE(ID, BS, TN) \
    case ID: \
        hipLaunchKernelGGL( \
            HIP_KERNEL_NAME(matmul_nbits_gemv_kernel_u2<BS, TN>), \
            grid, block, 0, stream, A, B, sc, zp, bi, out, M, N, K, bs); \
        break

    switch (config_id) {
        GEMV_U2_CASE( 0,  32,  1);  GEMV_U2_CASE( 1,  32,  2);
        GEMV_U2_CASE( 2,  32,  4);  GEMV_U2_CASE( 3,  32,  8);
        GEMV_U2_CASE( 4,  64,  1);  GEMV_U2_CASE( 5,  64,  2);
        GEMV_U2_CASE( 6,  64,  4);  GEMV_U2_CASE( 7,  64,  8);
        GEMV_U2_CASE( 8,  64, 16);
        GEMV_U2_CASE( 9, 128,  1);  GEMV_U2_CASE(10, 128,  2);
        GEMV_U2_CASE(11, 128,  4);  GEMV_U2_CASE(12, 128,  8);
        GEMV_U2_CASE(13, 128, 16);
        GEMV_U2_CASE(14, 256,  1);  GEMV_U2_CASE(15, 256,  2);
        GEMV_U2_CASE(16, 256,  4);  GEMV_U2_CASE(17, 256,  8);
        GEMV_U2_CASE(18, 256, 16);
        GEMV_U2_CASE(19, 512,  1);  GEMV_U2_CASE(20, 512,  2);
        GEMV_U2_CASE(21, 512,  4);  GEMV_U2_CASE(22, 512,  8);
        GEMV_U2_CASE(23, 512, 16);
        GEMV_U2_CASE(24, 1024,  1); GEMV_U2_CASE(25, 1024,  2);
        GEMV_U2_CASE(26, 1024,  4); GEMV_U2_CASE(27, 1024,  8);
        GEMV_U2_CASE(28, 1024, 16);
        GEMV_U2_CASE(29,  64, 32);  GEMV_U2_CASE(30, 128, 32);
        GEMV_U2_CASE(31, 256, 32);
        GEMV_U2_CASE(32,  64, 64);  GEMV_U2_CASE(33, 128, 64);
        GEMV_U2_CASE(34, 256, 64);
        default: break;
    }
#undef GEMV_U2_CASE
}

static int tuneGemvConfigU2(
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
    int   best_id = 8;
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    for (int cid = 0; cid < kNumGemvConfigs; cid++) {
        auto& cfg = kGemvConfigs[cid];
        if (cfg.tile_n >= 32 && N < 1024) continue;

        for (int w = 0; w < WARMUP; w++)
            launchGemvConfigU2(cid, stream, A, B, sc, zp, bi, out,
                               M, N, K, batch, bs);

        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchGemvConfigU2(cid, stream, A, B, sc, zp, bi, out,
                               M, N, K, batch, bs);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   u2 config[%2d] threads=%3d tile_n=%d : %.4f ms\n",
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
        "[custom_kernels] GEMV-u2 autotune: M=%lld N=%lld K=%lld bs=%lld -> "
        "best config[%d] threads=%d tile_n=%d (%.4f ms/iter)\n",
        (long long)M, (long long)N, (long long)K, (long long)bs,
        best_id, kGemvConfigs[best_id].threads,
        kGemvConfigs[best_id].tile_n, best_ms / ITERS);

    return best_id;
}

static std::unordered_map<std::string, int> gemv_u2_cache;
static std::mutex gemv_u2_tune_mutex;

static std::string gemvU2CacheKey(int64_t M, int64_t N, int64_t K,
                                  int64_t bs, bool has_zp) {
    return std::to_string(M) + "_" + std::to_string(N) + "_" +
           std::to_string(K) + "_" + std::to_string(bs) + "_" +
           (has_zp ? "1" : "0");
}

static int getOrTuneGemvConfigU2(
    hipStream_t stream,
    const __half* A, const uint8_t* B,
    const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t M, int64_t N, int64_t K,
    int64_t batch, int64_t bs)
{
    const bool has_zp = (zp != nullptr);
    std::string key = gemvU2CacheKey(M, N, K, bs, has_zp);

    std::lock_guard<std::mutex> lock(gemv_u2_tune_mutex);

    auto it = gemv_u2_cache.find(key);
    if (it != gemv_u2_cache.end())
        return it->second;

    int best = tuneGemvConfigU2(stream, A, B, sc, zp, bi, out,
                                M, N, K, batch, bs);
    gemv_u2_cache[key] = best;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] GEMV-u2 cache: stored key=\"%s\" -> config[%d] "
        "(cache size=%zu)\n",
        key.c_str(), best, gemv_u2_cache.size());

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
 * Section 2b: WMMA kernel (supports both fused dequant and fp16 B)
 * ======================================================================= */

typedef _Float16 half16 __attribute__((ext_vector_type(16)));
typedef float    float8 __attribute__((ext_vector_type(8)));

// The WMMA builtin below is RDNA3-family (wave32) only. Where __has_builtin
// reports it absent, trap instead so this file still compiles and a WMMA
// launch on an unsupported arch aborts loudly rather than returning garbage.
#if !defined(__has_builtin) ||                                                  \
    !__has_builtin(__builtin_amdgcn_wmma_f32_16x16x16_f16_w32)
static __device__ __forceinline__ float8
hipdnn_wmma_f32_16x16x16_f16_w32_unavailable(half16, half16, float8 c) {
  __builtin_trap();
  return c;
}
#define __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c)                     \
  hipdnn_wmma_f32_16x16x16_f16_w32_unavailable((a), (b), (c))
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

// K-tile width of a config, mirroring BK_T inside GemmFp16U4Impl.
static constexpr int wmmaConfigBkT(const WmmaConfig& c) {
    return c.bm / (c.wt_m * c.wt_n);
}

// Whether a config may serve this shape at all. Both the autotune and the
// launch consult this, so a config that cannot produce the right answer is
// never timed and never cached.
//
// K % BK_T == 0 is a correctness requirement, not a preference: the kernel
// walks K in whole BK_T-wide tiles with no tail predication, so a partial
// final tile reads past the end of the dequantized B and accumulates garbage.
// The dispatcher's own gate is only K % 32 == 0, which is enough for every
// config except the two BN=32 (WT2x1) tile shapes -- those have BK_T = 64.
// The gap was invisible for a long time because BN=32 is only considered when
// N <= 64, and no shape in a real checkpoint is that narrow: it reproduces at
// K=32 or K=96 with N=64, where it cost ~97% relative error.
static bool wmmaConfigSupported(const WmmaConfig& c, int N, int K) {
    if (c.bn < 64 && N > 2 * c.bn) return false;
    if (K % wmmaConfigBkT(c) != 0) return false;
    return true;
}

// Config to fall back on when the requested one cannot serve the shape.
// Config 9 ({128, 64, sw=4, fused}, BK_T = 32) is the historical default and
// is supported for every K the dispatcher lets through; the scan is there so
// this still returns something valid if the table is reordered.
static int wmmaFallbackConfig(int N, int K) {
    if (wmmaConfigSupported(kWmmaConfigs[9], N, K)) return 9;
    for (int cid = 0; cid < kNumWmmaConfigs; cid++)
        if (wmmaConfigSupported(kWmmaConfigs[cid], N, K)) return cid;
    return 9;
}

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
    if (!wmmaConfigSupported(kWmmaConfigs[config_id], N, K))
        config_id = wmmaFallbackConfig(N, K);

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
    int   best_id = wmmaFallbackConfig(N, K);
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    for (int cid = 0; cid < kNumWmmaConfigs; cid++) {
        auto& cfg = kWmmaConfigs[cid];
        if (!wmmaConfigSupported(cfg, N, K)) continue;

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

/* -----------------------------------------------------------------------
 * Section 2b-tier: WMMA-vs-GEMV tier selection for int4, by measurement
 *
 * Both tiers compute the same row-major output from the same inputs for any
 * M >= 2, so which one to use is purely a timing question -- and a fixed M
 * cutoff answers it with a guess that is wrong in both directions. Measured
 * crossovers on gfx1151 (int4, block 32), GEMV vs WMMA at equal M:
 *
 *   shape                    weight   GEMV wins   WMMA wins   crossover
 *   lm_head   4096x151936    311 MB   M <= 4      M >= 5      5
 *   mlp_down  12288x4096      25 MB   M <= 16     M >= 24     ~17
 *   attn_qkv  4096x4096        8 MB   M <= 16     M >= 24     ~22
 *   mlp_gate  4096x12288      25 MB   M <= 24     M >= 32     ~28
 *
 * The old fixed cutoff of 8 was too high for lm_head (WMMA is 1.39x faster at
 * M=7, which it never got to run) and far too low for the other three (GEMV is
 * 1.7-2.2x faster at M=8, which it was no longer allowed to run).
 *
 * The spread has a cause: GEMV re-reads the whole packed weight once per output
 * row, so its cost is M * W / B_eff, and B_eff depends on whether W fits the
 * last-level cache. lm_head at 311 MB cannot, so every row pays a full DRAM
 * pass (measured ~200 GB/s aggregate, ~79% of the 256 GB/s bus, flat in M). The
 * 8-25 MB shapes do fit, and their aggregate re-read rate measures 300-410 GB/s
 * -- above bus peak, i.e. served from cache -- so extra rows are nearly free and
 * GEMV stays ahead much longer. WMMA reads the weight once but computes a whole
 * row-tile whatever M is, giving it a per-shape fixed floor.
 *
 * That is enough to explain the ordering but not to predict it: mlp_gate and
 * mlp_down are both 25 MB yet cross at 28 and 17, because WMMA's floor also
 * depends on how much N-parallelism the grid has. So this does not become a
 * formula. It becomes a measurement, cached per shape like every other tuning
 * decision in this file.
 * ----------------------------------------------------------------------- */

// Median wall time of ITERS launches, each isolated by a stream sync.
//
// WHY isolate: timing a burst of back-to-back launches and dividing lets each
// launch's tail overlap the next one's head, which flatters exactly the
// low-occupancy candidates this decision has to discriminate between. Measured
// on a 4096x151936 lm_head at M=4, burst timing reported 5.5 ms/iter for a
// configuration that costs 14.7 ms per step in a real decode loop -- a 2.7x
// error, and more than enough to invert a tier choice. A decode step waits for
// its own result, so the tuner has to as well.
static float timeLaunchIsolated(hipStream_t stream,
                                const std::function<void()>& launch,
                                int iters = 5) {
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);

    launch();                       // warm up JIT, caches, and any lazy alloc
    hipStreamSynchronize(stream);

    float samples[16];
    if (iters > 16) iters = 16;
    for (int i = 0; i < iters; i++) {
        hipEventRecord(ev0, stream);
        launch();
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);
        hipEventElapsedTime(&samples[i], ev0, ev1);
    }

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    // Median, not min: min rewards a lucky run that found the weight already
    // resident, which is not what a steady decode loop sees.
    std::sort(samples, samples + iters);
    return samples[iters / 2];
}

static std::unordered_map<std::string, int> tier_cache;
static std::mutex tier_tune_mutex;
static std::once_flag tier_cache_init_flag;

static const std::string& getTierCachePath() {
    static std::string path = [] {
        const std::string& wmma = getWmmaCachePath();
        if (wmma.empty()) return std::string();
        // Same directory and arch suffix as the WMMA cache, different file.
        std::string p = wmma;
        const std::string needle = "morphizen_wmma_cache_";
        size_t at = p.find(needle);
        if (at == std::string::npos) return std::string();
        p.replace(at, needle.size(), "morphizen_tier_cache_");
        return p;
    }();
    return path;
}

static void loadTierCache() {
    const std::string& path = getTierCachePath();
    if (path.empty()) return;

    FILE* f = fopen(path.c_str(), "r");
    if (!f) return;

    char ts_line[64];
    if (!fgets(ts_line, sizeof(ts_line), f)) { fclose(f); return; }
    size_t ts_len = strlen(ts_line);
    if (ts_len > 0 && ts_line[ts_len - 1] == '\n') ts_line[ts_len - 1] = '\0';
    if (strcmp(ts_line, kWmmaBuildTimestamp) != 0) {
        fclose(f);
        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels] tier cache: build mismatch (file=\"%s\", "
            "binary=\"%s\"), will re-tune\n", ts_line, kWmmaBuildTimestamp);
        return;
    }

    char key[128];
    int choice;
    int loaded = 0;
    while (fscanf(f, "%127s %d", key, &choice) == 2) {
        if (choice == 0 || choice == 1) {
            tier_cache[key] = choice;
            loaded++;
        }
    }
    fclose(f);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] tier cache: loaded %d entries from %s\n",
        loaded, path.c_str());
}

static void saveTierCache() {
    const std::string& path = getTierCachePath();
    if (path.empty()) return;

    FILE* f = fopen(path.c_str(), "w");
    if (!f) return;

    fprintf(f, "%s\n", kWmmaBuildTimestamp);
    for (const auto& entry : tier_cache)
        fprintf(f, "%s %d\n", entry.first.c_str(), entry.second);
    fclose(f);
}

// Returns true if WMMA should serve this shape, false for the col-major GEMV.
// Times both once per (M,N,K,gs,zp) and remembers the answer.
static bool preferWmmaTier(hipStream_t stream,
                           int M, int N, int K, int gs, bool has_zp,
                           const std::function<void()>& wmma_launch,
                           const std::function<void()>& gemv_launch) {
    std::call_once(tier_cache_init_flag, loadTierCache);

    std::string key = wmmaCacheKey(M, N, K, gs, has_zp);

    std::lock_guard<std::mutex> lock(tier_tune_mutex);

    auto it = tier_cache.find(key);
    if (it != tier_cache.end())
        return it->second != 0;

    float wmma_ms = timeLaunchIsolated(stream, wmma_launch);
    float gemv_ms = timeLaunchIsolated(stream, gemv_launch);

    // Ties go to WMMA: it is the path whose cost stays flat as M grows, so it
    // is the safer side of a coin flip if M rises later in the same shape.
    bool wmma = wmma_ms <= gemv_ms;
    tier_cache[key] = wmma ? 1 : 0;
    saveTierCache();

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] tier autotune: M=%d N=%d K=%d gs=%d zp=%s -> %s "
        "(wmma %.4f ms vs gemv %.4f ms)\n",
        M, N, K, gs, has_zp ? "yes" : "no", wmma ? "WMMA" : "GEMV",
        wmma_ms, gemv_ms);

    return wmma;
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
 * Section 2d: WMMA kernel for uint3 weights (fused dequant)
 *
 * Mirrors GemmFp16I8Impl above, but decodes the continuous 3-bit bitstream
 * (Section 1c/1d) instead of one-byte-per-weight int8. This is the piece
 * that was missing before: bits=3 previously only had a GEMV + naive
 * fallback (no tensor-core path), so any M >= kBlockDim shape paid for a
 * scalar/GEMV kernel with no data reuse across M rows -- ~20x slower than
 * the uint4 WMMA path on the same shape.
 *
 * ---- Alignment / correctness analysis (the part that actually matters
 * here, since unlike uint4's clean nibble-per-byte packing, 3 bits doesn't
 * divide 8 evenly) ----
 *
 * Every thread decodes an 8-element K-strip per K-step, exactly like the
 * int4/int8 paths (`b_k8` is always a multiple of 8, and the K-loop stride
 * `t` is always a multiple of BK_T, which is itself always a multiple of
 * 8 for every tile config used here -- see kWmmaU3Configs below). The key
 * fact that makes this safe AND simple:
 *
 *     8 values x 3 bits = 24 bits = EXACTLY 3 bytes, no remainder.
 *
 * So for any k0 with k0 % 8 == 0, the 8 values [k0, k0+8) occupy bits
 * [3*k0, 3*k0+24) = bytes [3*k0/8, 3*k0/8 + 3) of the row's bitstream --
 * always a whole number of bytes, no partial-byte straddling at the
 * chunk's own boundaries. And because 24 bits fit inside a single 32-bit
 * register, extracting value i via `(chunk >> 3*i) & 0x7` for i in [0,8)
 * never crosses a *register* boundary either (max shift is 3*7=21, value
 * needs bits [21,24) -- well inside 32). This is simpler than the GEMV
 * kernel's 32-element/96-bit chunks, which DO need special-casing at
 * lanes 10 and 21 where a value straddles two 32-bit words; picking an
 * 8-element granularity here (to match the existing b_k8 convention) sidesteps
 * that entirely.
 *
 * The per-thread byte offset is loaded as a 16-bit read + an 8-bit read
 * (not a single 32-bit "fast" load) specifically to avoid ever touching a
 * 4th byte. A quick bound check confirms this is necessary: the last
 * 8-element chunk of the last row (n=N-1, k0=K-8) sits at byte offset
 * (N-1)*row_bytes + (K-8)*3/8, and reading 3 bytes from there touches
 * exactly up to index N*row_bytes - 1 -- the very last valid byte of the
 * whole B buffer. A naive 4-byte load at that offset would read one byte
 * past the end of the allocation (UB / potential fault). The 2-byte +
 * 1-byte split costs one extra instruction per 8-element chunk but is
 * unconditionally safe for every row, not just interior ones.
 *
 *   B      : uint8 [N, ceil(K*3/8)]              (continuous 3-bit stream)
 *   scales : fp16  [N, num_groups_k]
 *   zeros  : uint8 [N, num_groups_k]             (or NULL, default zp=4)
 *
 * Everything else (A load/prefetch, shared-memory double buffering, WMMA
 * compute, grid swizzle, boundary-checked writeback) is byte-for-byte
 * identical to GemmFp16I8Impl -- only the B decode differs.
 * ======================================================================= */

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
 * WMMA-u3 Autotune: same fused-only tile subset as WMMA-i8 (kWmmaI8Configs
 * only uses BM in {64,128} with default WT_M=WT_N=2, so BK_T = BM/4 is
 * always in {16,32} -- both divide 32, so the block_size>=32 && %32==0
 * dispatch gate below guarantees group_size compatibility with every
 * config in this table; no per-config skip logic needed).
 * ----------------------------------------------------------------------- */

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

static void launchDequantU3(
    hipStream_t stream,
    const unsigned char* B, const _Float16* scales,
    const unsigned char* zeros, _Float16* B_fp16,
    int N, int K, int gs, int ngk)
{
    constexpr int DQ_THR = 256;
    int k_groups = (K + DQ_THR * 8 - 1) / (DQ_THR * 8);
    dim3 dq_grid(static_cast<unsigned>(k_groups), static_cast<unsigned>(N));
    dim3 dq_block(DQ_THR);
    hipLaunchKernelGGL(dequant_u3_to_fp16, dq_grid, dq_block, 0, stream,
                       B, scales, zeros, B_fp16, N, K, gs, ngk);
}

struct WmmaConfigU3 {
    int  bm, bn;
    int  swizzle_n;
    bool fused = true;   // true: fused inline-decode; false: dequant + fp16 GEMM
    int  wt_m = 2, wt_n = 2;
};

static constexpr WmmaConfigU3 kWmmaU3Configs[] = {
    // --- fused inline-decode (bandwidth-optimized, best for small/medium M) ---
    { 64,  64,  2}, { 64,  64,  4}, { 64,  64,  8}, { 64,  64, 16},
    { 64, 128,  2}, { 64, 128,  4}, { 64, 128,  8}, { 64, 128, 16},
    {128,  64,  2}, {128,  64,  4}, {128,  64,  8}, {128,  64, 16},
    {128, 128,  2}, {128, 128,  4}, {128, 128,  8}, {128, 128, 16},
    {128, 128, 32}, {128, 128, 64},
    {128, 256,  4}, {128, 256,  8}, {128, 256, 16}, {128, 256, 32},
    // --- fused inline-decode, 256x128 WT4 (higher compute-to-smem ratio;
    //     matches the bits=4 fused winner on large-K compute-bound GEMMs) ---
    {256, 128,  4, true, 4, 2}, {256, 128,  8, true, 4, 2},
    {256, 128, 16, true, 4, 2}, {256, 128, 32, true, 4, 2},
    // --- separate dequant + plain fp16 WMMA GEMM (peak compute for large M;
    //     reuses the bits=4 MatMulNBitsFp16GEMM kernels, incl. 256x128 WT4) ---
    { 64, 128,  8, false}, {128,  64,  8, false},
    {128, 128,  2, false}, {128, 128,  8, false}, {128, 128, 16, false},
    {128, 128, 32, false},
    {128, 256,  4, false}, {128, 256,  8, false}, {128, 256, 16, false},
    {128, 256, 32, false},
    {256, 128,  4, false, 4, 2}, {256, 128,  8, false, 4, 2},
    {256, 128, 16, false, 4, 2}, {256, 128, 32, false, 4, 2},
};
static constexpr int kNumWmmaU3Configs =
    static_cast<int>(sizeof(kWmmaU3Configs) / sizeof(kWmmaU3Configs[0]));

static void launchWmmaConfigU3(
    int config_id, hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const unsigned char* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp,
    _Float16* B_fp16_buf = nullptr)
{
    auto& c = kWmmaU3Configs[config_id];
    bool need_bc = (M % c.bm != 0) || (N % c.bn != 0);

    /* Separate-dequant path: decode the 3-bit stream to a dense fp16 buffer,
     * then run the plain fp16 WMMA GEMM shared with the bits=4 path. The
     * decode leaves the hot GEMM loop free of any bit-unpacking, letting the
     * WMMA units hit peak throughput on compute-bound (large-M) shapes. */
    if (!c.fused) {
        launchDequantU3(stream, B, scales, zeros, B_fp16_buf, N, K, gs, ngk);

#define U3_FP16_GEMM_LAUNCH(BM_V, BN_V, BC_V) do {                            \
    constexpr int WM_  = BM_V / (2 * 16);                                     \
    constexpr int WN_  = BN_V / (2 * 16);                                     \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    hipLaunchKernelGGL(                                                       \
        HIP_KERNEL_NAME(MatMulNBitsFp16GEMM<BM_V, BN_V, 2, 2, 8, BC_V>),      \
        grid_, block_, 0, stream,                                             \
        M, N, K, A, lda, B_fp16_buf, C, ldc, c.swizzle_n);                    \
} while (0)

#define U3_FP16_TILE_CASE(BM_V, BN_V) \
    if (need_bc) U3_FP16_GEMM_LAUNCH(BM_V, BN_V, true);  \
    else         U3_FP16_GEMM_LAUNCH(BM_V, BN_V, false)

#define U3_FP16_GEMM_LAUNCH_WT4(BM_V, BN_V, BC_V) do {                        \
    constexpr int WM_  = BM_V / (4 * 16);                                     \
    constexpr int WN_  = BN_V / (2 * 16);                                     \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    hipLaunchKernelGGL(                                                       \
        HIP_KERNEL_NAME(MatMulNBitsFp16GEMM<BM_V, BN_V, 4, 2, 6, BC_V>),      \
        grid_, block_, 0, stream,                                             \
        M, N, K, A, lda, B_fp16_buf, C, ldc, c.swizzle_n);                    \
} while (0)

#define U3_FP16_TILE_CASE_WT4(BM_V, BN_V) \
    if (need_bc) U3_FP16_GEMM_LAUNCH_WT4(BM_V, BN_V, true);  \
    else         U3_FP16_GEMM_LAUNCH_WT4(BM_V, BN_V, false)

             if (c.wt_m == 4 && c.bm == 256 && c.bn == 128)
                                                 U3_FP16_TILE_CASE_WT4(256, 128);
        else if (c.bm ==  64 && c.bn ==  64) U3_FP16_TILE_CASE( 64,  64);
        else if (c.bm ==  64 && c.bn == 128) U3_FP16_TILE_CASE( 64, 128);
        else if (c.bm == 128 && c.bn ==  64) U3_FP16_TILE_CASE(128,  64);
        else if (c.bm == 128 && c.bn == 256) U3_FP16_TILE_CASE(128, 256);
        else                                  U3_FP16_TILE_CASE(128, 128);

#undef U3_FP16_TILE_CASE_WT4
#undef U3_FP16_GEMM_LAUNCH_WT4
#undef U3_FP16_TILE_CASE
#undef U3_FP16_GEMM_LAUNCH
        return;
    }

#define WMMA_U3_LAUNCH(BM_V, BN_V, BC_V) do {                                \
    constexpr int WM_  = BM_V / (2 * 16);                                     \
    constexpr int WN_  = BN_V / (2 * 16);                                     \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    if (has_zp) {                                                             \
        hipLaunchKernelGGL(                                                   \
            HIP_KERNEL_NAME(                                                  \
                MatMulNBitsWMMA_U3_ZP<BM_V, BN_V, 2, 2, 8, BC_V>),            \
            grid_, block_, 0, stream,                                         \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    } else {                                                                  \
        hipLaunchKernelGGL(                                                   \
            HIP_KERNEL_NAME(                                                  \
                MatMulNBitsWMMA_U3_NoZP<BM_V, BN_V, 2, 2, 8, BC_V>),          \
            grid_, block_, 0, stream,                                         \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    }                                                                         \
} while (0)

#define WMMA_U3_TILE_CASE(BM_V, BN_V) \
    if (need_bc) WMMA_U3_LAUNCH(BM_V, BN_V, true);  \
    else         WMMA_U3_LAUNCH(BM_V, BN_V, false)

#define WMMA_U3_LAUNCH_WT4(BM_V, BN_V, BC_V) do {                            \
    constexpr int WM_  = BM_V / (4 * 16);                                     \
    constexpr int WN_  = BN_V / (2 * 16);                                     \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    if (has_zp) {                                                             \
        hipLaunchKernelGGL(                                                   \
            HIP_KERNEL_NAME(                                                  \
                MatMulNBitsWMMA_U3_ZP<BM_V, BN_V, 4, 2, 6, BC_V>),            \
            grid_, block_, 0, stream,                                         \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    } else {                                                                  \
        hipLaunchKernelGGL(                                                   \
            HIP_KERNEL_NAME(                                                  \
                MatMulNBitsWMMA_U3_NoZP<BM_V, BN_V, 4, 2, 6, BC_V>),          \
            grid_, block_, 0, stream,                                         \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    }                                                                         \
} while (0)

#define WMMA_U3_TILE_CASE_WT4(BM_V, BN_V) \
    if (need_bc) WMMA_U3_LAUNCH_WT4(BM_V, BN_V, true);  \
    else         WMMA_U3_LAUNCH_WT4(BM_V, BN_V, false)

         if (c.wt_m == 4 && c.bm == 256 && c.bn == 128)
                                             WMMA_U3_TILE_CASE_WT4(256, 128);
    else if (c.bm ==  64 && c.bn ==  64) WMMA_U3_TILE_CASE( 64,  64);
    else if (c.bm ==  64 && c.bn == 128) WMMA_U3_TILE_CASE( 64, 128);
    else if (c.bm == 128 && c.bn ==  64) WMMA_U3_TILE_CASE(128,  64);
    else if (c.bm == 128 && c.bn == 256) WMMA_U3_TILE_CASE(128, 256);
    else                                  WMMA_U3_TILE_CASE(128, 128);

#undef WMMA_U3_TILE_CASE_WT4
#undef WMMA_U3_LAUNCH_WT4
#undef WMMA_U3_TILE_CASE
#undef WMMA_U3_LAUNCH
}

static int tuneWmmaConfigU3(
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
    constexpr int WARMUP = 2;
    constexpr int ITERS  = 7;

    // Scratch fp16 buffer for the separate-dequant configs (fused=false).
    _Float16* dq_buf = ensureDqBuffer(N, K);

    // --- Clock settle -------------------------------------------------------
    // Autotune compares configs by wall-time, so every config must be measured
    // at the SAME GPU clock. Right after a cold start the GPU runs at boost
    // clock and decays toward a lower sustained clock under load; a plain
    // sequential sweep therefore times early configs while boosted and late
    // configs while throttled, biasing the choice by table position (e.g. the
    // dq+gemm configs, which sit last, looked ~30-40% slower than they really
    // are). Drive a fixed compute load until per-iter latency plateaus (and for
    // at least ~60 ms, so we don't accept the still-boosted plateau of the very
    // first few iters) so all measurements happen at the steady-state clock.
    {
        float prev = 1e30f, cur = 0.0f, total = 0.0f;
        int stable = 0;
        for (int i = 0; i < 200 && stable < 3 && total < 200.0f; i++) {
            hipEventRecord(ev0, stream);
            launchWmmaConfigU3(12, stream, M, N, K, A, lda, B,
                               scales, zeros, gs, ngk, C, ldc, has_zp, dq_buf);
            hipEventRecord(ev1, stream);
            hipEventSynchronize(ev1);
            hipEventElapsedTime(&cur, ev0, ev1);
            if (total >= 60.0f && cur <= prev * 1.02f) stable++;
            else                                       stable = 0;
            prev = cur;
            total += cur;
        }
    }

    for (int cid = 0; cid < kNumWmmaU3Configs; cid++) {
        auto& cfg = kWmmaU3Configs[cid];
        if (cfg.bn > N * 2) continue;

        for (int w = 0; w < WARMUP; w++)
            launchWmmaConfigU3(cid, stream, M, N, K, A, lda, B,
                               scales, zeros, gs, ngk, C, ldc, has_zp, dq_buf);

        // Clocks are settled (above), so a single summed timing over ITERS is a
        // stable, order-independent estimate of steady-state latency.
        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchWmmaConfigU3(cid, stream, M, N, K, A, lda, B,
                               scales, zeros, gs, ngk, C, ldc, has_zp, dq_buf);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   wmma-u3 config[%2d] %s bm=%3d bn=%3d sw=%d : %.4f ms\n",
            cid, cfg.fused ? "fused " : "dq+gem",
            cfg.bm, cfg.bn, cfg.swizzle_n, ms / ITERS);

        if (ms < best_ms) {
            best_ms = ms;
            best_id = cid;
        }
    }

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] WMMA-u3 autotune: M=%d N=%d K=%d gs=%d zp=%s -> "
        "best config[%d] %s bm=%d bn=%d sw=%d (%.4f ms/iter)\n",
        M, N, K, gs, has_zp ? "yes" : "no",
        best_id, kWmmaU3Configs[best_id].fused ? "fused" : "dq+gemm",
        kWmmaU3Configs[best_id].bm, kWmmaU3Configs[best_id].bn,
        kWmmaU3Configs[best_id].swizzle_n, best_ms / ITERS);

    return best_id;
}

static std::unordered_map<std::string, int> wmma_u3_cache;
static std::mutex wmma_u3_tune_mutex;

static std::string wmmaU3CacheKey(int M, int N, int K, int gs, bool has_zp) {
    return std::to_string(M) + "_" + std::to_string(N) + "_" +
           std::to_string(K) + "_" + std::to_string(gs) + "_" +
           (has_zp ? "1" : "0");
}

static int getOrTuneWmmaConfigU3(
    hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const unsigned char* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp)
{
    std::string key = wmmaU3CacheKey(M, N, K, gs, has_zp);

    std::lock_guard<std::mutex> lock(wmma_u3_tune_mutex);

    auto it = wmma_u3_cache.find(key);
    if (it != wmma_u3_cache.end())
        return it->second;

    int best = tuneWmmaConfigU3(stream, M, N, K, A, lda, B,
                                 scales, zeros, gs, ngk, C, ldc, has_zp);
    wmma_u3_cache[key] = best;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] WMMA-u3 cache: stored key=\"%s\" -> config[%d] "
        "(cache size=%zu)\n",
        key.c_str(), best, wmma_u3_cache.size());

    return best;
}

/* =======================================================================
 * Section 2e: WMMA kernel for uint2 weights (fused dequant)
 *
 * Mirrors GemmFp16U3Impl above, but decodes the 2-bit stream instead of
 * the 3-bit one. Even simpler: 8 values x 2 bits = 16 bits = EXACTLY 2
 * bytes, so each thread's 8-element K-strip is a single naturally-aligned
 * uint16 load (b_k8 and t are multiples of 8, so byte offset (k*2)>>3 is
 * always even) and extracting value i via `(chunk >> 2*i) & 0x3` never
 * crosses a register boundary (max shift 2*7=14, value bits [14,16)).
 *
 *   B      : uint8 [N, (K*2)/8]                  (2-bit stream, 4 vals/byte)
 *   scales : fp16  [N, num_groups_k]
 *   zeros  : uint8 [N, num_groups_k]             (or NULL, default zp=2)
 *
 * Everything else is byte-for-byte identical to GemmFp16U3Impl.
 * ======================================================================= */

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

static void launchDequantU2(
    hipStream_t stream,
    const unsigned char* B, const _Float16* scales,
    const unsigned char* zeros, _Float16* B_fp16,
    int N, int K, int gs, int ngk)
{
    constexpr int DQ_THR = 256;
    int k_groups = (K + DQ_THR * 8 - 1) / (DQ_THR * 8);
    dim3 dq_grid(static_cast<unsigned>(k_groups), static_cast<unsigned>(N));
    dim3 dq_block(DQ_THR);
    hipLaunchKernelGGL(dequant_u2_to_fp16, dq_grid, dq_block, 0, stream,
                       B, scales, zeros, B_fp16, N, K, gs, ngk);
}

struct WmmaConfigU2 {
    int  bm, bn;
    int  swizzle_n;
    bool fused = true;   // true: fused inline-decode; false: dequant + fp16 GEMM
    int  wt_m = 2, wt_n = 2;
};

static constexpr WmmaConfigU2 kWmmaU2Configs[] = {
    // --- fused inline-decode (bandwidth-optimized, best for small/medium M) ---
    { 64,  64,  2}, { 64,  64,  4}, { 64,  64,  8}, { 64,  64, 16},
    { 64, 128,  2}, { 64, 128,  4}, { 64, 128,  8}, { 64, 128, 16},
    {128,  64,  2}, {128,  64,  4}, {128,  64,  8}, {128,  64, 16},
    {128, 128,  2}, {128, 128,  4}, {128, 128,  8}, {128, 128, 16},
    {128, 128, 32}, {128, 128, 64},
    {128, 256,  4}, {128, 256,  8}, {128, 256, 16}, {128, 256, 32},
    // --- fused inline-decode, 256x128 WT4 ---
    {256, 128,  4, true, 4, 2}, {256, 128,  8, true, 4, 2},
    {256, 128, 16, true, 4, 2}, {256, 128, 32, true, 4, 2},
    // --- separate dequant + plain fp16 WMMA GEMM (peak compute for large M) ---
    { 64, 128,  8, false}, {128,  64,  8, false},
    {128, 128,  2, false}, {128, 128,  8, false}, {128, 128, 16, false},
    {128, 128, 32, false},
    {128, 256,  4, false}, {128, 256,  8, false}, {128, 256, 16, false},
    {128, 256, 32, false},
    {256, 128,  4, false, 4, 2}, {256, 128,  8, false, 4, 2},
    {256, 128, 16, false, 4, 2}, {256, 128, 32, false, 4, 2},
};
static constexpr int kNumWmmaU2Configs =
    static_cast<int>(sizeof(kWmmaU2Configs) / sizeof(kWmmaU2Configs[0]));

static void launchWmmaConfigU2(
    int config_id, hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const unsigned char* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp,
    _Float16* B_fp16_buf = nullptr)
{
    auto& c = kWmmaU2Configs[config_id];
    bool need_bc = (M % c.bm != 0) || (N % c.bn != 0);

    if (!c.fused) {
        launchDequantU2(stream, B, scales, zeros, B_fp16_buf, N, K, gs, ngk);

#define U2_FP16_GEMM_LAUNCH(BM_V, BN_V, BC_V) do {                            \
    constexpr int WM_  = BM_V / (2 * 16);                                     \
    constexpr int WN_  = BN_V / (2 * 16);                                     \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    hipLaunchKernelGGL(                                                       \
        HIP_KERNEL_NAME(MatMulNBitsFp16GEMM<BM_V, BN_V, 2, 2, 8, BC_V>),      \
        grid_, block_, 0, stream,                                             \
        M, N, K, A, lda, B_fp16_buf, C, ldc, c.swizzle_n);                    \
} while (0)

#define U2_FP16_TILE_CASE(BM_V, BN_V) \
    if (need_bc) U2_FP16_GEMM_LAUNCH(BM_V, BN_V, true);  \
    else         U2_FP16_GEMM_LAUNCH(BM_V, BN_V, false)

#define U2_FP16_GEMM_LAUNCH_WT4(BM_V, BN_V, BC_V) do {                        \
    constexpr int WM_  = BM_V / (4 * 16);                                     \
    constexpr int WN_  = BN_V / (2 * 16);                                     \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    hipLaunchKernelGGL(                                                       \
        HIP_KERNEL_NAME(MatMulNBitsFp16GEMM<BM_V, BN_V, 4, 2, 6, BC_V>),      \
        grid_, block_, 0, stream,                                             \
        M, N, K, A, lda, B_fp16_buf, C, ldc, c.swizzle_n);                    \
} while (0)

#define U2_FP16_TILE_CASE_WT4(BM_V, BN_V) \
    if (need_bc) U2_FP16_GEMM_LAUNCH_WT4(BM_V, BN_V, true);  \
    else         U2_FP16_GEMM_LAUNCH_WT4(BM_V, BN_V, false)

             if (c.wt_m == 4 && c.bm == 256 && c.bn == 128)
                                                 U2_FP16_TILE_CASE_WT4(256, 128);
        else if (c.bm ==  64 && c.bn ==  64) U2_FP16_TILE_CASE( 64,  64);
        else if (c.bm ==  64 && c.bn == 128) U2_FP16_TILE_CASE( 64, 128);
        else if (c.bm == 128 && c.bn ==  64) U2_FP16_TILE_CASE(128,  64);
        else if (c.bm == 128 && c.bn == 256) U2_FP16_TILE_CASE(128, 256);
        else                                  U2_FP16_TILE_CASE(128, 128);

#undef U2_FP16_TILE_CASE_WT4
#undef U2_FP16_GEMM_LAUNCH_WT4
#undef U2_FP16_TILE_CASE
#undef U2_FP16_GEMM_LAUNCH
        return;
    }

#define WMMA_U2_LAUNCH(BM_V, BN_V, BC_V) do {                                \
    constexpr int WM_  = BM_V / (2 * 16);                                     \
    constexpr int WN_  = BN_V / (2 * 16);                                     \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    if (has_zp) {                                                             \
        hipLaunchKernelGGL(                                                   \
            HIP_KERNEL_NAME(                                                  \
                MatMulNBitsWMMA_U2_ZP<BM_V, BN_V, 2, 2, 8, BC_V>),            \
            grid_, block_, 0, stream,                                         \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    } else {                                                                  \
        hipLaunchKernelGGL(                                                   \
            HIP_KERNEL_NAME(                                                  \
                MatMulNBitsWMMA_U2_NoZP<BM_V, BN_V, 2, 2, 8, BC_V>),          \
            grid_, block_, 0, stream,                                         \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    }                                                                         \
} while (0)

#define WMMA_U2_TILE_CASE(BM_V, BN_V) \
    if (need_bc) WMMA_U2_LAUNCH(BM_V, BN_V, true);  \
    else         WMMA_U2_LAUNCH(BM_V, BN_V, false)

#define WMMA_U2_LAUNCH_WT4(BM_V, BN_V, BC_V) do {                            \
    constexpr int WM_  = BM_V / (4 * 16);                                     \
    constexpr int WN_  = BN_V / (2 * 16);                                     \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    if (has_zp) {                                                             \
        hipLaunchKernelGGL(                                                   \
            HIP_KERNEL_NAME(                                                  \
                MatMulNBitsWMMA_U2_ZP<BM_V, BN_V, 4, 2, 6, BC_V>),            \
            grid_, block_, 0, stream,                                         \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    } else {                                                                  \
        hipLaunchKernelGGL(                                                   \
            HIP_KERNEL_NAME(                                                  \
                MatMulNBitsWMMA_U2_NoZP<BM_V, BN_V, 4, 2, 6, BC_V>),          \
            grid_, block_, 0, stream,                                         \
            M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n); \
    }                                                                         \
} while (0)

#define WMMA_U2_TILE_CASE_WT4(BM_V, BN_V) \
    if (need_bc) WMMA_U2_LAUNCH_WT4(BM_V, BN_V, true);  \
    else         WMMA_U2_LAUNCH_WT4(BM_V, BN_V, false)

         if (c.wt_m == 4 && c.bm == 256 && c.bn == 128)
                                             WMMA_U2_TILE_CASE_WT4(256, 128);
    else if (c.bm ==  64 && c.bn ==  64) WMMA_U2_TILE_CASE( 64,  64);
    else if (c.bm ==  64 && c.bn == 128) WMMA_U2_TILE_CASE( 64, 128);
    else if (c.bm == 128 && c.bn ==  64) WMMA_U2_TILE_CASE(128,  64);
    else if (c.bm == 128 && c.bn == 256) WMMA_U2_TILE_CASE(128, 256);
    else                                  WMMA_U2_TILE_CASE(128, 128);

#undef WMMA_U2_TILE_CASE_WT4
#undef WMMA_U2_LAUNCH_WT4
#undef WMMA_U2_TILE_CASE
#undef WMMA_U2_LAUNCH
}

static int tuneWmmaConfigU2(
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
    constexpr int WARMUP = 2;
    constexpr int ITERS  = 7;

    _Float16* dq_buf = ensureDqBuffer(N, K);

    // --- Clock settle (see tuneWmmaConfigU3 for rationale) ---------------
    {
        float prev = 1e30f, cur = 0.0f, total = 0.0f;
        int stable = 0;
        for (int i = 0; i < 200 && stable < 3 && total < 200.0f; i++) {
            hipEventRecord(ev0, stream);
            launchWmmaConfigU2(12, stream, M, N, K, A, lda, B,
                               scales, zeros, gs, ngk, C, ldc, has_zp, dq_buf);
            hipEventRecord(ev1, stream);
            hipEventSynchronize(ev1);
            hipEventElapsedTime(&cur, ev0, ev1);
            if (total >= 60.0f && cur <= prev * 1.02f) stable++;
            else                                       stable = 0;
            prev = cur;
            total += cur;
        }
    }

    for (int cid = 0; cid < kNumWmmaU2Configs; cid++) {
        auto& cfg = kWmmaU2Configs[cid];
        if (cfg.bn > N * 2) continue;

        for (int w = 0; w < WARMUP; w++)
            launchWmmaConfigU2(cid, stream, M, N, K, A, lda, B,
                               scales, zeros, gs, ngk, C, ldc, has_zp, dq_buf);

        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchWmmaConfigU2(cid, stream, M, N, K, A, lda, B,
                               scales, zeros, gs, ngk, C, ldc, has_zp, dq_buf);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   wmma-u2 config[%2d] %s bm=%3d bn=%3d sw=%d : %.4f ms\n",
            cid, cfg.fused ? "fused " : "dq+gem",
            cfg.bm, cfg.bn, cfg.swizzle_n, ms / ITERS);

        if (ms < best_ms) {
            best_ms = ms;
            best_id = cid;
        }
    }

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] WMMA-u2 autotune: M=%d N=%d K=%d gs=%d zp=%s -> "
        "best config[%d] %s bm=%d bn=%d sw=%d (%.4f ms/iter)\n",
        M, N, K, gs, has_zp ? "yes" : "no",
        best_id, kWmmaU2Configs[best_id].fused ? "fused" : "dq+gemm",
        kWmmaU2Configs[best_id].bm, kWmmaU2Configs[best_id].bn,
        kWmmaU2Configs[best_id].swizzle_n, best_ms / ITERS);

    return best_id;
}

static std::unordered_map<std::string, int> wmma_u2_cache;
static std::mutex wmma_u2_tune_mutex;

static std::string wmmaU2CacheKey(int M, int N, int K, int gs, bool has_zp) {
    return std::to_string(M) + "_" + std::to_string(N) + "_" +
           std::to_string(K) + "_" + std::to_string(gs) + "_" +
           (has_zp ? "1" : "0");
}

static int getOrTuneWmmaConfigU2(
    hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const unsigned char* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp)
{
    std::string key = wmmaU2CacheKey(M, N, K, gs, has_zp);

    std::lock_guard<std::mutex> lock(wmma_u2_tune_mutex);

    auto it = wmma_u2_cache.find(key);
    if (it != wmma_u2_cache.end())
        return it->second;

    int best = tuneWmmaConfigU2(stream, M, N, K, A, lda, B,
                                 scales, zeros, gs, ngk, C, ldc, has_zp);
    wmma_u2_cache[key] = best;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] WMMA-u2 cache: stored key=\"%s\" -> config[%d] "
        "(cache size=%zu)\n",
        key.c_str(), best, wmma_u2_cache.size());

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

extern "C" void hip_matmul_nbits_unpack_zp_u8_2bit(
    void* stream, const void* zp_packed, void* dst_u8, int N, int groups_k)
{
    int packed_cols = (groups_k + 3) / 4;
    constexpr int THR = 256;
    dim3 grid(static_cast<unsigned>((packed_cols + THR - 1) / THR),
              static_cast<unsigned>(N));
    dim3 block(THR);
    hipLaunchKernelGGL(unpack_zp_2bit_to_u8_kernel,
                       grid, block, 0, static_cast<hipStream_t>(stream),
                       static_cast<const uint8_t*>(zp_packed),
                       static_cast<uint8_t*>(dst_u8),
                       N, groups_k, packed_cols);
}

extern "C" void hip_matmul_nbits_unpack_zp_u8_3bit(
    void* stream, const void* zp_packed, void* dst_u8, int N, int groups_k)
{
    int packed_cols = (groups_k * 3 + 7) / 8;
    constexpr int THR = 256;
    dim3 grid(static_cast<unsigned>((groups_k + THR - 1) / THR),
              static_cast<unsigned>(N));
    dim3 block(THR);
    hipLaunchKernelGGL(unpack_zp_3bit_to_u8_kernel,
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

// Full-weight int4 -> fp16 dequantization into a row-major [N, K] buffer, used
// by the CDNA/wave64 prefill fast path (lib/Runtime/real/matmul_nbits.cpp),
// which lacks WMMA and instead dequantizes B once (weights are constant) and
// calls a hipBLASLt fp16 GEMM. scales/zeros are fp16 [N, num_groups_k]; when
// zeros_fp16 is null the ONNX 4-bit default zero-point (8) is used.
extern "C" void hip_matmul_nbits_dequant_b_fp16(
    void* stream, const void* B_packed, const void* scales_fp16,
    const void* zeros_fp16, void* B_fp16_out, int N, int K, int group_size)
{
    int num_groups_k = (K + group_size - 1) / group_size;
    int threads_along_k = (K + 7) / 8;   // each thread emits 8 contiguous nibbles
    constexpr int THR = 256;
    dim3 grid(static_cast<unsigned>((threads_along_k + THR - 1) / THR),
              static_cast<unsigned>(N));
    dim3 block(THR);
    hipLaunchKernelGGL(dequant_u4_to_fp16, grid, block, 0,
                       static_cast<hipStream_t>(stream),
                       static_cast<const unsigned char*>(B_packed),
                       static_cast<const _Float16*>(scales_fp16),
                       static_cast<const _Float16*>(zeros_fp16),
                       static_cast<_Float16*>(B_fp16_out),
                       N, K, group_size, num_groups_k);
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
  if (bits != 2 && bits != 3 && bits != 4 && bits != 8) {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits: only bits=2, bits=3, bits=4 "
            "or bits=8 supported, got %lld\n",
            (long long)bits);
    return -1;
  }

  if (M <= 0 || N <= 0 || K <= 0 || batch_count <= 0) return 0;

  if (element_size_bytes != 2) {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits: only fp16 supported, "
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
        (block_size >= 32) && (block_size % 32 == 0) &&
        !hipdnn_device_is_wave64();  // WMMA is RDNA-only; wave64 uses GEMV/naive
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

      // 1 < M < kBlockDim is the speculative verify band: too many rows to
      // stream the weight once per row, too few for the WMMA-i8 tiles above.
      // The M-blocked kernel reads it once per tile of rows instead. M == 1 is
      // excluded because the two are then the same kernel, and the per-row
      // table has been tuned for that shape for longer.
      if (M > 1 && M < kBlockDim && gemvMBlockedEnabledI8()) {
        int cfg_id = getOrTuneGemvMConfigI8(
            hip_stream, a_ptr, b_ptr, sc_ptr, zp_ptr, bi_ptr, o_ptr,
            M, N, K, batch_count, block_size);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels] hip_matmul_nbits: int8 GEMV M-blocked M=%lld "
            "N=%lld K=%lld batch=%lld bs=%lld zp=%s bias=%s cfg=%d "
            "(tile_n=%d tile_m=%d threads=%d)\n",
            (long long)M, (long long)N, (long long)K, (long long)batch_count,
            (long long)block_size, zero_points ? "yes" : "null",
            bias ? "yes" : "null", cfg_id, kGemvMI8Configs[cfg_id].tile_n,
            kGemvMI8Configs[cfg_id].tile_m, kGemvMI8Configs[cfg_id].threads);

        launchGemvMConfigI8(cfg_id, hip_stream, a_ptr, b_ptr, sc_ptr, zp_ptr,
                            bi_ptr, o_ptr, M, N, K, batch_count, block_size);
      } else {
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
      }
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

  /* -------------------------------------------------------------------
   * bits=2 path: 2-bit packing, 4 values per byte, LSB-first (Section 1e/1f
   * and 2e above). Because 2 divides 8 evenly this is simultaneously the
   * ONNX MatMulNBits blockwise convention (blob_size = block_size/4 bytes)
   * and a plain continuous per-row 2-bit stream, so the two are byte-
   * identical when block_size % 4 == 0. zero_points (if present) are plain
   * uint8, one byte per group; default zp = 2. Same three dispatch paths as
   * bits=3 (WMMA fast path / GEMV / naive), specialized for the 2-bit decode.
   * ------------------------------------------------------------------- */
  if (bits == 2) {
    if (zero_points && zp_elem_size != 1) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits bits=2: zero_points must "
              "be uint8 (zp_elem_size=1), got zp_elem_size=%lld\n",
              (long long)zp_elem_size);
      return -1;
    }

    // ONNX MatMulNBits packs 2-bit zero_points 4-per-byte ([N, ceil(gk/4)]).
    // All u2 kernels below index zp one byte per group, so the runtime wrapper
    // unpacks the packed stream to one-byte-per-group and hands it back via
    // pre_unpacked_zp_u8. When that buffer is present prefer it; direct
    // callers that already supply one-byte-per-group zp leave it null and use
    // zero_points as-is.
    const void* z_u2 = (zero_points && zp_elem_size == 1 && pre_unpacked_zp_u8)
                           ? pre_unpacked_zp_u8
                           : zero_points;

    constexpr int kU2GemvMinBs = 32;
    const bool u2_data_format = (batch_count == 1) && (K % 32 == 0);
    const bool use_wmma_u2 =
        u2_data_format && (M >= kBlockDim) &&
        (block_size >= 32) && (block_size % 32 == 0) &&
        !hipdnn_device_is_wave64();  // WMMA is RDNA-only; wave64 uses GEMV/naive
    const bool is_pow2_bs = (block_size & (block_size - 1)) == 0;
    const bool use_gemv_u2 =
        !use_wmma_u2 && is_pow2_bs && (block_size >= kU2GemvMinBs) &&
        (K % 32 == 0);

    if (use_wmma_u2) {
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
      const auto* z_ptr = static_cast<const unsigned char*>(z_u2);
      auto*       o_ptr = static_cast<_Float16*>(output);

      bool has_zp = (z_ptr != nullptr);
      int cfg_id = getOrTuneWmmaConfigU2(
          hip_stream, iM, iN, iK, a_ptr, lda, b_ptr,
          s_ptr, z_ptr, gs, ngk, o_ptr, ldc, has_zp);

      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: u2 WMMA M=%d N=%d K=%d "
          "bs=%d zp=%s bias=%s cfg=%d\n",
          iM, iN, iK, gs, has_zp ? "yes" : "null",
          bias ? "yes" : "null", cfg_id);

      _Float16* dq_buf = kWmmaU2Configs[cfg_id].fused
                             ? nullptr : ensureDqBuffer(iN, iK);
      launchWmmaConfigU2(cfg_id, hip_stream, iM, iN, iK, a_ptr, lda, b_ptr,
                         s_ptr, z_ptr, gs, ngk, o_ptr, ldc, has_zp, dq_buf);

      if (bias) {
        dim3 bias_grid(static_cast<unsigned>((iN + 255) / 256),
                       static_cast<unsigned>(iM));
        dim3 bias_block(256);
        hipLaunchKernelGGL(
            matmul_nbits_add_bias_rowmajor, bias_grid, bias_block,
            0, hip_stream,
            o_ptr, static_cast<const _Float16*>(bias), iM, iN, ldc);
      }
    } else if (use_gemv_u2) {
      const auto* a_ptr  = static_cast<const __half*>(A);
      const auto* b_ptr  = static_cast<const uint8_t*>(B);
      const auto* sc_ptr = static_cast<const __half*>(scales);
      const auto* zp_ptr = static_cast<const uint8_t*>(z_u2);
      const auto* bi_ptr = static_cast<const __half*>(bias);
      auto*       o_ptr  = static_cast<__half*>(output);

      int cfg_id = getOrTuneGemvConfigU2(
          hip_stream, a_ptr, b_ptr, sc_ptr, zp_ptr, bi_ptr, o_ptr,
          M, N, K, batch_count, block_size);

      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: u2 GEMV M=%lld N=%lld "
          "K=%lld batch=%lld bs=%lld zp=%s bias=%s cfg=%d\n",
          (long long)M, (long long)N, (long long)K, (long long)batch_count,
          (long long)block_size, zero_points ? "yes" : "null",
          bias ? "yes" : "null", cfg_id);

      launchGemvConfigU2(cfg_id, hip_stream, a_ptr, b_ptr, sc_ptr, zp_ptr,
                         bi_ptr, o_ptr, M, N, K, batch_count, block_size);
    } else {
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: u2 naive M=%lld N=%lld K=%lld "
          "batch=%lld bs=%lld zp=%s bias=%s\n",
          (long long)M, (long long)N, (long long)K, (long long)batch_count,
          (long long)block_size, zero_points ? "yes" : "null",
          bias ? "yes" : "null");

      dim3 blk(kBlockDim, kBlockDim);
      dim3 grd(static_cast<unsigned>((N + kBlockDim - 1) / kBlockDim),
               static_cast<unsigned>((M + kBlockDim - 1) / kBlockDim),
               static_cast<unsigned>(batch_count));

      hipLaunchKernelGGL(
          matmul_nbits_kernel_u2, grd, blk, 0, hip_stream,
          static_cast<const __half*>(A), static_cast<const uint8_t*>(B),
          static_cast<const __half*>(scales),
          static_cast<const uint8_t*>(z_u2),
          static_cast<const __half*>(bias), static_cast<__half*>(output),
          M, N, K, block_size);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits u2 launch failed: %s\n",
              hipGetErrorString(err));
    }
    return static_cast<int>(err);
  }

  /* -------------------------------------------------------------------
   * bits=3 path: continuous-bitstream uint3 weight packing (Section 1c/1d
   * above) — a bandwidth-optimized format for sub-4-bit decode. zero_points
   * follow the ONNX MatMulNBits bits=3 convention: a continuous per-row
   * 3-bit stream ([N, ceil(groups_k*3/8)]), which the runtime wrapper
   * unpacks to one-byte-per-group and hands back via pre_unpacked_zp_u8.
   * Direct callers that already supply one-byte-per-group zp leave it null
   * and the raw zero_points are used as-is.
   *
   * Three paths, in priority order (mirrors the bits=8 dispatch):
   *   1. WMMA-u3 fast path (M >= 16, batch=1, K%32==0, group_size % 32 == 0
   *      and >= 32): autotuned 16x16x16 fp16 WMMA tiles with the 3-bit
   *      bitstream decoded inline (Section 2d) — no separate dequant
   *      buffer, no bandwidth given back. Used for prefill-shaped GEMMs.
   *   2. GEMV-u3 (block_size >= 32, power of two, and K % 32 == 0):
   *      autotuned BLOCK_SIZE x TILE_N reduction with vectorized 96-bit
   *      (uint3) loads, 32 weights per global memory transaction. Used for
   *      decode (M=1) and small-M shapes that don't meet WMMA's threshold.
   *   3. Naive fallback: thread-per-element kernel for shapes that don't
   *      meet either gate above.
   * ------------------------------------------------------------------- */
  if (bits == 3) {
    if (zero_points && zp_elem_size != 1) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits bits=3: zero_points must "
              "be uint8 (zp_elem_size=1), got zp_elem_size=%lld\n",
              (long long)zp_elem_size);
      return -1;
    }

    // ONNX MatMulNBits packs 3-bit zero_points as a continuous per-row stream
    // ([N, ceil(gk*3/8)]). All u3 kernels below index zp one byte per group,
    // so the runtime wrapper unpacks the stream to one-byte-per-group and
    // hands it back via pre_unpacked_zp_u8. Prefer it when present; direct
    // callers that already supply one-byte-per-group zp leave it null and use
    // zero_points as-is.
    const void* z_u3 = (zero_points && zp_elem_size == 1 && pre_unpacked_zp_u8)
                           ? pre_unpacked_zp_u8
                           : zero_points;

    constexpr int kU3GemvMinBs = 32;
    const bool u3_data_format = (batch_count == 1) && (K % 32 == 0);
    const bool use_wmma_u3 =
        u3_data_format && (M >= kBlockDim) &&
        (block_size >= 32) && (block_size % 32 == 0) &&
        !hipdnn_device_is_wave64();  // WMMA is RDNA-only; wave64 uses GEMV/naive
    const bool is_pow2_bs = (block_size & (block_size - 1)) == 0;
    const bool use_gemv_u3 =
        !use_wmma_u3 && is_pow2_bs && (block_size >= kU3GemvMinBs) &&
        (K % 32 == 0);

    if (use_wmma_u3) {
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
      const auto* z_ptr = static_cast<const unsigned char*>(z_u3);
      auto*       o_ptr = static_cast<_Float16*>(output);

      bool has_zp = (z_ptr != nullptr);
      int cfg_id = getOrTuneWmmaConfigU3(
          hip_stream, iM, iN, iK, a_ptr, lda, b_ptr,
          s_ptr, z_ptr, gs, ngk, o_ptr, ldc, has_zp);

      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: u3 WMMA M=%d N=%d K=%d "
          "bs=%d zp=%s bias=%s cfg=%d\n",
          iM, iN, iK, gs, has_zp ? "yes" : "null",
          bias ? "yes" : "null", cfg_id);

      _Float16* dq_buf = kWmmaU3Configs[cfg_id].fused
                             ? nullptr : ensureDqBuffer(iN, iK);
      launchWmmaConfigU3(cfg_id, hip_stream, iM, iN, iK, a_ptr, lda, b_ptr,
                         s_ptr, z_ptr, gs, ngk, o_ptr, ldc, has_zp, dq_buf);

      if (bias) {
        dim3 bias_grid(static_cast<unsigned>((iN + 255) / 256),
                       static_cast<unsigned>(iM));
        dim3 bias_block(256);
        hipLaunchKernelGGL(
            matmul_nbits_add_bias_rowmajor, bias_grid, bias_block,
            0, hip_stream,
            o_ptr, static_cast<const _Float16*>(bias), iM, iN, ldc);
      }
    } else if (use_gemv_u3) {
      const auto* a_ptr  = static_cast<const __half*>(A);
      const auto* b_ptr  = static_cast<const uint8_t*>(B);
      const auto* sc_ptr = static_cast<const __half*>(scales);
      const auto* zp_ptr = static_cast<const uint8_t*>(z_u3);
      const auto* bi_ptr = static_cast<const __half*>(bias);
      auto*       o_ptr  = static_cast<__half*>(output);

      int cfg_id = getOrTuneGemvConfigU3(
          hip_stream, a_ptr, b_ptr, sc_ptr, zp_ptr, bi_ptr, o_ptr,
          M, N, K, batch_count, block_size);

      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: u3 GEMV M=%lld N=%lld "
          "K=%lld batch=%lld bs=%lld zp=%s bias=%s cfg=%d\n",
          (long long)M, (long long)N, (long long)K, (long long)batch_count,
          (long long)block_size, zero_points ? "yes" : "null",
          bias ? "yes" : "null", cfg_id);

      launchGemvConfigU3(cfg_id, hip_stream, a_ptr, b_ptr, sc_ptr, zp_ptr,
                         bi_ptr, o_ptr, M, N, K, batch_count, block_size);
    } else {
      CUSTOM_KERNELS_DEBUG_LOG(
          "[custom_kernels] hip_matmul_nbits: u3 naive M=%lld N=%lld K=%lld "
          "batch=%lld bs=%lld zp=%s bias=%s\n",
          (long long)M, (long long)N, (long long)K, (long long)batch_count,
          (long long)block_size, zero_points ? "yes" : "null",
          bias ? "yes" : "null");

      dim3 blk(kBlockDim, kBlockDim);
      dim3 grd(static_cast<unsigned>((N + kBlockDim - 1) / kBlockDim),
               static_cast<unsigned>((M + kBlockDim - 1) / kBlockDim),
               static_cast<unsigned>(batch_count));

      hipLaunchKernelGGL(
          matmul_nbits_kernel_u3, grd, blk, 0, hip_stream,
          static_cast<const __half*>(A), static_cast<const uint8_t*>(B),
          static_cast<const __half*>(scales),
          static_cast<const uint8_t*>(z_u3),
          static_cast<const __half*>(bias), static_cast<__half*>(output),
          M, N, K, block_size);
    }

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits u3 launch failed: %s\n",
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
   * When M >= kWmmaMinM:   WMMA intrinsics (RDNA3 fast path)
   * When 1 < M < kWmmaMinM: GEMV kernel with COL_MAJOR=true
   * When M == 1:            row-major GEMV, further down
   * ------------------------------------------------------------------- */
  bool wmma_data_format = (batch_count == 1) && (K % 32 == 0);

  // Previously kBlockDim (16). WMMA reads B far more efficiently than the
  // col-major GEMV, but it computes a whole tile of rows whatever M is, so
  // the wasted rows grow as M shrinks and eventually outweigh the read. 8 is
  // the best of the three cutoffs measured on gpt-oss-20b, over interleaved
  // and order-reversed 16k-prefill pairs: -211 ms TTFT (-2.3%) against 16,
  // where moving the whole tier down to 2 is +47 ms, i.e. worse than leaving
  // it alone. 4 and 12 are untested.
  //
  // A single cutoff cannot be right for every shape, because the two sides
  // scale differently and by shape-dependent amounts -- see preferWmmaTier for
  // the measured crossovers (5 to ~28 across four shapes of one model). So the
  // cutoff below is only the fallback for when tuning is off; normally the tier
  // is timed per shape. Override with HIPDNN_EP_MATMUL_WMMA_MIN_M to pin it.
  //
  // Any cutoff is safe for partial tiles: need_bc in launchWmmaConfig selects
  // the bounds-checked variant whenever M is not a multiple of the tile.
  const int kWmmaMinM = wmmaMinM();

  // Both int4 tiers can serve any M >= 2 in the tuned window, so decide by
  // measurement there and by cutoff elsewhere. The two bodies are packaged as
  // callables so the same code runs whether it is being timed or used.
  const bool tier_is_tuned =
      wmma_data_format && !hipdnn_device_is_wave64() &&
      M > 1 && M <= kTierTuneMaxM && wmmaTierTuningEnabled();

  // WMMA is an RDNA3/RDNA4 (wave32) intrinsic. On CDNA (wave64, e.g. MI350)
  // there is no WMMA, so the prefill fast path is unavailable here; int4
  // M>=kWmmaMinM is instead served by the dequant-once + hipBLASLt path in
  // lib/Runtime/real/matmul_nbits.cpp, and any residual case falls through to
  // the correct GEMV/naive fallback below.
  auto run_wmma_int4 = [&]() -> int {
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
  };

  /* -------------------------------------------------------------------
   * GEMV path with col-major data — small M (>1), WMMA data format
   *
   * When batch==1, K%32==0, but 1 < M < kWmmaMinM: internally transpose
   * A row->col and output col->row so the caller sees row-major I/O.
   * GEMV kernel runs with COL_MAJOR=true for K-parallel reduction.
   *
   * M=1 skips this path entirely — row/col layouts are identical for a
   * single row, and the row-major GEMV avoids the ZP conversion kernel.
   * ------------------------------------------------------------------- */
  /* -------------------------------------------------------------------
   * M-blocked col-major GEMV — small M (>1), WMMA data format
   *
   * Same transposes and the same fp16 zero points as the col-major path
   * below, but one block covers up to tile_m rows, so the packed weight is
   * read once per tile of rows instead of once per row. See the GEMV-M
   * autotune section for why that matters and what it costs.
   * ------------------------------------------------------------------- */
  auto run_gemv_mblocked_int4 = [&]() -> int {
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_matmul_nbits: GEMV M-blocked path M=%lld "
        "N=%lld K=%lld bs=%lld zp=%s bias=%s\n",
        (long long)M, (long long)N, (long long)K, (long long)block_size,
        zero_points ? "yes" : "null", bias ? "yes" : "null");

    int iM = static_cast<int>(M);
    int iN = static_cast<int>(N);
    int iK = static_cast<int>(K);

    _Float16* a_buf = ensureAColBuffer(iM, iK);
    launchTranspose2DFp16(hip_stream, A, a_buf, iM, iK);
    const auto* a_col = reinterpret_cast<const __half*>(a_buf);
    auto* o_col = reinterpret_cast<__half*>(ensureOutColBuffer(iM, iN));

    const auto* b_u = static_cast<const uint8_t*>(B);
    const auto* s_h = static_cast<const __half*>(scales);
    const auto* z_u = static_cast<const uint8_t*>(zp_for_fp16_paths);
    const auto* bi_h = static_cast<const __half*>(bias);

    int cfg_id = getOrTuneGemvMConfig(hip_stream, a_col, b_u, s_h, z_u, bi_h,
                                      o_col, M, N, K, 1, block_size);
    launchGemvMConfig(cfg_id, hip_stream, a_col, b_u, s_h, z_u, bi_h, o_col,
                      M, N, K, 1, block_size);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits GEMV M-blocked launch "
              "failed: %s\n",
              hipGetErrorString(err));
      return static_cast<int>(err);
    }

    launchTranspose2DFp16(hip_stream, o_col, output, iN, iM);
    return static_cast<int>(hipGetLastError());
  };

  auto run_gemv_col_int4 = [&]() -> int {
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
  };

  /* -------------------------------------------------------------------
   * Tier selection between the two paths above.
   *
   * In the tuned window both are timed once for this shape and the winner is
   * cached (preferWmmaTier); outside it, or when tuning is pinned off, the
   * fixed kWmmaMinM cutoff decides. M == 1 belongs to neither -- it falls
   * through to the row-major GEMV below, which skips the transposes and the
   * zero-point conversion entirely.
   * ------------------------------------------------------------------- */
  if (wmma_data_format && M > 1 && !hipdnn_device_is_wave64()) {
    // Which GEMV serves this M. Below the WMMA cutoff the M-blocked variant
    // reads the packed weight once per tile of rows instead of once per row,
    // which is worth 1.05x on a Qwen3-8B verify call at M=4 (99.2 -> 94.4 ms,
    // v/step 2.72 -> 2.60). Above the cutoff it is untuned territory: the
    // config table stops at tile_m=8, so the per-row path stays.
    //
    // Note this is the GEMV *candidate*, not the decision. It is what the tier
    // probe times against WMMA, so M-blocking can never make a shape slower
    // than WMMA would have been -- which matters because forcing M-blocking
    // globally used to disable tier tuning and cost 1.54x at M=8 (168 -> 259 ms).
    const bool use_mblocked = (M < kWmmaMinM) && gemvMBlockedEnabled();
    auto run_gemv = [&]() -> int {
      return use_mblocked ? run_gemv_mblocked_int4() : run_gemv_col_int4();
    };

    bool use_wmma;
    if (tier_is_tuned) {
      use_wmma = preferWmmaTier(
          hip_stream, static_cast<int>(M), static_cast<int>(N),
          static_cast<int>(K), static_cast<int>(block_size),
          zp_for_fp16_paths != nullptr,
          [&]() { run_wmma_int4(); }, [&]() { run_gemv(); });
    } else {
      use_wmma = (M >= kWmmaMinM);
    }
    return use_wmma ? run_wmma_int4() : run_gemv();
  }

  // M == 1 with the cutoff pinned to 1: WMMA was explicitly asked for.
  if (wmma_data_format && M >= kWmmaMinM && !hipdnn_device_is_wave64())
    return run_wmma_int4();

  /* -------------------------------------------------------------------
   * GEMV path — small M, row-major data (non-WMMA format)
   *
   * For small M (≤ kBlockDim) with row-major A and uint8 zero points.
   * On the first call for each (M,N,K,block_size) shape, all configs
   * are benchmarked and the fastest is cached for subsequent calls.
   * ------------------------------------------------------------------- */
  if (M <= kBlockDim) {
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

  hipLaunchKernelGGL(
      matmul_nbits_kernel, grd, blk, 0, hip_stream,
      static_cast<const __half*>(A), static_cast<const uint8_t*>(B),
      static_cast<const __half*>(scales),
      static_cast<const uint8_t*>(zp_for_u8_paths),
      static_cast<const __half*>(bias), static_cast<__half*>(output),
      M, N, K, block_size);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits naive launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}
