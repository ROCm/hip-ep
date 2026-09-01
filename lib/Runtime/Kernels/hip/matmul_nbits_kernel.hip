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
// Offline autotune table. Only the lightweight lookup API is pulled in here;
// the flatbuffers dependency lives entirely in matmul_nbits_autotune.cpp.
#include "autotune/matmul_nbits/matmul_nbits_autotune.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rtc/hiprtc_module.h"

// Unlike the other RTC-enabled kernels, the device half of this file is not
// split into a header: 33 device blocks are interleaved with the autotune state
// machines. The build extracts it instead (cmake/rtc_extract_device.py) and
// embeds the result here.
extern "C" const unsigned char kMatMulNBitsDeviceSource[];
extern "C" const size_t kMatMulNBitsDeviceSource_size;

// nullptr whenever RTC is off or the module failed to build, which is the
// signal for HIPDNN_RTC_LAUNCH to use the AOT kernel. Defined at the bottom of
// the file, once every family has contributed its name expressions.
static hipFunction_t mnbRtcFunction(const char* name_expression);

/* -----------------------------------------------------------------------
 * Autotune config caches are in-process only.
 *
 * Each tuner (WMMA / GEMV int4 / int8 / dp4a) memoizes its winner in an
 * in-memory std::unordered_map for the lifetime of the process. There is no
 * disk persistence: the offline LUT (matmul_nbits_autotune) is the durable,
 * cross-process source of configs, so writing a per-shape tune cache to
 * %TEMP% was redundant (and rewrote the file on every newly seen shape). A
 * LUT miss still falls back to an on-device sweep, whose result lives only in
 * the in-process map.
 * ----------------------------------------------------------------------- */

/* =======================================================================
 * Section 1: Naive fallback kernel (row-major, any alignment, batched)
 * ======================================================================= */

// The device half lives in a header so the AOT build and hipRTC compile the
// same text. It also carries the constants both halves share (kBlockDim), so
// this include has to precede the host dispatch below.
#include "hip/rtc/matmul_nbits_device.h"

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


/* =======================================================================
 * Section 1b: GEMV-style kernel for small M (autotuned, vectorized)
 *
 * Template parameters:
 *   BLOCK_SIZE - threads per block (K-reduction parallelism)
 *   TILE_N     - N columns per block (1..16, amortizes A loads)
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
 *   - TILE_N up to 16; wider tiles are register-bound (see kGemvConfigs)
 *   - Compile-time WARP_SIZE=32 for RDNA 3 dead-code elimination
 *     (runtime __builtin_amdgcn_wavefrontsize() caused 13% regression)
 *   - Single-warp fast path: BLOCK_SIZE=32 uses shuffle-only reduction
 *     (no LDS sync needed)
 *
 * At runtime, an autotune pass benchmarks all 29 (BLOCK_SIZE, TILE_N)
 * combinations on the first call for each problem shape and caches the
 * fastest configuration for subsequent calls.
 * BLOCK_SIZE ∈ {32,64,128,256,512,1024}, TILE_N ∈ {1,2,4,8,16}.
 * BS=32 configs use single-warp shuffle-only reduction (no LDS).
 * See kGemvConfigs for why TILE_N stops at 16.
 *
 * Launch: grid(ceil(N/TILE_N), M, batch), block(BLOCK_SIZE)
 * ======================================================================= */


/* =======================================================================
 * Section 1c: W4A8 integer-dot-product (dp4a) GEMV path  [decode, M==1]
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
 * ======================================================================= */



/* -----------------------------------------------------------------------
 * GEMV Autotune: config table, dispatch, benchmark, cache
 * ----------------------------------------------------------------------- */

struct GemvConfig {
    int threads;
    int tile_n;
};

// One list drives the config table, the launch switch and the hipRTC name
// expressions, so a config cannot end up in one and not the others. The id must
// stay equal to the position, since it indexes kGemvConfigs.
//
// TILE_N stops at 16 on purpose.
//
// Every kernel that reads this table stages TILE_N B-vectors into registers
// before consuming any of them, so the hot loop holds a uint4 (or uint2/uint3)
// per column plus a scale and a running sum per column. At TILE_N=64 the B
// array alone is 256 VGPRs -- the entire per-thread budget on gfx1151 -- and
// the tail spills: 732-1436 B of scratch depending on bit width, and 68-364 B
// still at TILE_N=32. Those were the only kernels in this file with any
// scratch at all.
//
// They also never won. Across 50 autotune runs (u4 / i8 / u3 / u2 / dp4a, five
// shapes each, with and without zero points) the winner was always TILE_N <= 8;
// the best TILE_N>=32 candidate was 1.13x-3.1x slower and the best TILE_N=64
// candidate 2.45x-9.3x slower, never once inside the noise. They cost 31% of
// the sweep time to measure and discard, because being the slowest candidates
// they dominate it.
#define MNB_GEMV_CONFIG_LIST(X)                                               \
    /* BLOCK_SIZE=32: single warp (wave32) -- shuffle-only reduction, no LDS */ \
    X( 0,   32,  1) X( 1,   32,  2) X( 2,   32,  4) X( 3,   32,  8)           \
    X( 4,   64,  1) X( 5,   64,  2) X( 6,   64,  4) X( 7,   64,  8)           \
    X( 8,   64, 16)                                                           \
    X( 9,  128,  1) X(10,  128,  2) X(11,  128,  4) X(12,  128,  8)           \
    X(13,  128, 16)                                                           \
    X(14,  256,  1) X(15,  256,  2) X(16,  256,  4) X(17,  256,  8)           \
    X(18,  256, 16)                                                           \
    X(19,  512,  1) X(20,  512,  2) X(21,  512,  4) X(22,  512,  8)           \
    X(23,  512, 16)                                                           \
    X(24, 1024,  1) X(25, 1024,  2) X(26, 1024,  4) X(27, 1024,  8)           \
    X(28, 1024, 16)

static constexpr GemvConfig kGemvConfigs[] = {
#define X(ID, BS, TN) {BS, TN},
    MNB_GEMV_CONFIG_LIST(X)
#undef X
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
            HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid, block, 0, stream, \
                (A, B, sc, zp, bi, out, M, N, K, bs), \
                matmul_nbits_gemv_kernel<BS, TN, true>); \
        else \
            HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid, block, 0, stream, \
                (A, B, sc, zp, bi, out, M, N, K, bs), \
                matmul_nbits_gemv_kernel<BS, TN, false>); \
        break;

    switch (config_id) {
#define X(ID, BS, TN) GEMV_CASE(ID, BS, TN)
        MNB_GEMV_CONFIG_LIST(X)
#undef X
        default: break;
    }
#undef GEMV_CASE
}

static void appendGemvNames(std::vector<std::string>& names) {
#define X(ID, BS, TN) \
    names.push_back(HIPDNN_RTC_STR(matmul_nbits_gemv_kernel<BS, TN, true>)); \
    names.push_back(HIPDNN_RTC_STR(matmul_nbits_gemv_kernel<BS, TN, false>));
    MNB_GEMV_CONFIG_LIST(X)
#undef X
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
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);

    float best_ms = 1e30f;
    int   best_id = 8;
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    for (int cid = 0; cid < kNumGemvConfigs; cid++) {
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

/* -----------------------------------------------------------------------
 * Offline LUT bridge
 *
 * Order of consultation, for every tuner below:
 *
 *   1. process map cache (in-process only, per run)     -> use it
 *   2. offline LUT                                     -> use it, and store it
 *                                                         into the map cache so
 *                                                         step 1 answers next
 *                                                         time
 *   3. runtime sweep                                   -> as before
 *
 * The map stays in front because it holds measurements from this exact machine
 * and build; the LUT's job is to make the *first* encounter with a shape free,
 * not to override something already measured locally. Writing LUT hits into the
 * map means the lookup happens once per shape per process, not per call. The
 * map lives only for the process lifetime -- the tuners no longer persist
 * winners to a %TEMP% cache file; the offline LUT is the durable source now.
 *
 * The LUT stores tile geometry rather than a config index. These helpers turn a
 * geometry back into an index against the live tables, and double as the
 * validators the lookup calls while searching: a geometry the table no longer
 * contains (or that is illegal for this shape) makes the search move on to the
 * next-nearest point instead of returning something unlaunchable.
 * ----------------------------------------------------------------------- */

namespace matmul_lut = hipdnn_ep::matmul_nbits_autotune;

/* Config source, mirroring GQA's HIPDNN_GQA_AUTOTUNE_MODE:
 *   lookup (default) - resolve from the offline LUT, falling to the in-kernel
 *                      autotune sweep only on a miss.
 *   online           - bypass the LUT and always run the in-kernel sweep (the
 *                      behaviour that predates the table), for A/B comparison.
 * Both paths are compiled in; this only picks which one a process takes. Read
 * once and latched: the tuners sit on the dispatch path, and getenv per call
 * would serialize decode through the CRT environment lock. */
static bool matmulNbitsLutLookupEnabled() {
    static const bool lookup = [] {
        std::string raw;   // cross-platform env read (_dupenv_s / getenv)
#ifdef _WIN32
        char* v = nullptr;
        size_t len = 0;
        if (_dupenv_s(&v, &len, "HIPDNN_MATMUL_AUTOTUNE_MODE") == 0 && v) {
            raw = v;
            free(v);
        }
#else
        if (const char* v = getenv("HIPDNN_MATMUL_AUTOTUNE_MODE")) raw = v;
#endif
        std::string s;
        for (char c : raw) {
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n') continue;
            s.push_back(c >= 'A' && c <= 'Z' ? char(c - 'A' + 'a') : c);
        }
        if (s == "online") return false;
        if (!s.empty() && s != "lookup")
            fprintf(stderr,
                    "[custom_kernels] unrecognised HIPDNN_MATMUL_AUTOTUNE_MODE="
                    "\"%s\" (expected \"lookup\" or \"online\"); using lookup\n",
                    raw.c_str());
        return true;   // default, and explicit "lookup"
    }();
    return lookup;
}

static int gemvConfigFromGeometry(const matmul_lut::GemvAnswer& g) {
    for (int cid = 0; cid < kNumGemvConfigs; cid++)
        if (kGemvConfigs[cid].threads == g.threads &&
            kGemvConfigs[cid].tile_n == g.tile_n)
            return cid;
    return -1;
}

static bool matmulNbitsGemvValidator(void*, const matmul_lut::GemvAnswer& g) {
    return gemvConfigFromGeometry(g) >= 0;
}

/* Returns a kGemvConfigs index from the LUT, or -1 on a miss.
 *
 * The WMMA validator is null here: a Decode/DecodeDp4a lookup can only reach a
 * point whose payload is Gemv (pointConsistent enforces that at load), so a
 * Wmma answer on this path would be a malformed table and is rejected. */
static int matmulNbitsLutGemvConfig(matmul_lut::Phase phase, int M, int N, int K, int gs,
                           bool has_zp) {
    if (!matmulNbitsLutLookupEnabled()) return -1;   // online mode: sweep, skip the LUT
    matmul_lut::Request req;
    req.phase = phase;
    req.bits = 4;
    req.m = M;
    req.n = N;
    req.k = K;
    req.group_size = gs;
    req.has_zp = has_zp;
    const matmul_lut::Result r =
        matmul_lut::resolve(req, nullptr, matmulNbitsGemvValidator, nullptr);
    if (r.source == matmul_lut::Source::None) return -1;
    const int cid = gemvConfigFromGeometry(r.gemv);
    if (cid >= 0)
        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels] LUT hit (src=%d d=%.3f) M=%d N=%d K=%d gs=%d -> "
            "config[%d]\n",
            static_cast<int>(r.source), r.distance, M, N, K, gs, cid);
    return cid;
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

    // Map miss: try the offline table before paying for a sweep. A hit is
    // memoized in the in-process map so this shape never consults the LUT again.
    const int lut = matmulNbitsLutGemvConfig(matmul_lut::Phase::Decode, static_cast<int>(M),
                                    static_cast<int>(N), static_cast<int>(K),
                                    static_cast<int>(bs), zp != nullptr);
    if (lut >= 0) {
        gemv_cache[key] = lut;
        return lut;
    }

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
 * W4A8 dp4a GEMV Autotune + dispatch (decode, M==1, K%32==0)
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
        HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid, block, 0, stream, \
            (a_qb, a_scale, B, sc, zp, bi, out, N, K, bs), \
            matmul_nbits_gemv_dp4a_kernel<BS, TN>); \
        break;

    switch (config_id) {
#define X(ID, BS, TN) DP4A_CASE(ID, BS, TN)
        MNB_GEMV_CONFIG_LIST(X)
#undef X
        default: break;
    }
#undef DP4A_CASE
}

static void appendDp4aNames(std::vector<std::string>& names) {
#define X(ID, BS, TN) \
    names.push_back(HIPDNN_RTC_STR(matmul_nbits_gemv_dp4a_kernel<BS, TN>));
    MNB_GEMV_CONFIG_LIST(X)
#undef X
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

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   dp4a config[%2d] threads=%3d tile_n=%d : %.4f ms\n",
            cid, kGemvConfigs[cid].threads, kGemvConfigs[cid].tile_n,
            ms / ITERS);

        if (ms < best_ms) { best_ms = ms; best_id = cid; }
    }
    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] GEMV-dp4a autotune: N=%lld K=%lld bs=%lld zp=%s -> "
        "best config[%d] threads=%d tile_n=%d (%.4f ms/iter)\n",
        (long long)N, (long long)K, (long long)bs, zp ? "yes" : "no",
        best_id, kGemvConfigs[best_id].threads,
        kGemvConfigs[best_id].tile_n, best_ms / ITERS);

    return best_id;
}

static std::unordered_map<std::string, int> dp4a_cache;
static std::mutex dp4a_tune_mutex;

static int getOrTuneDp4aConfig(
    hipStream_t stream,
    const int8_t* a_qb, const float* a_scale,
    const uint8_t* B, const __half* sc, const uint8_t* zp,
    const __half* bi, __half* out,
    int64_t N, int64_t K, int64_t bs)
{
    std::string key = std::to_string(N) + "_" + std::to_string(K) + "_" +
                      std::to_string(bs) + "_" + (zp ? "1" : "0");

    std::lock_guard<std::mutex> lock(dp4a_tune_mutex);
    auto it = dp4a_cache.find(key);
    if (it != dp4a_cache.end())
        return it->second;

    // Keyed as DecodeDp4a, not Decode: this kernel ranks kGemvConfigs
    // differently from the fp GEMV, which is why the two keep separate caches.
    const int lut = matmulNbitsLutGemvConfig(matmul_lut::Phase::DecodeDp4a, 1,
                                    static_cast<int>(N), static_cast<int>(K),
                                    static_cast<int>(bs), zp != nullptr);
    if (lut >= 0) {
        dp4a_cache[key] = lut;
        return lut;
    }

    int best = tuneDp4aConfig(stream, a_qb, a_scale, B, sc, zp, bi, out,
                              N, K, bs);
    dp4a_cache[key] = best;
    return best;
}

// Public entry: W4A8 dp4a GEMV for a single decode row.
//   A          : fp16 activation row [K] (row-major, batch==M==1)
//   B          : packed int4 weights [N, K/2]
//   scales     : fp16 [N, k_blocks]
//   zp_u8      : pre-unpacked uint8 zero points [N, k_blocks] or nullptr (sym)
//   out        : fp16 [N]
//   a_qb_scratch/a_scale_scratch : per-session scratch, caller-sized to
//       >= K bytes and >= k_blocks floats respectively.
// Returns hip status (hipSuccess on success). K must be a multiple of 32.
extern "C" int hip_matmul_nbits_dp4a(
    void* stream_v,
    const void* A, const void* B, const void* scales, const void* zp_u8,
    const void* bias, void* out,
    int64_t N, int64_t K, int64_t block_size,
    void* a_qb_scratch, void* a_scale_scratch)
{
    if (K % 32 != 0) return hipErrorInvalidValue;
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

    // Step 1: dynamic per-group int8 quantization of the activation row.
    // One block per K-group, fixed 128 threads (handles bs in {32,64,128,256}).
    (void)hipGetLastError();
    HIPDNN_RTC_LAUNCH(mnbRtcFunction,
                      dim3(static_cast<unsigned>(k_blocks)), dim3(128), 0,
                      stream, (Ah, a_qb, a_sc, K, block_size),
                      matmul_nbits_quant_act_i8_kernel<128>);

    // Step 2: dp4a GEMV (autotuned config).
    int cfg = getOrTuneDp4aConfig(stream, a_qb, a_sc, Bp, sc, zp, bi, outp,
                                  N, K, block_size);
    launchDp4aConfig(cfg, stream, a_qb, a_sc, Bp, sc, zp, bi, outp,
                     N, K, block_size);
    return hipGetLastError();
}

/* -----------------------------------------------------------------------
 * GEMV (int8) Autotune: dispatch, benchmark, cache
 *
 * Reuses kGemvConfigs (29 configs of BLOCK_SIZE x TILE_N) and the same
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

#define GEMV_I8_CASE(ID, BS, TN) \
    case ID: \
        HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid, block, 0, stream, \
            (A, B, sc, zp, bi, out, M, N, K, bs), \
            matmul_nbits_gemv_kernel_i8<BS, TN>); \
        break;

    switch (config_id) {
#define X(ID, BS, TN) GEMV_I8_CASE(ID, BS, TN)
        MNB_GEMV_CONFIG_LIST(X)
#undef X
        default: break;
    }
#undef GEMV_I8_CASE
}

static void appendGemvI8Names(std::vector<std::string>& names) {
#define X(ID, BS, TN) \
    names.push_back(HIPDNN_RTC_STR(matmul_nbits_gemv_kernel_i8<BS, TN>));
    MNB_GEMV_CONFIG_LIST(X)
#undef X
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
        HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid, block, 0, stream, \
            (A, B, sc, zp, bi, out, M, N, K, bs), \
            matmul_nbits_gemv_kernel_u3<BS, TN>); \
        break;

    switch (config_id) {
#define X(ID, BS, TN) GEMV_U3_CASE(ID, BS, TN)
        MNB_GEMV_CONFIG_LIST(X)
#undef X
        default: break;
    }
#undef GEMV_U3_CASE
}

static void appendGemvU3Names(std::vector<std::string>& names) {
#define X(ID, BS, TN) \
    names.push_back(HIPDNN_RTC_STR(matmul_nbits_gemv_kernel_u3<BS, TN>));
    MNB_GEMV_CONFIG_LIST(X)
#undef X
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
        HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid, block, 0, stream, \
            (A, B, sc, zp, bi, out, M, N, K, bs), \
            matmul_nbits_gemv_kernel_u2<BS, TN>); \
        break;

    switch (config_id) {
#define X(ID, BS, TN) GEMV_U2_CASE(ID, BS, TN)
        MNB_GEMV_CONFIG_LIST(X)
#undef X
        default: break;
    }
#undef GEMV_U2_CASE
}

static void appendGemvU2Names(std::vector<std::string>& names) {
#define X(ID, BS, TN) \
    names.push_back(HIPDNN_RTC_STR(matmul_nbits_gemv_kernel_u2<BS, TN>));
    MNB_GEMV_CONFIG_LIST(X)
#undef X
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


/* =======================================================================
 * Section 2b: WMMA kernel (supports both fused dequant and fp16 B)
 * ======================================================================= */

/* WMMA kernel entry points with per-variant GPU attributes */



/* fp16 GEMM kernel entry (for pre-dequantized B, FUSED_DQ=false) */


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
    // 0 means the default bm/(wt_m*wt_n). WT2x4 needs it set explicitly: the
    // default would be 16, and a K-step of 16 doubles the barrier count.
    int  bk = 0;
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
    // --- WT_N=4 fused, explicit BK=32 (fewer LDS reads per WMMA, and half the
    //     threads per workgroup so two workgroups share a WGP and one can issue
    //     WMMA while the other sits at its barrier). Requires group_size % 16.
    {128, 128,  4, true, 2, 4, 32}, {128, 128,  8, true, 2, 4, 32},
    {128, 128, 16, true, 2, 4, 32}, {128, 128, 32, true, 2, 4, 32},
    {128, 256,  4, true, 2, 4, 32}, {128, 256,  8, true, 2, 4, 32},
    {128, 256, 16, true, 2, 4, 32}, {128, 256, 32, true, 2, 4, 32},
    // BK=64 is not an option here: double-buffered (BM+BN)x(BK+8) halfs exceeds
    // the 64 KB per-workgroup LDS limit for both of these tiles.
    // --- WT_M=4, WT_N=4 fused, BK=32: 16 WMMA per 8 fragment loads, the best
    //     Matrix-Core-to-LDS ratio in the table, at 4 waves and ~190 VGPR per
    //     workgroup. Requires group_size % 32. ---
    {128, 128,  4, true, 4, 4, 32}, {128, 128,  8, true, 4, 4, 32},
    {128, 128, 16, true, 4, 4, 32}, {128, 128, 32, true, 4, 4, 32},
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

static constexpr int wmmaThreads(const WmmaConfig& c) {
    return (c.bm / (16 * c.wt_m)) * (c.bn / (16 * c.wt_n)) * 32;
}
static constexpr int wmmaBk(const WmmaConfig& c) {
    return c.bk ? c.bk : c.bm / (c.wt_m * c.wt_n);
}
// Nibbles of B one thread stages per K-step. They are contiguous in k and share
// a single scale/zero fetch, so the config is only valid when a quant group
// covers them: group_size % this == 0.
static constexpr int wmmaBNibblesPerThread(const WmmaConfig& c) {
    return c.bn * wmmaBk(c) / wmmaThreads(c);
}
static bool wmmaConfigApplies(const WmmaConfig& c, int N, int gs) {
    if (c.bn < 64 && N > 2 * c.bn) return false;
    return gs % wmmaBNibblesPerThread(c) == 0;
}

/* WMMA half of the LUT bridge; see the GEMV half above kGemvConfigs for the
 * cache-then-LUT-then-sweep ordering this serves. */

// wmmaConfigApplies needs the shape, so the validator carries it through ctx.
struct MatmulNbitsWmmaLutCtx {
    int n;
    int gs;
};

static int wmmaConfigFromGeometry(const matmul_lut::WmmaAnswer& g, int N, int gs) {
    for (int cid = 0; cid < kNumWmmaConfigs; cid++) {
        const WmmaConfig& c = kWmmaConfigs[cid];
        if (c.bm != g.bm || c.bn != g.bn || c.swizzle_n != g.swizzle_n ||
            c.wt_m != g.wt_m || c.wt_n != g.wt_n || c.fused != g.fused)
            continue;
        if (wmmaBk(c) != g.bk) continue;
        if (!wmmaConfigApplies(c, N, gs)) continue;
        return cid;
    }
    return -1;
}

static bool matmulNbitsWmmaValidator(void* ctx, const matmul_lut::WmmaAnswer& g) {
    const auto* c = static_cast<const MatmulNbitsWmmaLutCtx*>(ctx);
    return wmmaConfigFromGeometry(g, c->n, c->gs) >= 0;
}

// Returns a kWmmaConfigs index from the LUT, or -1 on a miss.
static int matmulNbitsLutWmmaConfig(int M, int N, int K, int gs, bool has_zp,
                           int b_row_bytes) {
    if (!matmulNbitsLutLookupEnabled()) return -1;   // online mode: sweep, skip the LUT
    matmul_lut::Request req;
    req.phase = matmul_lut::Phase::Prefill;
    req.bits = 4;
    req.m = M;
    req.n = N;
    req.k = K;
    req.group_size = gs;
    req.has_zp = has_zp;
    req.b_row_bytes = b_row_bytes;
    MatmulNbitsWmmaLutCtx ctx{N, gs};
    const matmul_lut::Result r =
        matmul_lut::resolve(req, matmulNbitsWmmaValidator, nullptr, &ctx);
    if (r.source == matmul_lut::Source::None) return -1;
    const int cid = wmmaConfigFromGeometry(r.wmma, N, gs);
    // A padded stride can only be served by a fused config; the dequant pass
    // behind a !fused one has no stride argument.
    if (cid >= 0 && b_row_bytes != 0 && !kWmmaConfigs[cid].fused) return -1;
    if (cid >= 0)
        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels] LUT hit (src=%d d=%.3f) M=%d N=%d K=%d gs=%d "
            "stride=%d -> config[%d]\n",
            static_cast<int>(r.source), r.distance, M, N, K, gs, b_row_bytes,
            cid);
    return cid;
}

// The (BM, BN) tiles the dispatch below can reach, each with the warp-tile,
// occupancy and BK template arguments it is instantiated with. Several
// kWmmaConfigs entries (one per swizzle_n) map onto the same tile, so this is a
// separate list; it drives both the dispatch and the hipRTC name expressions.
// BK is spelled out rather than left to the default because the WT_N=4 forms
// need one that differs from BM/(WT_M*WT_N).
#define MNB_WMMA_TILE_LIST(X)                                                 \
    X(128, 128, 2, 4, 8, 32) X(128, 256, 2, 4, 8, 32)                         \
    X(256, 128, 4, 2, 6, 32)                                                  \
    X(128,  32, 2, 1, 8, 64) X( 64,  32, 2, 1, 8, 32)                         \
    X( 64,  64, 2, 2, 8, 16) X( 64, 128, 2, 2, 8, 16)                         \
    X(128,  64, 2, 2, 8, 32) X(128, 256, 2, 2, 8, 32) X(128, 128, 2, 2, 8, 32)

// The fused dispatch reaches one more tile than the dequant+GEMM one, so its
// MatMulNBitsFp16GEMM instantiation would never be launched.
#define MNB_WMMA_FUSED_ONLY_TILE_LIST(X)                                      \
    X(128, 128, 4, 4, 8, 32)

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
    HIPDNN_RTC_LAUNCH(mnbRtcFunction, dq_grid, dq_block, 0, stream,
                      (B_packed, scales, zeros, B_fp16, N, K, gs, ngk),
                      dequant_u4_to_fp16);
}

static void launchWmmaConfig(
    int config_id, hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const _Float16* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp,
    _Float16* B_fp16_buf = nullptr, int b_row_bytes = 0)
{
    auto& c = kWmmaConfigs[config_id];
    bool need_bc = (M % c.bm != 0) || (N % c.bn != 0);

    if (!c.fused) {
        launchDequant(stream, B, scales, zeros, B_fp16_buf, N, K, gs, ngk);

#define FP16_GEMM_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, BK_V, BC_V) do { \
    constexpr int WM_ = BM_V / (WTM_V * 16); \
    constexpr int WN_ = BN_V / (WTN_V * 16); \
    constexpr int THR_ = WM_ * WN_ * 32; \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V), \
               static_cast<unsigned>((M + BM_V - 1) / BM_V)); \
    dim3 block_(THR_); \
    HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid_, block_, 0, stream, \
        (M, N, K, A, lda, B_fp16_buf, C, ldc, c.swizzle_n), \
        MatMulNBitsFp16GEMM<BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V, BK_V>); \
} while(0)

#define FP16_TILE_CASE(BM_V, BN_V, WTM_V, WTN_V, WEU_V, BK_V) \
    if (need_bc) FP16_GEMM_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, BK_V, true); \
    else         FP16_GEMM_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, BK_V, false)

        // The WT_N=4 forms carry a BK that differs from the default, so each
        // (tile, BK) pair needs its own case. Matching on wmmaBk() rather than
        // assuming 32 keeps a future table entry with a different BK from
        // silently launching the wrong instantiation.
        const int c_bk = wmmaBk(c);
        if      (c.wt_m == 2 && c.wt_n == 4 && c.bm == 128 && c.bn == 128 && c_bk == 32)
                                          FP16_TILE_CASE(128, 128, 2, 4, 8, 32);
        else if (c.wt_m == 2 && c.wt_n == 4 && c.bm == 128 && c.bn == 256 && c_bk == 32)
                                          FP16_TILE_CASE(128, 256, 2, 4, 8, 32);
        else if (c.wt_m == 4 && c.bm == 256 && c.bn == 128)
                                          FP16_TILE_CASE(256, 128, 4, 2, 6, 32);
        else if (c.wt_n == 1 && c.bm == 128 && c.bn == 32)
                                          FP16_TILE_CASE(128,  32, 2, 1, 8, 64);
        else if (c.wt_n == 1 && c.bm ==  64 && c.bn == 32)
                                          FP16_TILE_CASE( 64,  32, 2, 1, 8, 32);
        else if (c.bm ==  64 && c.bn ==  64) FP16_TILE_CASE( 64,  64, 2, 2, 8, 16);
        else if (c.bm ==  64 && c.bn == 128) FP16_TILE_CASE( 64, 128, 2, 2, 8, 16);
        else if (c.bm == 128 && c.bn ==  64) FP16_TILE_CASE(128,  64, 2, 2, 8, 32);
        else if (c.bm == 128 && c.bn == 256) FP16_TILE_CASE(128, 256, 2, 2, 8, 32);
        else                                 FP16_TILE_CASE(128, 128, 2, 2, 8, 32);

#undef FP16_TILE_CASE
#undef FP16_GEMM_LAUNCH
        return;
    }

#define WMMA_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, BK_V, BC_V) do { \
    constexpr int WM_ = BM_V / (WTM_V * 16); \
    constexpr int WN_ = BN_V / (WTN_V * 16); \
    constexpr int THR_ = WM_ * WN_ * 32; \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V), \
               static_cast<unsigned>((M + BM_V - 1) / BM_V)); \
    dim3 block_(THR_); \
    if (has_zp) { \
        HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid_, block_, 0, stream, \
            (M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n, \
             b_row_bytes), \
            MatMulNBitsWMMA_ZP<BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V, BK_V>); \
    } else { \
        HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid_, block_, 0, stream, \
            (M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc, c.swizzle_n, \
             b_row_bytes), \
            MatMulNBitsWMMA_NoZP<BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V, BK_V>); \
    } \
} while(0)

#define WMMA_TILE_CASE(BM_V, BN_V, WTM_V, WTN_V, WEU_V, BK_V) \
    if (need_bc) WMMA_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, BK_V, true); \
    else         WMMA_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, BK_V, false)

    // The WT_N=4 forms carry a BK that differs from the default, so each
    // (tile, BK) pair needs its own case. Matching on wmmaBk() rather than
    // assuming 32 keeps a future table entry with a different BK from silently
    // launching the wrong instantiation.
    const int c_bk = wmmaBk(c);
    if      (c.wt_m == 4 && c.wt_n == 4 && c.bm == 128 && c.bn == 128 && c_bk == 32)
                                      WMMA_TILE_CASE(128, 128, 4, 4, 8, 32);
    else if (c.wt_m == 2 && c.wt_n == 4 && c.bm == 128 && c.bn == 128 && c_bk == 32)
                                      WMMA_TILE_CASE(128, 128, 2, 4, 8, 32);
    else if (c.wt_m == 2 && c.wt_n == 4 && c.bm == 128 && c.bn == 256 && c_bk == 32)
                                      WMMA_TILE_CASE(128, 256, 2, 4, 8, 32);
    else if (c.wt_m == 4 && c.bm == 256 && c.bn == 128)
                                      WMMA_TILE_CASE(256, 128, 4, 2, 6, 32);
    else if (c.wt_n == 1 && c.bm == 128 && c.bn == 32)
                                      WMMA_TILE_CASE(128,  32, 2, 1, 8, 64);
    else if (c.wt_n == 1 && c.bm ==  64 && c.bn == 32)
                                      WMMA_TILE_CASE( 64,  32, 2, 1, 8, 32);
    else if (c.bm ==  64 && c.bn ==  64) WMMA_TILE_CASE( 64,  64, 2, 2, 8, 16);
    else if (c.bm ==  64 && c.bn == 128) WMMA_TILE_CASE( 64, 128, 2, 2, 8, 16);
    else if (c.bm == 128 && c.bn ==  64) WMMA_TILE_CASE(128,  64, 2, 2, 8, 32);
    else if (c.bm == 128 && c.bn == 256) WMMA_TILE_CASE(128, 256, 2, 2, 8, 32);
    else                                 WMMA_TILE_CASE(128, 128, 2, 2, 8, 32);

#undef WMMA_TILE_CASE
#undef WMMA_LAUNCH
}

static void appendWmmaNames(std::vector<std::string>& names) {
#define MNB_WMMA_FUSED_NAMES(BM, BN, WTM, WTN, WEU, BK) \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_ZP<BM, BN, WTM, WTN, WEU, false, BK>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_ZP<BM, BN, WTM, WTN, WEU, true, BK>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_NoZP<BM, BN, WTM, WTN, WEU, false, BK>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_NoZP<BM, BN, WTM, WTN, WEU, true, BK>));

#define X(BM, BN, WTM, WTN, WEU, BK) \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsFp16GEMM<BM, BN, WTM, WTN, WEU, false, BK>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsFp16GEMM<BM, BN, WTM, WTN, WEU, true, BK>)); \
    MNB_WMMA_FUSED_NAMES(BM, BN, WTM, WTN, WEU, BK)
    MNB_WMMA_TILE_LIST(X)
#undef X

#define X(BM, BN, WTM, WTN, WEU, BK) \
    MNB_WMMA_FUSED_NAMES(BM, BN, WTM, WTN, WEU, BK)
    MNB_WMMA_FUSED_ONLY_TILE_LIST(X)
#undef X
#undef MNB_WMMA_FUSED_NAMES
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


static void launchTranspose2DFp16(hipStream_t stream,
                                  const void* src, void* dst,
                                  int rows, int cols)
{
    constexpr int TILE = 32;
    dim3 block(TILE, 8);
    dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid, block, 0, stream,
                      (static_cast<const _Float16*>(src),
                       static_cast<_Float16*>(dst), rows, cols),
                      transpose2d_fp16_kernel);
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

// Zero-padded activation copy for the K%32!=0 int4 WMMA path (see the dispatch
// site). Sized [M, K_pad]; the caller writes the real [M, K] columns and zeroes
// the tail every call, so the buffer may be reused across shapes with different
// K_pad without stale tail data leaking into the padded reduction steps.
static _Float16* g_a_pad_buf = nullptr;
static size_t    g_a_pad_buf_elems = 0;

static _Float16* ensureAPadBuffer(int M, int K_pad) {
    size_t need = static_cast<size_t>(M) * K_pad;
    if (g_a_pad_buf && g_a_pad_buf_elems >= need) return g_a_pad_buf;
    if (g_a_pad_buf) hipFree(g_a_pad_buf);
    hipMalloc(&g_a_pad_buf, need * sizeof(_Float16));
    g_a_pad_buf_elems = need;
    return g_a_pad_buf;
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

/* b_row_bytes = 0 means the arrival stride, num_groups_k * (group_size/2).
 *
 * A non-zero stride has to be measured with rather than assumed away: a
 * power-of-two stride makes small tiles look up to 8x worse than they are
 * (64x64 at N=5120 K=4096: 1575 us unpadded against 193 us padded), so a table
 * tuned at one stride ranks the tiles wrongly for the other. Tuning at the
 * stride that will actually be used is the whole point of threading it here. */
static int tuneWmmaConfig(
    hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const _Float16* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp, int b_row_bytes = 0)
{
    hipEvent_t ev0, ev1;
    hipEventCreate(&ev0);
    hipEventCreate(&ev1);

    _Float16* dq_buf = ensureDqBuffer(N, K);

    // Separate-dequant configs read B only through launchDequant, which has no
    // stride argument, so at a padded stride they would decode the padding as
    // weights. Measuring them here would let the tuner crown a config the
    // padded path cannot serve.
    const bool fused_only = (b_row_bytes != 0);

    float best_ms = 1e30f;
    int   best_id = 9;
    constexpr int WARMUP = 1;
    constexpr int ITERS  = 5;

    /* Single in-order pass, keeping each config's batch time. Deliberately the
     * original, minimal measurement: the kernel tuner is on the runtime dispatch
     * path, so it stays simple and fast. Any more careful measurement (clock
     * settling, multi-pass, structural tie-breaks) belongs in the offline LUT
     * sweep, not here -- see autotune/matmul_nbits. */
    for (int cid = 0; cid < kNumWmmaConfigs; cid++) {
        auto& cfg = kWmmaConfigs[cid];
        if (!wmmaConfigApplies(cfg, N, gs)) continue;
        if (fused_only && !cfg.fused) continue;

        for (int w = 0; w < WARMUP; w++)
            launchWmmaConfig(cid, stream, M, N, K, A, lda, B,
                             scales, zeros, gs, ngk, C, ldc, has_zp, dq_buf,
                             b_row_bytes);

        hipEventRecord(ev0, stream);
        for (int i = 0; i < ITERS; i++)
            launchWmmaConfig(cid, stream, M, N, K, A, lda, B,
                             scales, zeros, gs, ngk, C, ldc, has_zp, dq_buf,
                             b_row_bytes);
        hipEventRecord(ev1, stream);
        hipEventSynchronize(ev1);

        float ms = 0.0f;
        hipEventElapsedTime(&ms, ev0, ev1);

        CUSTOM_KERNELS_DEBUG_LOG(
            "[custom_kernels]   wmma config[%2d] %s bm=%3d bn=%3d sw=%d "
            "wt=%dx%d bk=%2d : %.4f ms\n",
            cid, kWmmaConfigs[cid].fused ? "fused " : "dq+gem",
            kWmmaConfigs[cid].bm, kWmmaConfigs[cid].bn,
            kWmmaConfigs[cid].swizzle_n, kWmmaConfigs[cid].wt_m,
            kWmmaConfigs[cid].wt_n, wmmaBk(kWmmaConfigs[cid]),
            ms / ITERS);

        if (ms < best_ms) {
            best_ms = ms;
            best_id = cid;
        }
    }

    hipEventDestroy(ev0);
    hipEventDestroy(ev1);

    // The full tile geometry is logged, not just the config index, so an
    // offline consumer can record what won without depending on this table's
    // ordering -- indices shift whenever the table is edited.
    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] WMMA autotune: M=%d N=%d K=%d gs=%d zp=%s -> "
        "best config[%d] %s bm=%d bn=%d sw=%d wt=%dx%d bk=%d (%.4f ms/iter)\n",
        M, N, K, gs, has_zp ? "yes" : "no",
        best_id, kWmmaConfigs[best_id].fused ? "fused" : "dq+gemm",
        kWmmaConfigs[best_id].bm, kWmmaConfigs[best_id].bn,
        kWmmaConfigs[best_id].swizzle_n, kWmmaConfigs[best_id].wt_m,
        kWmmaConfigs[best_id].wt_n, wmmaBk(kWmmaConfigs[best_id]),
        best_ms / ITERS);

    return best_id;
}

static std::unordered_map<std::string, int> wmma_cache;
static std::mutex wmma_tune_mutex;

/* The stride is part of the key because it changes the answer, not just the
 * timing: the winner at the arrival stride is the most alias-tolerant tile,
 * which is a different tile from the fastest one at a padded stride (at
 * N=5120 K=4096, cfg11 -> cfg3). Appended only when padding is in use, so
 * entries written by earlier builds for the arrival stride still hit. */
static std::string wmmaCacheKey(int M, int N, int K, int gs, bool has_zp,
                                int b_row_bytes = 0) {
    std::string key = std::to_string(M) + "_" + std::to_string(N) + "_" +
                      std::to_string(K) + "_" + std::to_string(gs) + "_" +
                      (has_zp ? "1" : "0");
    if (b_row_bytes != 0) key += "_p" + std::to_string(b_row_bytes);
    return key;
}

// HIPDNN_MATMUL_NBITS_WMMA_CFG pins one entry of kWmmaConfigs and skips the
// autotune entirely. Profilers that place their capture window by raw dispatch
// index need this: the tuner's few hundred candidate dispatches would otherwise
// shift every later index, and the window would land on a different kernel
// without reporting an error. Unset or out of range keeps the autotuner.
static int forcedWmmaConfig() {
    static const int forced = [] {
        const char* v = nullptr;
#ifdef _WIN32
        char* dup = nullptr;
        size_t len = 0;
        if (_dupenv_s(&dup, &len, "HIPDNN_MATMUL_NBITS_WMMA_CFG") != 0)
            return -1;
        v = dup;
#else
        v = getenv("HIPDNN_MATMUL_NBITS_WMMA_CFG");
#endif
        int id = -1;
        if (v && v[0] != '\0') id = atoi(v);
#ifdef _WIN32
        free(dup);
#endif
        if (id < 0 || id >= kNumWmmaConfigs) return -1;
        return id;
    }();
    return forced;
}

static int getOrTuneWmmaConfig(
    hipStream_t stream,
    int M, int N, int K,
    const _Float16* A, int lda,
    const unsigned char* B,
    const _Float16* scales, const _Float16* zeros,
    int gs, int ngk,
    _Float16* C, int ldc, bool has_zp, int b_row_bytes = 0)
{
    const int forced = forcedWmmaConfig();
    // A forced config at a padded stride must still read B directly; the
    // dequant pass behind a !fused config has no stride argument.
    if (forced >= 0 && wmmaConfigApplies(kWmmaConfigs[forced], N, gs) &&
        (b_row_bytes == 0 || kWmmaConfigs[forced].fused))
        return forced;

    std::string key = wmmaCacheKey(M, N, K, gs, has_zp, b_row_bytes);

    std::lock_guard<std::mutex> lock(wmma_tune_mutex);

    auto it = wmma_cache.find(key);
    if (it != wmma_cache.end())
        return it->second;

    // Map miss: the offline table before the sweep. The stride is part of the
    // LUT key for the same reason it is part of the cache key -- a padded
    // buffer has a different winner.
    const int lut = matmulNbitsLutWmmaConfig(M, N, K, gs, has_zp, b_row_bytes);
    if (lut >= 0) {
        wmma_cache[key] = lut;
        return lut;
    }

    int best = tuneWmmaConfig(stream, M, N, K, A, lda, B,
                               scales, zeros, gs, ngk, C, ldc, has_zp,
                               b_row_bytes);
    wmma_cache[key] = best;

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] WMMA cache: stored key=\"%s\" -> config[%d] "
        "(cache size=%zu)\n",
        key.c_str(), best, wmma_cache.size());

    return best;
}

/* =======================================================================
 * Section 2b-padrow: row-stride padding for the fused int4 WMMA path
 *
 * The arrival row stride is num_groups_k * (group_size/2) bytes, which is K/2
 * whenever group_size divides K -- so a power-of-two K gives a power-of-two
 * stride. A 128-column tile then reads 128 rows that are an exact power-of-two
 * multiple of the 128-byte cache line apart, so they land in a handful of the
 * 64 cache sets and evict each other even though the 2 KB working set fits in
 * L1 easily. Adding one line to the stride makes it an odd number of lines and
 * spreads the rows over every set.
 *
 * Nothing is reordered, so the padded buffer is readable by any tile and the
 * autotuner stays free -- which is what separates this from a repack.
 * ======================================================================= */

/* Copies B into a buffer whose row stride is padded by one cache line, leaving
 * the within-row layout untouched. One thread per 16 bytes of a row. */
__global__ void u4_prepack_padrow_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int N, int row_bytes, int pad_bytes)
{
    const int chunk = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int n = static_cast<int>(blockIdx.y);
    const int chunks = (row_bytes + 15) / 16;
    if (n >= N || chunk >= chunks) return;

    const size_t off = static_cast<size_t>(chunk) * 16;
    const unsigned char* in = src + static_cast<size_t>(n) * row_bytes + off;
    unsigned char* out =
        dst + static_cast<size_t>(n) * (row_bytes + pad_bytes) + off;
    const int bytes = (off + 16 <= static_cast<size_t>(row_bytes))
                          ? 16 : (row_bytes - static_cast<int>(off));
    if (bytes == 16)
        *reinterpret_cast<u32x4*>(out) = *reinterpret_cast<const u32x4*>(in);
    else
        for (int i = 0; i < bytes; i++) out[i] = in[i];
}

// One cache line. Enough to take the stride off a power of two, and the
// smallest amount that does: the pathology is the line-granular set index.
static constexpr int kU4PadRowBytes = 128;

// Row-padded B keeps the arrival layout, so its size is just the padded rows.
extern "C" size_t hip_matmul_nbits_u4_padrow_bytes(int N, int K) {
    return static_cast<size_t>(N) * ((K / 2) + kU4PadRowBytes);
}

/* Callers are gated on group_size dividing K (see the host wrapper), so the
 * arrival stride num_groups_k*(group_size/2) equals K/2 here and the copy can
 * take the shape without the quant geometry. */
extern "C" int hip_matmul_nbits_u4_prepack_padrow(
    void* stream, const void* B, void* dst, int N, int K)
{
    const int row_bytes = K / 2;
    constexpr int BLK = 256;
    const int chunks = (row_bytes + 15) / 16;
    hipLaunchKernelGGL(
        u4_prepack_padrow_kernel,
        dim3(static_cast<unsigned>((chunks + BLK - 1) / BLK),
             static_cast<unsigned>(N)),
        dim3(BLK), 0, static_cast<hipStream_t>(stream),
        static_cast<const unsigned char*>(B), static_cast<unsigned char*>(dst),
        N, row_bytes, kU4PadRowBytes);
    return static_cast<int>(hipGetLastError());
}

/* Fused int4 WMMA prefill against a B whose row stride may be padded.
 *
 * cfg < 0 autotunes, and because the stride is passed through to the tuning
 * launches and into the cache key, the winner is chosen for the layout that
 * will actually be used. Returns -1 when no fused config can serve the shape,
 * which lets the caller fall back to the arrival-layout path. */
extern "C" int hip_matmul_nbits_u4_wmma_stride(
    void* stream, int cfg, int M, int N, int K, int group_size,
    const void* A, int lda, const void* B, const void* scales,
    const void* zeros, void* C, int ldc, int b_row_bytes)
{
    const int ngk = (K + group_size - 1) / group_size;
    const bool has_zp = (zeros != nullptr);
    auto s  = static_cast<hipStream_t>(stream);
    auto ap = static_cast<const _Float16*>(A);
    auto bp = static_cast<const unsigned char*>(B);
    auto sp = static_cast<const _Float16*>(scales);
    auto zp = static_cast<const _Float16*>(zeros);
    auto cp = static_cast<_Float16*>(C);

    int id = cfg;
    if (id < 0 || id >= kNumWmmaConfigs ||
        !wmmaConfigApplies(kWmmaConfigs[id], N, group_size))
        id = getOrTuneWmmaConfig(s, M, N, K, ap, lda, bp, sp, zp, group_size,
                                 ngk, cp, ldc, has_zp, b_row_bytes);

    // Only fused configs read B directly; a separate-dequant config would need
    // the dequant pass taught about the stride too.
    if (!kWmmaConfigs[id].fused) return -1;

    launchWmmaConfig(id, s, M, N, K, ap, lda, bp, sp, zp, group_size, ngk, cp,
                     ldc, has_zp, nullptr, b_row_bytes);
    return static_cast<int>(hipGetLastError());
}

// One reusable scratch for the padded-row-stride B copy, shared across every
// MatMulNBits layer. The padded buffer is only live for the duration of one
// layer's prefill -- it is a WMMA input, not persistent state -- and layers run
// in order on the stream, so a single grow-to-max buffer (like g_dq_buf above)
// replaces the former per-weight persistent cache that kept a full padded copy
// of every weight matrix alive for the whole session.
//
// EXTRA GPU MEMORY (the only allocation this feature adds):
//   bytes = max over padrow-eligible layers of  N * (K/2 + kU4PadRowBytes)
// It is a MAX over layers, not a sum: the scratch grows to the largest single
// layer it ever sees, then is reused (overwritten) by every later layer, so the
// footprint does not scale with layer count and never grows during a run.
//   N, K here are the weight's out/in dims (independent of prompt length).
//   Padrow-eligible = fused int4 prefill with M>=16, K%32==0, gs|K, shouldPadRow(K).
// Worked peaks (one loaded model at a time):
//   * Generation (lm_head runs at M=1, excluded): ~6 MB (small) up to ~115 MB
//     (DeepSeek-70B, N=28672 K=8192). Most 7-14B models are 15-70 MB.
//   * Full-sequence logits (e.g. PPL eval) pulls the vocab layer in at M>=16;
//     it then dominates at N_vocab*(K/2+128) ~= 0.5 GB (e.g. DeepSeek/Qwen vocab).
// Contrast: the old per-weight cache held ~all padrow weights at once (the SUM),
// e.g. +32 GB on DeepSeek-70B; this bounds it to the single MAX layer above.
//
// Hard cap: a layer whose padded copy would exceed the cap falls back to the
// arrival layout instead of sizing the scratch to it, so a vocab/lm_head
// projection or any other extreme shape cannot balloon this buffer. padrow is a
// bandwidth optimization, so a capped-out layer only loses that speedup, never
// correctness. Override with HIPDNN_MATMUL_PADROW_MAX_MB (0 disables padrow
// entirely; unset = default).
//
// 256 MB is chosen from measured benefit-vs-size on gfx1151, not arbitrarily:
//   * Every real *dense* layer (N <= 32768) is <= ~115 MB padded (DeepSeek-70B
//     FFN N=28672 K=8192) and gets 1.1-2.2x from padrow -- clearly worth it.
//     256 MB covers them all with ~2x headroom.
//   * The only shapes above the cap are vocab/lm_head (N=128k-256k, ~0.5 GB
//     padded), which only reach M>=16 in full-sequence-logits paths (e.g. PPL,
//     or [batch,seq,vocab] exports). There padrow's benefit at realistic
//     prefill M is small -- ~1.02x (K=4096) to ~1.13x (K=8192) at M=512 -- so
//     spending ~0.5 GB on them is a poor trade; skipping them is deliberate.
// Machines that want a different balance (e.g. lots of memory + PPL-heavy) can
// raise the cap via the env var above.
static constexpr size_t kU4PadRowDefaultMaxBytes = 256ull * 1024 * 1024;

static size_t matmulNbitsPadRowMaxBytes() {
    static const size_t cap = []() -> size_t {
        std::string raw;   // cross-platform env read, as in matmulNbitsLutLookupEnabled
#ifdef _WIN32
        char* v = nullptr;
        size_t len = 0;
        if (_dupenv_s(&v, &len, "HIPDNN_MATMUL_PADROW_MAX_MB") == 0 && v) {
            raw = v;
            free(v);
        }
#else
        if (const char* v = getenv("HIPDNN_MATMUL_PADROW_MAX_MB")) raw = v;
#endif
        if (raw.empty()) return kU4PadRowDefaultMaxBytes;
        unsigned long long mb = std::strtoull(raw.c_str(), nullptr, 10);
        return static_cast<size_t>(mb) * 1024ull * 1024ull;  // 0 => padrow off
    }();
    return cap;
}

static unsigned char* g_bpad_buf   = nullptr;
static size_t         g_bpad_bytes = 0;

static unsigned char* ensureBPadBuffer(size_t bytes) {
    if (g_bpad_buf && g_bpad_bytes >= bytes) return g_bpad_buf;
    if (g_bpad_buf) hipFree(g_bpad_buf);
    if (hipMalloc(&g_bpad_buf, bytes) != hipSuccess) {
        g_bpad_buf = nullptr;
        g_bpad_bytes = 0;
        return nullptr;
    }
    g_bpad_bytes = bytes;
    return g_bpad_buf;
}

/* Fused int4 WMMA prefill against a row-padded B, repacking into the shared
 * scratch on each call. This bounds padded-B memory to one layer rather than a
 * persistent per-weight copy: the repack is a single strided copy (cheap next
 * to the prefill GEMM), and the config autotune/cache still keys on the padded
 * stride via hip_matmul_nbits_u4_wmma_stride. Returns -1 (caller falls back to
 * the arrival layout) on scratch/repack failure or when no fused config fits. */
extern "C" int hip_matmul_nbits_u4_wmma_padrow(
    void* stream, int cfg, int M, int N, int K, int group_size,
    const void* A, int lda, const void* B, const void* scales,
    const void* zeros, void* C, int ldc)
{
    const size_t need = hip_matmul_nbits_u4_padrow_bytes(N, K);
    // Too big to pad (vocab/lm_head or other extreme shape): fall back to the
    // arrival layout so the shared scratch never balloons past the cap.
    if (need > matmulNbitsPadRowMaxBytes()) return -1;
    unsigned char* dst = ensureBPadBuffer(need);
    if (!dst) return -1;
    if (hip_matmul_nbits_u4_prepack_padrow(stream, B, dst, N, K) != 0) return -1;
    const int b_row_bytes = (K / 2) + kU4PadRowBytes;
    return hip_matmul_nbits_u4_wmma_stride(stream, cfg, M, N, K, group_size, A,
                                           lda, dst, scales, zeros, C, ldc,
                                           b_row_bytes);
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


/* int8 WMMA entry kernels: fused dequant only (no separate-DQ variant in v1) */



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

#define MNB_WMMA_I8_TILE_LIST(X)                                              \
    X( 64,  64) X( 64, 128) X(128,  64) X(128, 256) X(128, 128)

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
        HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid_, block_, 0, stream,           \
            (M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc,              \
             c.swizzle_n),                                                    \
            MatMulNBitsWMMA_I8_ZP<BM_V, BN_V, 2, 2, 8, BC_V>);                \
    } else {                                                                  \
        HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid_, block_, 0, stream,           \
            (M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc,              \
             c.swizzle_n),                                                    \
            MatMulNBitsWMMA_I8_NoZP<BM_V, BN_V, 2, 2, 8, BC_V>);              \
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

static void appendWmmaI8Names(std::vector<std::string>& names) {
#define X(BM, BN) \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_I8_ZP<BM, BN, 2, 2, 8, false>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_I8_ZP<BM, BN, 2, 2, 8, true>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_I8_NoZP<BM, BN, 2, 2, 8, false>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_I8_NoZP<BM, BN, 2, 2, 8, true>));
    MNB_WMMA_I8_TILE_LIST(X)
#undef X
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




/* -----------------------------------------------------------------------
 * WMMA-u3 Autotune: same fused-only tile subset as WMMA-i8 (kWmmaI8Configs
 * only uses BM in {64,128} with default WT_M=WT_N=2, so BK_T = BM/4 is
 * always in {16,32} -- both divide 32, so the block_size>=32 && %32==0
 * dispatch gate below guarantees group_size compatibility with every
 * config in this table; no per-config skip logic needed).
 * ----------------------------------------------------------------------- */


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
    HIPDNN_RTC_LAUNCH(mnbRtcFunction, dq_grid, dq_block, 0, stream,
                      (B, scales, zeros, B_fp16, N, K, gs, ngk),
                      dequant_u3_to_fp16);
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

#define MNB_WMMA_U3_TILE_LIST(X)                                              \
    X(256, 128, 4, 2, 6)                                                      \
    X( 64,  64, 2, 2, 8) X( 64, 128, 2, 2, 8)                                 \
    X(128,  64, 2, 2, 8) X(128, 256, 2, 2, 8) X(128, 128, 2, 2, 8)

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

#define U3_FP16_GEMM_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V) do {       \
    constexpr int WM_  = BM_V / (WTM_V * 16);                                 \
    constexpr int WN_  = BN_V / (WTN_V * 16);                                 \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid_, block_, 0, stream,               \
        (M, N, K, A, lda, B_fp16_buf, C, ldc, c.swizzle_n),                   \
        MatMulNBitsFp16GEMM<BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V>);          \
} while (0)

#define U3_FP16_TILE_CASE(BM_V, BN_V, WTM_V, WTN_V, WEU_V) \
    if (need_bc) U3_FP16_GEMM_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, true); \
    else         U3_FP16_GEMM_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, false)

             if (c.wt_m == 4 && c.bm == 256 && c.bn == 128)
                                    U3_FP16_TILE_CASE(256, 128, 4, 2, 6);
        else if (c.bm ==  64 && c.bn ==  64)
                                    U3_FP16_TILE_CASE( 64,  64, 2, 2, 8);
        else if (c.bm ==  64 && c.bn == 128)
                                    U3_FP16_TILE_CASE( 64, 128, 2, 2, 8);
        else if (c.bm == 128 && c.bn ==  64)
                                    U3_FP16_TILE_CASE(128,  64, 2, 2, 8);
        else if (c.bm == 128 && c.bn == 256)
                                    U3_FP16_TILE_CASE(128, 256, 2, 2, 8);
        else                        U3_FP16_TILE_CASE(128, 128, 2, 2, 8);

#undef U3_FP16_TILE_CASE
#undef U3_FP16_GEMM_LAUNCH
        return;
    }

#define WMMA_U3_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V) do {            \
    constexpr int WM_  = BM_V / (WTM_V * 16);                                 \
    constexpr int WN_  = BN_V / (WTN_V * 16);                                 \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    if (has_zp) {                                                             \
        HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid_, block_, 0, stream,           \
            (M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc,              \
             c.swizzle_n),                                                    \
            MatMulNBitsWMMA_U3_ZP<BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V>);    \
    } else {                                                                  \
        HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid_, block_, 0, stream,           \
            (M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc,              \
             c.swizzle_n),                                                    \
            MatMulNBitsWMMA_U3_NoZP<BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V>);  \
    }                                                                         \
} while (0)

#define WMMA_U3_TILE_CASE(BM_V, BN_V, WTM_V, WTN_V, WEU_V) \
    if (need_bc) WMMA_U3_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, true);  \
    else         WMMA_U3_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, false)

         if (c.wt_m == 4 && c.bm == 256 && c.bn == 128)
                                WMMA_U3_TILE_CASE(256, 128, 4, 2, 6);
    else if (c.bm ==  64 && c.bn ==  64)
                                WMMA_U3_TILE_CASE( 64,  64, 2, 2, 8);
    else if (c.bm ==  64 && c.bn == 128)
                                WMMA_U3_TILE_CASE( 64, 128, 2, 2, 8);
    else if (c.bm == 128 && c.bn ==  64)
                                WMMA_U3_TILE_CASE(128,  64, 2, 2, 8);
    else if (c.bm == 128 && c.bn == 256)
                                WMMA_U3_TILE_CASE(128, 256, 2, 2, 8);
    else                        WMMA_U3_TILE_CASE(128, 128, 2, 2, 8);

#undef WMMA_U3_TILE_CASE
#undef WMMA_U3_LAUNCH
}

static void appendWmmaU3Names(std::vector<std::string>& names) {
#define X(BM, BN, WTM, WTN, WEU) \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_U3_ZP<BM, BN, WTM, WTN, WEU, false>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_U3_ZP<BM, BN, WTM, WTN, WEU, true>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_U3_NoZP<BM, BN, WTM, WTN, WEU, false>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_U3_NoZP<BM, BN, WTM, WTN, WEU, true>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsFp16GEMM<BM, BN, WTM, WTN, WEU, false>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsFp16GEMM<BM, BN, WTM, WTN, WEU, true>));
    MNB_WMMA_U3_TILE_LIST(X)
#undef X
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
    HIPDNN_RTC_LAUNCH(mnbRtcFunction, dq_grid, dq_block, 0, stream,
                      (B, scales, zeros, B_fp16, N, K, gs, ngk),
                      dequant_u2_to_fp16);
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

#define MNB_WMMA_U2_TILE_LIST(X)                                              \
    X(256, 128, 4, 2, 6)                                                      \
    X( 64,  64, 2, 2, 8) X( 64, 128, 2, 2, 8)                                 \
    X(128,  64, 2, 2, 8) X(128, 256, 2, 2, 8) X(128, 128, 2, 2, 8)

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

#define U2_FP16_GEMM_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V) do {       \
    constexpr int WM_  = BM_V / (WTM_V * 16);                                 \
    constexpr int WN_  = BN_V / (WTN_V * 16);                                 \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid_, block_, 0, stream,               \
        (M, N, K, A, lda, B_fp16_buf, C, ldc, c.swizzle_n),                   \
        MatMulNBitsFp16GEMM<BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V>);          \
} while (0)

#define U2_FP16_TILE_CASE(BM_V, BN_V, WTM_V, WTN_V, WEU_V) \
    if (need_bc) U2_FP16_GEMM_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, true); \
    else         U2_FP16_GEMM_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, false)

             if (c.wt_m == 4 && c.bm == 256 && c.bn == 128)
                                    U2_FP16_TILE_CASE(256, 128, 4, 2, 6);
        else if (c.bm ==  64 && c.bn ==  64)
                                    U2_FP16_TILE_CASE( 64,  64, 2, 2, 8);
        else if (c.bm ==  64 && c.bn == 128)
                                    U2_FP16_TILE_CASE( 64, 128, 2, 2, 8);
        else if (c.bm == 128 && c.bn ==  64)
                                    U2_FP16_TILE_CASE(128,  64, 2, 2, 8);
        else if (c.bm == 128 && c.bn == 256)
                                    U2_FP16_TILE_CASE(128, 256, 2, 2, 8);
        else                        U2_FP16_TILE_CASE(128, 128, 2, 2, 8);

#undef U2_FP16_TILE_CASE
#undef U2_FP16_GEMM_LAUNCH
        return;
    }

#define WMMA_U2_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V) do {            \
    constexpr int WM_  = BM_V / (WTM_V * 16);                                 \
    constexpr int WN_  = BN_V / (WTN_V * 16);                                 \
    constexpr int THR_ = WM_ * WN_ * 32;                                      \
    dim3 grid_(static_cast<unsigned>((N + BN_V - 1) / BN_V),                  \
               static_cast<unsigned>((M + BM_V - 1) / BM_V));                 \
    dim3 block_(THR_);                                                        \
    if (has_zp) {                                                             \
        HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid_, block_, 0, stream,           \
            (M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc,              \
             c.swizzle_n),                                                    \
            MatMulNBitsWMMA_U2_ZP<BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V>);    \
    } else {                                                                  \
        HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid_, block_, 0, stream,           \
            (M, N, K, A, lda, B, scales, zeros, gs, ngk, C, ldc,              \
             c.swizzle_n),                                                    \
            MatMulNBitsWMMA_U2_NoZP<BM_V, BN_V, WTM_V, WTN_V, WEU_V, BC_V>);  \
    }                                                                         \
} while (0)

#define WMMA_U2_TILE_CASE(BM_V, BN_V, WTM_V, WTN_V, WEU_V) \
    if (need_bc) WMMA_U2_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, true);  \
    else         WMMA_U2_LAUNCH(BM_V, BN_V, WTM_V, WTN_V, WEU_V, false)

         if (c.wt_m == 4 && c.bm == 256 && c.bn == 128)
                                WMMA_U2_TILE_CASE(256, 128, 4, 2, 6);
    else if (c.bm ==  64 && c.bn ==  64)
                                WMMA_U2_TILE_CASE( 64,  64, 2, 2, 8);
    else if (c.bm ==  64 && c.bn == 128)
                                WMMA_U2_TILE_CASE( 64, 128, 2, 2, 8);
    else if (c.bm == 128 && c.bn ==  64)
                                WMMA_U2_TILE_CASE(128,  64, 2, 2, 8);
    else if (c.bm == 128 && c.bn == 256)
                                WMMA_U2_TILE_CASE(128, 256, 2, 2, 8);
    else                        WMMA_U2_TILE_CASE(128, 128, 2, 2, 8);

#undef WMMA_U2_TILE_CASE
#undef WMMA_U2_LAUNCH
}

static void appendWmmaU2Names(std::vector<std::string>& names) {
#define X(BM, BN, WTM, WTN, WEU) \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_U2_ZP<BM, BN, WTM, WTN, WEU, false>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_U2_ZP<BM, BN, WTM, WTN, WEU, true>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_U2_NoZP<BM, BN, WTM, WTN, WEU, false>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsWMMA_U2_NoZP<BM, BN, WTM, WTN, WEU, true>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsFp16GEMM<BM, BN, WTM, WTN, WEU, false>)); \
    names.push_back( \
        HIPDNN_RTC_STR(MatMulNBitsFp16GEMM<BM, BN, WTM, WTN, WEU, true>));
    MNB_WMMA_U2_TILE_LIST(X)
#undef X
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
    HIPDNN_RTC_LAUNCH(mnbRtcFunction,
                      grid, block, 0, static_cast<hipStream_t>(stream),
                      (static_cast<const uint8_t*>(zp_packed),
                       static_cast<uint8_t*>(dst_u8),
                       N, groups_k, packed_cols),
                      unpack_zp_nibbles_to_u8_kernel);
}

extern "C" void hip_matmul_nbits_unpack_zp_u8_2bit(
    void* stream, const void* zp_packed, void* dst_u8, int N, int groups_k)
{
    int packed_cols = (groups_k + 3) / 4;
    constexpr int THR = 256;
    dim3 grid(static_cast<unsigned>((packed_cols + THR - 1) / THR),
              static_cast<unsigned>(N));
    dim3 block(THR);
    HIPDNN_RTC_LAUNCH(mnbRtcFunction,
                      grid, block, 0, static_cast<hipStream_t>(stream),
                      (static_cast<const uint8_t*>(zp_packed),
                       static_cast<uint8_t*>(dst_u8),
                       N, groups_k, packed_cols),
                      unpack_zp_2bit_to_u8_kernel);
}

extern "C" void hip_matmul_nbits_unpack_zp_u8_3bit(
    void* stream, const void* zp_packed, void* dst_u8, int N, int groups_k)
{
    int packed_cols = (groups_k * 3 + 7) / 8;
    constexpr int THR = 256;
    dim3 grid(static_cast<unsigned>((groups_k + THR - 1) / THR),
              static_cast<unsigned>(N));
    dim3 block(THR);
    HIPDNN_RTC_LAUNCH(mnbRtcFunction,
                      grid, block, 0, static_cast<hipStream_t>(stream),
                      (static_cast<const uint8_t*>(zp_packed),
                       static_cast<uint8_t*>(dst_u8),
                       N, groups_k, packed_cols),
                      unpack_zp_3bit_to_u8_kernel);
}

extern "C" void hip_matmul_nbits_convert_zp_fp16(
    void* stream, const void* zp_packed, void* dst_fp16, int N, int groups_k)
{
    int packed_cols = (groups_k + 1) / 2;
    constexpr int THR = 256;
    dim3 grid(static_cast<unsigned>((packed_cols + THR - 1) / THR),
              static_cast<unsigned>(N));
    dim3 block(THR);
    HIPDNN_RTC_LAUNCH(mnbRtcFunction,
                      grid, block, 0, static_cast<hipStream_t>(stream),
                      (static_cast<const unsigned char*>(zp_packed),
                       static_cast<_Float16*>(dst_fp16),
                       N, groups_k, packed_cols),
                      unpack_zp_u8_to_fp16_kernel);
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
    HIPDNN_RTC_LAUNCH(mnbRtcFunction, grid, block, 0,
                      static_cast<hipStream_t>(stream),
                      (static_cast<const unsigned char*>(B_packed),
                       static_cast<const _Float16*>(scales_fp16),
                       static_cast<const _Float16*>(zeros_fp16),
                       static_cast<_Float16*>(B_fp16_out),
                       N, K, group_size, num_groups_k),
                      dequant_u4_to_fp16);
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
        HIPDNN_RTC_LAUNCH(
            mnbRtcFunction, bias_grid, bias_block, 0, hip_stream,
            (o_ptr, static_cast<const _Float16*>(bias), iM, iN, ldc),
            matmul_nbits_add_bias_rowmajor);
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

      HIPDNN_RTC_LAUNCH(
          mnbRtcFunction, grd, blk, 0, hip_stream,
          (static_cast<const __half*>(A), static_cast<const uint8_t*>(B),
           static_cast<const __half*>(scales),
           static_cast<const uint8_t*>(zero_points),
           static_cast<const __half*>(bias), static_cast<__half*>(output),
           M, N, K, block_size),
          matmul_nbits_kernel_i8);
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
        HIPDNN_RTC_LAUNCH(
            mnbRtcFunction, bias_grid, bias_block, 0, hip_stream,
            (o_ptr, static_cast<const _Float16*>(bias), iM, iN, ldc),
            matmul_nbits_add_bias_rowmajor);
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

      HIPDNN_RTC_LAUNCH(
          mnbRtcFunction, grd, blk, 0, hip_stream,
          (static_cast<const __half*>(A), static_cast<const uint8_t*>(B),
           static_cast<const __half*>(scales),
           static_cast<const uint8_t*>(z_u2),
           static_cast<const __half*>(bias), static_cast<__half*>(output),
           M, N, K, block_size),
          matmul_nbits_kernel_u2);
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
        HIPDNN_RTC_LAUNCH(
            mnbRtcFunction, bias_grid, bias_block, 0, hip_stream,
            (o_ptr, static_cast<const _Float16*>(bias), iM, iN, ldc),
            matmul_nbits_add_bias_rowmajor);
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

      HIPDNN_RTC_LAUNCH(
          mnbRtcFunction, grd, blk, 0, hip_stream,
          (static_cast<const __half*>(A), static_cast<const uint8_t*>(B),
           static_cast<const __half*>(scales),
           static_cast<const uint8_t*>(z_u3),
           static_cast<const __half*>(bias), static_cast<__half*>(output),
           M, N, K, block_size),
          matmul_nbits_kernel_u3);
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
  //   - FP16 for WMMA and col-major GEMV paths
  //   - Individual uint8 for row-major GEMV and naive fallback paths
  // When zp_elem_size == 2, zero_points is already FP16 [N, num_groups_k]
  // (not nibble-packed) and is used as zp_for_fp16_paths directly, with no
  // conversion. zp_for_u8_paths is deliberately left unset in that case: the
  // dispatch below never reads it when zp_elem_size==2, because M==1 is
  // routed to the FP16-capable col-major GEMV kernel instead of the
  // uint8-only row-major one (see fp16_zp_ready below) -- reinterpreting the
  // FP16 buffer as raw uint8 bytes there previously produced garbage
  // zero-points.
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
  // True once zp_for_fp16_paths is a genuinely-ready FP16 buffer (or there
  // are no zero points at all): zp_elem_size==2 supplies it directly above;
  // zp_elem_size==1 only pre-builds pre_unpacked_zp_fp16 for M>1 (see the
  // wrapper), so M==1 with zp_elem_size==1 is deliberately excluded here and
  // keeps using the row-major kernel's pre_unpacked_zp_u8 below instead of
  // paying for an extra fp16 conversion on the M==1 decode hot path.
  const bool fp16_zp_ready =
      !zero_points || zp_elem_size == 2 ||
      (zp_elem_size == 1 && pre_unpacked_zp_fp16 != nullptr);

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
  // Any cutoff is safe for partial tiles: need_bc in launchWmmaConfig selects
  // the bounds-checked variant whenever M is not a multiple of the tile.
  static constexpr int kWmmaMinM = 8;

  /* int4 prefill with K%32!=0 (e.g. a model whose intermediate_size is not a
   * multiple of 32, so down_proj has K=4304): the WMMA kernel requires K%32==0,
   * so this shape would otherwise fall to the naive kernel on every prefill.
   *
   * Pad K up to the next multiple of 32 and run the *unchanged* WMMA path on a
   * zero-padded copy of A. The extra reduction steps read zero activations, so
   * they contribute nothing and the [M,N] output is unchanged -- no output
   * cropping, since only the contraction dim is padded. The weight rows are
   * already stored padded to full groups (num_groups_k*(group_size/2) bytes),
   * so B needs nothing. K%32==0 shapes never enter here and are byte-for-byte
   * unaffected. Verified against the CPU reference for K=4304. */
  if ((batch_count == 1) && (K % 32 != 0) && M >= kWmmaMinM &&
      !hipdnn_device_is_wave64()) {
    int iM = static_cast<int>(M);
    int iN = static_cast<int>(N);
    int iK = static_cast<int>(K);
    int gs = static_cast<int>(block_size);
    int K_pad = (iK + 31) & ~31;
    // ceil(K/gs); the last group is partial but its storage is padded to full.
    int ngk = static_cast<int>((K + block_size - 1) / block_size);
    int ldc = iN;

    const auto* b_ptr = static_cast<const unsigned char*>(B);
    const auto* s_ptr = static_cast<const _Float16*>(scales);
    const auto* z_ptr = static_cast<const _Float16*>(zp_for_fp16_paths);
    auto* out_ptr = static_cast<_Float16*>(output);
    const bool has_zp = (z_ptr != nullptr);

    CUSTOM_KERNELS_DEBUG_LOG(
        "[custom_kernels] hip_matmul_nbits: WMMA K-pad path M=%d N=%d K=%d -> "
        "K_pad=%d bs=%d zp=%s bias=%s\n",
        iM, iN, iK, K_pad, gs, zero_points ? "yes" : "null",
        bias ? "yes" : "null");

    // Zero-padded copy of A: real [M,K] columns, then the K_pad-K tail zeroed.
    // The tail is re-zeroed every call so the buffer can be reused across
    // shapes with a different K_pad.
    _Float16* a_pad = ensureAPadBuffer(iM, K_pad);
    hipMemcpy2DAsync(a_pad, static_cast<size_t>(K_pad) * sizeof(_Float16), A,
                     static_cast<size_t>(iK) * sizeof(_Float16),
                     static_cast<size_t>(iK) * sizeof(_Float16), iM,
                     hipMemcpyDeviceToDevice, hip_stream);
    hipMemset2DAsync(
        reinterpret_cast<char*>(a_pad) + static_cast<size_t>(iK) * sizeof(_Float16),
        static_cast<size_t>(K_pad) * sizeof(_Float16), 0,
        static_cast<size_t>(K_pad - iK) * sizeof(_Float16), iM, hip_stream);

    int wmma_cfg = getOrTuneWmmaConfig(hip_stream, iM, iN, K_pad, a_pad, K_pad,
                                       b_ptr, s_ptr, z_ptr, gs, ngk, out_ptr,
                                       ldc, has_zp);
    _Float16* dq_buf =
        kWmmaConfigs[wmma_cfg].fused ? nullptr : ensureDqBuffer(iN, K_pad);
    launchWmmaConfig(wmma_cfg, hip_stream, iM, iN, K_pad, a_pad, K_pad, b_ptr,
                     s_ptr, z_ptr, gs, ngk, out_ptr, ldc, has_zp, dq_buf);

    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
      fprintf(stderr,
              "[custom_kernels] hip_matmul_nbits WMMA K-pad launch failed: %s\n",
              hipGetErrorString(err));
      return static_cast<int>(err);
    }

    if (bias) {
      dim3 bias_grid(static_cast<unsigned>((iN + 255) / 256),
                     static_cast<unsigned>(iM));
      hipLaunchKernelGGL(matmul_nbits_add_bias_rowmajor, bias_grid, dim3(256), 0,
                         hip_stream, out_ptr,
                         static_cast<const _Float16*>(bias), iM, iN, ldc);
    }
    return static_cast<int>(hipGetLastError());
  }

  // WMMA is an RDNA3/RDNA4 (wave32) intrinsic. On CDNA (wave64, e.g. MI350)
  // there is no WMMA, so the prefill fast path is unavailable here; int4
  // M>=kWmmaMinM is instead served by the dequant-once + hipBLASLt path in
  // lib/Runtime/real/matmul_nbits.cpp, and any residual case falls through to
  // the correct GEMV/naive fallback below.
  if (wmma_data_format && M >= kWmmaMinM && !hipdnn_device_is_wave64()) {
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
      HIPDNN_RTC_LAUNCH(
          mnbRtcFunction, bias_grid, bias_block, 0, hip_stream,
          (out_ptr, static_cast<const _Float16*>(bias), iM, iN, ldc),
          matmul_nbits_add_bias_rowmajor);

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
   * GEMV path with col-major data — small M, WMMA data format
   *
   * When batch==1, K%32==0, M < kWmmaMinM: internally transpose A row->col
   * and output col->row so the caller sees row-major I/O (skipped for M==1,
   * where row-major and col-major are the identical single row/column).
   * GEMV kernel runs with COL_MAJOR=true, reading zero_points as FP16
   * directly -- the only kernel variant that can, so M==1 must also take
   * this path whenever an FP16 zp buffer is actually available
   * (fp16_zp_ready above); M==1 with zp_elem_size==1 and no pre-built
   * pre_unpacked_zp_fp16 instead falls through to the row-major path below,
   * which already has the uint8 zp it needs at no extra cost.
   * ------------------------------------------------------------------- */
  if (wmma_data_format && M < kWmmaMinM && (M > 1 || fp16_zp_ready)) {
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

  HIPDNN_RTC_LAUNCH(
      mnbRtcFunction, grd, blk, 0, hip_stream,
      (static_cast<const __half*>(A), static_cast<const uint8_t*>(B),
       static_cast<const __half*>(scales),
       static_cast<const uint8_t*>(zp_for_u8_paths),
       static_cast<const __half*>(bias), static_cast<__half*>(output),
       M, N, K, block_size),
      matmul_nbits_kernel);

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    fprintf(stderr,
            "[custom_kernels] hip_matmul_nbits naive launch failed: %s\n",
            hipGetErrorString(err));
  }
  return static_cast<int>(err);
}

// ---------------------------------------------------------------------------
// hipRTC module
//
// Every family contributes its name expressions through the same X-macro lists
// that drive its launch switch, so a registered name and a launched one cannot
// drift apart. All of them are registered before the single compile: hipRTC
// resolves name expressions while parsing.
// ---------------------------------------------------------------------------

// The kernels launched outside a config-list family: one instance each, so the
// name expression is spelled here rather than generated.
static void appendMiscNames(std::vector<std::string>& names) {
  names.push_back(HIPDNN_RTC_STR(matmul_nbits_kernel));
  names.push_back(HIPDNN_RTC_STR(matmul_nbits_kernel_i8));
  names.push_back(HIPDNN_RTC_STR(matmul_nbits_kernel_u3));
  names.push_back(HIPDNN_RTC_STR(matmul_nbits_kernel_u2));
  names.push_back(HIPDNN_RTC_STR(matmul_nbits_quant_act_i8_kernel<128>));
  names.push_back(HIPDNN_RTC_STR(matmul_nbits_add_bias_rowmajor));
  names.push_back(HIPDNN_RTC_STR(dequant_u4_to_fp16));
  names.push_back(HIPDNN_RTC_STR(dequant_u3_to_fp16));
  names.push_back(HIPDNN_RTC_STR(dequant_u2_to_fp16));
  names.push_back(HIPDNN_RTC_STR(transpose2d_fp16_kernel));
  names.push_back(HIPDNN_RTC_STR(unpack_zp_nibbles_to_u8_kernel));
  names.push_back(HIPDNN_RTC_STR(unpack_zp_2bit_to_u8_kernel));
  names.push_back(HIPDNN_RTC_STR(unpack_zp_3bit_to_u8_kernel));
  names.push_back(HIPDNN_RTC_STR(unpack_zp_u8_to_fp16_kernel));
}

static hipdnn_ep::rtc::Module& mnbRtcModule() {
  static hipdnn_ep::rtc::Module module(
      "matmul_nbits", reinterpret_cast<const char*>(kMatMulNBitsDeviceSource),
      kMatMulNBitsDeviceSource_size, [] {
        std::vector<std::string> names;
        appendGemvNames(names);
        appendDp4aNames(names);
        appendGemvI8Names(names);
        appendGemvU3Names(names);
        appendGemvU2Names(names);
        appendWmmaNames(names);
        appendWmmaI8Names(names);
        appendWmmaU3Names(names);
        appendWmmaU2Names(names);
        appendMiscNames(names);
        // The u2/u3 dequant paths reuse the bits=4 fp16 GEMM tiles, so the
        // lists overlap; a repeated name expression would be registered twice.
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());
        return names;
      }());
  return module;
}

static hipFunction_t mnbRtcFunction(const char* name_expression) {
  if (!hipdnn_ep::rtc::enabled()) {
    return nullptr;
  }
  return mnbRtcModule().getFunction(name_expression);
}
