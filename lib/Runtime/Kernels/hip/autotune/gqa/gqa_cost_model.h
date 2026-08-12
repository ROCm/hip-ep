/*
 * Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ============================================================================
// GQA dispatch cost model -- tier 2 of the autotune policy.
//
// Tier 1  offline LUT       lib/Runtime/real/gqa_autotune.cpp, keyed off
//                           gqa_autotune.fb shipped with EPContext. Exact or
//                           bucketed hit -> use it verbatim.
// Tier 2  THIS FILE         no LUT entry: score every candidate analytically
//                           and take the best. No GPU launch, no timing.
// Tier 3  heuristic         model declines (unknown geometry, missing resource
//                           entry): a fixed config that is merely known to run.
//
// One function:
//
//   double hip_gqa_config_score(shape, candidate);   // higher is better
//
// Score every candidate, keep the highest:
//
//   for (cand : candidates)
//     if (hip_gqa_config_score(&shape, &cand) > best) best = ...;
//
// The score is predicted useful MACs per unit time, i.e. a throughput, so it is
// comparable across shapes and its ranking within one shape is exactly the
// ranking of predicted runtime. It returns 0 for a candidate that is not a real
// kernel instantiation, so callers can score blindly and let those rank last --
// that zero is also the signal to fall through to tier 3.
//
// ---------------------------------------------------------------------------
// The model
//
//   makespan  =  block_waves x per_block_cost   (+ c5 * reduce, decode only)
//
//     block_waves    = ceil(blocks / (CUs * blocks_per_cu))
//     per_block_cost = c0                                fixed / launch
//                    + c1 * kv_tiles * BKV * d / waves_per_block   math, split over the block's waves
//                    + c2 * kv_tiles * BKV * d                     KV bytes, shared by the block, NOT split
//                    + c3 * kv_tiles * scratch_bytes               spill traffic, once per KV loop trip
//                    + c4 * scratch_bytes                          spill traffic that does NOT scale with
//                                                                  the loop: staging Q into registers on
//                                                                  entry and draining the accumulators on
//                                                                  exit happens once per block
//
// `blocks_per_cu`, `waves_per_block` and `scratch_bytes` are compile-time facts
// about each kernel instantiation, read out of
//   hipcc -Rpass-analysis=kernel-resource-usage
// and tabulated in the .cpp. Only c0..c5 are fitted. That split matters: the
// occupancy numbers move by themselves when a kernel is edited, so the model
// tracks kernel changes instead of silently going stale.
//
// Why these terms. Every tile-size knob trades the same two things:
//   * bigger tiles  -> fewer blocks, more KV reuse per block, but more registers
//                      (and past ~256 VGPRs, spill traffic on every loop trip)
//   * smaller tiles -> more blocks to fill the CUs, no spill, but the KV tile is
//                      re-read by more blocks
// c1 vs c2 prices that trade; c3/c4 make a spilling config lose while its KV
// loop is too short to pay for the spill. c4 is what reproduces the sliding
// window behaviour: a short loop leaves the fixed spill dominating, so the
// spill-free config wins despite launching twice the blocks. Without c4 the
// model got the windowed case right and full attention wrong by ~20%.
//
// ---------------------------------------------------------------------------
// Accuracy, measured on gfx1151 / 20 CU over 372 shapes (captured from models +
// a synthetic grid + threshold sweeps; see RdpCapture/ops_analyze/gqa/)
//
//   path            picks the true best   mean loss   worst loss
//   prefill_v8      41/41  (100%)           +0.00%       +0.0%
//   flash_decode    48/71  ( 68%)           +2.70%      +22.1%
//   prefill_v7      64/144 ( 44%)           +2.92%      +77.2%
//   prefill_v5      88/116 ( 76%)           +3.77%      +57.0%
//
// On the 59 shapes actually captured from models: 69% exact, +2.49% mean, 92%
// within 5% of the measured optimum.
//
// The hit rate understates the model -- most misses are ties. What matters is
// the mean loss, and ~3% is well inside the 1.3x-3.4x spread between the best
// and worst candidate that the autotuner exists to avoid. v7's low hit rate is
// not a modelling failure either: its two live configs are within 5% of each
// other on almost every shape, so which one "wins" is largely measurement noise.
//
// The coefficients are device-specific (they encode this part's math:bandwidth
// ratio). Refit with RdpCapture/ops_analyze/gqa/tools/fit_cost_model.py after
// running the sweeps on a new part; a wrong-device fit degrades the pick, it
// does not produce wrong results.
// ============================================================================

#ifndef HIP_GQA_COST_MODEL_H
#define HIP_GQA_COST_MODEL_H

// Same export contract as the kernel launchers in hip_custom_kernels.h: the
// implementation ships in custom_kernels_<arch>.{dll,so} and the runtime side
// (gqa_autotune.cpp) resolves it from there.
#if defined(_WIN32)
  #if defined(HIP_CUSTOM_KERNELS_EXPORTS)
    #define HIP_GQA_COST_MODEL_API __declspec(dllexport)
  #else
    #define HIP_GQA_COST_MODEL_API
  #endif
#elif defined(__GNUC__) || defined(__clang__)
  #define HIP_GQA_COST_MODEL_API __attribute__((visibility("default")))
#else
  #define HIP_GQA_COST_MODEL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Which fused kernel the shape lands on. hip_gqa_flash_prefill_v2 picks by head
// dim (64 -> v5, 128 -> v7, 256 -> v8); decode always uses the split-K path.
typedef enum {
  HIP_GQA_PATH_PREFILL_V5 = 0,
  HIP_GQA_PATH_PREFILL_V7 = 1,
  HIP_GQA_PATH_PREFILL_V8 = 2,
  HIP_GQA_PATH_DECODE     = 3
} hip_gqa_path_t;

typedef struct {
  int batch;         // B
  int num_heads;     // Hq
  int kv_heads;      // G   (Hq % G == 0)
  int head_dim;      // d   (64 / 128 / 256)
  int q_len;         // sq  (1 for decode)
  int kv_len;        // skv (total KV length, i.e. past + sq)
  int window;        // sliding window; <= 0 means full attention
  int cu_count;      // device CUs; <= 0 falls back to 20 (gfx1151)
} hip_gqa_shape_t;

typedef struct {
  hip_gqa_path_t path;
  // Prefill knobs. Unused entries are ignored; set them to 0.
  int m_tiles;       // v5 M_TILES, v7 MT, v8 MT
  int bkv;           // KV tile width (32 / 64)
  int num_waves;     // v7 NW, v8 ND
  // Decode knobs.
  int use_wmma;      // 0 = scalar kernel, 1 = WMMA kernel
  int splits;        // split-K count
} hip_gqa_config_t;

// Predicted efficiency of running `shape` with `cand`. Higher is better.
// Returns 0 when the candidate is not a real instantiation for this shape.
HIP_GQA_COST_MODEL_API double hip_gqa_config_score(
    const hip_gqa_shape_t *shape, const hip_gqa_config_t *cand);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // HIP_GQA_COST_MODEL_H
