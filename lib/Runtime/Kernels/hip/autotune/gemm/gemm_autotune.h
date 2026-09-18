/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_GEMM_AUTOTUNE_H
#define HIPDNN_EP_GEMM_AUTOTUNE_H

#include <cstdint>

/* Gemm offline autotune lookup. Mirrors matmul_nbits_autotune (see
 * ../matmul_nbits/matmul_nbits_autotune.h) and the GQA autotune: an offline
 * table, embedded per-arch into custom_kernels_<arch>, consulted between the
 * in-process tune map and the runtime sweep:
 *
 *     map hit              -> use it
 *     map miss, LUT hit    -> use it, and write it into the map
 *     both miss            -> run the sweep / heuristic (unchanged behaviour)
 *
 * The table is gated on GPU arch, schema version and kernel ABI, so a
 * mismatched table is ignored rather than silently applied. When no table is
 * embedded (the size-0 stub) resolve() reports a miss and the caller falls back
 * exactly as before, so the wiring is safe to land with an empty table.
 *
 * These types are the vocabulary gemm_kernel.hip already dispatches in; the
 * kernel aliases `gemm_lut = hipdnn_ep::gemm_autotune` and calls resolve().
 */

namespace hipdnn_ep {
namespace gemm_autotune {

/* Which kernel is being configured. Wmma = the prefill tiled path (kWmma);
 * GemvNt = the decode N-major GEMV path (kGemvNt). They keep separate tune
 * caches at runtime, so they are separate keys here. */
enum class Kind : uint8_t { Wmma = 1, GemvNt = 2 };

/* Where a resolved config came from (reported for logs/tests). Exact is the
 * degenerate Nearest with distance 0 (this shape was measured). */
enum class Source : uint8_t {
  None = 0,     // nothing resolved; caller sweeps / uses the heuristic
  Exact = 1,    // a measured point with this exact (M, N, K)
  Nearest = 2,  // the closest measured point in log space
  Fallback = 3, // phase + type only; the group had no usable point
};

/* One Gemm invocation, in the terms gemm_kernel.hip dispatch already has. */
struct Request {
  Kind kind = Kind::Wmma;
  int type_bytes = 2; /* sizeof(T): 2=f16/bf16, 4=f32, 8=f64 */
  int trans_b = 1;
  int m = 0, n = 0, k = 0;
};

/* WMMA tile geometry, matching the WmmaCfg fields the dispatch keys on. A
 * geometry the live kWmma table no longer contains is refused by the validator
 * and the search moves on, so a stale table is never worse than no table. */
struct WmmaAnswer {
  int bm = 0, bn = 0, wt_m = 0, wt_n = 0, swizzle_n = 0, split_k = 1, bk = 32;
};

/* (threads, tile_n) for the GEMV NT decode path, matched against kGemvNt. */
struct GemvAnswer {
  int threads = 0, tile_n = 0;
};

struct Result {
  Source source = Source::None;
  WmmaAnswer wmma;
  GemvAnswer gemv;
  float distance = 0.0f; /* weighted log2 distance to the answering point */
};

/* The validators report whether the live config table still contains the
 * candidate geometry AND whether it is legal for this shape. Returning false
 * makes the search move to the next-nearest point. `ctx` is passed through. */
using WmmaValidator = bool (*)(void *ctx, const WmmaAnswer &);
using GemvValidator = bool (*)(void *ctx, const GemvAnswer &);

/* Resolve `request` against the embedded table. Returns Source::None when the
 * table is absent/incompatible or every candidate was rejected. */
Result resolve(const Request &request, WmmaValidator wmma_valid,
               GemvValidator gemv_valid, void *ctx);

/* Diagnostics for the debug log. */
struct Stats {
  bool table_loaded = false;
  uint32_t points = 0;
  uint32_t invalid_points = 0;
  uint32_t rejected_answers = 0;
};

Stats stats();

} // namespace gemm_autotune
} // namespace hipdnn_ep

#endif // HIPDNN_EP_GEMM_AUTOTUNE_H
