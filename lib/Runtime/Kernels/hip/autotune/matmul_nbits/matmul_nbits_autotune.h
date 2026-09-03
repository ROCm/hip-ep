/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIPDNN_EP_MATMUL_NBITS_AUTOTUNE_H
#define HIPDNN_EP_MATMUL_NBITS_AUTOTUNE_H

#include <cstdint>

/* MatMulNBits offline autotune lookup.
 *
 * Sits between the in-process tune map and the runtime sweep:
 *
 *     map hit              -> use it
 *     map miss, LUT hit    -> use it, and write it into the map
 *     both miss            -> run the sweep (unchanged behaviour)
 *
 * Map-first rather than LUT-first because the map holds measurements from
 * this exact machine and build, which beat a table tuned on a reference part;
 * the LUT's job is to make the first encounter with a shape free, not to
 * override something already measured locally. The map is in-process only and
 * lives for the process lifetime -- there is no on-disk tune cache.
 *
 * The table is compiled into custom_kernels_<arch> (see
 * lib/Runtime/Kernels/CMakeLists.txt) and gated on GPU arch, schema version and
 * kernel ABI, so a mismatched table is ignored rather than silently applied.
 */

namespace hipdnn_ep {
namespace matmul_nbits_autotune {

/* Where a resolved config came from. Reported so a caller (or a test) can tell
 * a real table hit from the compiled-in last resort.
 *
 * Exact and Nearest are the same lookup: Exact is the case where the closest
 * measured point sits at distance 0, i.e. this shape was measured. They are
 * reported apart only because the distinction matters when reading a log or
 * judging how much of a workload the table actually covers. */
enum class Source : uint8_t {
  None = 0,      // nothing resolved; caller must sweep
  Exact = 1,     // a measured point with this exact (M, N, K)
  Nearest = 2,   // the closest measured point in log space
  Fallback = 3,  // phase + bits only; the group had no usable point
};

/* Which kernel is being configured. Decode and DecodeDp4a both index
 * kGemvConfigs but rank it differently, so they are separate keys -- mirroring
 * the separate tune caches the two already keep at runtime. */
enum class Phase : uint8_t { Prefill = 1, Decode = 2, DecodeDp4a = 3 };

/* One MatMulNBits invocation, in the terms the kernel dispatch already has. */
struct Request {
  Phase phase = Phase::Prefill;
  int bits = 4;
  int m = 1;
  int n = 0;
  int k = 0;
  int group_size = 128;
  bool has_zp = false;
  /* Bytes between B rows, 0 for the arrival layout. Prefill only: the padded
   * layout has a different winner, so it must not share a row. */
  int b_row_bytes = 0;
};

/* Tile geometry for the WMMA prefill path, as stored in the row. The caller
 * turns this back into a kWmmaConfigs index; a geometry the live table no
 * longer contains means the row is stale and is skipped during the probe. */
struct WmmaAnswer {
  int bm = 0;
  int bn = 0;
  int swizzle_n = 0;
  int wt_m = 0;
  int wt_n = 0;
  int bk = 0;
  bool fused = true;
};

/* (BLOCK_SIZE, TILE_N) for the GEMV decode path, matched against
 * kGemvConfigs the same way. */
struct GemvAnswer {
  int threads = 0;
  int tile_n = 0;
};

struct Result {
  Source source = Source::None;
  WmmaAnswer wmma;
  GemvAnswer gemv;
  /* Weighted log2 distance to the point that answered, in octaves. 0 on an
   * Exact hit, meaningless on Fallback. Exposed so the debug log can show how
   * far the answer was extrapolated, which is the signal for "this shape is
   * worth adding to the sweep". */
  float distance = 0.0f;
};

/* Resolves `request` against the embedded table.
 *
 * The validators are called with a candidate geometry and must report whether
 * the live config table still contains it AND whether it is legal for this
 * shape (kWmmaConfigs has both constraints in wmmaConfigApplies). Returning
 * false makes the search move on to the next-nearest point rather than hand
 * back an unusable answer, which is what keeps a stale table from being worse
 * than no table. `ctx` is passed through untouched.
 *
 * Returns Source::None only when the table is absent or incompatible, or when
 * every point in the group and the fallback were all rejected; the caller then
 * sweeps as before. A well-formed table has a fallback per (phase, bits).
 */
using WmmaValidator = bool (*)(void* ctx, const WmmaAnswer&);
using GemvValidator = bool (*)(void* ctx, const GemvAnswer&);

Result resolve(const Request& request, WmmaValidator wmma_valid,
               GemvValidator gemv_valid, void* ctx);

/* Diagnostics for the debug log: whether a table loaded at all, how many
 * points it has, and how many were dropped as unusable. */
struct Stats {
  bool table_loaded = false;
  uint32_t points = 0;
  uint32_t invalid_points = 0;    // malformed key or a dangling config index
  uint32_t rejected_answers = 0;  // validator said the geometry is not live
};

Stats stats();

}  // namespace matmul_nbits_autotune
}  // namespace hipdnn_ep

#endif  // HIPDNN_EP_MATMUL_NBITS_AUTOTUNE_H
