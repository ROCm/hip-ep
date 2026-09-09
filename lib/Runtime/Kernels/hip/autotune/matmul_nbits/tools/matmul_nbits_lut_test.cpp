/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
/* Invariant tests for the MatMulNBits LUT lookup.
 *
 * Asserts properties that hold for *any* well-formed table, not the contents of
 * one particular table. An earlier version hard-coded "this shape resolves to
 * the middle tier", which started failing the moment the table got more
 * coverage and the answer improved to Exact -- a test that fails when the thing
 * under test gets better is worse than no test. Distance bounds are used
 * instead: they only get easier to satisfy as coverage grows.
 *
 * GPU-free: only the table and the lookup are exercised.
 *
 * Build (from the repo root):
 *   clang++ -std=c++17 -I <flatbuffers include> -I <generated header dir> \
 *       -I lib/Runtime/Kernels/hip/autotune/matmul_nbits \
 *       lib/Runtime/Kernels/hip/autotune/matmul_nbits/tools/matmul_nbits_lut_test.cpp \
 *       lib/Runtime/Kernels/hip/autotune/matmul_nbits/matmul_nbits_autotune.cpp \
 *       <generated matmul_nbits_lut_data_gfx1151.cpp> -o matmul_nbits_lut_test
 */
#include "matmul_nbits_autotune.h"

#include <cstdio>

using namespace hipdnn_ep::matmul_nbits_autotune;

static bool acceptWmma(void*, const WmmaAnswer&) { return true; }
static bool acceptGemv(void*, const GemvAnswer&) { return true; }
static bool rejectWmma(void*, const WmmaAnswer&) { return false; }
static bool rejectGemv(void*, const GemvAnswer&) { return false; }

static int failures = 0;

static const char* name(Source s) {
  switch (s) {
  case Source::Exact: return "Exact";
  case Source::Nearest: return "Nearest";
  case Source::Fallback: return "Fallback";
  default: return "None";
  }
}

// Specificity order, for "at least as specific as" assertions.
static int rank(Source s) {
  switch (s) {
  case Source::Exact: return 3;
  case Source::Nearest: return 2;
  case Source::Fallback: return 1;
  default: return 0;
  }
}

static Result run(const Request& r, bool accept = true) {
  return accept ? resolve(r, acceptWmma, acceptGemv, nullptr)
                : resolve(r, rejectWmma, rejectGemv, nullptr);
}

static void expectAtLeast(const char* what, const Request& r, Source floor_) {
  const Result got = run(r);
  const bool ok = rank(got.source) >= rank(floor_);
  if (!ok) ++failures;
  std::printf("  %-52s %-8s (>= %-8s) %s\n", what, name(got.source),
              name(floor_), ok ? "OK" : "FAIL");
}

static void expectExactly(const char* what, const Request& r, Source want,
                          bool accept = true) {
  const Result got = run(r, accept);
  const bool ok = got.source == want;
  if (!ok) ++failures;
  std::printf("  %-52s %-8s (== %-8s) %s\n", what, name(got.source),
              name(want), ok ? "OK" : "FAIL");
}

/* Asserts the answer came from a point within `octaves` of the query. Without
 * this a "Nearest" result says nothing: the fallback is also technically an
 * answer, and so is a point on the far side of the space. */
static void expectWithin(const char* what, const Request& r, float octaves) {
  const Result got = run(r);
  const bool ok = got.source == Source::Nearest && got.distance <= octaves;
  if (!ok) ++failures;
  std::printf("  %-52s %-8s d=%.3f (<= %.2f) %s\n", what, name(got.source),
              got.distance, octaves, ok ? "OK" : "FAIL");
}

static Request prefill(int m, int n, int k, int gs, bool zp, int stride = 0) {
  Request r;
  r.phase = Phase::Prefill;
  r.m = m; r.n = n; r.k = k; r.group_size = gs; r.has_zp = zp;
  r.b_row_bytes = stride;
  return r;
}

static Request decode(Phase p, int n, int k, int gs, bool zp) {
  Request r;
  r.phase = p;
  r.m = 1; r.n = n; r.k = k; r.group_size = gs; r.has_zp = zp;
  return r;
}

int main() {
  const Stats s = stats();
  std::printf("table_loaded=%d points=%u invalid_points=%u\n\n",
              int(s.table_loaded), s.points, s.invalid_points);
  if (!s.table_loaded) {
    // Not a failure: an arch with no measured table is a supported state, and
    // every shape must then resolve to None so the caller sweeps.
    std::printf("no table for this arch; checking the no-table contract\n");
    expectExactly("no table -> None", prefill(128, 5120, 4096, 128, false),
                  Source::None);
    std::printf("%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
  }

  // A malformed point would be dropped at load; none should be.
  if (s.invalid_points != 0) {
    std::printf("  %-52s %u points FAIL\n", "table has inconsistent points",
                s.invalid_points);
    ++failures;
  }

  std::printf("-- nothing resolves to None --\n");
  // Shapes chosen to be nowhere near anything measured. Nearest-neighbour has
  // no notion of "too far", so these are expected to resolve off a distant
  // point rather than the fallback; either is acceptable, None is not.
  expectAtLeast("prefill, absurd shape", prefill(333, 99991, 7777, 128, false),
                Source::Fallback);
  expectAtLeast("prefill, tiny", prefill(16, 32, 32, 32, true),
                Source::Fallback);
  expectAtLeast("decode, absurd shape",
                decode(Phase::Decode, 99991, 7777, 128, false),
                Source::Fallback);
  expectAtLeast("dp4a, absurd shape",
                decode(Phase::DecodeDp4a, 99991, 7777, 128, false),
                Source::Fallback);
  expectAtLeast("prefill, padded stride, absurd shape",
                prefill(128, 99991, 7777, 128, false, 7777 / 2 + 128),
                Source::Fallback);

  std::printf("\n-- measured shapes resolve exactly --\n");
  // Verified against shapes/oga_models_bits4.csv -- Llama-3.1-8B has separate
  // q/k/v projections, so there is no fused N=6144 to test with. M values come
  // from the sweep ladder.
  expectExactly("prefill Llama-3.1-8B gate/up M=128 N=14336 K=4096",
                prefill(128, 14336, 4096, 128, true), Source::Exact);
  expectExactly("prefill Llama-3.1-8B down   M=512 N=4096  K=14336",
                prefill(512, 4096, 14336, 128, true), Source::Exact);
  expectExactly("prefill Llama-3.1-8B kv     M=32  N=1024  K=4096",
                prefill(32, 1024, 4096, 128, true), Source::Exact);
  expectExactly("decode  Llama-3.1-8B gate/up      N=14336 K=4096",
                decode(Phase::Decode, 14336, 4096, 128, true), Source::Exact);
  expectExactly("dp4a    Llama-3.1-8B gate/up      N=14336 K=4096",
                decode(Phase::DecodeDp4a, 14336, 4096, 128, true),
                Source::Exact);

  std::printf("\n-- an exact hit is a zero-distance neighbour --\n");
  {
    const Result got = run(prefill(128, 14336, 4096, 128, true));
    const bool ok = got.source == Source::Exact && got.distance == 0.0f;
    if (!ok) ++failures;
    std::printf("  %-52s d=%.3f %s\n", "measured shape reports distance 0",
                got.distance, ok ? "OK" : "FAIL");
  }

  std::printf("\n-- unmeasured shapes land on a close measured point --\n");
  // N nudged 2.3%% off a measured point: the answer must come from that point,
  // not from whatever won most often somewhere in the neighbourhood. 0.1
  // octaves is far tighter than the old octave buckets could promise.
  expectWithin("prefill, N=14000 vs measured 14336",
               prefill(128, 14000, 4096, 128, true), 0.1f);
  expectWithin("decode, N=14000 vs measured 14336",
               decode(Phase::Decode, 14000, 4096, 128, true), 0.1f);
  // An M the sweep ladder never visited must snap to an adjacent rung rather
  // than drag the match onto a different (N, K).
  expectWithin("prefill, M=100 between ladder rungs 64 and 128",
               prefill(100, 14336, 4096, 128, true), 0.7f);

  std::printf("\n-- keys that must not be conflated --\n");
  // Decode and DecodeDp4a rank kGemvConfigs differently; if they shared a group
  // one would silently answer for the other. They may legitimately agree, so
  // this only asserts that both resolve -- the separation is enforced by
  // pointConsistent at load and by invalid_points being 0 above.
  {
    const Result a = run(decode(Phase::Decode, 14336, 4096, 128, true));
    const Result b = run(decode(Phase::DecodeDp4a, 14336, 4096, 128, true));
    const bool ok = a.source != Source::None && b.source != Source::None;
    if (!ok) ++failures;
    std::printf("  %-52s fp=[%d,%d] dp4a=[%d,%d] %s\n",
                "decode vs dp4a both resolve", a.gemv.threads, a.gemv.tile_n,
                b.gemv.threads, b.gemv.tile_n, ok ? "OK" : "FAIL");
  }

  std::printf("\n-- a validator that rejects everything yields None --\n");
  // The point of storing geometry rather than a config index: an answer the
  // live config table cannot serve must make the search move outwards, and if
  // every point and the fallback are unusable the caller must be told to
  // sweep, not handed something unlaunchable.
  expectExactly("all geometries rejected",
                prefill(128, 14336, 4096, 128, true), Source::None,
                /*accept=*/false);

  const Stats after = stats();
  std::printf("\nrejected_answers=%u\n", after.rejected_answers);
  std::printf("%s (%d failures)\n", failures ? "FAILED" : "ALL PASSED",
              failures);
  return failures ? 1 : 0;
}
