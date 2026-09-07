/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/* Invariant tests for the GQA nearest-neighbour LUT lookup.
 *
 * Asserts properties that hold for *any* well-formed table, not the contents of
 * one particular table: a test that fails when the table gains coverage (and an
 * answer improves from Nearest to Exact) would be worse than no test. It links
 * the real embedded table (lut/gfx1151.fb, xxd'd by CMake) and drives the same
 * resolver that ships in custom_kernels_<arch>, through the exported extern "C"
 * shims -- so it exercises the real load + hierarchical match, GPU-free.
 *
 * The distance metric and the split clamp are the parts easy to break silently;
 * the assertions below bound them rather than pin exact configs.
 */
#include "gqa_autotune.h"

#include <cstdio>
#include <cstdlib>

using hipdnn_ep::GqaAutotuneMode;
using hipdnn_ep::GqaDecodeRequest;
using hipdnn_ep::GqaDecodeResult;
using hipdnn_ep::GqaPrefillConfig;
using hipdnn_ep::GqaPrefillRequest;
using hipdnn_ep::GqaPrefillResult;
using hipdnn_ep::GqaPrefillVariant;
using hipdnn_ep::GqaTuneSource;

static int failures = 0;

#define CHECK(cond, msg)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL: %s (%s:%d)\n", (msg), __FILE__, __LINE__);   \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static const char *srcName(GqaTuneSource s) {
  return hipdnn_ep::gqa_tune_source_name(s);
}

static GqaDecodeResult decode(void *p, int H, int G, int d, int skv,
                              int window = 0, int batch = 1) {
  GqaDecodeRequest req{/*kv_dtype=*/0,    batch, H, G, d, skv,
                       /*max_splits=*/64, window};
  GqaDecodeResult out;
  hip_gqa_autotune_resolve_decode(p, &req, &out);
  return out;
}

static GqaPrefillResult prefill(void *p, GqaPrefillVariant v, int H, int G,
                                int d, int sq, int skv) {
  GqaPrefillRequest req{v, /*batch=*/1, H, G, d, sq, skv, /*window=*/0};
  GqaPrefillResult out;
  hip_gqa_autotune_resolve_prefill(p, &req, &out);
  return out;
}

// A decode config the kernel would accept: split count in range, and a legal
// KV tile height. This is what "fallback always runnable" has to mean.
static bool runnableDecode(const GqaDecodeResult &r) {
  return r.config.splits >= 1 && r.config.splits <= 64 &&
         (r.config.bkv == 16 || r.config.bkv == 32);
}

int main() {
  // The embedded table is stamped gfx1151; the GPU-free arch check reads this.
  // Set it before the first resolve, which is when the table loads.
#ifdef _WIN32
  _putenv_s("HIPDNN_EP_GQA_ARCH", "gfx1151");
#else
  setenv("HIPDNN_EP_GQA_ARCH", "gfx1151", 1);
#endif

  void *policy = hip_gqa_autotune_create();
  CHECK(policy != nullptr, "create returned null");
  CHECK(hip_gqa_autotune_mode(policy) ==
            static_cast<int>(GqaAutotuneMode::Lookup),
        "default mode should be Lookup");

  // The table has to load and be self-consistent: every row it kept is one it
  // will actually answer from. A rejected row is silent coverage loss.
  CHECK(hip_gqa_autotune_table_loaded() == 1, "embedded table did not load");
  CHECK(hip_gqa_autotune_point_count() > 0, "table has no points");
  CHECK(hip_gqa_autotune_invalid_points() == 0,
        "table has rows the loader rejected");

  // A measured geometry+length resolves from the table, not the heuristic, and
  // its config is runnable. 32:8:128 (Llama-3.1-8B) at a swept length.
  {
    GqaDecodeResult r = decode(policy, 32, 8, 128, 4096);
    CHECK(r.source != GqaTuneSource::Heuristic,
          "measured decode shape fell to the heuristic");
    CHECK(r.distance >= 0.0f, "distance must be non-negative");
    CHECK(runnableDecode(r), "measured decode config not runnable");
  }

  // A length nobody swept still resolves from the table (nearest), never the
  // heuristic while a table is present, and the split clamp keeps it legal.
  {
    GqaDecodeResult r = decode(policy, 32, 8, 128, 5000);
    CHECK(r.source == GqaTuneSource::Nearest ||
              r.source == GqaTuneSource::Exact ||
              r.source == GqaTuneSource::Fallback,
          "unswept decode length should resolve from the table");
    CHECK(runnableDecode(r), "nearest decode config not runnable");
  }

  // The split clamp: a short context cannot be handed more splits than it has
  // 16-key tiles. ceil(48/16) = 3.
  {
    GqaDecodeResult r = decode(policy, 32, 8, 128, 48);
    CHECK(r.config.splits <= 3, "splits not clamped to useful work at skv=48");
    CHECK(runnableDecode(r), "clamped decode config not runnable");
  }

  // A head_dim the kernels are not instantiated for has no group; the resolver
  // must still return a runnable config (the heuristic), never crash or hang.
  {
    GqaDecodeResult r = decode(policy, 32, 8, 48, 4096);
    CHECK(r.source == GqaTuneSource::Heuristic,
          "untemplated head_dim should reach the heuristic");
    CHECK(runnableDecode(r), "heuristic decode config not runnable");
  }

  // An unmeasured head pair with a measured heads-per-group ratio stays within
  // that ratio rather than dropping to the heuristic. 16:4:128 is hpg=4.
  {
    GqaDecodeResult r = decode(policy, 16, 4, 128, 4096);
    CHECK(r.source != GqaTuneSource::Heuristic,
          "same-hpg decode should resolve from the table");
    CHECK(runnableDecode(r), "same-hpg decode config not runnable");
  }

  // Prefill resolves from the table for a measured variant/geometry, and a
  // prefill config always carries a positive KV tile height.
  {
    GqaPrefillResult r =
        prefill(policy, GqaPrefillVariant::V7, 32, 8, 128, 512, 8192);
    CHECK(r.source != GqaTuneSource::Heuristic,
          "measured prefill shape fell to the heuristic");
    CHECK(r.config.bkv > 0, "prefill config has no bkv");
  }

  // The explicit heuristic recovery path (used after the kernel rejects a
  // resolved config) always yields a config.
  {
    GqaPrefillRequest req{GqaPrefillVariant::V8, 1, 16, 4, 256, 1024, 4096, 0};
    GqaPrefillConfig cfg;
    hip_gqa_autotune_fallback_prefill(&req, &cfg);
    CHECK(cfg.bkv > 0, "fallback prefill config has no bkv");
  }

  hip_gqa_autotune_destroy(policy);

  if (failures == 0) {
    std::printf("test-gqa-autotune: all checks passed (%u points loaded)\n",
                hip_gqa_autotune_point_count());
    return 0;
  }
  std::fprintf(stderr, "test-gqa-autotune: %d checks FAILED\n", failures);
  return 1;
}
