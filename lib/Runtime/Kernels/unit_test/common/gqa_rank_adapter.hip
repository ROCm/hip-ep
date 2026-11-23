/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Standalone UT catalog uses the production candidate enumeration verbatim.
// Every timed candidate goes through the existing configured public entry.
#include "../../hip/gqa_kernel.hip"
#include "../../hip/autotune/gqa/gqa_autotune.h"
#include "ut_gqa_rank.h"

// Reuse the actual resolver without an installed table, for the exact
// production lookup-mode fallback. These bytes are private to RANK=1 builds.
extern "C" const unsigned char* const kGqaLutBlobs[1] = {nullptr};
extern "C" const size_t kGqaLutBlobSizes[1] = {0};
extern "C" const size_t kGqaLutBlobCount = 0;

namespace {
std::vector<UtGqaConfig> rankGqaConfigs(const UtGqaShape& shape) {
  const auto& s = shape;
  std::vector<UtGqaConfig> result;
  if (s.b < 1 || s.h < 1 || s.g < 1 || s.h % s.g || s.skv < 1 || s.sq < 1) return result;
  if (s.sq == 1) {
    void* policy = hip_gqa_autotune_create("lookup");
    const hipdnn_ep::GqaDecodeRequest request{s.kv_dtype, s.b, s.h, s.g, s.d,
        s.window > 0 ? std::min(s.skv, s.window) : s.skv, s.max_splits, s.window};
    hipdnn_ep::GqaDecodeResult fallback;
    hip_gqa_autotune_resolve_decode(policy, &request, &fallback);
    hip_gqa_autotune_destroy(policy);
    result.push_back({fallback.config.use_wmma ? 1 : 0, fallback.config.splits,
                      fallback.config.bkv, 0, 0, 0, 0});
    int splits[7], count = 0;
    const int effective = s.window > 0 ? std::min(s.skv, s.window) : s.skv;
    flashDecodeCandidateSplits(effective, s.max_splits, splits, count);
    for (int i = 0; i < count; ++i) {
      result.push_back({0, splits[i], 16, 0, 0, 0, 0});
      if (flash_decode_wmma_supported(s.d, s.h / s.g)) {
        result.push_back({1, splits[i], 16, 0, 0, 0, 0});
        if (s.d == 64 && flashDecodeTallTileUseful(32, effective, splits[i]))
          result.push_back({1, splits[i], 32, 0, 0, 0, 0});
      }
    }
  } else {
    const hipdnn_ep::GqaPrefillRequest request{
        s.d == 256 ? hipdnn_ep::GqaPrefillVariant::V8x : hipdnn_ep::GqaPrefillVariant::V6x,
        s.b, s.h, s.g, s.d, s.sq, s.skv, s.window};
    hipdnn_ep::GqaPrefillConfig fallback;
    hip_gqa_autotune_fallback_prefill(&request, &fallback);
    result.push_back({0, 0, fallback.bkv, fallback.hb, fallback.qw, fallback.mt, fallback.nd});
    if (s.d == 256) {
      if (s.window > 0) return {};
      PrefillV8xCfg configs[kPrefillV8xMaxCandidates];
      const int count = prefillV8xCandidates(s.d, s.h, s.g, configs);
      for (int i = 0; i < count; ++i)
        result.push_back({0, 0, configs[i].bkv, configs[i].hb, 0, 0, configs[i].nd});
    } else {
      PrefillV6xCfg configs[kPrefillV6xMaxCandidates];
      const int count = prefillV6xCandidates(s.d, s.h, s.g, s.window, configs);
      for (int i = 0; i < count; ++i)
        result.push_back({0, 0, configs[i].bkv, configs[i].hb, configs[i].qw, configs[i].mt, 0});
    }
  }
  return result;
}
}
extern "C" int hipdnn_ut_gqa_config_count(const UtGqaShape* shape) {
  return static_cast<int>(rankGqaConfigs(*shape).size());
}
extern "C" int hipdnn_ut_gqa_config(const UtGqaShape* shape, int id, UtGqaConfig* output) {
  const auto configs = rankGqaConfigs(*shape);
  if (id < 0 || id >= static_cast<int>(configs.size()) || !output) return -1;
  *output = configs[id]; return 0;
}
