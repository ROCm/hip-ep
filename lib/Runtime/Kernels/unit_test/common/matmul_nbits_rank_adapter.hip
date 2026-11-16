/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// This is a standalone UT build adapter, not a second kernel implementation.
// Include the production TU verbatim to access its private config descriptors
// and caches without adding a production ABI or changing source/dispatch.
#include "../../hip/matmul_nbits_kernel.hip"
#include "ut_matmul_nbits_rank.h"

namespace {
std::string rankKey(const UtNbitsShape& s, int route) {
  const bool zp = s.zero != 0;
  const bool fp16 = s.zp_bytes == 2;
  if (route == 1) {
    if (s.bits == 4) return wmmaCacheKey(s.m, s.n, (s.k + 31) & ~31, s.gs, zp);
    if (s.bits == 8) return wmmaI8CacheKey(s.m, s.n, s.k, s.gs, zp);
    if (s.bits == 3) return wmmaU3CacheKey(s.m, s.n, s.k, s.gs, zp);
    return wmmaU2CacheKey(s.m, s.n, s.k, s.gs, zp, zp && fp16);
  }
  if (s.bits == 4) return gemvCacheKey(s.m, s.n, s.k, s.gs, false, zp);
  if (s.bits == 8) return gemvI8CacheKey(s.m, s.n, s.k, s.gs, zp);
  if (s.bits == 3) return gemvU3CacheKey(s.m, s.n, s.k, s.gs, zp);
  return gemvU2CacheKey(s.m, s.n, s.k, s.gs, zp, zp && fp16);
}
std::unordered_map<std::string, int>& rankCache(int bits, int route) {
  if (route == 1) {
    if (bits == 4) return wmma_cache;
    if (bits == 8) return wmma_i8_cache;
    if (bits == 3) return wmma_u3_cache;
    return wmma_u2_cache;
  }
  if (bits == 4) return gemv_cache;
  if (bits == 8) return gemv_i8_cache;
  if (bits == 3) return gemv_u3_cache;
  return gemv_u2_cache;
}
int rankRoute(const UtNbitsShape& s) {
  if (s.bits == 4) {
    if (s.m >= wmmaMinM() && !hipdnn_device_is_wave64()) return 1;
    if (s.m > 1 && s.k % 32 == 0 && gemvMBlockedEnabled()) return 3;
    return 2;
  }
  if (s.m >= kBlockDim && s.k % 32 == 0 && s.gs >= 32 && s.gs % 32 == 0 &&
      !hipdnn_device_is_wave64()) return 1;
  if (s.bits == 8 && s.m > 1 && s.m < kBlockDim && gemvMBlockedEnabledI8()) return 3;
  if (s.bits == 8 || (s.gs >= 32 && (s.gs & (s.gs - 1)) == 0 && s.k % 32 == 0)) return 2;
  return 4;
}
}

extern "C" int hipdnn_ut_nbits_config(const UtNbitsShape* shape, int id, UtNbitsConfig* out) {
  if (!shape || !out || shape->m <= 0 || shape->n <= 0 || shape->k <= 0) return -1;
  const auto& s = *shape;
  UtNbitsConfig config;
  config.route = rankRoute(s);
  if (config.route == 3) {
    // This route has no tile_m representation in the existing LUT schema.
    // Only its actual default is eligible; do not invent lossy table configs.
    if (id >= 0) return -2;
    const auto* table = s.bits == 4 ? kGemvMConfigs : kGemvMI8Configs;
    const int count = s.bits == 4 ? kNumGemvMConfigs : kNumGemvMI8Configs;
    const auto candidates = gemvMCandidates(table, count, s.m, s.n);
    config.id = candidates.heuristic >= 0 ? candidates.heuristic : (s.bits == 4 ? 25 : 29);
    const auto& c = table[config.id];
    config.threads = c.threads; config.tile_n = c.tile_n; config.tile_m = c.tile_m;
  } else if (config.route == 4) {
    if (id >= 0) return -2;
    config.id = 0;
  } else {
    if (id < 0) {
      const auto& cache = rankCache(s.bits, config.route);
      const auto it = cache.find(rankKey(s, config.route));
      if (it == cache.end()) return -3;  // caller must execute no-LUT default first
      id = it->second;
    }
    config.id = id;
    if (config.route == 2) {
      if (id >= kNumGemvConfigs) return -2;
      config.threads = kGemvConfigs[id].threads;
      config.tile_n = kGemvConfigs[id].tile_n;
    } else if (s.bits == 4) {
      if (id >= kNumWmmaConfigs) return -2;
      const auto& c = kWmmaConfigs[id];
      if (!wmmaConfigLaunchable(c, s.m, s.n, (s.k + 31) & ~31, s.gs, 0)) return -2;
      config.bm = c.bm; config.bn = c.bn; config.swizzle = c.swizzle_n;
      config.wt_m = c.wt_m; config.wt_n = c.wt_n; config.bk = wmmaBk(c);
      config.fused = c.fused; config.ext = c.ext; config.variant = c.variant;
    } else if (s.bits == 8) {
      if (id >= kNumWmmaI8Configs) return -2;
      const auto& c = kWmmaI8Configs[id];
      if (c.bn > s.n * 2) return -2;
      config.bm = c.bm; config.bn = c.bn; config.swizzle = c.swizzle_n;
      config.wt_m = 2; config.wt_n = 2; config.bk = 8;
    } else {
      if (id >= (s.bits == 2 ? kNumWmmaU2Configs : kNumWmmaU3Configs)) return -2;
      // Both retained tables have the same descriptor fields, but different types.
      auto copy = [&](const auto& c) {
        config.bm = c.bm; config.bn = c.bn; config.swizzle = c.swizzle_n;
        config.wt_m = c.wt_m; config.wt_n = c.wt_n; config.fused = c.fused;
        config.bk = config.wt_m == 4 ? 6 : 8;
      };
      if (s.bits == 2) copy(kWmmaU2Configs[id]); else copy(kWmmaU3Configs[id]);
      if (config.bn > s.n * 2 || (!config.fused && s.n > dequantMaxGridY())) return -2;
    }
  }
  *out = config;
  return 0;
}

extern "C" int hipdnn_ut_nbits_select(const UtNbitsShape* shape, const UtNbitsConfig* config) {
  if (!shape || !config) return -1;
  if (!matmulNbitsLutLookupEnabled() || matmul_lut::stats().table_loaded) return -4;
  if (config->route == 3 || config->route == 4) return 0;  // unchanged singleton default
  UtNbitsConfig verified;
  if (hipdnn_ut_nbits_config(shape, config->id, &verified) || verified.route != config->route) return -2;
  rankCache(shape->bits, config->route)[rankKey(*shape, config->route)] = config->id;
  return 0;
}

extern "C" int hipdnn_ut_nbits_config_count(const UtNbitsShape* shape) {
  const int route = rankRoute(*shape);
  if (route == 3 || route == 4) return 0;
  if (route == 2) return kNumGemvConfigs;
  if (shape->bits == 4) return kNumWmmaConfigs;
  if (shape->bits == 8) return kNumWmmaI8Configs;
  return shape->bits == 2 ? kNumWmmaU2Configs : kNumWmmaU3Configs;
}
