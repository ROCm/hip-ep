/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <hip/hip_runtime.h>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include "ut_gemm_rank.h"

namespace {
thread_local bool rankDryDispatch = false;
thread_local std::string rankFinalLaunch;
template <class T>
void rankArgument(std::ostringstream& text, T value) {
  if constexpr (std::is_pointer<T>::value || std::is_same<T, std::nullptr_t>::value)
    text << ";ptr";
  else
    text << ';' << std::setprecision(17) << value;
}
template <auto Kernel>
const char* rankKernelName() {
  // The compiler resolves type/value template arguments. Token stringification
  // would retain T/TransB, or differ for equivalent constant expressions.
  return __PRETTY_FUNCTION__;
}
template <auto Kernel, class... Args>
void rankRecordLaunch(dim3 grid, dim3 block, size_t shared, Args... args) {
  if (!rankDryDispatch) return;
  std::ostringstream text;
  text << rankKernelName<Kernel>() << ";grid=" << grid.x << ':' << grid.y << ':' << grid.z
       << ";block=" << block.x << ':' << block.y << ':' << block.z << ";shared=" << shared;
  (rankArgument(text, args), ...);
  rankFinalLaunch += text.str() + '\n';
}
}

// Include the production TU verbatim. Only this standalone adapter records
// host launches. The real kernel launch and every device body stay unchanged.
#undef hipLaunchKernelGGL
#define hipLaunchKernelGGL(kernel, grid, block, shared, stream, ...)          \
  do {                                                                     \
    rankRecordLaunch<kernel>(grid, block, shared, __VA_ARGS__);               \
    if (!rankDryDispatch)                                                  \
      kernel<<<grid, block, shared, stream>>>(__VA_ARGS__);                  \
  } while (0)
#include "../../hip/gemm_kernel.hip"
#undef hipLaunchKernelGGL

namespace {
const void* rankOperand() { return reinterpret_cast<const void*>(uintptr_t(0x10000)); }
int rankFamily(const UtGemmShape& s) {
  if (s.ta || s.m < 1 || s.n < 1 || s.k < 1) return 0;
  if (s.m == 1) return s.tb ? 2 : 3;
  if (s.m < 16) return 0;
  if (s.dtype != kTypeFloat32 && !hipdnn_device_is_wave64()) return 1;
  if (s.dtype == kTypeFloat32 && !hipdnn_device_is_wave64() &&
      f32TileEligible(static_cast<const float*>(rankOperand()), static_cast<const float*>(rankOperand()),
                      s.m, s.n, s.k, s.ta, s.tb)) return 4;
  return 5;
}
void rankInstall(int id) {
  std::lock_guard<std::mutex> lock(g_tune_mu);
  // A dry production dispatch creates the real key. Do not duplicate its key
  // packing or use environment pins latched by other production families.
  for (auto& entry : g_lookup_first_cache) entry.second.cfg = id;
}
std::string rankProbe(const UtGemmShape& s, int id) {
  rankInstall(id);
  rankFinalLaunch.clear();
  rankDryDispatch = true;
  const int status = hip_gemm(reinterpret_cast<void*>(uintptr_t(0x20000)), rankOperand(), rankOperand(),
      s.beta == 0 ? nullptr : rankOperand(), const_cast<void*>(rankOperand()), s.m, s.n, s.k,
      s.alpha, s.beta, s.ta, s.tb, s.dtype, s.crows, s.ccols);
  rankDryDispatch = false;
  if (status || rankFinalLaunch.empty()) return {};
  return rankFinalLaunch;
}
std::string rankGeometry(const UtGemmShape& s, int id) {
  const int family = rankFamily(s);
  int bm = 0, bn = 0, sw = 0, wtm = 0, wtn = 0, bk = 0, sk = 0, threads = 0, tn = 0;
  const char* route = "default";
  if (family == 1) {
    const auto& c = kWmma[id];
    if (!wmmaApplies(c, s.m, s.n, s.k)) return {};
    route = "wmma"; bm = c.bm; bn = c.bn; sw = c.swizzle_n;
    wtm = c.wt_m; wtn = c.wt_n; bk = c.bk; sk = c.split_k;
  } else if (family == 2 || family == 3) {
    const auto& c = family == 2 ? kGemvNt[id] : kGemvNn[id];
    route = family == 2 ? "gemv_nt" : "gemv_nn"; threads = c.threads; tn = c.tile_n;
  } else if (family == 4 || family == 5) {
    route = "tiled_fma";
    auto copy = [&](const auto& c) {
      bm = c.bm; bn = c.bn; bk = c.bk; wtm = c.tm; wtn = c.tn; threads = c.threads;
    };
    if (family == 4) copy(kF32Tile[id]); else copy(kTiledFma[id]);
  }
  char text[256];
  std::snprintf(text, sizeof(text), "route=%s;bm=%d;bn=%d;swizzle=%d;wt_m=%d;wt_n=%d;bk=%d;split_k=%d;threads=%d;tile_n=%d",
                route, bm, bn, sw, wtm, wtn, bk, sk, threads, tn);
  return text;
}
}

extern "C" int hipdnn_ut_gemm_config_count(const UtGemmShape* shape) {
  if (!shape) return 0;
  switch (rankFamily(*shape)) {
    case 1: return kNumWmma;
    case 2: return kNumGemvNt;
    case 3: return kNumGemvNn;
    case 4: return kNumF32Tile;
    case 5: return kNumTiledFma;
    default: return 0;
  }
}
extern "C" int hipdnn_ut_gemm_config(const UtGemmShape* shape, int id, UtGemmConfig* out) {
  if (!shape || !out || shape->m <= 0 || shape->n <= 0 || shape->k <= 0 ||
      (shape->dtype != kTypeFloat16 && shape->dtype != kTypeBFloat16 && shape->dtype != kTypeFloat32)) return -1;
  if (!gemmLutLookupEnabled() || gemm_lut::stats().table_loaded || gemmForcedRoute()) return -4;
  {
    std::lock_guard<std::mutex> lock(g_tune_mu);
    g_lookup_first_cache.clear();
  }
  // Seed the production lookup key with a genuine empty-resolver default.
  const std::string baseline = rankProbe(*shape, -1);
  if (baseline.empty()) return -3;
  if (id < 0) {
    *out = {-1, {}, baseline};
    // The actual default may be a measured/skinny route not in the LUT pool.
    // Only call rankGeometry for real descriptor IDs.
    out->geometry = "route=default;bm=0;bn=0;swizzle=0;wt_m=0;wt_n=0;bk=0;split_k=0;threads=0;tile_n=0";
    for (int candidate = 0; candidate < hipdnn_ut_gemm_config_count(shape); ++candidate) {
      const std::string geometry = rankGeometry(*shape, candidate);
      if (!geometry.empty() && rankProbe(*shape, candidate) == baseline) {
        *out = {-1, geometry, baseline};
        break;
      }
    }
  } else {
    if (id >= hipdnn_ut_gemm_config_count(shape)) return -2;
    const std::string geometry = rankGeometry(*shape, id);
    if (geometry.empty()) return -2;
    const std::string launch = rankProbe(*shape, id);
    if (launch.empty()) return -3;
    *out = {id, geometry, launch};
  }
  rankInstall(-1);
  return 0;
}
extern "C" int hipdnn_ut_gemm_select(const UtGemmShape* shape, const UtGemmConfig* config) {
  if (!shape || !config) return -1;
  if (!gemmLutLookupEnabled() || gemm_lut::stats().table_loaded || gemmForcedRoute()) return -4;
  // An unrepresentable baseline is the unchanged default; legal candidates
  // enter the same production LUT-first path as a stored table geometry.
  rankInstall(config->id);
  return 0;
}
