/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include <hip/hip_runtime.h>
#include <iomanip>
#include <sstream>
#include <stdexcept>
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
  // Clang can abbreviate a function-valued template argument in
  // __PRETTY_FUNCTION__, omitting the kernel's template arguments. The HIP
  // registration name identifies the actual instantiated device entry instead.
  const char* name = hipKernelNameRefByPtr(
      reinterpret_cast<const void*>(Kernel), nullptr);
  if (!name || !*name)
    throw std::runtime_error("UT rank: instantiated HIP kernel name unavailable");
  return name;
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

// The general family passes one argument struct. Found by argument-dependent
// lookup at the recorder's instantiation: its runtime values (not pointers)
// enter the final launch identity like any scalar kernel argument.
namespace gemm_gen {
inline void rankArgument(std::ostringstream& text, const Args& p) {
  text << ";m=" << p.m << ";n=" << p.n << ";k=" << p.k << ";lda=" << p.lda
       << ";ldb=" << p.ldb << ";ldy=" << p.ldy << ";sa=" << p.sa << ";sb=" << p.sb
       << ";sc=" << p.sc << ";sy=" << p.sy << ";alpha=" << std::setprecision(17)
       << p.alpha << ";bias=" << p.bias << ";group=" << p.group_m
       << ";stagger=" << p.stagger_mod << ":" << p.stagger_tiles
       << ";splits=" << p.splits << ";a=ptr;b=ptr;c=" << (p.c ? "ptr" : "null")
       << ";ws=" << (p.ws ? "ptr" : "null");
}
}  // namespace gemm_gen

namespace {
const void* rankOperand() { return reinterpret_cast<const void*>(uintptr_t(0x10000)); }
int rankFamily(const UtGemmShape& s) {
  const int output = s.output < 0 ? s.dtype : s.output;
  const int family = gemmBatchFamily(s.m,s.n,s.k,s.ta,s.tb,s.dtype,output,
                                    s.batches,s.a_group,s.b_group);
  if (family == 1) return 6;
  if (family == 2) return 7;
  if (family == 3 && s.k) return 8;
  if (s.ta || s.m < 1 || s.n < 1 || s.k < 1) return 0;
  if (s.m == 1) return s.tb ? 2 : 3;
  if (s.m < 16) return 0;
  if (s.dtype != kTypeFloat32 && !hipdnn_device_is_wave64()) return 1;
  if (s.dtype == kTypeFloat32 && !hipdnn_device_is_wave64() &&
      f32TileEligible(static_cast<const float*>(rankOperand()), static_cast<const float*>(rankOperand()),
                      s.m, s.n, s.k, s.ta, s.tb)) return 4;
  return 5;
}
// General-family candidates follow the legacy family's descriptor IDs: ID
// legacyCount + i selects the i-th catalog entry legal for this shape.
int rankLegacyCount(const UtGemmShape& s);
int rankGeneral(const UtGemmShape& s, int id) {
  const int base = rankLegacyCount(s);
  if (id < base) return -1;
  const int output = s.output < 0 ? s.dtype : s.output;
  if (s.batches < 1 || s.a_group != 1 || s.b_group != 1 || s.ta) return -1;
  if (s.beta != 0 && !(s.beta == 1 && s.crows == 1 && s.ccols == s.n)) return -1;
  int index = id - base;
  for (int g = 0; g < gemm_gen::kNumCfg; ++g)
    if (gemm_gen::legal(gemm_gen::kCfg[g], s.m, s.n, s.k, s.ta, s.tb, s.dtype, output, s.batches) &&
        index-- == 0)
      return g;
  return -1;
}
void rankInstall(const UtGemmShape& shape, int id) {
  const int general = id >= 0 ? rankGeneral(shape, id) : -1;
  gemmGenRankConfig = general;
  if (general >= 0) id = -1;
  const int family = rankFamily(shape);
  gemmBatchRankConfig = family >= 6 ? id : -2;
  std::lock_guard<std::mutex> lock(g_tune_mu);
  // A dry production dispatch creates the real key. Do not duplicate its key
  // packing or use environment pins latched by other production families.
  for (auto& entry : g_lookup_first_cache) entry.second.cfg = id;
}
std::string rankProbe(const UtGemmShape& s, int id) {
  rankInstall(s, id);
  rankFinalLaunch.clear();
  rankDryDispatch = true;
  const int output = s.output < 0 ? s.dtype : s.output;
  size_t workspace = 0;
  int status = hip_gemm_workspace_size(s.m, s.n, s.k, s.ta, s.tb, s.dtype,
      output, s.batches, s.a_group, s.b_group, &workspace);
  // Disjoint virtual address ranges pass the production overlap guards. Dry
  // dispatch never dereferences operands or workspace and records each launch.
  if (!status) status = hip_gemm_strided_batched(
      reinterpret_cast<void*>(uintptr_t(0x20000)),
      reinterpret_cast<const void*>(uintptr_t(0x100000000000)),
      reinterpret_cast<const void*>(uintptr_t(0x200000000000)),
      s.beta == 0 ? nullptr : reinterpret_cast<const void*>(uintptr_t(0x300000000000)),
      reinterpret_cast<void*>(uintptr_t(0x400000000000)), s.m, s.n, s.k,
      s.alpha, s.beta, s.ta, s.tb, s.dtype, output, s.crows, s.ccols,
      s.batches, s.stride_a < 0 ? int64_t(s.m)*s.k : s.stride_a,
      s.stride_b < 0 ? int64_t(s.n)*s.k : s.stride_b,
      s.stride_c < 0 ? int64_t(s.crows)*s.ccols : s.stride_c,
      s.stride_y < 0 ? int64_t(s.m)*s.n : s.stride_y,
      s.a_group, s.b_group,
      reinterpret_cast<void*>(uintptr_t(0x500000000000)), workspace);
  rankDryDispatch = false;
  if (status || rankFinalLaunch.empty()) return {};
  return rankFinalLaunch;
}
std::string rankGeometry(const UtGemmShape& s, int id) {
  const int general = rankGeneral(s, id);
  if (general >= 0) {
    const auto& c = gemm_gen::kCfg[general];
    const char* route = c.family == gemm_gen::kWmma ? "gen_wmma" : c.family == gemm_gen::kGemv ? "gen_gemv" : "gen_f32";
    char text[256];
    std::snprintf(text, sizeof(text), "route=%s;bm=%d;bn=%d;swizzle=%d;wt_m=%d;wt_n=%d;bk=%d;split_k=%d;threads=%d;tile_n=%d;family=%d",
                  route, c.bm, c.bn, c.smod, c.wt_m, c.wt_n, c.bk, c.splits, c.threads,
                  c.family == gemm_gen::kGemv ? c.tile_n : c.stiles, c.family);
    return text;
  }
  const int family = rankFamily(s);
  int bm = 0, bn = 0, sw = 0, wtm = 0, wtn = 0, bk = 0, sk = 0, threads = 0, tn = 0;
  const char* route = "default";
  if (family == 1) {
    const auto& c = kWmma[id];
    if (!wmmaApplies(c, s.m, s.n, s.k)) return {};
    route = "wmma"; bm = c.bm; bn = c.bn; sw = c.swizzle_n;
    wtm = c.wt_m; wtn = c.wt_n; bk = c.bk; sk = c.split_k;
  } else if (family == 2 || family == 3 || family == 6) {
    const auto& c = family == 2 ? kGemvNt[id] : kGemvNn[id];
    if (family == 6 && s.output >= 0 && s.output != s.dtype && c.tile_n == 0)
      return {};
    route = family == 2 ? "gemv_nt" : "gemv_nn"; threads = c.threads; tn = c.tile_n;
  } else if (family == 4 || family == 5) {
    route = "tiled_fma";
    auto copy = [&](const auto& c) {
      bm = c.bm; bn = c.bn; bk = c.bk; wtm = c.tm; wtn = c.tn; threads = c.threads;
    };
    if (family == 4) copy(kF32Tile[id]); else copy(kTiledFma[id]);
  } else if (family == 7) {
    const auto& c = kBatchedWmma[id];
    if (!gemmBatchedWmmaLegal(c, s.m, s.n, s.k, s.batches, s.ta, s.tb)) return {};
    route="wmma";bm=c.bm;bn=c.bn;wtm=c.wm;wtn=c.wn;bk=c.bk;sw=8;sk=c.splits;
  } else if (family == 8) {
    const auto& c=kBatchedFma[id];
    route="tiled_fma";bm=c.bm;bn=c.bn;wtm=c.tm;wtn=c.tn;bk=c.bk;threads=c.threads;
  }
  char text[256];
  std::snprintf(text, sizeof(text), "route=%s;bm=%d;bn=%d;swizzle=%d;wt_m=%d;wt_n=%d;bk=%d;split_k=%d;threads=%d;tile_n=%d",
                route, bm, bn, sw, wtm, wtn, bk, sk, threads, tn);
  return text;
}

int rankLegacyCount(const UtGemmShape& s) {
  switch (rankFamily(s)) {
    case 1: return kNumWmma;
    case 2: return kNumGemvNt;
    case 3: return kNumGemvNn;
    case 4: return kNumF32Tile;
    case 5: return kNumTiledFma;
    case 6: return kNumGemvNn;
    case 7: return kNumBatchedWmma;
    case 8: return kNumBatchedFma;
    default: return 0;
  }
}
}

extern "C" int hipdnn_ut_gemm_config_count(const UtGemmShape* shape) {
  if (!shape) return 0;
  int general = 0;
  while (rankGeneral(*shape, rankLegacyCount(*shape) + general) >= 0) ++general;
  return rankLegacyCount(*shape) + general;
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
  rankInstall(*shape, -1);
  return 0;
}
extern "C" int hipdnn_ut_gemm_select(const UtGemmShape* shape, const UtGemmConfig* config) {
  if (!shape || !config) return -1;
  if (!gemmLutLookupEnabled() || gemm_lut::stats().table_loaded || gemmForcedRoute()) return -4;
  // An unrepresentable baseline is the unchanged default; legal candidates
  // enter the same production LUT-first path as a stored table geometry.
  rankInstall(*shape, config->id);
  return 0;
}
