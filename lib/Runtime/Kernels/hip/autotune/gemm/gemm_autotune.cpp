/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "gemm_autotune.h"

#include "gemm_autotune_generated.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace hipdnn_ep {
namespace gemm_autotune {
namespace {

namespace fbs = hipdnn_ep::gemm_autotune::fbs;
namespace common_fbs = hipdnn_ep::common::fbs;

constexpr uint32_t kSchemaVersion = 3;
// Bump when the config tables or the meaning of a stored geometry changes.
constexpr const char kKernelAbi[] = "gemm-v2";

bool logOn() {
  static const bool on = [] {
#ifdef _WIN32
    char buf[8];
    return GetEnvironmentVariableA("HIPDNN_GEMM_LUT_LOG", buf, sizeof(buf)) >
               0 &&
           buf[0] >= '1';
#else
    const char *v = getenv("HIPDNN_GEMM_LUT_LOG");
    return v && v[0] >= '1';
#endif
  }();
  return on;
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

fbs::GemmPhase phaseClass(Kind kind) {
  switch (kind) {
  case Kind::Wmma: return fbs::GemmPhase::Wmma;
  case Kind::GemvNt: return fbs::GemmPhase::GemvNt;
  case Kind::GemvNn: return fbs::GemmPhase::GemvNn;
  case Kind::TiledFma: return fbs::GemmPhase::TiledFma;
  default: return fbs::GemmPhase::Any;
  }
}

/* D4 fix: an explicit phase -> config-kind table. A boolean "is it Wmma"
 * inference silently misclassifies a third/fourth phase; this must be
 * extended by hand whenever GemmPhase grows. */
fbs::GemmConfigKind phaseConfigKind(fbs::GemmPhase phase) {
  switch (phase) {
  case fbs::GemmPhase::Wmma: return fbs::GemmConfigKind::Wmma;
  case fbs::GemmPhase::GemvNt: return fbs::GemmConfigKind::Gemv;
  case fbs::GemmPhase::GemvNn: return fbs::GemmConfigKind::Gemv;
  case fbs::GemmPhase::TiledFma: return fbs::GemmConfigKind::TiledFma;
  default: return fbs::GemmConfigKind::None;
  }
}

/* The C++ and flatbuffers HipdnnDType enums are hand-kept in lock-step (see
 * hipdnn_dtype.h) with identical numeric values, so this is a plain
 * reinterpretation, not a lookup table that can drift out of sync silently --
 * a mismatch would be caught by any round-trip test comparing names. */
common_fbs::HipdnnDType toFbsDType(HipdnnDType d) {
  return static_cast<common_fbs::HipdnnDType>(d);
}
HipdnnDType fromFbsDType(common_fbs::HipdnnDType d) {
  return static_cast<HipdnnDType>(d);
}

fbs::GemmTransB transBClass(int trans_b) {
  return trans_b ? fbs::GemmTransB::NT : fbs::GemmTransB::NN;
}

/* The exact-match part of the key: 8-bit-per-lane so appending a phase or
 * dtype value never collides with an existing lane (fixes D3's 2-bit fields,
 * which the old GemvNn/TiledFma phases would have overflowed). Lanes 4-7 are
 * reserved for a future field (matmul_nbits reuses this layout with more
 * lanes -- see plan.md §2.4). */
uint64_t groupKey(fbs::GemmPhase phase, common_fbs::HipdnnDType act,
                  common_fbs::HipdnnDType wts, fbs::GemmTransB trb) {
  return static_cast<uint64_t>(phase) |
         (static_cast<uint64_t>(act) << 8) |
         (static_cast<uint64_t>(wts) << 16) |
         (static_cast<uint64_t>(trb) << 24);
}

/* Same layout as groupKey (D5 fix: v1 dropped trans_b from the fallback key,
 * conflating NN and NT winners). Named separately from groupKey because it
 * indexes a different table (fallbacks, not groups), even though the bit
 * layout happens to coincide today. */
uint64_t fallbackKey(fbs::GemmPhase phase, common_fbs::HipdnnDType act,
                     common_fbs::HipdnnDType wts, fbs::GemmTransB trb) {
  return groupKey(phase, act, wts, trb);
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

struct Answer {
  fbs::GemmConfigKind kind;
  WmmaAnswer wmma;
  GemvAnswer gemv;
  TiledFmaAnswer tiled_fma;
};

struct Point {
  float lm, ln, lk;
  uint32_t m, n, k;
  uint16_t config;
};

struct Table {
  std::vector<Answer> pool;
  std::unordered_map<uint64_t, std::vector<Point>> groups;
  std::unordered_map<uint64_t, uint16_t> fallbacks;
  float wm = 1.0f, wn = 1.0f, wk = 1.0f;
  bool bf16_aliases_f16 = false;
  bool loaded = false;
  uint32_t points = 0;
  uint32_t invalid_points = 0;
  std::atomic<uint32_t> rejected{0};
};

std::string currentGpuArch() {
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, 0) != hipSuccess)
    return std::string();
  std::string arch(props.gcnArchName);
  const size_t colon = arch.find(':');
  return colon == std::string::npos ? arch : arch.substr(0, colon);
}

bool pointConsistent(const fbs::GemmTunePoint &p, fbs::GemmConfigKind kind) {
  if (p.phase() == fbs::GemmPhase::Any ||
      p.act_dtype() == common_fbs::HipdnnDType::Any ||
      p.wts_dtype() == common_fbs::HipdnnDType::Any ||
      p.out_dtype() == common_fbs::HipdnnDType::Any ||
      p.trans_b() == fbs::GemmTransB::Any)
    return false;
  if (p.m() == 0 || p.n() == 0 || p.k() == 0)
    return false;
  return kind == phaseConfigKind(p.phase());
}

bool compatible(const fbs::GemmAutotuneLut *lut) {
  if (lut->schema_version() != kSchemaVersion) {
    if (logOn())
      fprintf(stderr, "[gemm-lut] schema %u != %u, ignoring table\n",
              lut->schema_version(), kSchemaVersion);
    return false;
  }
  if (!lut->kernel_abi() || lut->kernel_abi()->str() != kKernelAbi) {
    if (logOn())
      fprintf(stderr, "[gemm-lut] kernel_abi mismatch, ignoring table\n");
    return false;
  }
  if (lut->gpu_arch() && !lut->gpu_arch()->str().empty()) {
    const std::string actual = currentGpuArch();
    if (actual.empty() || actual != lut->gpu_arch()->str()) {
      if (logOn())
        fprintf(stderr, "[gemm-lut] arch \"%s\" != device \"%s\", ignoring\n",
                lut->gpu_arch()->c_str(), actual.c_str());
      return false;
    }
  }
  return true;
}

void loadBuffer(Table &t, const unsigned char *data, size_t size) {
  if (!data || size == 0) {
    if (logOn())
      fprintf(stderr, "[gemm-lut] no embedded table (size 0); shapes will "
                      "use the heuristic / autotune\n");
    return;
  }
  flatbuffers::Verifier verifier(data, size);
  if (!fbs::VerifyGemmAutotuneLutBuffer(verifier)) {
    if (logOn())
      fprintf(stderr, "[gemm-lut] buffer failed verification\n");
    return;
  }
  const fbs::GemmAutotuneLut *lut = fbs::GetGemmAutotuneLut(data);
  if (!lut || !lut->points() || !lut->configs() || !compatible(lut)) {
    if (logOn())
      fprintf(stderr, "[gemm-lut] table present (%zu bytes) but unusable\n",
              size);
    return;
  }
  if (!(lut->weight_m() > 0.0f) || !(lut->weight_n() > 0.0f) ||
      !(lut->weight_k() > 0.0f)) {
    if (logOn())
      fprintf(stderr, "[gemm-lut] non-positive metric weight\n");
    return;
  }
  t.wm = lut->weight_m();
  t.wn = lut->weight_n();
  t.wk = lut->weight_k();
  t.bf16_aliases_f16 = lut->bf16_aliases_f16();

  const auto *configs = lut->configs();
  if (configs->size() > 255) {
    if (logOn())
      fprintf(stderr, "[gemm-lut] config pool %zu exceeds uint8 index; ignoring table\n", configs->size());
    return;
  }
  t.pool.reserve(configs->size());
  for (const fbs::GemmTuneConfig *c : *configs) {
    Answer a{};
    a.kind = c->kind();
    if (a.kind == fbs::GemmConfigKind::Wmma) {
      a.wmma.bm = static_cast<int>(c->bm16()) * 16;
      a.wmma.bn = static_cast<int>(c->bn16()) * 16;
      a.wmma.swizzle_n = c->swizzle();
      a.wmma.wt_m = c->wt_m();
      a.wmma.wt_n = c->wt_n();
      a.wmma.bk = c->bk();
      a.wmma.split_k = c->split_k() ? c->split_k() : 1;
    } else if (a.kind == fbs::GemmConfigKind::Gemv) {
      a.gemv.threads = c->threads();
      a.gemv.tile_n = c->tile_n();
    } else if (a.kind == fbs::GemmConfigKind::TiledFma) {
      a.tiled_fma.bm = static_cast<int>(c->bm16()) * 16;
      a.tiled_fma.bn = static_cast<int>(c->bn16()) * 16;
      a.tiled_fma.tm = c->wt_m();
      a.tiled_fma.tn = c->wt_n();
      a.tiled_fma.bk = c->bk();
      a.tiled_fma.threads = c->threads();
    }
    t.pool.push_back(a);
  }

  for (const fbs::GemmTunePoint *p : *lut->points()) {
    const unsigned idx = p->config();
    if (idx >= t.pool.size()) {
      ++t.invalid_points;
      continue;
    }
    if (!pointConsistent(*p, t.pool[idx].kind)) {
      ++t.invalid_points;
      continue;
    }
    Point point;
    point.m = p->m();
    point.n = p->n();
    point.k = p->k();
    point.lm = std::log2(static_cast<float>(point.m));
    point.ln = std::log2(static_cast<float>(point.n));
    point.lk = std::log2(static_cast<float>(point.k));
    point.config = static_cast<uint16_t>(idx);
    t.groups[groupKey(p->phase(), p->act_dtype(), p->wts_dtype(),
                      p->trans_b())]
        .push_back(point);
    ++t.points;
  }

  if (lut->fallbacks()) {
    for (const fbs::GemmFallback *f : *lut->fallbacks()) {
      const unsigned idx = f->config();
      if (idx >= t.pool.size() || f->phase() == fbs::GemmPhase::Any ||
          f->act_dtype() == common_fbs::HipdnnDType::Any ||
          f->wts_dtype() == common_fbs::HipdnnDType::Any ||
          f->trans_b() == fbs::GemmTransB::Any)
        continue;
      if (t.pool[idx].kind != phaseConfigKind(f->phase()))
        continue;
      t.fallbacks.emplace(
          fallbackKey(f->phase(), f->act_dtype(), f->wts_dtype(),
                     f->trans_b()),
          static_cast<uint16_t>(idx));
    }
  }

  t.loaded = true;
  if (logOn())
    fprintf(stderr,
            "[gemm-lut] loaded %u points in %zu groups (%u rejected), "
            "weights m=%.2f n=%.2f k=%.2f, bf16_aliases_f16=%d, for %s\n",
            t.points, t.groups.size(), t.invalid_points, t.wm, t.wn, t.wk,
            t.bf16_aliases_f16 ? 1 : 0,
            lut->gpu_arch() ? lut->gpu_arch()->c_str() : "?");
}

} // namespace
} // namespace gemm_autotune
} // namespace hipdnn_ep

// Defined by the CMake-generated byte array (embedded from the checked-in
// lut/<arch>.fb via file(READ ... HEX)) or by tools/empty_lut_data.cpp for an
// arch with no table; see lib/Runtime/Kernels/CMakeLists.txt. Each
// custom_kernels_<arch> DLL links exactly one such payload.
extern "C" const unsigned char kGemmLutData[];
extern "C" const size_t kGemmLutData_size;

namespace hipdnn_ep {
namespace gemm_autotune {
namespace {

Table &table() {
  static Table *t = [] {
    auto *fresh = new Table();
    loadBuffer(*fresh, kGemmLutData, kGemmLutData_size);
    return fresh;
  }();
  return *t;
}

bool accept(const Answer &answer, WmmaValidator wmma_valid,
            GemvValidator gemv_valid, TiledFmaValidator tiled_fma_valid,
            Result &out, void *ctx) {
  if (answer.kind == fbs::GemmConfigKind::Wmma) {
    if (!wmma_valid || !wmma_valid(ctx, answer.wmma))
      return false;
    out.wmma = answer.wmma;
    return true;
  }
  if (answer.kind == fbs::GemmConfigKind::Gemv) {
    if (!gemv_valid || !gemv_valid(ctx, answer.gemv))
      return false;
    out.gemv = answer.gemv;
    return true;
  }
  if (answer.kind == fbs::GemmConfigKind::TiledFma) {
    if (!tiled_fma_valid || !tiled_fma_valid(ctx, answer.tiled_fma))
      return false;
    out.tiled_fma = answer.tiled_fma;
    return true;
  }
  return false;
}

/* Search one group's points for the nearest usable answer, log-space over
 * (M, N, K). Returns true and fills `result`/`chosen_d2` on a usable hit. */
bool searchGroup(Table &t, const std::vector<Point> &pts, float qm, float qn,
                 float qk, WmmaValidator wv, GemvValidator gv,
                 TiledFmaValidator tv, void *ctx, Result &result,
                 const Point *&chosen, float &chosen_d2) {
  auto dist2 = [&](const Point &p) {
    const float dm = t.wm * (qm - p.lm);
    const float dn = t.wn * (qn - p.ln);
    const float dk = t.wk * (qk - p.lk);
    return dm * dm + dn * dn + dk * dk;
  };

  size_t best = 0;
  float best_d2 = dist2(pts[0]);
  for (size_t i = 1; i < pts.size(); ++i) {
    const float d2 = dist2(pts[i]);
    if (d2 < best_d2) {
      best_d2 = d2;
      best = i;
    }
  }

  if (accept(t.pool[pts[best].config], wv, gv, tv, result, ctx)) {
    chosen = &pts[best];
    chosen_d2 = best_d2;
    return true;
  }
  t.rejected.fetch_add(1, std::memory_order_relaxed);

  std::vector<std::pair<float, size_t>> order;
  order.reserve(pts.size());
  for (size_t i = 0; i < pts.size(); ++i)
    if (i != best)
      order.emplace_back(dist2(pts[i]), i);
  std::sort(order.begin(), order.end());
  for (const auto &cand : order) {
    if (accept(t.pool[pts[cand.second].config], wv, gv, tv, result, ctx)) {
      chosen = &pts[cand.second];
      chosen_d2 = cand.first;
      return true;
    }
    t.rejected.fetch_add(1, std::memory_order_relaxed);
  }
  return false;
}

} // namespace

Result resolve(const Request &request, WmmaValidator wmma_valid,
               GemvValidator gemv_valid, TiledFmaValidator tiled_fma_valid,
               void *ctx) {
  Result result;
  Table &t = table();
  if (!t.loaded)
    return result;

  const fbs::GemmPhase phase = phaseClass(request.kind);
  if (phase == fbs::GemmPhase::Any)
    return result;
  if (request.act_dtype == HipdnnDType::Any ||
      request.wts_dtype == HipdnnDType::Any)
    return result;
  if (request.n <= 0 || request.k <= 0)
    return result;
  const common_fbs::HipdnnDType act = toFbsDType(request.act_dtype);
  const common_fbs::HipdnnDType wts = toFbsDType(request.wts_dtype);
  const fbs::GemmTransB trb = transBClass(request.trans_b);

  const float qm = std::log2(static_cast<float>(request.m < 1 ? 1 : request.m));
  const float qn = std::log2(static_cast<float>(request.n));
  const float qk = std::log2(static_cast<float>(request.k));

  auto logHeader = [&](const char *tag) {
    fprintf(stderr,
            "[gemm-lut] %s phase=%d act=%s wts=%s out=%s trb=%d M=%d N=%d "
            "K=%d",
            tag, static_cast<int>(phase),
            common::toString(request.act_dtype),
            common::toString(request.wts_dtype),
            common::toString(request.out_dtype), request.trans_b, request.m,
            request.n, request.k);
  };

  const Point *chosen = nullptr;
  float chosen_d2 = 0.0f;
  bool via_alias = false;

  const auto git = t.groups.find(groupKey(phase, act, wts, trb));
  if (git != t.groups.end() && !git->second.empty() &&
      searchGroup(t, git->second, qm, qn, qk, wmma_valid, gemv_valid,
                 tiled_fma_valid, ctx, result, chosen, chosen_d2)) {
    // exact/nearest below.
  } else if (request.act_dtype == HipdnnDType::BF16 &&
            request.wts_dtype == HipdnnDType::BF16 && t.bf16_aliases_f16) {
    /* Stage C aliasing: only fires for an equal-dtype bf16 request (act==wts,
     * the only shape this round measured), and only when the table says the
     * F16 and BF16 winners were close enough to merge (plan.md §5 Stage C).
     * Always reported as Source::Alias, never disguised as Exact/Nearest. */
    const auto ait =
        t.groups.find(groupKey(phase, common_fbs::HipdnnDType::F16,
                               common_fbs::HipdnnDType::F16, trb));
    if (ait != t.groups.end() && !ait->second.empty() &&
        searchGroup(t, ait->second, qm, qn, qk, wmma_valid, gemv_valid,
                   tiled_fma_valid, ctx, result, chosen, chosen_d2)) {
      via_alias = true;
    }
  }

  if (chosen) {
    const bool exact =
        chosen->m == static_cast<uint32_t>(request.m < 1 ? 1 : request.m) &&
        chosen->n == static_cast<uint32_t>(request.n) &&
        chosen->k == static_cast<uint32_t>(request.k);
    result.source = via_alias
                        ? Source::Alias
                        : (exact ? Source::Exact : Source::Nearest);
    result.distance = std::sqrt(chosen_d2);
    if (logOn()) {
      logHeader(via_alias ? "alias" : (exact ? "exact" : "nearest"));
      fprintf(stderr, " -> point M=%u N=%u K=%u d=%.3f\n", chosen->m,
              chosen->n, chosen->k, result.distance);
    }
    return result;
  }

  const auto fit = t.fallbacks.find(fallbackKey(phase, act, wts, trb));
  if (fit != t.fallbacks.end() &&
      accept(t.pool[fit->second], wmma_valid, gemv_valid, tiled_fma_valid,
            result, ctx)) {
    result.source = Source::Fallback;
    if (logOn()) {
      logHeader("fallback");
      fprintf(stderr, "\n");
    }
    return result;
  }

  if (logOn()) {
    logHeader("miss");
    fprintf(stderr, "\n");
  }
  return result;
}

Stats stats() {
  Table &t = table();
  Stats s;
  s.table_loaded = t.loaded;
  s.points = t.points;
  s.invalid_points = t.invalid_points;
  s.rejected_answers = t.rejected.load(std::memory_order_relaxed);
  return s;
}

} // namespace gemm_autotune
} // namespace hipdnn_ep
