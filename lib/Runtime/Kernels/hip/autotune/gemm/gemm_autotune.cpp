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

constexpr uint32_t kSchemaVersion = 1;
// Bump when the config tables or the meaning of a stored geometry changes.
constexpr const char kKernelAbi[] = "gemm-wmma-v1";

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
  default: return fbs::GemmPhase::Any;
  }
}

fbs::GemmTypeBytes typeBytesClass(int bytes) {
  switch (bytes) {
  case 2: return fbs::GemmTypeBytes::B2;
  case 4: return fbs::GemmTypeBytes::B4;
  case 8: return fbs::GemmTypeBytes::B8;
  default: return fbs::GemmTypeBytes::Any;
  }
}

fbs::GemmTransB transBClass(int trans_b) {
  return trans_b ? fbs::GemmTransB::NT : fbs::GemmTransB::NN;
}

/* The exact-match part of the key. */
uint32_t groupKey(fbs::GemmPhase phase, fbs::GemmTypeBytes tb,
                  fbs::GemmTransB trb) {
  return (static_cast<uint32_t>(phase) & 0x3u) |
         ((static_cast<uint32_t>(tb) & 0x3u) << 2) |
         ((static_cast<uint32_t>(trb) & 0x3u) << 4);
}

uint32_t fallbackKey(fbs::GemmPhase phase, fbs::GemmTypeBytes tb) {
  return (static_cast<uint32_t>(phase) & 0x3u) |
         ((static_cast<uint32_t>(tb) & 0x3u) << 2);
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

struct Answer {
  fbs::GemmConfigKind kind;
  WmmaAnswer wmma;
  GemvAnswer gemv;
};

struct Point {
  float lm, ln, lk;
  uint32_t m, n, k;
  uint16_t config;
};

struct Table {
  std::vector<Answer> pool;
  std::unordered_map<uint32_t, std::vector<Point>> groups;
  std::unordered_map<uint32_t, uint16_t> fallbacks;
  float wm = 1.0f, wn = 1.0f, wk = 1.0f;
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
      p.type_bytes() == fbs::GemmTypeBytes::Any ||
      p.trans_b() == fbs::GemmTransB::Any)
    return false;
  if (p.m() == 0 || p.n() == 0 || p.k() == 0)
    return false;
  const bool wmma = p.phase() == fbs::GemmPhase::Wmma;
  return kind == (wmma ? fbs::GemmConfigKind::Wmma : fbs::GemmConfigKind::Gemv);
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

  const auto *configs = lut->configs();
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
    } else {
      a.gemv.threads = c->threads();
      a.gemv.tile_n = c->tile_n();
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
    t.groups[groupKey(p->phase(), p->type_bytes(), p->trans_b())].push_back(
        point);
    ++t.points;
  }

  if (lut->fallbacks()) {
    for (const fbs::GemmFallback *f : *lut->fallbacks()) {
      const unsigned idx = f->config();
      if (idx >= t.pool.size() || f->phase() == fbs::GemmPhase::Any ||
          f->type_bytes() == fbs::GemmTypeBytes::Any)
        continue;
      const bool wmma = f->phase() == fbs::GemmPhase::Wmma;
      if (t.pool[idx].kind !=
          (wmma ? fbs::GemmConfigKind::Wmma : fbs::GemmConfigKind::Gemv))
        continue;
      t.fallbacks.emplace(fallbackKey(f->phase(), f->type_bytes()),
                          static_cast<uint16_t>(idx));
    }
  }

  t.loaded = true;
  if (logOn())
    fprintf(stderr,
            "[gemm-lut] loaded %u points in %zu groups (%u rejected), "
            "weights m=%.2f n=%.2f k=%.2f, for %s\n",
            t.points, t.groups.size(), t.invalid_points, t.wm, t.wn, t.wk,
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
            GemvValidator gemv_valid, void *ctx, Result &out) {
  if (answer.kind == fbs::GemmConfigKind::Wmma) {
    if (!wmma_valid || !wmma_valid(ctx, answer.wmma))
      return false;
    out.wmma = answer.wmma;
    return true;
  }
  if (!gemv_valid || !gemv_valid(ctx, answer.gemv))
    return false;
  out.gemv = answer.gemv;
  return true;
}

} // namespace

Result resolve(const Request &request, WmmaValidator wmma_valid,
               GemvValidator gemv_valid, void *ctx) {
  Result result;
  Table &t = table();
  if (!t.loaded)
    return result;

  const fbs::GemmPhase phase = phaseClass(request.kind);
  const fbs::GemmTypeBytes tb = typeBytesClass(request.type_bytes);
  if (phase == fbs::GemmPhase::Any || tb == fbs::GemmTypeBytes::Any)
    return result;
  if (request.n <= 0 || request.k <= 0)
    return result;
  const fbs::GemmTransB trb = transBClass(request.trans_b);

  const float qm = std::log2(static_cast<float>(request.m < 1 ? 1 : request.m));
  const float qn = std::log2(static_cast<float>(request.n));
  const float qk = std::log2(static_cast<float>(request.k));

  const auto git = t.groups.find(groupKey(phase, tb, trb));
  if (git != t.groups.end()) {
    const std::vector<Point> &pts = git->second;
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

    size_t chosen = pts.size();
    if (accept(t.pool[pts[best].config], wmma_valid, gemv_valid, ctx, result)) {
      chosen = best;
    } else {
      t.rejected.fetch_add(1, std::memory_order_relaxed);
      std::vector<std::pair<float, size_t>> order;
      order.reserve(pts.size());
      for (size_t i = 0; i < pts.size(); ++i)
        if (i != best)
          order.emplace_back(dist2(pts[i]), i);
      std::sort(order.begin(), order.end());
      for (const auto &cand : order) {
        if (accept(t.pool[pts[cand.second].config], wmma_valid, gemv_valid, ctx,
                   result)) {
          chosen = cand.second;
          best_d2 = cand.first;
          break;
        }
        t.rejected.fetch_add(1, std::memory_order_relaxed);
      }
    }

    if (chosen != pts.size()) {
      const Point &p = pts[chosen];
      const bool exact =
          p.m == static_cast<uint32_t>(request.m < 1 ? 1 : request.m) &&
          p.n == static_cast<uint32_t>(request.n) &&
          p.k == static_cast<uint32_t>(request.k);
      result.source = exact ? Source::Exact : Source::Nearest;
      result.distance = std::sqrt(best_d2);
      if (logOn())
        fprintf(stderr,
                "[gemm-lut] %s kind=%d tb=%d trb=%d M=%d N=%d K=%d -> point "
                "M=%u N=%u K=%u d=%.3f\n",
                exact ? "exact" : "nearest", static_cast<int>(phase),
                request.type_bytes, request.trans_b, request.m, request.n,
                request.k, p.m, p.n, p.k, result.distance);
      return result;
    }
  }

  const auto fit = t.fallbacks.find(fallbackKey(phase, tb));
  if (fit != t.fallbacks.end() &&
      accept(t.pool[fit->second], wmma_valid, gemv_valid, ctx, result)) {
    result.source = Source::Fallback;
    if (logOn())
      fprintf(stderr, "[gemm-lut] fallback kind=%d tb=%d M=%d N=%d K=%d\n",
              static_cast<int>(phase), request.type_bytes, request.m, request.n,
              request.k);
    return result;
  }

  if (logOn())
    fprintf(stderr, "[gemm-lut] miss kind=%d tb=%d M=%d N=%d K=%d\n",
            static_cast<int>(phase), request.type_bytes, request.m, request.n,
            request.k);
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
