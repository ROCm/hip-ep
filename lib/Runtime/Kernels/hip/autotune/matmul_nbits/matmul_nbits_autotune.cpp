/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "matmul_nbits_autotune.h"

#include "matmul_nbits_autotune_generated.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if !defined(HIPDNN_EP_MATMUL_NBITS_AUTOTUNE_GPU_FREE)
#include <hip/hip_runtime.h>
#endif

#ifdef _WIN32
#include <windows.h>
#endif

namespace hipdnn_ep {
namespace matmul_nbits_autotune {
namespace {

namespace fbs = hipdnn_ep::matmul_nbits_autotune::fbs;

constexpr uint32_t kSchemaVersion = 5;
constexpr uint32_t kLegacySchemaVersion = 4;
// Bump when the config tables or the meaning of a stored geometry changes.
// A table stamped with an older ABI is rejected outright rather than allowed to
// pick configs that no longer mean what they meant when they were measured.
constexpr const char kKernelAbi[] = "matmul_nbits-v1";

bool logOn() {
  static const bool on = [] {
#ifdef _WIN32
    char buf[8];
    return GetEnvironmentVariableA("HIPDNN_MATMUL_LUT_LOG", buf,
                                   sizeof(buf)) > 0 &&
           buf[0] >= '1';
#else
    const char* v = getenv("HIPDNN_MATMUL_LUT_LOG");
    return v && v[0] >= '1';
#endif
  }();
  return on;
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

fbs::MatmulNbitsBits bitsClass(int bits) {
  switch (bits) {
  case 4: return fbs::MatmulNbitsBits::B4;
  case 8: return fbs::MatmulNbitsBits::B8;
  case 3: return fbs::MatmulNbitsBits::B3;
  case 2: return fbs::MatmulNbitsBits::B2;
  default: return fbs::MatmulNbitsBits::Any;
  }
}

fbs::MatmulNbitsGroupSize groupSizeClass(int gs) {
  switch (gs) {
  case 16: return fbs::MatmulNbitsGroupSize::G16;
  case 32: return fbs::MatmulNbitsGroupSize::G32;
  case 64: return fbs::MatmulNbitsGroupSize::G64;
  case 128: return fbs::MatmulNbitsGroupSize::G128;
  case 256: return fbs::MatmulNbitsGroupSize::G256;
  case 512: return fbs::MatmulNbitsGroupSize::G512;
  default: return fbs::MatmulNbitsGroupSize::Any;
  }
}

/* The exact-match part of the key. Points sharing one of these are mutually
 * comparable by distance; points across two of them are not. */
uint32_t groupKey(fbs::MatmulNbitsPhase phase, fbs::MatmulNbitsBits bits, fbs::MatmulNbitsGroupSize gs,
                  fbs::MatmulNbitsZeroPoint zp, fbs::MatmulNbitsRowStride stride) {
  return (static_cast<uint32_t>(phase) & 0x3u) |
         ((static_cast<uint32_t>(bits) & 0x7u) << 2) |
         ((static_cast<uint32_t>(gs) & 0x7u) << 5) |
         ((static_cast<uint32_t>(zp) & 0x3u) << 8) |
         ((static_cast<uint32_t>(stride) & 0x3u) << 10);
}

uint32_t fallbackKey(fbs::MatmulNbitsPhase phase, fbs::MatmulNbitsBits bits) {
  return (static_cast<uint32_t>(phase) & 0x3u) |
         ((static_cast<uint32_t>(bits) & 0x7u) << 2);
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

struct Answer {
  fbs::MatmulNbitsConfigKind kind;
  WmmaAnswer wmma;
  GemvAnswer gemv;
};

/* A measured point, with the logs taken once at load rather than per lookup. */
struct Point {
  float lm, ln, lk;
  uint32_t m, n, k;
  uint16_t config;
};

struct MInterval {
  uint32_t m_lo, m_hi, n, k;
  uint16_t config;
};

struct Table {
  std::vector<Answer> pool;
  std::vector<bool> invalid_config_metadata;
  std::unordered_map<uint32_t, std::vector<Point>> groups;
  std::unordered_map<uint32_t, uint16_t> fallbacks;
  std::unordered_map<uint32_t, std::vector<MInterval>> intervals;
  float wm = 1.0f, wn = 1.0f, wk = 1.0f;
  bool extended_configs = false;   // some config names an extended entry
  bool loaded = false;
  uint32_t points = 0;
  uint32_t invalid_points = 0;
  uint32_t invalid_extended_configs = 0;
  uint32_t interval_count = 0;
  uint32_t invalid_intervals = 0;
  std::atomic<uint32_t> rejected{0};
};

Table& table();

std::string currentGpuArch() {
#if defined(HIPDNN_EP_MATMUL_NBITS_AUTOTUNE_GPU_FREE)
  // No device to ask; the arch a table is checked against comes from the
  // environment, so a build machine can query a shipped .fb without a GPU.
#ifdef _WIN32
  char buf[64];
  DWORD n = GetEnvironmentVariableA("HIPDNN_EP_MATMUL_NBITS_ARCH", buf, sizeof(buf));
  return (n > 0 && n < sizeof(buf)) ? std::string(buf, n) : std::string();
#else
  const char* v = getenv("HIPDNN_EP_MATMUL_NBITS_ARCH");
  return v ? std::string(v) : std::string();
#endif
#else
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, 0) != hipSuccess)
    return std::string();
  std::string arch(props.gcnArchName);
  // gcnArchName carries feature suffixes like "gfx1151:xnack-"; the table is
  // stamped with the bare arch.
  const size_t colon = arch.find(':');
  return colon == std::string::npos ? arch : arch.substr(0, colon);
#endif
}

/* A point must classify every field it is keyed on, carry positive dims, and
 * name a config of the kind its phase consumes. Enforcing it at load is what
 * lets the lookup trust what it finds: a point that slipped through with an
 * unclassified field would sit in a group nothing ever queries, or worse, in
 * one it does. */
bool pointConsistent(const fbs::MatmulNbitsTunePoint& p, fbs::MatmulNbitsConfigKind kind) {
  if (p.phase() == fbs::MatmulNbitsPhase::Any || p.bits() == fbs::MatmulNbitsBits::Any ||
      p.group_size() == fbs::MatmulNbitsGroupSize::Any ||
      p.zero_point() == fbs::MatmulNbitsZeroPoint::Any)
    return false;
  if (p.m() == 0 || p.n() == 0 || p.k() == 0) return false;

  // Only prefill distinguishes row stride, and it must: a decode point naming
  // one would land in a group no decode lookup builds.
  const bool prefill = p.phase() == fbs::MatmulNbitsPhase::Prefill;
  if (prefill != (p.row_stride() != fbs::MatmulNbitsRowStride::Any)) return false;
  if (p.row_stride() > fbs::MatmulNbitsRowStride::MAX) return false;

  // The referenced config has to match the path it will be handed to.
  return kind == (prefill ? fbs::MatmulNbitsConfigKind::Wmma : fbs::MatmulNbitsConfigKind::Gemv);
}

bool compatible(const fbs::MatmulNbitsAutotuneLut* lut) {
  if ((lut->schema_version() != kSchemaVersion &&
       lut->schema_version() != kLegacySchemaVersion) ||
      (lut->schema_version() == kLegacySchemaVersion && lut->m_intervals() &&
       lut->m_intervals()->size() != 0)) {
    if (logOn())
      fprintf(stderr, "[matmul-lut] schema %u != %u, ignoring table\n",
              lut->schema_version(), kSchemaVersion);
    return false;
  }
  if (!lut->kernel_abi() || lut->kernel_abi()->str() != kKernelAbi) {
    if (logOn())
      fprintf(stderr, "[matmul-lut] kernel_abi mismatch, ignoring table\n");
    return false;
  }
  if (lut->gpu_arch() && !lut->gpu_arch()->str().empty()) {
    const std::string actual = currentGpuArch();
    if (actual.empty() || actual != lut->gpu_arch()->str()) {
      if (logOn())
        fprintf(stderr, "[matmul-lut] arch \"%s\" != device \"%s\", ignoring\n",
                lut->gpu_arch()->c_str(), actual.c_str());
      return false;
    }
  }
  return true;
}

void loadBuffer(Table& t, const unsigned char* data, size_t size) {
  // No table embedded for this arch (the size-0 stub cmake emits). Logged, not
  // silent, so HIPDNN_MATMUL_LUT_LOG=1 distinguishes "no table in this build"
  // from "log not wired": every shape will fall through to the runtime sweep.
  if (!data || size == 0) {
    if (logOn())
      fprintf(stderr, "[matmul-lut] no embedded table (size 0); shapes will "
                      "autotune at runtime\n");
    return;
  }
  flatbuffers::Verifier verifier(data, size);
  if (!fbs::VerifyMatmulNbitsAutotuneLutBuffer(verifier)) {
    if (logOn()) fprintf(stderr, "[matmul-lut] buffer failed verification\n");
    return;
  }
  const fbs::MatmulNbitsAutotuneLut* lut = fbs::GetMatmulNbitsAutotuneLut(data);
  if (!lut || !lut->points() || !lut->configs() || !compatible(lut)) {
    if (logOn())
      fprintf(stderr, "[matmul-lut] table present (%zu bytes) but unusable "
                      "(malformed or incompatible); shapes will autotune\n",
              size);
    return;
  }

  // A non-positive weight would collapse a dimension out of the metric, which
  // is never what a fit meant to express; treat the table as unusable rather
  // than quietly matching across it.
  if (!(lut->weight_m() > 0.0f) || !(lut->weight_n() > 0.0f) ||
      !(lut->weight_k() > 0.0f) || !std::isfinite(lut->weight_m()) ||
      !std::isfinite(lut->weight_n()) || !std::isfinite(lut->weight_k())) {
    if (logOn()) fprintf(stderr, "[matmul-lut] non-positive metric weight\n");
    return;
  }
  t.wm = lut->weight_m();
  t.wn = lut->weight_n();
  t.wk = lut->weight_k();

  // Decode the de-duplicated config pool once; points reference it by index.
  const auto* configs = lut->configs();
  t.pool.reserve(configs->size());
  for (const fbs::MatmulNbitsTuneConfig* c : *configs) {
    Answer a{};
    a.kind = c->kind();
    if (a.kind == fbs::MatmulNbitsConfigKind::Wmma) {
      a.wmma.bm = static_cast<int>(c->bm16()) * 16;
      a.wmma.bn = static_cast<int>(c->bn16()) * 16;
      a.wmma.swizzle_n = c->swizzle();
      a.wmma.wt_m = c->wt_m();
      a.wmma.wt_n = c->wt_n();
      a.wmma.bk = c->bk();
      a.wmma.fused = c->fused() != 0;
    } else {
      a.gemv.threads = c->threads();
      a.gemv.tile_n = c->tile_n();
    }
    t.pool.push_back(a);
  }

  t.invalid_config_metadata.assign(t.pool.size(), false);
  if (lut->extended_configs()) {
    for (const fbs::MatmulNbitsExtendedConfig* e : *lut->extended_configs()) {
      const unsigned idx = e->config();
      if (idx >= t.pool.size() || t.pool[idx].kind != fbs::MatmulNbitsConfigKind::Wmma ||
          t.pool[idx].wmma.ext) {
        ++t.invalid_extended_configs;
        if (idx < t.pool.size()) t.invalid_config_metadata[idx] = true;
        continue;
      }
      t.pool[idx].wmma.ext = true;
      t.pool[idx].wmma.variant = e->variant();
      t.extended_configs = true;
    }
  }

  for (const fbs::MatmulNbitsTunePoint* p : *lut->points()) {
    const unsigned idx = p->config();
    if (idx >= t.pool.size()) {   // dangling index: same treatment as a bad key
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
    t.groups[groupKey(p->phase(), p->bits(), p->group_size(), p->zero_point(),
                      p->row_stride())]
        .push_back(point);
    ++t.points;
  }

  // Validate range keys independently. Malformed or overlapping ranges cannot
  // override old points. The ordinary point/fallback table remains available.
  if (lut->m_intervals()) {
    for (const auto* r : *lut->m_intervals()) {
      const unsigned idx = r->config();
      const bool prefill = r->phase() == fbs::MatmulNbitsPhase::Prefill;
      const bool valid = idx < t.pool.size() && !t.invalid_config_metadata[idx] &&
          r->m_lo() < r->m_hi() &&
          r->n() > 0 && r->k() > 0 &&
          r->phase() >= fbs::MatmulNbitsPhase::Prefill &&
          r->phase() <= fbs::MatmulNbitsPhase::MAX &&
          r->bits() > fbs::MatmulNbitsBits::Any &&
          r->bits() <= fbs::MatmulNbitsBits::MAX &&
          r->group_size() > fbs::MatmulNbitsGroupSize::Any &&
          r->group_size() <= fbs::MatmulNbitsGroupSize::MAX &&
          r->zero_point() > fbs::MatmulNbitsZeroPoint::Any &&
          r->zero_point() <= fbs::MatmulNbitsZeroPoint::MAX &&
          (prefill ? (r->row_stride() == fbs::MatmulNbitsRowStride::Arrival ||
                      r->row_stride() == fbs::MatmulNbitsRowStride::Padded)
                   : r->row_stride() == fbs::MatmulNbitsRowStride::Any) &&
          t.pool[idx].kind == (prefill ? fbs::MatmulNbitsConfigKind::Wmma
                                      : fbs::MatmulNbitsConfigKind::Gemv);
      if (!valid) { ++t.invalid_intervals; continue; }
      t.intervals[groupKey(r->phase(), r->bits(), r->group_size(),
                           r->zero_point(), r->row_stride())]
          .push_back({r->m_lo(), r->m_hi(), r->n(), r->k(),
                      static_cast<uint16_t>(idx)});
    }
    for (auto& group : t.intervals) {
      auto& ranges = group.second;
      std::vector<bool> bad(ranges.size(), false);
      for (size_t i = 0; i < ranges.size(); ++i)
        for (size_t j = i + 1; j < ranges.size(); ++j)
          if (ranges[i].n == ranges[j].n && ranges[i].k == ranges[j].k &&
              (std::max)(ranges[i].m_lo, ranges[j].m_lo) <
                  (std::min)(ranges[i].m_hi, ranges[j].m_hi))
            bad[i] = bad[j] = true;
      std::vector<MInterval> usable;
      for (size_t i = 0; i < ranges.size(); ++i) {
        if (bad[i]) ++t.invalid_intervals;
        else { usable.push_back(ranges[i]); ++t.interval_count; }
      }
      ranges.swap(usable);
    }
  }

  if (lut->fallbacks()) {
    for (const fbs::MatmulNbitsFallback* f : *lut->fallbacks()) {
      const unsigned idx = f->config();
      if (idx >= t.pool.size() || f->phase() == fbs::MatmulNbitsPhase::Any ||
          f->bits() == fbs::MatmulNbitsBits::Any)
        continue;
      const bool prefill = f->phase() == fbs::MatmulNbitsPhase::Prefill;
      if (t.pool[idx].kind !=
          (prefill ? fbs::MatmulNbitsConfigKind::Wmma : fbs::MatmulNbitsConfigKind::Gemv))
        continue;
      t.fallbacks.emplace(fallbackKey(f->phase(), f->bits()),
                          static_cast<uint16_t>(idx));
    }
  }

  t.loaded = true;
  if (logOn())
    fprintf(stderr, "[matmul-lut] intervals %u (%u rejected)\n",
            t.interval_count, t.invalid_intervals);
  if (logOn())
    fprintf(stderr,
            "[matmul-lut] loaded %u points in %zu groups (%u rejected), "
            "extended configs %s (%u rejected), weights m=%.2f n=%.2f k=%.2f, "
            "for %s\n",
            t.points, t.groups.size(), t.invalid_points,
            t.extended_configs ? "yes" : "no", t.invalid_extended_configs, t.wm,
            t.wn, t.wk,
            lut->gpu_arch() ? lut->gpu_arch()->c_str() : "?");
}

}  // namespace
}  // namespace matmul_nbits_autotune
}  // namespace hipdnn_ep

// Generated by lib/Runtime/Kernels/CMakeLists.txt (_emit_lut_registry, pure
// CMake, no Python), or tools/empty_lut_data.cpp for a build with no table.
// A gfx*-generic DLL embeds one blob per family-member ISA whose lut/<arch>.fb
// exists; a concrete-arch DLL embeds one; an unmeasured build embeds none. The
// table() loop below keeps the first blob compatible() accepts (its gpu_arch ==
// this device), so each arch gets its own table with no cross-arch mixing.
extern "C" const unsigned char* const kMatmulNbitsLutBlobs[];
extern "C" const size_t kMatmulNbitsLutBlobSizes[];
extern "C" const size_t kMatmulNbitsLutBlobCount;

namespace hipdnn_ep {
namespace matmul_nbits_autotune {
namespace {

Table& table() {
  static const std::unique_ptr<Table> t = [] {
    auto fresh = std::make_unique<Table>();
    for (size_t i = 0; i < kMatmulNbitsLutBlobCount && !fresh->loaded; ++i)
      loadBuffer(*fresh, kMatmulNbitsLutBlobs[i], kMatmulNbitsLutBlobSizes[i]);
    return fresh;
  }();
  return *t;
}

/* Hands `answer` to whichever validator matches its kind. A geometry the live
 * config table no longer contains, or that is illegal for this shape, has to be
 * refused here rather than returned: the caller would launch it. */
bool accept(const Answer& answer, WmmaValidator wmma_valid,
            GemvValidator gemv_valid, void* ctx, Result& out) {
  if (answer.kind == fbs::MatmulNbitsConfigKind::Wmma) {
    if (!wmma_valid || !wmma_valid(ctx, answer.wmma)) return false;
    out.wmma = answer.wmma;
    return true;
  }
  if (!gemv_valid || !gemv_valid(ctx, answer.gemv)) return false;
  out.gemv = answer.gemv;
  return true;
}

}  // namespace

Result resolve(const Request& request, WmmaValidator wmma_valid,
               GemvValidator gemv_valid, void* ctx) {
  Result result;
  Table& t = table();
  if (!t.loaded) return result;

  fbs::MatmulNbitsPhase phase = fbs::MatmulNbitsPhase::Prefill;
  switch (request.phase) {
  case Phase::Prefill:    phase = fbs::MatmulNbitsPhase::Prefill; break;
  case Phase::Decode:     phase = fbs::MatmulNbitsPhase::Decode; break;
  case Phase::DecodeDp4a: phase = fbs::MatmulNbitsPhase::DecodeDp4a; break;
  }
  const fbs::MatmulNbitsBits bits = bitsClass(request.bits);
  if (bits == fbs::MatmulNbitsBits::Any) return result;
  if (request.n <= 0 || request.k <= 0) return result;

  const bool prefill = phase == fbs::MatmulNbitsPhase::Prefill;
  const fbs::MatmulNbitsGroupSize gs = groupSizeClass(request.group_size);
  const fbs::MatmulNbitsZeroPoint zp =
      request.has_zp ? fbs::MatmulNbitsZeroPoint::Asymmetric : fbs::MatmulNbitsZeroPoint::Symmetric;
  fbs::MatmulNbitsRowStride stride = fbs::MatmulNbitsRowStride::Any;
  bool classified = true;
  if (prefill && request.b_row_bytes == 0) {
    stride = fbs::MatmulNbitsRowStride::Arrival;
  } else if (prefill) {
    const int arrival = request.group_size > 0
                            ? (request.k + request.group_size - 1) /
                                  request.group_size * (request.group_size / 2)
                            : request.k / 2;
    switch (request.b_row_bytes - arrival) {
    case 0:   stride = fbs::MatmulNbitsRowStride::Arrival; break;
    case 128: stride = fbs::MatmulNbitsRowStride::Padded; break;
    default:  classified = false; break;
    }
  }

  if (classified && request.m > 0) {
    const auto it = t.intervals.find(groupKey(phase, bits, gs, zp, stride));
    if (it != t.intervals.end()) {
      for (const auto& r : it->second) {
        if (r.n != static_cast<uint32_t>(request.n) ||
            r.k != static_cast<uint32_t>(request.k) ||
            static_cast<uint32_t>(request.m) <= r.m_lo ||
            static_cast<uint32_t>(request.m) > r.m_hi) continue;
        if (accept(t.pool[r.config], wmma_valid, gemv_valid, ctx, result)) {
          result.source = Source::Interval;
          if (logOn())
            fprintf(stderr, "[matmul-lut] interval phase=%d M=%d N=%d K=%d "
                            "gs=%d zp=%d -> (%u,%u] cfg=%u\n",
                    static_cast<int>(phase), request.m, request.n, request.k,
                    request.group_size, static_cast<int>(request.has_zp),
                    r.m_lo, r.m_hi, r.config);
          return result;
        }
        t.rejected.fetch_add(1, std::memory_order_relaxed);
        break;
      }
    }
  }

  const float qm = std::log2(static_cast<float>(request.m < 1 ? 1 : request.m));
  const float qn = std::log2(static_cast<float>(request.n));
  const float qk = std::log2(static_cast<float>(request.k));

  const auto git = classified ? t.groups.find(groupKey(phase, bits, gs, zp, stride))
                              : t.groups.end();
  if (git != t.groups.end()) {
    const std::vector<Point>& pts = git->second;

    auto dist2 = [&](const Point& p) {
      const float dm = t.wm * (qm - p.lm);
      const float dn = t.wn * (qn - p.ln);
      const float dk = t.wk * (qk - p.lk);
      return dm * dm + dn * dn + dk * dk;
    };

    // The nearest point is almost always usable, so find it without building
    // or sorting anything. Ties go to the earlier point, and the builder emits
    // them in a sorted order, so the answer does not depend on load order.
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
      // Rejected: only now is it worth ordering the rest to walk outwards.
      t.rejected.fetch_add(1, std::memory_order_relaxed);
      std::vector<std::pair<float, size_t>> order;
      order.reserve(pts.size());
      for (size_t i = 0; i < pts.size(); ++i)
        if (i != best) order.emplace_back(dist2(pts[i]), i);
      std::sort(order.begin(), order.end());
      for (const auto& cand : order) {
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
      const Point& p = pts[chosen];
      const bool exact =
          p.m == static_cast<uint32_t>(request.m < 1 ? 1 : request.m) &&
          p.n == static_cast<uint32_t>(request.n) &&
          p.k == static_cast<uint32_t>(request.k);
      result.source = exact ? Source::Exact : Source::Nearest;
      result.distance = std::sqrt(best_d2);
      if (logOn())
        fprintf(stderr,
                "[matmul-lut] %s phase=%d M=%d N=%d K=%d gs=%d zp=%d -> "
                "point M=%u N=%u K=%u cfg=%u d=%.3f\n",
                exact ? "exact" : "nearest", static_cast<int>(phase), request.m,
                request.n, request.k, request.group_size,
                static_cast<int>(request.has_zp), p.m, p.n, p.k,
                p.config, result.distance);
      return result;
    }
  }

  const auto fit = t.fallbacks.find(fallbackKey(phase, bits));
  if (fit != t.fallbacks.end() &&
      accept(t.pool[fit->second], wmma_valid, gemv_valid, ctx, result)) {
    result.source = Source::Fallback;
    if (logOn())
      fprintf(stderr, "[matmul-lut] fallback phase=%d M=%d N=%d K=%d gs=%d zp=%d\n",
              static_cast<int>(phase), request.m, request.n, request.k,
              request.group_size, static_cast<int>(request.has_zp));
    return result;
  }

  if (logOn())
    fprintf(stderr, "[matmul-lut] miss phase=%d M=%d N=%d K=%d gs=%d zp=%d\n",
            static_cast<int>(phase), request.m, request.n, request.k,
            request.group_size, static_cast<int>(request.has_zp));
  return result;
}

bool extendedConfigsMeasured() {
  return table().extended_configs;
}

Stats stats() {
  Table& t = table();
  Stats s;
  s.table_loaded = t.loaded;
  s.points = t.points;
  s.invalid_points = t.invalid_points;
  s.rejected_answers = t.rejected.load(std::memory_order_relaxed);
  s.intervals = t.interval_count;
  s.invalid_intervals = t.invalid_intervals;
  return s;
}

}  // namespace matmul_nbits_autotune
}  // namespace hipdnn_ep
