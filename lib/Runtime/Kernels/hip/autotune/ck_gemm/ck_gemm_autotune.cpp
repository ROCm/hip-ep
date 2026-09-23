/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hip_custom_kernels.h"

#include "ck_gemm_autotune_generated.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <hip/hip_runtime.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace hipdnn_ep {
namespace ck_gemm_autotune {
namespace {

namespace fbs = hipdnn_ep::ck_gemm_autotune::fbs;

constexpr uint32_t kSchemaVersion = 1;
constexpr const char kKernelAbi[] = "ck_gemm-v1";

std::string readEnv(const char *name) {
#ifdef _WIN32
  char buf[64];
  const DWORD n = GetEnvironmentVariableA(name, buf, sizeof(buf));
  return (n > 0 && n < sizeof(buf)) ? std::string(buf, n) : std::string();
#else
  const char *v = getenv(name);
  return v ? std::string(v) : std::string();
#endif
}

bool logOn() {
  static const bool on = [] {
    const std::string v = readEnv("HIPDNN_CK_GEMM_LUT_LOG");
    return !v.empty() && v[0] >= '1';
  }();
  return on;
}

bool onlineMode() {
  static const bool on = readEnv("HIPDNN_CK_GEMM_AUTOTUNE_MODE") == "online";
  return on;
}

fbs::CkGemmDtype dtypeClass(int dtype) {
  switch (dtype) {
  case HIP_DTYPE_FLOAT16:
    return fbs::CkGemmDtype::F16;
  case HIP_DTYPE_FLOAT32:
    return fbs::CkGemmDtype::F32;
  default:
    return fbs::CkGemmDtype::Any;
  }
}

uint32_t groupKey(fbs::CkGemmDtype ab, fbs::CkGemmDtype d, bool transA,
                  bool bias) {
  return static_cast<uint32_t>(ab) | (static_cast<uint32_t>(d) << 4) |
         (transA ? 1u << 8 : 0u) | (bias ? 1u << 9 : 0u);
}

// 0..3 for a dim divisible by at most 1, 2, 4, 8.
uint32_t alignClass(int64_t x) {
  return (x % 8 == 0) ? 3u : (x % 4 == 0) ? 2u : (x % 2 == 0) ? 1u : 0u;
}

// CK tiles are gated on vector widths that must divide m, n or k, so the
// winner follows their alignment rather than their magnitude: shapes one apart
// in m can need different tiles. Distance is only meaningful within one of
// these.
uint32_t alignedKey(uint32_t group, int64_t m, int64_t n, int64_t k) {
  return group | (alignClass(m) << 12) | (alignClass(n) << 14) |
         (alignClass(k) << 16);
}

struct Point {
  float lm, ln, lk, lb;
  uint32_t m, n, k, batch;
  int instance;
};

struct Table {
  std::unordered_map<uint32_t, std::vector<Point>> aligned;
  std::unordered_map<uint32_t, std::vector<Point>> groups;
  float wm = 1.0f, wn = 1.0f, wk = 1.0f, wb = 1.0f;
  bool loaded = false;
  uint32_t points = 0;
  uint32_t invalid_points = 0;
};

std::string currentGpuArch() {
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, 0) != hipSuccess) {
    return std::string();
  }
  std::string arch(props.gcnArchName);
  // gcnArchName carries feature suffixes like "gfx1151:xnack-"; the table is
  // stamped with the bare arch.
  const size_t colon = arch.find(':');
  return colon == std::string::npos ? arch : arch.substr(0, colon);
}

bool compatible(const fbs::CkGemmAutotuneLut *lut) {
  if (lut->schema_version() != kSchemaVersion) {
    if (logOn()) {
      fprintf(stderr, "[ck-gemm-lut] schema %u != %u, ignoring table\n",
              lut->schema_version(), kSchemaVersion);
    }
    return false;
  }
  if (!lut->kernel_abi() || lut->kernel_abi()->str() != kKernelAbi) {
    if (logOn()) {
      fprintf(stderr, "[ck-gemm-lut] kernel_abi mismatch, ignoring table\n");
    }
    return false;
  }
  if (lut->gpu_arch() && !lut->gpu_arch()->str().empty()) {
    const std::string actual = currentGpuArch();
    if (actual.empty() || actual != lut->gpu_arch()->str()) {
      if (logOn()) {
        fprintf(stderr,
                "[ck-gemm-lut] arch \"%s\" != device \"%s\", ignoring\n",
                lut->gpu_arch()->c_str(), actual.c_str());
      }
      return false;
    }
  }
  return true;
}

void loadBuffer(Table &t, const unsigned char *data, size_t size) {
  if (!data || size == 0) {
    return;
  }
  flatbuffers::Verifier verifier(data, size);
  if (!fbs::VerifyCkGemmAutotuneLutBuffer(verifier)) {
    if (logOn()) {
      fprintf(stderr, "[ck-gemm-lut] buffer failed verification\n");
    }
    return;
  }
  const fbs::CkGemmAutotuneLut *lut = fbs::GetCkGemmAutotuneLut(data);
  if (!lut || !lut->points() || !lut->instances() || !compatible(lut)) {
    return;
  }
  if (!(lut->weight_m() > 0.0f) || !(lut->weight_n() > 0.0f) ||
      !(lut->weight_k() > 0.0f) || !(lut->weight_batch() > 0.0f)) {
    if (logOn()) {
      fprintf(stderr, "[ck-gemm-lut] non-positive metric weight\n");
    }
    return;
  }
  t.wm = lut->weight_m();
  t.wn = lut->weight_n();
  t.wk = lut->weight_k();
  t.wb = lut->weight_batch();

  std::unordered_map<std::string, int> byName;
  const int count = hip_ck_gemm_num_instances();
  for (int i = 0; i < count; ++i) {
    byName.emplace(hip_ck_gemm_instance_name(i), i);
  }
  std::vector<int> live;
  live.reserve(lut->instances()->size());
  for (const flatbuffers::String *name : *lut->instances()) {
    const auto it = byName.find(name->str());
    live.push_back(it == byName.end() ? -1 : it->second);
  }

  for (const fbs::CkGemmTunePoint *p : *lut->points()) {
    const unsigned idx = p->instance();
    if (idx >= live.size() || live[idx] < 0 ||
        p->ab_dtype() == fbs::CkGemmDtype::Any ||
        p->d_dtype() == fbs::CkGemmDtype::Any || p->m() == 0 || p->n() == 0 ||
        p->k() == 0 || p->batch() == 0) {
      ++t.invalid_points;
      continue;
    }
    Point point;
    point.m = p->m();
    point.n = p->n();
    point.k = p->k();
    point.batch = p->batch();
    point.lm = std::log2(static_cast<float>(point.m));
    point.ln = std::log2(static_cast<float>(point.n));
    point.lk = std::log2(static_cast<float>(point.k));
    point.lb = std::log2(static_cast<float>(point.batch));
    point.instance = live[idx];
    const uint32_t group = groupKey(p->ab_dtype(), p->d_dtype(),
                                    p->trans_a() != 0, p->bias() != 0);
    t.groups[group].push_back(point);
    t.aligned[alignedKey(group, point.m, point.n, point.k)].push_back(point);
    ++t.points;
  }

  t.loaded = true;
  if (logOn()) {
    fprintf(stderr,
            "[ck-gemm-lut] loaded %u points in %zu groups (%u dropped) for "
            "%s\n",
            t.points, t.groups.size(), t.invalid_points,
            lut->gpu_arch() ? lut->gpu_arch()->c_str() : "?");
  }
}

} // namespace
} // namespace ck_gemm_autotune
} // namespace hipdnn_ep

// Generated by lib/Runtime/Kernels/CMakeLists.txt (_emit_lut_registry).
extern "C" const unsigned char *const kCkGemmLutBlobs[];
extern "C" const size_t kCkGemmLutBlobSizes[];
extern "C" const size_t kCkGemmLutBlobCount;

namespace hipdnn_ep {
namespace ck_gemm_autotune {
namespace {

Table &table() {
  static Table *t = [] {
    auto *fresh = new Table();
    for (size_t i = 0; i < kCkGemmLutBlobCount && !fresh->loaded; ++i) {
      loadBuffer(*fresh, kCkGemmLutBlobs[i], kCkGemmLutBlobSizes[i]);
    }
    if (!fresh->loaded && logOn()) {
      fprintf(stderr,
              "[ck-gemm-lut] no usable table (%zu embedded); every shape "
              "sweeps\n",
              kCkGemmLutBlobCount);
    }
    return fresh;
  }();
  return *t;
}

} // namespace
} // namespace ck_gemm_autotune
} // namespace hipdnn_ep

extern "C" int hip_ck_gemm_lut_candidates(int64_t m, int64_t n, int64_t k,
                                          int64_t batch, int transA,
                                          int abDtype, int dDtype, int hasBias,
                                          int *out, int cap) {
  using namespace hipdnn_ep::ck_gemm_autotune;
  if (!out || cap <= 0 || onlineMode()) {
    return 0;
  }
  Table &t = table();
  if (!t.loaded || m <= 0 || n <= 0 || k <= 0) {
    return 0;
  }
  const fbs::CkGemmDtype ab = dtypeClass(abDtype);
  const fbs::CkGemmDtype d = dtypeClass(dDtype);
  if (batch < 1) {
    batch = 1;
  }

  const uint32_t group = groupKey(ab, d, transA != 0, hasBias != 0);
  const auto git = t.groups.find(group);
  if (ab == fbs::CkGemmDtype::Any || d == fbs::CkGemmDtype::Any ||
      git == t.groups.end()) {
    if (logOn()) {
      fprintf(stderr,
              "[ck-gemm-lut] miss m=%lld n=%lld k=%lld batch=%lld transA=%d "
              "ab=%d d=%d bias=%d\n",
              (long long)m, (long long)n, (long long)k, (long long)batch,
              transA, abDtype, dDtype, hasBias);
    }
    return 0;
  }

  const float qm = std::log2(static_cast<float>(m));
  const float qn = std::log2(static_cast<float>(n));
  const float qk = std::log2(static_cast<float>(k));
  const float qb = std::log2(static_cast<float>(batch));
  auto ranked = [&](const std::vector<Point> &pts) {
    std::vector<std::pair<float, size_t>> order;
    order.reserve(pts.size());
    for (size_t i = 0; i < pts.size(); ++i) {
      const Point &p = pts[i];
      const float dm = t.wm * (qm - p.lm);
      const float dn = t.wn * (qn - p.ln);
      const float dk = t.wk * (qk - p.lk);
      const float db = t.wb * (qb - p.lb);
      order.emplace_back(dm * dm + dn * dn + dk * dk + db * db, i);
    }
    std::sort(order.begin(), order.end());
    return order;
  };

  int found = 0;
  const Point *nearest = nullptr;
  float nearest_d2 = 0.0f;
  bool aligned_hit = false;
  auto take = [&](const std::vector<Point> &pts) {
    const auto order = ranked(pts);
    for (const auto &cand : order) {
      const Point &p = pts[cand.second];
      if (!nearest) {
        nearest = &p;
        nearest_d2 = cand.first;
      }
      if (std::find(out, out + found, p.instance) != out + found) {
        continue;
      }
      out[found++] = p.instance;
      if (found == cap) {
        return;
      }
    }
  };
  const auto ait = t.aligned.find(alignedKey(group, m, n, k));
  if (ait != t.aligned.end()) {
    aligned_hit = true;
    take(ait->second);
  }
  if (found < cap) {
    take(git->second);
  }

  if (logOn()) {
    fprintf(stderr,
            "[ck-gemm-lut] m=%lld n=%lld k=%lld batch=%lld transA=%d ab=%d "
            "d=%d bias=%d -> %s (%s point m=%u n=%u k=%u batch=%u d=%.3f), "
            "%d candidate(s)\n",
            (long long)m, (long long)n, (long long)k, (long long)batch, transA,
            abDtype, dDtype, hasBias, hip_ck_gemm_instance_name(out[0]),
            aligned_hit ? "aligned" : "unaligned", nearest->m, nearest->n,
            nearest->k, nearest->batch, std::sqrt(nearest_d2), found);
  }
  return found;
}
