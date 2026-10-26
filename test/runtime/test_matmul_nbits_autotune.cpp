/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

/* Row-stride invariants of the MatMulNBits LUT lookup, on the real embedded
 * table (lut/gfx1151.fb, embedded by CMake) and the resolver that ships in
 * custom_kernels_<arch>, built GPU-free.
 *
 * Asserts properties of any well-formed table rather than its contents: every
 * prefill point is read back from the .fb, and each (M, N, K, group size,
 * zero points) is queried at the arrival pitch, arrival + 128 B and arrival +
 * 256 B. A query is an Exact hit exactly when the table has a point at that
 * pitch, and then returns that point's config: a lookup never answers with a
 * point measured at another pitch. A pad with no row-stride category goes to
 * the phase's fallback.
 */
#include "matmul_nbits_autotune.h"
#include "matmul_nbits_autotune_generated.h"

#include <cstdio>
#include <map>
#include <set>
#include <tuple>

extern "C" const unsigned char *const kMatmulNbitsLutBlobs[];
extern "C" const size_t kMatmulNbitsLutBlobSizes[];

namespace mna = hipdnn_ep::matmul_nbits_autotune;
namespace fbs = hipdnn_ep::matmul_nbits_autotune::fbs;

static int failures = 0;

#define CHECK(cond, ...)                                                       \
  do {                                                                         \
    if (!(cond)) {                                                             \
      std::fprintf(stderr, "FAIL (%s:%d): ", __FILE__, __LINE__);              \
      std::fprintf(stderr, __VA_ARGS__);                                       \
      std::fprintf(stderr, "\n");                                              \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

static bool acceptAll(void *, const mna::WmmaAnswer &) { return true; }

static int groupSize(fbs::MatmulNbitsGroupSize g) {
  return 8 << static_cast<int>(g); // G16 = 1 ... G512 = 6
}

using Key = std::tuple<int, bool, int, int, int>; // gs, zp, m, n, k

int main() {
  const auto *lut = fbs::GetMatmulNbitsAutotuneLut(kMatmulNbitsLutBlobs[0]);
  std::map<int, int> ext_variant; // configs index -> variant
  if (lut->extended_configs())
    for (const auto *e : *lut->extended_configs())
      ext_variant[e->config()] = e->variant();

  // Prefill points by stride, with the geometry each one names.
  std::map<std::pair<Key, fbs::MatmulNbitsRowStride>, mna::WmmaAnswer> rows;
  std::set<Key> keys;
  for (const auto *p : *lut->points()) {
    if (p->phase() != fbs::MatmulNbitsPhase::Prefill ||
        p->bits() != fbs::MatmulNbitsBits::B4)
      continue;
    const auto *c = lut->configs()->Get(p->config());
    mna::WmmaAnswer a;
    a.bm = c->bm16() * 16;
    a.bn = c->bn16() * 16;
    a.swizzle_n = c->swizzle();
    a.wt_m = c->wt_m();
    a.wt_n = c->wt_n();
    a.bk = c->bk();
    a.fused = c->fused() != 0;
    auto ev = ext_variant.find(p->config());
    a.ext = ev != ext_variant.end();
    a.variant = a.ext ? ev->second : 0;
    const Key key{groupSize(p->group_size()),
                  p->zero_point() == fbs::MatmulNbitsZeroPoint::Asymmetric,
                  p->m(), static_cast<int>(p->n()), p->k()};
    rows[{key, p->row_stride()}] = a;
    keys.insert(key);
  }
  CHECK(!keys.empty(), "no prefill points in the embedded table");

  const std::pair<int, fbs::MatmulNbitsRowStride> pads[] = {
      {0, fbs::MatmulNbitsRowStride::Arrival},
      {128, fbs::MatmulNbitsRowStride::Padded},
      {256, fbs::MatmulNbitsRowStride::Padded256}};
  int exact = 0;
  for (const Key &key : keys) {
    const auto [gs, zp, m, n, k] = key;
    mna::Request req;
    req.phase = mna::Phase::Prefill;
    req.bits = 4;
    req.m = m;
    req.n = n;
    req.k = k;
    req.group_size = gs;
    req.has_zp = zp;
    const int arrival = (k + gs - 1) / gs * (gs / 2);
    for (const auto &[pad, stride] : pads) {
      req.b_row_bytes = pad ? arrival + pad : 0;
      const mna::Result r = mna::resolve(req, acceptAll, nullptr, nullptr);
      const auto row = rows.find({key, stride});
      const bool want_exact = row != rows.end();
      CHECK((r.source == mna::Source::Exact) == want_exact,
            "M=%d N=%d K=%d gs=%d zp=%d pad=%d: source %d, table %s a point "
            "at this pitch",
            m, n, k, gs, zp, pad, static_cast<int>(r.source),
            want_exact ? "has" : "has no");
      if (want_exact && r.source == mna::Source::Exact) {
        const mna::WmmaAnswer &w = row->second;
        CHECK(r.wmma.bm == w.bm && r.wmma.bn == w.bn &&
                  r.wmma.swizzle_n == w.swizzle_n && r.wmma.wt_m == w.wt_m &&
                  r.wmma.wt_n == w.wt_n && r.wmma.bk == w.bk &&
                  r.wmma.fused == w.fused && r.wmma.ext == w.ext &&
                  r.wmma.variant == w.variant,
              "M=%d N=%d K=%d gs=%d pad=%d: answer differs from the point", m,
              n, k, gs, pad);
        ++exact;
      }
    }
    // A pad with no row-stride category may only reach the fallback.
    req.b_row_bytes = arrival + 192;
    const mna::Result r = mna::resolve(req, acceptAll, nullptr, nullptr);
    CHECK(r.source == mna::Source::Fallback,
          "M=%d N=%d K=%d gs=%d pad=192: source %d, expected Fallback", m, n, k,
          gs, static_cast<int>(r.source));
  }

  const mna::Stats s = mna::stats();
  CHECK(s.table_loaded, "table not loaded");
  CHECK(s.invalid_points == 0, "%u invalid points", s.invalid_points);
  std::printf("%zu prefill shapes x 3 pitches, %d exact hits, %d failure(s)\n",
              keys.size(), exact, failures);
  return failures ? 1 : 0;
}
