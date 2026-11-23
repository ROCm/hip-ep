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
 * zero points) is queried at the arrival pitch and arrival + 128 B. A query is
 * an Interval hit when an explicit range covers its full key; otherwise it
 * is an Exact hit exactly when the table has a point at that pitch. Both
 * sources return the geometry named by their payload record. A pad with no
 * row-stride category goes to the phase's fallback.
 */
#include "matmul_nbits_autotune.h"
#include "matmul_nbits_autotune_generated.h"
// Keep synthetic-buffer checks on the real private loader without a product
// test API.
#include "../../lib/Runtime/Kernels/hip/autotune/matmul_nbits/matmul_nbits_autotune.cpp"
#include <memory>
#include <new>

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

static bool acceptOriginal(void *, const mna::WmmaAnswer &a) { return !a.ext; }

static void loadFixture(const fbs::MatmulNbitsAutotuneLutT &doc) {
  flatbuffers::FlatBufferBuilder b;
  fbs::FinishMatmulNbitsAutotuneLutBuffer(
      b, fbs::MatmulNbitsAutotuneLut::Pack(b, &doc));
  auto &t = mna::table();
  t.~Table();
  new (&t) mna::Table();
  mna::loadBuffer(t, b.GetBufferPointer(), b.GetSize());
}

static void intervalChecks() {
  fbs::MatmulNbitsAutotuneLutT doc;
  doc.schema_version = 4;
  doc.gpu_arch = "gfx1151";
  doc.kernel_abi = "matmul_nbits-v1";
  doc.configs.emplace_back(fbs::MatmulNbitsConfigKind::Wmma, 4, 4, 2, 2, 2, 16,
                           1, 0, 0);
  doc.configs.emplace_back(fbs::MatmulNbitsConfigKind::Wmma, 2, 2, 4, 1, 2, 64,
                           1, 0, 0);
  doc.extended_configs.emplace_back(1, 0);
  doc.points.emplace_back(
      fbs::MatmulNbitsPhase::Prefill, fbs::MatmulNbitsBits::B4,
      fbs::MatmulNbitsGroupSize::G32, fbs::MatmulNbitsZeroPoint::Symmetric,
      fbs::MatmulNbitsRowStride::Arrival, 0, 64, 512, 2560);
  doc.fallbacks.emplace_back(fbs::MatmulNbitsPhase::Prefill,
                             fbs::MatmulNbitsBits::B4, 0);
  mna::Request q;
  q.m = 64;
  q.n = 512;
  q.k = 2560;
  q.group_size = 32;
  loadFixture(doc);
  CHECK(mna::resolve(q, acceptAll, nullptr, nullptr).source ==
            mna::Source::Exact,
        "old v4 payload (omitted default) must load");
  doc.schema_version = 5;
  loadFixture(doc);
  CHECK(mna::resolve(q, acceptAll, nullptr, nullptr).source ==
            mna::Source::Exact,
        "v5 with no ranges must retain v4 exact lookup");
  doc.m_intervals.emplace_back(
      fbs::MatmulNbitsPhase::Prefill, fbs::MatmulNbitsBits::B4,
      fbs::MatmulNbitsGroupSize::G32, fbs::MatmulNbitsZeroPoint::Symmetric,
      fbs::MatmulNbitsRowStride::Arrival, 1, 60, 70, 2560, 512);
  loadFixture(doc);
  for (int m : {60, 61, 64, 70, 71}) {
    q.m = m;
    const auto a = mna::resolve(q, acceptAll, nullptr, nullptr);
    const bool in = m > 60 && m <= 70;
    CHECK(
        (a.source == mna::Source::Interval) == in && a.wmma.ext == in,
        "exclusive/inclusive endpoint or explicit exact override failed at M%d",
        m);
    if (in)
      CHECK(a.wmma.bm == 32 && a.wmma.bn == 32 && a.wmma.variant == 0,
            "interval must retain full extended geometry");
  }
  q.m = 64;
  CHECK(mna::resolve(q, acceptOriginal, nullptr, nullptr).source ==
            mna::Source::Exact,
        "illegal interval config must fall through to original exact point");
  for (int axis = 0; axis < 7; ++axis) {
    auto different = q;
    switch (axis) {
    case 0:
      different.n++;
      break;
    case 1:
      different.k += 32;
      break;
    case 2:
      different.bits = 2;
      break;
    case 3:
      different.group_size = 128;
      break;
    case 4:
      different.has_zp = true;
      break;
    case 5:
      different.b_row_bytes = 2560 / 2 + 128;
      break;
    case 6:
      different.phase = mna::Phase::DecodeDp4a;
      break;
    }
    CHECK(mna::resolve(different, acceptAll, nullptr, nullptr).source !=
              mna::Source::Interval,
          "interval crossed full-key axis %d", axis);
  }
  const auto good = doc.m_intervals[0];
  doc.m_intervals.emplace_back(
      fbs::MatmulNbitsPhase::Prefill, fbs::MatmulNbitsBits::B4,
      fbs::MatmulNbitsGroupSize::G32, fbs::MatmulNbitsZeroPoint::Symmetric,
      fbs::MatmulNbitsRowStride::Arrival, 1, 65, 80, 2560, 512);
  loadFixture(doc);
  CHECK(mna::stats().invalid_intervals == 2 && mna::stats().intervals == 0 &&
            mna::resolve(q, acceptAll, nullptr, nullptr).source ==
                mna::Source::Exact,
        "overlapping ranges must be refused without losing old exact lookup");
  doc.m_intervals.clear();
  doc.m_intervals.emplace_back(
      fbs::MatmulNbitsPhase::Prefill, fbs::MatmulNbitsBits::B4,
      fbs::MatmulNbitsGroupSize::G32, fbs::MatmulNbitsZeroPoint::Symmetric,
      fbs::MatmulNbitsRowStride::Arrival, 255, 60, 70, 2560, 512);
  loadFixture(doc);
  CHECK(mna::stats().invalid_intervals == 1 &&
            mna::resolve(q, acceptAll, nullptr, nullptr).source ==
                mna::Source::Exact,
        "dangling interval must fall through");
  doc.m_intervals = {good};
  doc.extended_configs.emplace_back(1, 1);
  loadFixture(doc);
  CHECK(mna::stats().invalid_intervals == 1 &&
            mna::resolve(q, acceptAll, nullptr, nullptr).source ==
                mna::Source::Exact,
        "duplicate extended metadata must not override an old point");
  doc.extended_configs.pop_back();
  doc.schema_version = 4;
  loadFixture(doc);
  CHECK(!mna::stats().table_loaded, "v4 must not carry version-5 ranges");
  doc.schema_version = 5;
  doc.gpu_arch = "gfx1152";
  loadFixture(doc);
  CHECK(!mna::stats().table_loaded, "wrong-arch interval table accepted");
  doc.gpu_arch = "gfx1151";
  doc.kernel_abi = "stale";
  loadFixture(doc);
  CHECK(!mna::stats().table_loaded, "wrong-ABI interval table accepted");
}

int main() {
  const auto *lut = fbs::GetMatmulNbitsAutotuneLut(kMatmulNbitsLutBlobs[0]);
  std::map<int, int> ext_variant; // configs index -> variant
  if (lut->extended_configs())
    for (const auto *e : *lut->extended_configs())
      ext_variant[e->config()] = e->variant();

  const auto geometry = [&](int index) {
    const auto *c = lut->configs()->Get(index);
    mna::WmmaAnswer a;
    a.bm = c->bm16() * 16;
    a.bn = c->bn16() * 16;
    a.swizzle_n = c->swizzle();
    a.wt_m = c->wt_m();
    a.wt_n = c->wt_n();
    a.bk = c->bk();
    a.fused = c->fused() != 0;
    const auto ev = ext_variant.find(index);
    a.ext = ev != ext_variant.end();
    a.variant = a.ext ? ev->second : 0;
    return a;
  };

  // Prefill points by stride, with the geometry each one names.
  std::map<std::pair<Key, fbs::MatmulNbitsRowStride>, mna::WmmaAnswer> rows;
  std::set<Key> keys;
  for (const auto *p : *lut->points()) {
    if (p->phase() != fbs::MatmulNbitsPhase::Prefill ||
        p->bits() != fbs::MatmulNbitsBits::B4)
      continue;
    const Key key{groupSize(p->group_size()),
                  p->zero_point() == fbs::MatmulNbitsZeroPoint::Asymmetric,
                  p->m(), static_cast<int>(p->n()), p->k()};
    rows[{key, p->row_stride()}] = geometry(p->config());
    keys.insert(key);
  }
  CHECK(!keys.empty(), "no prefill points in the embedded table");

  const std::pair<int, fbs::MatmulNbitsRowStride> pads[] = {
      {0, fbs::MatmulNbitsRowStride::Arrival},
      {128, fbs::MatmulNbitsRowStride::Padded}};
  int exact = 0, interval_hits = 0;
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
      const fbs::MatmulNbitsMInterval *interval = nullptr;
      if (lut->m_intervals()) {
        for (const auto *candidate : *lut->m_intervals()) {
          if (candidate->phase() != fbs::MatmulNbitsPhase::Prefill ||
              candidate->bits() != fbs::MatmulNbitsBits::B4 ||
              groupSize(candidate->group_size()) != gs ||
              (candidate->zero_point() ==
               fbs::MatmulNbitsZeroPoint::Asymmetric) != zp ||
              candidate->row_stride() != stride || candidate->n() != n ||
              candidate->k() != k || m <= candidate->m_lo() ||
              m > candidate->m_hi())
            continue;
          CHECK(!interval,
                "overlapping embedded intervals at M=%d N=%d K=%d gs=%d pad=%d",
                m, n, k, gs, pad);
          interval = candidate;
        }
      }
      const bool want_exact = row != rows.end();
      CHECK(interval ? r.source == mna::Source::Interval
                     : (r.source == mna::Source::Exact) == want_exact,
            "M=%d N=%d K=%d gs=%d zp=%d pad=%d: source %d, table %s a point "
            "at this pitch",
            m, n, k, gs, zp, pad, static_cast<int>(r.source),
            want_exact ? "has" : "has no");
      if (interval || (want_exact && r.source == mna::Source::Exact)) {
        const mna::WmmaAnswer w =
            interval ? geometry(interval->config()) : row->second;
        CHECK(r.wmma.bm == w.bm && r.wmma.bn == w.bn &&
                  r.wmma.swizzle_n == w.swizzle_n && r.wmma.wt_m == w.wt_m &&
                  r.wmma.wt_n == w.wt_n && r.wmma.bk == w.bk &&
                  r.wmma.fused == w.fused && r.wmma.ext == w.ext &&
                  r.wmma.variant == w.variant,
              "M=%d N=%d K=%d gs=%d pad=%d: answer differs from the payload "
              "record",
              m, n, k, gs, pad);
        if (interval)
          ++interval_hits;
        else
          ++exact;
      }
    }
    // A pad with no row-stride category may only reach the fallback.
    for (const int pad : {192, 256}) {
      req.b_row_bytes = arrival + pad;
      const mna::Result r = mna::resolve(req, acceptAll, nullptr, nullptr);
      CHECK(r.source == mna::Source::Fallback,
            "M=%d N=%d K=%d gs=%d pad=%d: source %d, expected Fallback", m, n,
            k, gs, pad, static_cast<int>(r.source));
    }
  }

  const mna::Stats s = mna::stats();
  CHECK(s.table_loaded, "table not loaded");
  CHECK(s.invalid_points == 0, "%u invalid points", s.invalid_points);
  CHECK(s.invalid_intervals == 0, "%u invalid intervals", s.invalid_intervals);
  std::printf("%zu prefill shapes x 2 pitches, %d exact hits, %d interval "
              "hits, %d failure(s)\n",
              keys.size(), exact, interval_hits, failures);
  intervalChecks();
  std::printf("interval/compatibility checks: %d failure(s)\n", failures);
  return failures ? 1 : 0;
}
