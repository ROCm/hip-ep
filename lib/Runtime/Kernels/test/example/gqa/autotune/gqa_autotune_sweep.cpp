// ============================================================
// GQA autotune sweep: enumerate EVERY autotune candidate for a list of real
// model shapes and report the winner per shape.
//
// The production launchers (hip_gqa_flash_prefill_v2 / hip_gqa_flash_decode)
// tune once per shape and then hide the choice inside a process-local cache, so
// there is no way from the public ABI to see the full candidate ranking or to
// control the timing budget. This TU therefore #includes the kernel source and
// drives the internal dispatch helpers directly:
//
//   d == 64  -> gqa_flash_prefill_v5   candidates: M_TILES x BKV
//   d == 128 -> gqa_flash_prefill_v7   candidates: NW x BKV x MT
//   d == 256 -> gqa_flash_prefill_v8   candidates: ND x MT x BKV
//   decode   -> flash split-K decode   candidates: {scalar,wmma} x split-count
//
// The candidate sets are the ones the production tuners use
// (prefillV*Candidates / flashDecodeCandidateSplits), so a winner reported here
// is a config the runtime can actually land on. Decode additionally sweeps
// split counts outside the production guard, flagged `prod=0`, to show what the
// guard costs.
//
// Because this file owns the kernel TU, do NOT link gqa_kernel.obj with it.
//
// Output: one CSV row per (shape, candidate) on stdout, plus a human-readable
// per-shape summary on stderr.
// ============================================================

#include "../../../../hip/gqa_kernel.hip"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#define SWEEP_CHECK(expr)                                                      \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),        \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

namespace {

struct Shape {
  std::string id;
  std::string group;
  std::string phase; // "prefill" | "decode"
  int B, H, G, d, sq, skv, window;
  // KV cache capacity, i.e. the stride between heads in [B, G, max_seq, d]. Not
  // the same axis as skv, and worth controlling separately: a power-of-two
  // stride aliases in cache badly enough to reorder the candidates, and a real
  // cache capacity usually *is* a power of two while skv is whatever the step
  // is at. Defaults to skv when the shape file omits the column.
  int max_seq;
  int sink; // 1 = head_sink tensor + smooth_softmax (gpt-oss layers)
  std::string src_rows;
  std::string note;
};

struct Result {
  std::string cfg; // canonical config string, e.g. "MT2_BKV32"
  double ms;
  bool prod_candidate; // in the production tuner's candidate set
  double rel_l2;       // vs. the first candidate's output (-1 = not checked)
  // The same config split into its individual knobs, so the summary table can
  // give each tunable its own column instead of a string a reader has to parse.
  // -1 = the knob does not exist for this kernel.
  int impl_wmma = -1, splits = -1, mt = -1, bkv = -1, nw = -1, nd = -1;
};

// What the *production* tuner would spend on its first call for this shape.
// The runtime pays this once per cache key before any real work is produced, so
// it is the number that matters when judging whether a tuner is worth its cost.
struct TuneCost {
  double ms;    // wall time of the tuning launches
  int launches; // kernel launches issued while tuning
};

// tunePrefillV5/V7/V8Cfg: kWarmup=30 launches cycling the candidate list, then
// kRounds=4 x (1 untimed + kIters=30 timed) launches per candidate.
TuneCost prodPrefillTuneCost(const std::vector<Result> &res) {
  TuneCost c{0.0, 0};
  const int nc = (int)res.size();
  if (nc == 0)
    return c;
  for (int w = 0; w < 30; ++w) {
    c.ms += res[w % nc].ms;
    ++c.launches;
  }
  for (const Result &r : res) {
    c.ms += 4 * 31 * r.ms;
    c.launches += 4 * 31;
  }
  return c;
}

// tuneFlashDecode: WARMUP=2 + ITERS=10 launches per candidate, and only the
// production candidate set is ever timed.
TuneCost prodDecodeTuneCost(const std::vector<Result> &res) {
  TuneCost c{0.0, 0};
  for (const Result &r : res) {
    if (!r.prod_candidate)
      continue;
    c.ms += 12 * r.ms;
    c.launches += 12;
  }
  return c;
}

// ---- timing -----------------------------------------------------------------

// Time `launch` with an iteration count sized to the shape: one untimed probe
// call sets the per-call cost, then enough iterations to fill target_ms are run
// for each of `rounds` measurements. Big prefill shapes (0.5 s a call) would
// otherwise inherit the production tuner's fixed 30x4x30 launches.
//
// The rounds are reduced by MEDIAN, not min. hipEventElapsedTime occasionally
// returns ~0 for a round; with min that single reading becomes the config's
// time and wins the sweep outright (observed: a decode candidate "timing" at
// 160 ns, 49x below every sibling, while its output was bit-comparable). The
// median needs two bad readings in a row to be fooled, and costs nothing --
// interference makes rounds slower, and the median rejects those too.
template <typename F>
double timeConfig(F &&launch, double target_ms, int min_iters, int max_iters,
                  int rounds) {
  (void)hipGetLastError(); // don't attribute an earlier config's error here
  launch();
  SWEEP_CHECK(hipDeviceSynchronize());
  // A rejected launch returns instantly and would otherwise look like the
  // fastest config in the sweep.
  SWEEP_CHECK(hipGetLastError());

  hipEvent_t e0, e1;
  SWEEP_CHECK(hipEventCreate(&e0));
  SWEEP_CHECK(hipEventCreate(&e1));

  SWEEP_CHECK(hipEventRecord(e0, nullptr));
  launch();
  SWEEP_CHECK(hipEventRecord(e1, nullptr));
  SWEEP_CHECK(hipEventSynchronize(e1));
  float probe_ms = 0.0f;
  SWEEP_CHECK(hipEventElapsedTime(&probe_ms, e0, e1));

  int iters = min_iters;
  if (probe_ms > 1e-6f)
    iters = (int)(target_ms / probe_ms);
  iters = std::max(min_iters, std::min(max_iters, iters));
  // A call that already runs for tens of ms is far above the noise floor that
  // repeated rounds exist to suppress, and the 32k-prompt shapes cost minutes
  // if every round is honoured.
  // A median needs an odd sample count to be a real sample, and three rounds is
  // the cheapest count that survives one bad reading.
  rounds = std::max(3, rounds | 1);
  if (probe_ms > 50.0f)
    rounds = 3;

  std::vector<double> samples;
  samples.reserve(rounds);
  for (int r = 0; r < rounds; ++r) {
    launch(); // warm the caches for this round
    SWEEP_CHECK(hipEventRecord(e0, nullptr));
    for (int i = 0; i < iters; ++i)
      launch();
    SWEEP_CHECK(hipEventRecord(e1, nullptr));
    SWEEP_CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    SWEEP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    samples.push_back((double)ms / iters);
  }
  SWEEP_CHECK(hipEventDestroy(e0));
  SWEEP_CHECK(hipEventDestroy(e1));

  std::sort(samples.begin(), samples.end());
  // SWEEP_TRACE=1 prints the raw rounds. Worth having: a config whose rounds
  // disagree by 2x is not a config that lost, it is a reading that did not
  // measure anything, and the median hides that.
  static const bool trace = getenv("SWEEP_TRACE") != nullptr;
  if (trace) {
    fprintf(stderr, "      iters=%d rounds:", iters);
    for (double s : samples)
      fprintf(stderr, " %.5f", s);
    fprintf(stderr, " ms\n");
  }
  return samples[samples.size() / 2];
}

// Median duration of a dispatch timed on its own, which is a different quantity
// from the one above and the right one for decode.
//
// Dividing a loop of N launches by N measures the throughput of back-to-back
// launches. For a short decode kernel that is neither what the runtime will see
// nor a stable number: consecutive launches pipeline, so the marginal cost of
// one falls below its own latency -- 16:4:256 at seq_kv=160 read 0.34 us that
// way, under a launch, and its candidates spread 120x -- and `iters` is derived
// from a cold probe, so two candidates of the same shape end up measured over
// different window lengths. RGP capture, which is what the base geometries were
// measured with, reports each dispatch's own duration; so does this.
//
// The cost is a host sync per sample. At 20-200 samples of a 10 us kernel that
// is a few milliseconds a candidate, which is cheaper than the loop form it
// replaces.
template <typename F>
double timeIsolated(F &&launch, double budget_ms, int min_samples,
                    int max_samples) {
  (void)hipGetLastError();
  launch();
  SWEEP_CHECK(hipDeviceSynchronize());
  // A rejected launch returns instantly and would otherwise look like the
  // fastest config in the sweep.
  SWEEP_CHECK(hipGetLastError());

  hipEvent_t e0, e1;
  SWEEP_CHECK(hipEventCreate(&e0));
  SWEEP_CHECK(hipEventCreate(&e1));
  const auto once = [&]() {
    SWEEP_CHECK(hipEventRecord(e0, nullptr));
    launch();
    SWEEP_CHECK(hipEventRecord(e1, nullptr));
    SWEEP_CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    SWEEP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    return (double)ms;
  };

  const double probe = once();
  int samples = max_samples;
  if (probe > 1e-6)
    samples = (int)(budget_ms / probe);
  samples = std::max(min_samples, std::min(max_samples, samples));

  std::vector<double> got;
  got.reserve(samples);
  for (int i = 0; i < samples; ++i)
    got.push_back(once());
  SWEEP_CHECK(hipEventDestroy(e0));
  SWEEP_CHECK(hipEventDestroy(e1));

  std::sort(got.begin(), got.end());
  static const bool trace = getenv("SWEEP_TRACE") != nullptr;
  if (trace)
    fprintf(stderr,
            "      n=%d  min %.5f  p25 %.5f  median %.5f  p75 %.5f  "
            "max %.5f ms\n",
            (int)got.size(), got.front(), got[got.size() / 4],
            got[got.size() / 2], got[3 * got.size() / 4], got.back());
  return got[got.size() / 2];
}

double relL2(const std::vector<float> &a, const std::vector<float> &b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double diff = (double)a[i] - (double)b[i];
    num += diff * diff;
    den += (double)b[i] * (double)b[i];
  }
  return std::sqrt(num / (den + 1e-12));
}

std::vector<float> fetchHalf(const __half *dev, size_t n) {
  std::vector<__half> tmp(n);
  SWEEP_CHECK(
      hipMemcpy(tmp.data(), dev, n * sizeof(__half), hipMemcpyDeviceToHost));
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i)
    out[i] = __half2float(tmp[i]);
  return out;
}

// What the production online tuner decided, and what deciding cost.
//
// Filled by calling the same tune* function the runtime calls -- not a model of
// it
// -- so the decision time is the real thing: its own warmup, rounds and
// iteration counts, on this machine, in whatever state the GPU is in. That last
// part matters, because the tuner runs once when a cache key is first seen,
// which in production is early and on a cold GPU.
struct OnlineResult {
  bool measured = false;
  std::string cfg;
  double decide_ms = 0.0;
  int launches = 0; // kernel launches the tuner issued
};

// ---- prefill sweep ----------------------------------------------------------

std::vector<Result> sweepPrefill(const Shape &s, bool verify, double target_ms,
                                 int rounds, OnlineResult *online = nullptr,
                                 bool print_realtime = true) {
  const int B = s.B, H = s.H, G = s.G, d = s.d, sq = s.sq, skv = s.skv;
  const int max_seq = s.max_seq;
  const int past_len = skv - sq;
  const float scale = 1.0f / std::sqrt((float)d);

  const size_t qn = (size_t)B * sq * H * d;
  const size_t kn = (size_t)B * G * max_seq * d;

  std::mt19937 rng(1234u + (unsigned)sq + (unsigned)d);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<__half> hQ(qn), hK(kn), hV(kn), hSink(H);
  for (auto &x : hQ)
    x = __float2half(dist(rng));
  for (auto &x : hK)
    x = __float2half(dist(rng));
  for (auto &x : hV)
    x = __float2half(dist(rng));
  for (int h = 0; h < H; ++h)
    hSink[h] = __float2half(-2.0f + 4.0f * (float)h / (float)H);

  __half *dQ, *dK, *dV, *dO, *dSink;
  SWEEP_CHECK(hipMalloc(&dQ, qn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dK, kn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dV, kn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dO, qn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dSink, (size_t)H * sizeof(__half)));
  SWEEP_CHECK(
      hipMemcpy(dQ, hQ.data(), qn * sizeof(__half), hipMemcpyHostToDevice));
  SWEEP_CHECK(
      hipMemcpy(dK, hK.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
  SWEEP_CHECK(
      hipMemcpy(dV, hV.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
  SWEEP_CHECK(hipMemcpy(dSink, hSink.data(), (size_t)H * sizeof(__half),
                        hipMemcpyHostToDevice));

  const _Float16 *Q = reinterpret_cast<const _Float16 *>(dQ);
  const _Float16 *K = reinterpret_cast<const _Float16 *>(dK);
  const _Float16 *V = reinterpret_cast<const _Float16 *>(dV);
  _Float16 *O = reinterpret_cast<_Float16 *>(dO);
  // gqa.cpp sends the sink tensor and smooth_softmax=1 together on gpt-oss.
  const _Float16 *sink =
      s.sink ? reinterpret_cast<const _Float16 *>(dSink) : nullptr;
  const int smooth = s.sink ? 1 : 0;
  const int window = s.window > 0 ? s.window : 0;

  // Verifying every candidate needs a host copy of O per candidate; skip it on
  // the multi-hundred-MB prompts where it dominates the sweep's wall time.
  const bool do_verify = verify && qn <= (size_t)8 << 20;
  std::vector<float> ref;

  std::vector<Result> out;
  auto measure = [&](Result r, auto &&launch) {
    r.ms = timeConfig(launch, target_ms, 3, 2000, rounds);
    r.prod_candidate = true;
    r.rel_l2 = -1.0;
    if (do_verify) {
      launch();
      SWEEP_CHECK(hipDeviceSynchronize());
      std::vector<float> cur = fetchHalf(dO, qn);
      if (ref.empty())
        ref = cur;
      else
        r.rel_l2 = relL2(cur, ref);
    }
    out.push_back(r);
    if (print_realtime) {
      // Real-time print: show each candidate as soon as it is measured.
      double best_so_far = out[0].ms;
      for (const auto &x : out) best_so_far = std::min(best_so_far, x.ms);
      fprintf(stderr, "   %-18s %9.5f ms%s\n", r.cfg.c_str(), r.ms,
              r.ms <= best_so_far + 1e-9 ? "  <-- best so far" : "");
      fflush(stderr);
    }
  };

  if (d == 64) {
    PrefillV5Cfg cands[8];
    const int nc = prefillV5Candidates(d, cands);
    for (int i = 0; i < nc; ++i) {
      const PrefillV5Cfg c = cands[i];
      char cfg[64];
      snprintf(cfg, sizeof(cfg), "MT%d_BKV%d", c.m_tiles, c.bkv);
      Result r;
      r.cfg = cfg;
      r.mt = c.m_tiles;
      r.bkv = c.bkv;
      measure(r, [&]() {
        dispatchPrefillV5(c.m_tiles, c.bkv, d, nullptr, Q, K, V, O, B, H, G, sq,
                          skv, max_seq, past_len, scale, sink, H, smooth,
                          window);
      });
    }
  } else if (d == 128) {
    PrefillV7Cfg cands[12];
    const int nc = prefillV7Candidates(d, cands);
    for (int i = 0; i < nc; ++i) {
      const PrefillV7Cfg c = cands[i];
      char cfg[64];
      snprintf(cfg, sizeof(cfg), "NW%d_BKV%d_MT%d", c.nw, c.bkv, c.mt);
      Result r;
      r.cfg = cfg;
      r.nw = c.nw;
      r.bkv = c.bkv;
      r.mt = c.mt;
      measure(r, [&]() {
        dispatchPrefillV7(c.nw, c.bkv, c.mt, d, nullptr, Q, K, V, O, B, H, G,
                          sq, skv, max_seq, past_len, scale);
      });
    }
  } else {
    PrefillV8Cfg cands[8];
    const int nc = prefillV8Candidates(d, cands);
    for (int i = 0; i < nc; ++i) {
      const PrefillV8Cfg c = cands[i];
      char cfg[64];
      snprintf(cfg, sizeof(cfg), "ND%d_MT%d_BKV%d", c.nd, c.mt, c.bkv);
      Result r;
      r.cfg = cfg;
      r.nd = c.nd;
      r.mt = c.mt;
      r.bkv = c.bkv;
      measure(r, [&]() {
        dispatchPrefillV8(c.nd, c.mt, c.bkv, d, nullptr, Q, K, V, O, B, H, G,
                          sq, skv, max_seq, past_len, scale);
      });
    }
  }

  if (online) {
    // The production tuner, called exactly as the runtime calls it. Timed
    // around a full sync so the number includes everything it makes the GPU do.
    SWEEP_CHECK(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    int cfg = 0;
    char buf[64] = {0};
    if (d == 64) {
      cfg = tunePrefillV5Cfg(d, nullptr, Q, K, V, O, B, H, G, sq, skv, max_seq,
                             past_len, scale, window);
      snprintf(buf, sizeof(buf), "MT%d_BKV%d", cfg / 1000, cfg % 1000);
    } else if (d == 128) {
      cfg = tunePrefillV7Cfg(d, nullptr, Q, K, V, O, B, H, G, sq, skv, max_seq,
                             past_len, scale);
      snprintf(buf, sizeof(buf), "NW%d_BKV%d_MT%d", (cfg / 1000) % 1000,
               cfg % 1000, cfg / 1000000);
    } else {
      cfg = tunePrefillV8Cfg(d, nullptr, Q, K, V, O, B, H, G, sq, skv, max_seq,
                             past_len, scale);
      snprintf(buf, sizeof(buf), "ND%d_MT%d_BKV%d", cfg / 1000000,
               (cfg / 1000) % 1000, cfg % 1000);
    }
    SWEEP_CHECK(hipDeviceSynchronize());
    online->decide_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
    online->cfg = buf;
    // kWarmup=30 warmup launches cycling the candidates, then kRounds=4 rounds
    // of (1 untimed + kIters=30 timed) per candidate. Constants live in the
    // tune* functions in gqa_kernel.hip.
    online->launches = 30 + (int)out.size() * 4 * 31;
    online->measured = true;
  }

  hipFree(dQ);
  hipFree(dK);
  hipFree(dV);
  hipFree(dO);
  hipFree(dSink);
  return out;
}

// How a reading was taken, stamped on every row of the results CSV. Bump it
// whenever a change makes new readings incomparable with old ones -- a different
// timer, a different cache regime, a different candidate clamp. The measurement
// store keys on it and refuses to pool across values, which is the only thing
// stopping a stale reading from quietly deciding a row.
//
//   iso1   decode: each dispatch timed on its own, KV cache rotated past the
//          last-level cache (2026-08-18). Before it: a loop of N launches over one
//          cache, which measured cache bandwidth and launch pipelining instead of
//          the kernel, up to 20x fast and not reproducible.
//   loop1  prefill: a loop of N launches divided by N. A 10 ms kernel does not care
//          about launch overhead (measured at 1.011x) and its own tiles miss cache
//          regardless, so neither problem applies.
constexpr const char *kDecodeHarnessVersion = "iso1";
constexpr const char *kPrefillHarnessVersion = "loop1";

// What the reading measured, as opposed to how. **Bump the entry for a kernel
// whenever it changes in a way that can move which config wins** -- a different
// tile shape, a different inner loop, a new or removed candidate, anything that is
// not a pure refactor.
//
// This is the mechanism that makes a kernel change maintainable rather than a
// re-measure of everything. tools/measurement_store.py parses this table, so a bump
// stops the store reusing readings of *that* kernel and nothing else: change the v7
// prefill kernel and the decode grid stays valid. Leaving it alone after a real
// change is the failure this exists to prevent, and it is silent, so it belongs in
// the same commit as the kernel change.
struct KernelVersion {
  const char *kernel;
  const char *version;
};
constexpr KernelVersion kKernelVersions[] = {
    {"flash_decode", "decode-2"},
    {"prefill_v5", "v5-1"},
    {"prefill_v7", "v7-1"},
    {"prefill_v8", "v8-1"},
};

static const char *kernelVersion(const char *kernel) {
  for (const KernelVersion &kv : kKernelVersions)
    if (strcmp(kv.kernel, kernel) == 0)
      return kv.version;
  return "unknown";
}

// ---- decode sweep -----------------------------------------------------------

// How many copies of the KV cache a decode measurement may rotate over. See the
// comment where `copies` is computed: the point is to make a timing loop read
// as much distinct memory as a sequence of layers does, and 48 copies covers
// the smallest shapes in the grid without spending more than a few hundred MB.
constexpr size_t kKvCopiesMax = 48;

std::vector<Result> sweepDecode(const Shape &s, bool verify, double target_ms,
                                int rounds, int split_cap,
                                OnlineResult *online = nullptr,
                                bool print_realtime = true) {
  const int B = s.B, H = s.H, G = s.G, d = s.d, skv = s.skv;
  const int hpg = H / G;
  const int max_seq = s.max_seq;
  // The decode kernels take the split count as a runtime argument, so a cap
  // above kFlashDecodeMaxSplits only needs a bigger partials workspace.
  // Allowing it here answers whether the production ceiling is leaving anything
  // on the table at 32k context, where 64 splits is the only candidate left.
  const int cap = split_cap;
  const float scale = 1.0f / std::sqrt((float)d);
  const int window = s.window > 0 ? s.window : -1;
  // Mirrors hip_gqa_flash_decode: a sliding layer tunes over the window, not
  // the full context, so its candidate clamp uses the window length.
  const int tune_len = (window > 0 && skv > window) ? window : skv;

  const size_t qn = (size_t)B * H * d;
  const size_t kn = (size_t)B * G * max_seq * d;
  const size_t pn = (size_t)B * H * cap * (d + 2);

  std::mt19937 rng(4321u + (unsigned)skv + (unsigned)d);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<__half> hQ(qn), hK(kn), hV(kn), hSink(H);
  for (auto &x : hQ)
    x = __float2half(dist(rng));
  for (auto &x : hK)
    x = __float2half(dist(rng));
  for (auto &x : hV)
    x = __float2half(dist(rng));
  for (int h = 0; h < H; ++h)
    hSink[h] = __float2half(-2.0f + 4.0f * (float)h / (float)H);
  std::vector<int> hSeq(B, skv - 1); // kernels read seqlens_k + 1

  // One KV cache is not what a decode step reads. A layer reads its own K and V
  // once per token, and a 32-layer model at 8 k context walks well over a
  // hundred megabytes doing it, so every layer's read comes from memory. Timing
  // a loop over a *single* cache measures the opposite: at 2 MB the whole thing
  // sits in the 32 MB last-level cache after the first iteration, and what
  // comes out is cache bandwidth. On 16:4:128 at seq_kv=1055 that reads as low
  // as 1.26 us, which is 1.4 TB/s -- above what the memory can do -- and it is
  // not stable either: three back-to-back sweeps of that shape measured
  // scalar_SPLITS16 at 6.03, 1.57 and 19.57 us depending on what stayed
  // resident.
  //
  // So the loop rotates over enough copies of the cache to exceed the
  // last-level cache, which is both what a real layer sequence does and what
  // makes the reading reproducible. Small shapes get many copies and large ones
  // get one.
  const size_t kv_bytes = kn * sizeof(__half);
  const size_t kKvRotationBytes = 192u << 20; // > 32 MB LLC, with margin
  const int copies = (int)std::min<size_t>(
      kKvCopiesMax, std::max<size_t>(1, kKvRotationBytes / (2 * kv_bytes + 1)));

  __half *dQ, *dO, *dSink;
  __half *dK[kKvCopiesMax];
  __half *dV[kKvCopiesMax];
  float *dPart;
  int *dSeq;
  SWEEP_CHECK(hipMalloc(&dQ, qn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dO, qn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dSink, (size_t)H * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dPart, pn * sizeof(float)));
  SWEEP_CHECK(hipMalloc(&dSeq, (size_t)B * sizeof(int)));
  for (int i = 0; i < copies; ++i) {
    SWEEP_CHECK(hipMalloc(&dK[i], kv_bytes));
    SWEEP_CHECK(hipMalloc(&dV[i], kv_bytes));
    SWEEP_CHECK(hipMemcpy(dK[i], hK.data(), kv_bytes, hipMemcpyHostToDevice));
    SWEEP_CHECK(hipMemcpy(dV[i], hV.data(), kv_bytes, hipMemcpyHostToDevice));
  }
  SWEEP_CHECK(
      hipMemcpy(dQ, hQ.data(), qn * sizeof(__half), hipMemcpyHostToDevice));
  SWEEP_CHECK(hipMemcpy(dSink, hSink.data(), (size_t)H * sizeof(__half),
                        hipMemcpyHostToDevice));
  SWEEP_CHECK(hipMemcpy(dSeq, hSeq.data(), (size_t)B * sizeof(int),
                        hipMemcpyHostToDevice));

  const __half *sink = s.sink ? dSink : nullptr;
  const int smooth = s.sink ? 1 : 0;

  // Production candidate set, used to flag which rows the runtime can pick. It
  // is always computed at the production ceiling, even when the sweep is
  // exploring above it.
  int prod_splits[7];
  int n_prod = 0;
  flashDecodeCandidateSplits(tune_len, kFlashDecodeMaxSplits, prod_splits,
                             n_prod);
  auto is_prod = [&](int sp) {
    if (sp > kFlashDecodeMaxSplits)
      return false;
    for (int i = 0; i < n_prod; ++i)
      if (prod_splits[i] == sp)
        return true;
    return false;
  };

  // Sweep the full ladder, not just the production subset: the guard in
  // flashDecodeCandidateSplits drops small splits at long context, and the
  // dropped timings are what justify the guard.
  static const int kAllSplits[] = {1,  2,  4,  8,   16,  32,
                                   48, 64, 96, 128, 192, 256};
  const int max_useful = std::max(1, (tune_len + 15) / 16);

  const bool wmma_ok = flash_decode_wmma_supported(d, hpg);
  std::vector<float> ref;
  std::vector<Result> out;

  for (int impl = 0; impl < 2; ++impl) {
    const bool use_wmma = (impl == 1);
    if (use_wmma && !wmma_ok)
      continue;
    // PR #675 makes BKV=16 versus BKV=32 a d64 WMMA choice. d128 has one
    // BKV=32 instantiation, so its established `wmma_SPLITS*` spelling stays
    // a single candidate.
    const int bkv_values[2] = {16, 32};
    const int bkv_count = use_wmma && d == 64 ? 2 : 1;
    for (int bi = 0; bi < bkv_count; ++bi) {
      const int bkv = bkv_values[bi];
      int last = -1;
      for (int sp_base : kAllSplits) {
        const int sp = std::min(sp_base, std::min(cap, max_useful));
        if (sp == last)
          continue; // clamping collapsed this rung
        last = sp;
        // The online tuner never considers the tall tile when the per-split
        // range cannot amortize its register and ragged-tail cost. Do not let
        // an offline row name a candidate production would not consider.
        if (use_wmma && d == 64 &&
            !flashDecodeTallTileUseful(bkv, tune_len, sp))
          continue;
        const FlashDecodeCfg cfg{use_wmma, sp, bkv};
        int turn = 0;
        auto launch = [&]() {
          // Next cache in the rotation, so consecutive launches read different
          // memory the way consecutive layers do.
          const int i = turn++ % copies;
          launchFlashDecodeConfig<KvDtype::kF16>(
              cfg, nullptr, dQ, dK[i], dV[i], nullptr, nullptr, dO, dPart, B,
              H, G, d, hpg, max_seq, scale, dSeq, window, sink, smooth);
        };
        // Each dispatch on its own, and each one reading the next cache in the
        // rotation. `rounds` is spent on samples instead: the median of 40-800
        // isolated dispatches rather than the median of a handful of averages.
        const double ms = timeIsolated(launch, target_ms * rounds, 40, 800);
        double err = -1.0;
        if (verify) {
          launch();
          SWEEP_CHECK(hipDeviceSynchronize());
          std::vector<float> cur = fetchHalf(dO, qn);
          if (ref.empty())
            ref = cur;
          else
            err = relL2(cur, ref);
        }
        char name[64];
        if (use_wmma && d == 64)
          snprintf(name, sizeof(name), "wmma_BKV%d_SPLITS%d", bkv, sp);
        else
          snprintf(name, sizeof(name), "%s_SPLITS%d",
                   use_wmma ? "wmma" : "scalar", sp);
        Result r;
        r.cfg = name;
        r.ms = ms;
        r.prod_candidate =
            is_prod(sp) &&
            (!use_wmma || d != 64 ||
             flashDecodeTallTileUseful(bkv, tune_len, sp));
        r.rel_l2 = err;
        r.impl_wmma = use_wmma ? 1 : 0;
        r.splits = sp;
        r.bkv = use_wmma ? bkv : -1;
        out.push_back(r);
        if (print_realtime) {
          // Real-time print.
          double best_so_far = out[0].ms;
          for (const auto &x : out) best_so_far = std::min(best_so_far, x.ms);
          fprintf(stderr, "   %-18s %9.5f ms%s\n", r.cfg.c_str(), r.ms,
                  r.ms <= best_so_far + 1e-9 ? "  <-- best so far" : "");
          fflush(stderr);
        }
      }
    }
  }

  if (online) {
    SWEEP_CHECK(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    // The production tuner sees one cache, because that is what it gets in
    // production too: it tunes on the layer it was called for.
    const FlashDecodeCfg cfg = tuneFlashDecodeConfig(
        nullptr, d, hpg, tune_len, kFlashDecodeMaxSplits, "sweep_online",
        [&](const FlashDecodeCfg &candidate) {
          launchFlashDecodeConfig<KvDtype::kF16>(
              candidate, nullptr, dQ, dK[0], dV[0], nullptr, nullptr, dO,
              dPart, B, H, G, d, hpg, max_seq, scale, dSeq, window, sink,
              smooth);
        });
    SWEEP_CHECK(hipDeviceSynchronize());
    online->decide_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
    char buf[64];
    if (cfg.use_wmma && d == 64)
      snprintf(buf, sizeof(buf), "wmma_BKV%d_SPLITS%d", cfg.bkv, cfg.splits);
    else
      snprintf(buf, sizeof(buf), "%s_SPLITS%d",
               cfg.use_wmma ? "wmma" : "scalar", cfg.splits);
    online->cfg = buf;
    // PR #675's production tuner adapts the iteration count to the candidate's
    // duration, so unlike the former fixed 2+10 loop it has no stable launch
    // count worth reporting.
    online->launches = 0;
    online->measured = true;
  }

  for (int i = 0; i < copies; ++i) {
    hipFree(dK[i]);
    hipFree(dV[i]);
  }
  hipFree(dQ);
  hipFree(dO);
  hipFree(dSink);
  hipFree(dPart);
  hipFree(dSeq);
  return out;
}

// ---- shape list -------------------------------------------------------------

std::vector<std::string> splitCsv(const std::string &line) {
  std::vector<std::string> f;
  std::string cur;
  bool in_q = false;
  for (char c : line) {
    if (c == '"') {
      in_q = !in_q;
      continue;
    }
    if (c == ',' && !in_q) {
      f.push_back(cur);
      cur.clear();
      continue;
    }
    cur.push_back(c);
  }
  f.push_back(cur);
  return f;
}

// Columns are looked up by name, not by position. The shape files this reads
// have grown columns over time (the LUT grid adds max_seq and grid_role in the
// middle of the row), and a positional reader answers that by silently loading
// max_seq as the window rather than by failing.
std::vector<Shape> loadShapes(const std::string &path) {
  std::ifstream in(path);
  if (!in) {
    fprintf(stderr, "cannot open shape file: %s\n", path.c_str());
    std::exit(1);
  }
  std::string line;
  if (!std::getline(in, line)) {
    fprintf(stderr, "empty shape file: %s\n", path.c_str());
    std::exit(1);
  }
  if (!line.empty() && line.back() == '\r')
    line.pop_back();
  const std::vector<std::string> header = splitCsv(line);
  auto column = [&](const char *name) -> int {
    for (size_t i = 0; i < header.size(); ++i)
      if (header[i] == name)
        return (int)i;
    return -1;
  };
  for (const char *required :
       {"id", "phase", "B", "H", "G", "d", "sq", "skv", "window"}) {
    if (column(required) < 0) {
      fprintf(stderr, "shape file %s is missing required column '%s'\n",
              path.c_str(), required);
      std::exit(1);
    }
  }
  // Looked up by name at the point of use, so adding a column here cannot shift
  // the meaning of another one.
  auto text = [&](const std::vector<std::string> &f, const char *name,
                  const char *dflt) -> std::string {
    const int idx = column(name);
    if (idx < 0 || idx >= (int)f.size() || f[idx].empty())
      return dflt;
    return f[idx];
  };
  auto number = [&](const std::vector<std::string> &f, const char *name,
                    int dflt) -> int {
    const int idx = column(name);
    if (idx < 0 || idx >= (int)f.size() || f[idx].empty())
      return dflt;
    return std::atoi(f[idx].c_str());
  };

  std::vector<Shape> out;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    if (line.empty())
      continue;
    const std::vector<std::string> f = splitCsv(line);
    Shape s;
    s.id = text(f, "id", "");
    if (s.id.empty() || s.id[0] == '#')
      continue;
    s.group = text(f, "group", "");
    s.phase = text(f, "phase", "");
    s.B = number(f, "B", 1);
    s.H = number(f, "H", 0);
    s.G = number(f, "G", 0);
    s.d = number(f, "d", 0);
    s.sq = number(f, "sq", 0);
    s.skv = number(f, "skv", 0);
    s.window = number(f, "window", -1);
    s.sink = number(f, "sink", 0);
    s.max_seq = number(f, "max_seq", 0);
    if (s.max_seq < s.skv)
      s.max_seq = s.skv;
    // src_rows on the model shape files, grid_role on the LUT grid: both say
    // where the row came from, which is all this harness echoes back.
    s.src_rows = text(f, "src_rows", text(f, "grid_role", "").c_str());
    s.note = text(f, "note", "");
    out.push_back(s);
  }
  return out;
}

} // namespace

int main(int argc, char **argv) {
  std::string shape_file;
  std::string only;
  std::string csv_path;
  std::string best_path;
  bool verify = true;
  double target_ms = 1000.0;
  int rounds = 3;
  int decode_split_cap = kFlashDecodeMaxSplits;
  double warmup_s = 4.0;
  bool tune_compare = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--shapes")
      shape_file = next();
    else if (a == "--only")
      only = next();
    else if (a == "--csv")
      csv_path = next();
    else if (a == "--best-csv")
      best_path = next();
    else if (a == "--no-verify")
      verify = false;
    else if (a == "--target-ms")
      target_ms = atof(next());
    else if (a == "--rounds")
      rounds = atoi(next());
    else if (a == "--decode-split-cap")
      decode_split_cap = atoi(next());
    else if (a == "--warmup-s")
      warmup_s = atof(next());
    else if (a == "--tune-compare")
      tune_compare = true;
    else {
      fprintf(
          stderr,
          "usage: %s --shapes <csv> [--only <id-prefix>] [--csv <all>]\n"
          "          [--best-csv <winners>] [--no-verify] [--target-ms N]\n"
          "          [--rounds N] [--decode-split-cap N] [--warmup-s N]\n"
          "\n"
          "  --tune-compare        also call the production online tuner per\n"
          "                        shape and record what it chose and how "
          "long\n"
          "                        deciding took, for the LUT-vs-autotune "
          "table.\n"
          "                        Roughly doubles the run time.\n"
          "  --warmup-s N          seconds of untimed work before the first\n"
          "                        measurement, so a cold clock cannot land "
          "on\n"
          "                        whichever shapes the file happens to list\n"
          "                        first. Default 4; 0 disables.\n"
          "  --decode-split-cap N  explore decode split counts above the\n"
          "     production ceiling (%d). Rows above it are written with\n"
          "     prod_candidate=0 and never win.\n",
          argv[0], kFlashDecodeMaxSplits);
      return 2;
    }
  }
  if (shape_file.empty()) {
    fprintf(stderr, "--shapes <csv> is required\n");
    return 2;
  }

  int dev = 0;
  SWEEP_CHECK(hipGetDevice(&dev));
  hipDeviceProp_t prop;
  SWEEP_CHECK(hipGetDeviceProperties(&prop, dev));
  fprintf(
      stderr, "Device: %s (%s, %d CUs)  target=%.0f ms/candidate rounds=%d\n\n",
      prop.name, prop.gcnArchName, prop.multiProcessorCount, target_ms, rounds);

  const std::vector<Shape> shapes = loadShapes(shape_file);

  FILE *csv = stdout;
  if (!csv_path.empty()) {
    csv = fopen(csv_path.c_str(), "w");
    if (!csv) {
      fprintf(stderr, "cannot write %s\n", csv_path.c_str());
      return 1;
    }
  }
  // How this reading was taken, on every row, and it differs by phase: decode
  // times each dispatch on its own over a rotating KV cache, prefill still divides
  // a loop by its iteration count, which is fine for a 10 ms kernel and was not
  // fine for a 20 us one. A measurement is only comparable to another taken the
  // same way -- the same decode shapes read up to 20x faster under the loop timer
  // -- so tools/measurement_store.py keys on this and never pools across values.
  char harness_decode[64], harness_prefill[64];
  snprintf(harness_decode, sizeof(harness_decode), "%s-t%g-r%d",
           kDecodeHarnessVersion, target_ms, rounds);
  snprintf(harness_prefill, sizeof(harness_prefill), "%s-t%g-r%d",
           kPrefillHarnessVersion, target_ms, rounds);

  fprintf(csv, "id,group,phase,B,H,G,HpG,d,sq,skv,window,sink,kernel,config,ms,"
               "rel_ms_vs_best,is_best,prod_candidate,rel_l2_vs_first,"
               "best_config,src_rows,harness,kernel_ver\n");

  FILE *best_csv = nullptr;
  if (!best_path.empty()) {
    best_csv = fopen(best_path.c_str(), "w");
    if (!best_csv) {
      fprintf(stderr, "cannot write %s\n", best_path.c_str());
      return 1;
    }
    fprintf(
        best_csv,
        "id,group,phase,B,H,G,HpG,d,sq,skv,window,sink,kernel,"
        "best_config,cfg_impl,cfg_splits,cfg_MT,cfg_BKV,cfg_NW,cfg_ND,"
        "best_ms,worst_ms,best_vs_worst_x,n_candidates,"
        "sweep_wall_s,autotune_prod_ms,autotune_prod_launches,"
        "autotune_prod_vs_one_call_x,"
        // Present only with --tune-compare: what the production tuner actually
        // chose, what that choice measures at, and what deciding cost.
        "online_config,online_ms,online_vs_best_x,online_decide_ms,"
        "online_decide_launches,"
        "src_rows,note\n");
  }

  // Bring the GPU to a steady clock before anything is timed. Without this the
  // first shapes of a run read slow, and because a shape file groups shapes by
  // role, the entire boundary set of a grid can land inside that window:
  // measured on gfx1151, the first ~27 shapes of a cold prefill run came out up
  // to 13x slow and flipped 11 of 93 winners, while the shapes measured later
  // reproduced within 8%. A few seconds of the real kernels is enough; the
  // results are discarded.
  if (warmup_s > 0.0 && !shapes.empty()) {
    const Shape *seed = nullptr;
    for (const Shape &s : shapes)
      if (only.empty() || s.id.compare(0, only.size(), only) == 0) {
        seed = &s;
        break;
      }
    if (seed) {
      fprintf(stderr, "warming up for %.1f s on %s ...\n", warmup_s,
              seed->id.c_str());
      const auto t_warm = std::chrono::steady_clock::now();
      do {
        if (seed->phase == "prefill")
          (void)sweepPrefill(*seed, /*verify=*/false, /*target_ms=*/5.0, 1,
                             /*online=*/nullptr, /*print_realtime=*/false);
        else
          (void)sweepDecode(*seed, /*verify=*/false, /*target_ms=*/5.0, 1,
                            decode_split_cap, /*online=*/nullptr,
                            /*print_realtime=*/false);
      } while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             t_warm)
                   .count() < warmup_s);
    }
  }

  for (const Shape &s : shapes) {
    if (!only.empty() && s.id.compare(0, only.size(), only) != 0)
      continue;
    const int hpg = s.H / s.G;
    const bool prefill = (s.phase == "prefill");
    const char *kernel = prefill ? (s.d == 64    ? "prefill_v5"
                                    : s.d == 128 ? "prefill_v7"
                                                 : "prefill_v8")
                                 : "flash_decode";

    fprintf(stderr,
            "[%s] %s %s  B%d H%d G%d(HpG%d) d%d sq=%d skv=%d win=%d sink=%d\n",
            s.id.c_str(), s.group.c_str(), s.phase.c_str(), s.B, s.H, s.G, hpg,
            s.d, s.sq, s.skv, s.window, s.sink);

    const auto t0 = std::chrono::steady_clock::now();
    OnlineResult online;
    std::vector<Result> res =
        prefill ? sweepPrefill(s, verify, target_ms, rounds,
                               tune_compare ? &online : nullptr)
                : sweepDecode(s, verify, target_ms, rounds, decode_split_cap,
                              tune_compare ? &online : nullptr);
    const double sweep_wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
            .count();
    if (res.empty()) {
      fprintf(stderr, "   (no candidates)\n\n");
      continue;
    }

    // Winner = fastest config the production tuner is allowed to pick.
    int best = -1;
    double worst_ms = 0.0;
    for (size_t i = 0; i < res.size(); ++i) {
      if (res[i].prod_candidate && (best < 0 || res[i].ms < res[best].ms))
        best = (int)i;
      if (res[i].prod_candidate)
        worst_ms = std::max(worst_ms, res[i].ms);
    }
    if (best < 0)
      best = 0;
    const Result &win = res[best];

    for (size_t i = 0; i < res.size(); ++i) {
      const Result &r = res[i];
      fprintf(csv,
              "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s,%.6f,%.3f,%d,%d,%.3e,%"
              "s,%s,%s,%s\n",
              s.id.c_str(), s.group.c_str(), s.phase.c_str(), s.B, s.H, s.G,
              hpg, s.d, s.sq, s.skv, s.window, s.sink, kernel, r.cfg.c_str(),
              r.ms, r.ms / win.ms, (int)i == best ? 1 : 0,
              r.prod_candidate ? 1 : 0, r.rel_l2, win.cfg.c_str(),
              s.src_rows.c_str(),
              s.phase == "decode" ? harness_decode : harness_prefill,
              kernelVersion(kernel));
      fprintf(stderr, "   %-18s %9.5f ms  x%.2f%s%s\n", r.cfg.c_str(), r.ms,
              r.ms / win.ms, (int)i == best ? "   <== BEST" : "",
              r.prod_candidate ? "" : "   (outside prod candidates)");
    }
    fflush(csv);

    const TuneCost tc =
        prefill ? prodPrefillTuneCost(res) : prodDecodeTuneCost(res);
    fprintf(stderr,
            "   best=%s  %.5f ms (%.2fx vs worst)   sweep=%.1f s   "
            "prod-autotune=%.2f ms / %d launches (= %.0f normal calls)\n",
            win.cfg.c_str(), win.ms, worst_ms / win.ms, sweep_wall_s, tc.ms,
            tc.launches, tc.ms / win.ms);
    if (online.measured) {
      double online_ms = 0.0;
      for (const Result &r : res)
        if (r.cfg == online.cfg) {
          online_ms = r.ms;
          break;
        }
      fprintf(
          stderr,
          "   online tuner chose %s (%.5f ms, %+.1f%% vs best), deciding took "
          "%.1f ms / %d launches\n",
          online.cfg.c_str(), online_ms,
          online_ms > 0.0 ? (online_ms / win.ms - 1.0) * 100.0 : 0.0,
          online.decide_ms, online.launches);
    }
    fprintf(stderr, "\n");

    if (best_csv) {
      auto opt = [](int v) {
        return v < 0 ? std::string("") : std::to_string(v);
      };
      // The measured time of whatever the online tuner chose, looked up in this
      // shape's own sweep so both strategies are scored on one set of timings.
      double online_ms = 0.0;
      if (online.measured)
        for (const Result &r : res)
          if (r.cfg == online.cfg) {
            online_ms = r.ms;
            break;
          }
      fprintf(best_csv,
              "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,"
              "%.6f,%.6f,%.2f,%d,%.1f,%.2f,%d,%.0f,%s,%.6f,%.4f,%.2f,%d,"
              "%s,\"%s\"\n",
              s.id.c_str(), s.group.c_str(), s.phase.c_str(), s.B, s.H, s.G,
              hpg, s.d, s.sq, s.skv, s.window, s.sink, kernel, win.cfg.c_str(),
              win.impl_wmma < 0 ? "" : (win.impl_wmma ? "wmma" : "scalar"),
              opt(win.splits).c_str(), opt(win.mt).c_str(),
              opt(win.bkv).c_str(), opt(win.nw).c_str(), opt(win.nd).c_str(),
              win.ms, worst_ms, worst_ms / win.ms, (int)res.size(),
              sweep_wall_s, tc.ms, tc.launches, tc.ms / win.ms,
              online.measured ? online.cfg.c_str() : "", online_ms,
              online_ms > 0.0 ? online_ms / win.ms : 0.0, online.decide_ms,
              online.launches, s.src_rows.c_str(), s.note.c_str());
      fflush(best_csv);
    }
  }

  if (csv != stdout)
    fclose(csv);
  if (best_csv)
    fclose(best_csv);
  return 0;
}
