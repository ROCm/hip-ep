// ============================================================
// GQA autotune sweep: enumerate EVERY autotune candidate for a list of real
// model shapes and report the winner per shape.
//
// The production launchers (hip_gqa_flash_prefill_v2 / hip_gqa_flash_decode_v2)
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
// The candidate sets are the ones the production tuners use (prefillV*Candidates
// / flashDecodeCandidateSplits), so a winner reported here is a config the
// runtime can actually land on. Decode additionally sweeps split counts outside
// the production guard, flagged `prod=0`, to show what the guard costs.
//
// Because this file owns the kernel TU, do NOT link gqa_kernel.obj with it.
//
// Output: one CSV row per (shape, candidate) on stdout, plus a human-readable
// per-shape summary on stderr.
// ============================================================

#include "../../../../hip/gqa_kernel.hip"
// The cost model is a separate translation unit in the real build; this harness
// is a single TU, so pull the implementation in directly.
#include "../../../../hip/autotune/gqa/gqa_cost_model.cpp"

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
  std::string phase;  // "prefill" | "decode"
  int B, H, G, d, sq, skv, window;
  int sink;           // 1 = head_sink tensor + smooth_softmax (gpt-oss layers)
  std::string src_rows;
  std::string note;
};

struct Result {
  std::string cfg;        // canonical config string, e.g. "MT2_BKV32"
  double ms;
  bool prod_candidate;    // in the production tuner's candidate set
  double rel_l2;          // vs. the first candidate's output (-1 = not checked)
  // The same config split into its individual knobs, so the summary table can
  // give each tunable its own column instead of a string a reader has to parse.
  // -1 = the knob does not exist for this kernel.
  int impl_wmma = -1, splits = -1, mt = -1, bkv = -1, nw = -1, nd = -1;
};

// Score every measured candidate with the analytic model in
// include/gqa_dispatch_model.h and return the index of the one it would pick.
// Running this alongside the sweep is what keeps the model honest: the same run
// that measures the true winner also records what the formula would have chosen.
int modelPick(const Shape& s, const std::vector<Result>& res, int cu_count) {
  hip_gqa_shape_t sh;
  sh.batch = s.B;
  sh.num_heads = s.H;
  sh.kv_heads = s.G;
  sh.head_dim = s.d;
  sh.q_len = s.sq;
  sh.kv_len = s.skv;
  sh.window = s.window;
  sh.cu_count = cu_count;

  const bool prefill = (s.phase == "prefill");
  hip_gqa_path_t path = HIP_GQA_PATH_DECODE;
  if (prefill)
    path = (s.d == 64)    ? HIP_GQA_PATH_PREFILL_V5
         : (s.d == 128)   ? HIP_GQA_PATH_PREFILL_V7
                          : HIP_GQA_PATH_PREFILL_V8;

  int best = -1;
  double best_score = 0.0;
  for (size_t i = 0; i < res.size(); ++i) {
    // The model is a dispatch-time predictor, so it only ranks configs the
    // production tuner could actually land on.
    if (!res[i].prod_candidate) continue;
    hip_gqa_config_t c;
    c.path = path;
    c.m_tiles = res[i].mt > 0 ? res[i].mt : 0;
    c.bkv = res[i].bkv > 0 ? res[i].bkv : 0;
    c.num_waves = res[i].nw > 0 ? res[i].nw : (res[i].nd > 0 ? res[i].nd : 0);
    c.use_wmma = res[i].impl_wmma > 0 ? 1 : 0;
    c.splits = res[i].splits > 0 ? res[i].splits : 0;
    const double score = hip_gqa_config_score(&sh, &c);
    if (score > best_score) { best_score = score; best = (int)i; }
  }
  return best;
}

// What the *production* tuner would spend on its first call for this shape.
// The runtime pays this once per cache key before any real work is produced, so
// it is the number that matters when judging whether a tuner is worth its cost.
struct TuneCost {
  double ms;    // wall time of the tuning launches
  int launches; // kernel launches issued while tuning
};

// tunePrefillV5/V7/V8Cfg: kWarmup=30 launches cycling the candidate list, then
// kRounds=4 x (1 untimed + kIters=30 timed) launches per candidate.
TuneCost prodPrefillTuneCost(const std::vector<Result>& res) {
  TuneCost c{0.0, 0};
  const int nc = (int)res.size();
  if (nc == 0) return c;
  for (int w = 0; w < 30; ++w) { c.ms += res[w % nc].ms; ++c.launches; }
  for (const Result& r : res) { c.ms += 4 * 31 * r.ms; c.launches += 4 * 31; }
  return c;
}

// tuneFlashDecode: WARMUP=2 + ITERS=10 launches per candidate, and only the
// production candidate set is ever timed.
TuneCost prodDecodeTuneCost(const std::vector<Result>& res) {
  TuneCost c{0.0, 0};
  for (const Result& r : res) {
    if (!r.prod_candidate) continue;
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
double timeConfig(F&& launch, double target_ms, int min_iters, int max_iters,
                  int rounds) {
  (void)hipGetLastError();  // don't attribute an earlier config's error here
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
  if (probe_ms > 1e-6f) iters = (int)(target_ms / probe_ms);
  iters = std::max(min_iters, std::min(max_iters, iters));
  // A call that already runs for tens of ms is far above the noise floor that
  // repeated rounds exist to suppress, and the 32k-prompt shapes cost minutes
  // if every round is honoured.
  // A median needs an odd sample count to be a real sample, and three rounds is
  // the cheapest count that survives one bad reading.
  rounds = std::max(3, rounds | 1);
  if (probe_ms > 50.0f) rounds = 3;

  std::vector<double> samples;
  samples.reserve(rounds);
  for (int r = 0; r < rounds; ++r) {
    launch();  // warm the caches for this round
    SWEEP_CHECK(hipEventRecord(e0, nullptr));
    for (int i = 0; i < iters; ++i) launch();
    SWEEP_CHECK(hipEventRecord(e1, nullptr));
    SWEEP_CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    SWEEP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    samples.push_back((double)ms / iters);
  }
  SWEEP_CHECK(hipEventDestroy(e0));
  SWEEP_CHECK(hipEventDestroy(e1));

  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

double relL2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double diff = (double)a[i] - (double)b[i];
    num += diff * diff;
    den += (double)b[i] * (double)b[i];
  }
  return std::sqrt(num / (den + 1e-12));
}

std::vector<float> fetchHalf(const __half* dev, size_t n) {
  std::vector<__half> tmp(n);
  SWEEP_CHECK(hipMemcpy(tmp.data(), dev, n * sizeof(__half), hipMemcpyDeviceToHost));
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = __half2float(tmp[i]);
  return out;
}

// ---- prefill sweep ----------------------------------------------------------

std::vector<Result> sweepPrefill(const Shape& s, bool verify, double target_ms,
                                 int rounds) {
  const int B = s.B, H = s.H, G = s.G, d = s.d, sq = s.sq, skv = s.skv;
  const int max_seq = skv;
  const int past_len = skv - sq;
  const float scale = 1.0f / std::sqrt((float)d);

  const size_t qn = (size_t)B * sq * H * d;
  const size_t kn = (size_t)B * G * max_seq * d;

  std::mt19937 rng(1234u + (unsigned)sq + (unsigned)d);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<__half> hQ(qn), hK(kn), hV(kn), hSink(H);
  for (auto& x : hQ) x = __float2half(dist(rng));
  for (auto& x : hK) x = __float2half(dist(rng));
  for (auto& x : hV) x = __float2half(dist(rng));
  for (int h = 0; h < H; ++h)
    hSink[h] = __float2half(-2.0f + 4.0f * (float)h / (float)H);

  __half *dQ, *dK, *dV, *dO, *dSink;
  SWEEP_CHECK(hipMalloc(&dQ, qn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dK, kn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dV, kn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dO, qn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dSink, (size_t)H * sizeof(__half)));
  SWEEP_CHECK(hipMemcpy(dQ, hQ.data(), qn * sizeof(__half), hipMemcpyHostToDevice));
  SWEEP_CHECK(hipMemcpy(dK, hK.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
  SWEEP_CHECK(hipMemcpy(dV, hV.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
  SWEEP_CHECK(hipMemcpy(dSink, hSink.data(), (size_t)H * sizeof(__half),
                        hipMemcpyHostToDevice));

  const _Float16* Q = reinterpret_cast<const _Float16*>(dQ);
  const _Float16* K = reinterpret_cast<const _Float16*>(dK);
  const _Float16* V = reinterpret_cast<const _Float16*>(dV);
  _Float16* O = reinterpret_cast<_Float16*>(dO);
  // gqa.cpp sends the sink tensor and smooth_softmax=1 together on gpt-oss.
  const _Float16* sink = s.sink ? reinterpret_cast<const _Float16*>(dSink) : nullptr;
  const int smooth = s.sink ? 1 : 0;
  const int window = s.window > 0 ? s.window : 0;

  // Verifying every candidate needs a host copy of O per candidate; skip it on
  // the multi-hundred-MB prompts where it dominates the sweep's wall time.
  const bool do_verify = verify && qn <= (size_t)8 << 20;
  std::vector<float> ref;

  std::vector<Result> out;
  auto measure = [&](Result r, auto&& launch) {
    r.ms = timeConfig(launch, target_ms, 3, 200, rounds);
    r.prod_candidate = true;
    r.rel_l2 = -1.0;
    if (do_verify) {
      launch();
      SWEEP_CHECK(hipDeviceSynchronize());
      std::vector<float> cur = fetchHalf(dO, qn);
      if (ref.empty()) ref = cur; else r.rel_l2 = relL2(cur, ref);
    }
    out.push_back(r);
  };

  if (d == 64) {
    PrefillV5Cfg cands[8];
    const int nc = prefillV5Candidates(d, cands);
    for (int i = 0; i < nc; ++i) {
      const PrefillV5Cfg c = cands[i];
      char cfg[64];
      snprintf(cfg, sizeof(cfg), "MT%d_BKV%d", c.m_tiles, c.bkv);
      Result r; r.cfg = cfg; r.mt = c.m_tiles; r.bkv = c.bkv;
      measure(r, [&]() {
        dispatchPrefillV5(c.m_tiles, c.bkv, d, nullptr, Q, K, V, O, B, H, G, sq,
                          skv, max_seq, past_len, scale, sink, H, smooth, window);
      });
    }
  } else if (d == 128) {
    PrefillV7Cfg cands[12];
    const int nc = prefillV7Candidates(d, cands);
    for (int i = 0; i < nc; ++i) {
      const PrefillV7Cfg c = cands[i];
      char cfg[64];
      snprintf(cfg, sizeof(cfg), "NW%d_BKV%d_MT%d", c.nw, c.bkv, c.mt);
      Result r; r.cfg = cfg; r.nw = c.nw; r.bkv = c.bkv; r.mt = c.mt;
      measure(r, [&]() {
        dispatchPrefillV7(c.nw, c.bkv, c.mt, d, nullptr, Q, K, V, O, B, H, G, sq,
                          skv, max_seq, past_len, scale);
      });
    }
  } else {
    PrefillV8Cfg cands[8];
    const int nc = prefillV8Candidates(d, cands);
    for (int i = 0; i < nc; ++i) {
      const PrefillV8Cfg c = cands[i];
      char cfg[64];
      snprintf(cfg, sizeof(cfg), "ND%d_MT%d_BKV%d", c.nd, c.mt, c.bkv);
      Result r; r.cfg = cfg; r.nd = c.nd; r.mt = c.mt; r.bkv = c.bkv;
      measure(r, [&]() {
        dispatchPrefillV8(c.nd, c.mt, c.bkv, d, nullptr, Q, K, V, O, B, H, G, sq,
                          skv, max_seq, past_len, scale);
      });
    }
  }

  hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO); hipFree(dSink);
  return out;
}

// ---- decode sweep -----------------------------------------------------------

std::vector<Result> sweepDecode(const Shape& s, bool verify, double target_ms,
                                int rounds, int split_cap) {
  const int B = s.B, H = s.H, G = s.G, d = s.d, skv = s.skv;
  const int hpg = H / G;
  const int max_seq = skv;
  // The decode kernels take the split count as a runtime argument, so a cap
  // above kFlashDecodeMaxSplits only needs a bigger partials workspace. Allowing
  // it here answers whether the production ceiling is leaving anything on the
  // table at 32k context, where 64 splits is the only candidate left.
  const int cap = split_cap;
  const float scale = 1.0f / std::sqrt((float)d);
  const int window = s.window > 0 ? s.window : -1;
  // Mirrors hip_gqa_flash_decode_v2: a sliding layer tunes over the window, not
  // the full context, so its candidate clamp uses the window length.
  const int tune_len = (window > 0 && skv > window) ? window : skv;

  const size_t qn = (size_t)B * H * d;
  const size_t kn = (size_t)B * G * max_seq * d;
  const size_t pn = (size_t)B * H * cap * (d + 2);

  std::mt19937 rng(4321u + (unsigned)skv + (unsigned)d);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<__half> hQ(qn), hK(kn), hV(kn), hSink(H);
  for (auto& x : hQ) x = __float2half(dist(rng));
  for (auto& x : hK) x = __float2half(dist(rng));
  for (auto& x : hV) x = __float2half(dist(rng));
  for (int h = 0; h < H; ++h)
    hSink[h] = __float2half(-2.0f + 4.0f * (float)h / (float)H);
  std::vector<int> hSeq(B, skv - 1);  // kernels read seqlens_k + 1

  __half *dQ, *dK, *dV, *dO, *dSink;
  float* dPart;
  int* dSeq;
  SWEEP_CHECK(hipMalloc(&dQ, qn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dK, kn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dV, kn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dO, qn * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dSink, (size_t)H * sizeof(__half)));
  SWEEP_CHECK(hipMalloc(&dPart, pn * sizeof(float)));
  SWEEP_CHECK(hipMalloc(&dSeq, (size_t)B * sizeof(int)));
  SWEEP_CHECK(hipMemcpy(dQ, hQ.data(), qn * sizeof(__half), hipMemcpyHostToDevice));
  SWEEP_CHECK(hipMemcpy(dK, hK.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
  SWEEP_CHECK(hipMemcpy(dV, hV.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
  SWEEP_CHECK(hipMemcpy(dSink, hSink.data(), (size_t)H * sizeof(__half),
                        hipMemcpyHostToDevice));
  SWEEP_CHECK(hipMemcpy(dSeq, hSeq.data(), (size_t)B * sizeof(int),
                        hipMemcpyHostToDevice));

  const __half* sink = s.sink ? dSink : nullptr;
  const int smooth = s.sink ? 1 : 0;

  // Production candidate set, used to flag which rows the runtime can pick. It
  // is always computed at the production ceiling, even when the sweep is
  // exploring above it.
  int prod_splits[7]; int n_prod = 0;
  flashDecodeCandidateSplits(tune_len, kFlashDecodeMaxSplits, prod_splits, n_prod);
  auto is_prod = [&](int sp) {
    if (sp > kFlashDecodeMaxSplits) return false;
    for (int i = 0; i < n_prod; ++i) if (prod_splits[i] == sp) return true;
    return false;
  };

  // Sweep the full ladder, not just the production subset: the guard in
  // flashDecodeCandidateSplits drops small splits at long context, and the
  // dropped timings are what justify the guard.
  static const int kAllSplits[] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 192, 256};
  const int max_useful = std::max(1, (tune_len + 15) / 16);

  const bool wmma_ok = flash_decode_wmma_supported(d, hpg);
  std::vector<float> ref;
  std::vector<Result> out;

  for (int impl = 0; impl < 2; ++impl) {
    const bool use_wmma = (impl == 1);
    if (use_wmma && !wmma_ok) continue;
    int last = -1;
    for (int sp_base : kAllSplits) {
      const int sp = std::min(sp_base, std::min(cap, max_useful));
      if (sp == last) continue;  // clamping collapsed this rung
      last = sp;
      const FlashDecodeCfg cfg{use_wmma, sp};
      auto launch = [&]() {
        launchFlashDecodeConfig(cfg, nullptr, dQ, dK, dV, dO, dPart, B, H, G, d,
                                hpg, max_seq, scale, dSeq, window, sink, smooth);
      };
      const double ms = timeConfig(launch, target_ms, 5, 500, rounds);
      double err = -1.0;
      if (verify) {
        launch();
        SWEEP_CHECK(hipDeviceSynchronize());
        std::vector<float> cur = fetchHalf(dO, qn);
        if (ref.empty()) ref = cur; else err = relL2(cur, ref);
      }
      char name[64];
      snprintf(name, sizeof(name), "%s_SPLITS%d", use_wmma ? "wmma" : "scalar", sp);
      Result r;
      r.cfg = name; r.ms = ms; r.prod_candidate = is_prod(sp); r.rel_l2 = err;
      r.impl_wmma = use_wmma ? 1 : 0; r.splits = sp;
      out.push_back(r);
    }
  }

  hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);
  hipFree(dSink); hipFree(dPart); hipFree(dSeq);
  return out;
}

// ---- shape list -------------------------------------------------------------

std::vector<std::string> splitCsv(const std::string& line) {
  std::vector<std::string> f;
  std::string cur;
  bool in_q = false;
  for (char c : line) {
    if (c == '"') { in_q = !in_q; continue; }
    if (c == ',' && !in_q) { f.push_back(cur); cur.clear(); continue; }
    cur.push_back(c);
  }
  f.push_back(cur);
  return f;
}

std::vector<Shape> loadShapes(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    fprintf(stderr, "cannot open shape file: %s\n", path.c_str());
    std::exit(1);
  }
  std::vector<Shape> out;
  std::string line;
  bool header = true;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty()) continue;
    if (header) { header = false; continue; }
    const std::vector<std::string> f = splitCsv(line);
    if (f.size() < 12) continue;
    Shape s;
    s.id = f[0]; s.group = f[1]; s.phase = f[2];
    s.B = std::atoi(f[3].c_str());
    s.H = std::atoi(f[4].c_str());
    s.G = std::atoi(f[5].c_str());
    s.d = std::atoi(f[6].c_str());
    s.sq = std::atoi(f[7].c_str());
    s.skv = std::atoi(f[8].c_str());
    s.window = std::atoi(f[9].c_str());
    s.sink = std::atoi(f[10].c_str());
    s.src_rows = f[11];
    s.note = f.size() > 12 ? f[12] : "";
    out.push_back(s);
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  std::string shape_file;
  std::string only;
  std::string csv_path;
  std::string best_path;
  bool verify = true;
  double target_ms = 40.0;
  int rounds = 3;
  int decode_split_cap = kFlashDecodeMaxSplits;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
    if (a == "--shapes") shape_file = next();
    else if (a == "--only") only = next();
    else if (a == "--csv") csv_path = next();
    else if (a == "--best-csv") best_path = next();
    else if (a == "--no-verify") verify = false;
    else if (a == "--target-ms") target_ms = atof(next());
    else if (a == "--rounds") rounds = atoi(next());
    else if (a == "--decode-split-cap") decode_split_cap = atoi(next());
    else {
      fprintf(stderr,
              "usage: %s --shapes <csv> [--only <id-prefix>] [--csv <all>]\n"
              "          [--best-csv <winners>] [--no-verify] [--target-ms N]\n"
              "          [--rounds N] [--decode-split-cap N]\n"
              "\n"
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
  fprintf(stderr, "Device: %s (%s, %d CUs)  target=%.0f ms/candidate rounds=%d\n\n",
          prop.name, prop.gcnArchName, prop.multiProcessorCount, target_ms, rounds);

  const std::vector<Shape> shapes = loadShapes(shape_file);

  FILE* csv = stdout;
  if (!csv_path.empty()) {
    csv = fopen(csv_path.c_str(), "w");
    if (!csv) { fprintf(stderr, "cannot write %s\n", csv_path.c_str()); return 1; }
  }
  fprintf(csv,
          "id,group,phase,B,H,G,HpG,d,sq,skv,window,sink,kernel,config,ms,"
          "rel_ms_vs_best,is_best,prod_candidate,rel_l2_vs_first,"
          "best_config,src_rows\n");

  FILE* best_csv = nullptr;
  if (!best_path.empty()) {
    best_csv = fopen(best_path.c_str(), "w");
    if (!best_csv) { fprintf(stderr, "cannot write %s\n", best_path.c_str()); return 1; }
    fprintf(best_csv,
            "id,group,phase,B,H,G,HpG,d,sq,skv,window,sink,kernel,"
            "best_config,cfg_impl,cfg_splits,cfg_MT,cfg_BKV,cfg_NW,cfg_ND,"
            "best_ms,worst_ms,best_vs_worst_x,n_candidates,"
            "sweep_wall_s,autotune_prod_ms,autotune_prod_launches,"
            "autotune_prod_vs_one_call_x,"
            "model_config,model_ms,model_vs_best_x,model_exact,"
            "src_rows,note\n");
  }

  for (const Shape& s : shapes) {
    if (!only.empty() && s.id.compare(0, only.size(), only) != 0) continue;
    const int hpg = s.H / s.G;
    const bool prefill = (s.phase == "prefill");
    const char* kernel = prefill ? (s.d == 64 ? "prefill_v5"
                                  : s.d == 128 ? "prefill_v7" : "prefill_v8")
                                 : "flash_decode";

    fprintf(stderr, "[%s] %s %s  B%d H%d G%d(HpG%d) d%d sq=%d skv=%d win=%d sink=%d\n",
            s.id.c_str(), s.group.c_str(), s.phase.c_str(), s.B, s.H, s.G, hpg,
            s.d, s.sq, s.skv, s.window, s.sink);

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<Result> res =
        prefill ? sweepPrefill(s, verify, target_ms, rounds)
                : sweepDecode(s, verify, target_ms, rounds, decode_split_cap);
    const double sweep_wall_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (res.empty()) { fprintf(stderr, "   (no candidates)\n\n"); continue; }

    // Winner = fastest config the production tuner is allowed to pick.
    int best = -1;
    double worst_ms = 0.0;
    for (size_t i = 0; i < res.size(); ++i) {
      if (res[i].prod_candidate && (best < 0 || res[i].ms < res[best].ms))
        best = (int)i;
      if (res[i].prod_candidate) worst_ms = std::max(worst_ms, res[i].ms);
    }
    if (best < 0) best = 0;
    const Result& win = res[best];

    for (size_t i = 0; i < res.size(); ++i) {
      const Result& r = res[i];
      fprintf(csv,
              "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s,%.6f,%.3f,%d,%d,%.3e,%s,%s\n",
              s.id.c_str(), s.group.c_str(), s.phase.c_str(), s.B, s.H, s.G, hpg,
              s.d, s.sq, s.skv, s.window, s.sink, kernel, r.cfg.c_str(), r.ms,
              r.ms / win.ms, (int)i == best ? 1 : 0,
              r.prod_candidate ? 1 : 0, r.rel_l2, win.cfg.c_str(),
              s.src_rows.c_str());
      fprintf(stderr, "   %-18s %9.5f ms  x%.2f%s%s\n", r.cfg.c_str(), r.ms,
              r.ms / win.ms, (int)i == best ? "   <== BEST" : "",
              r.prod_candidate ? "" : "   (outside prod candidates)");
    }
    fflush(csv);

    const TuneCost tc = prefill ? prodPrefillTuneCost(res) : prodDecodeTuneCost(res);
    const int mi = modelPick(s, res, prop.multiProcessorCount);
    const Result* mr = (mi >= 0) ? &res[mi] : nullptr;
    fprintf(stderr,
            "   best=%s  %.5f ms (%.2fx vs worst)   sweep=%.1f s   "
            "prod-autotune=%.2f ms / %d launches (= %.0f normal calls)\n",
            win.cfg.c_str(), win.ms, worst_ms / win.ms, sweep_wall_s, tc.ms,
            tc.launches, tc.ms / win.ms);
    if (mr)
      fprintf(stderr, "   model=%s  %.5f ms  (%+.1f%% vs best)%s\n\n",
              mr->cfg.c_str(), mr->ms, (mr->ms / win.ms - 1.0) * 100.0,
              mi == best ? "   exact" : "");
    else
      fprintf(stderr, "   model: no scorable candidate\n\n");

    if (best_csv) {
      auto opt = [](int v) { return v < 0 ? std::string("") : std::to_string(v); };
      fprintf(best_csv,
              "%s,%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%s,%s,%s,%s,%s,%s,%s,%s,"
              "%.6f,%.6f,%.2f,%d,%.1f,%.2f,%d,%.0f,%s,%.6f,%.3f,%d,%s,\"%s\"\n",
              s.id.c_str(), s.group.c_str(), s.phase.c_str(), s.B, s.H, s.G, hpg,
              s.d, s.sq, s.skv, s.window, s.sink, kernel, win.cfg.c_str(),
              win.impl_wmma < 0 ? "" : (win.impl_wmma ? "wmma" : "scalar"),
              opt(win.splits).c_str(), opt(win.mt).c_str(), opt(win.bkv).c_str(),
              opt(win.nw).c_str(), opt(win.nd).c_str(),
              win.ms, worst_ms, worst_ms / win.ms, (int)res.size(),
              sweep_wall_s, tc.ms, tc.launches, tc.ms / win.ms,
              mr ? mr->cfg.c_str() : "", mr ? mr->ms : 0.0,
              mr ? mr->ms / win.ms : 0.0, (mi == best) ? 1 : 0,
              s.src_rows.c_str(), s.note.c_str());
      fflush(best_csv);
    }
  }

  if (csv != stdout) fclose(csv);
  if (best_csv) fclose(best_csv);
  return 0;
}
