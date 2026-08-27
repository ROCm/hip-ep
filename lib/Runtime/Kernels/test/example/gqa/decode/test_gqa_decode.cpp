// ============================================================
// custom_kernels GQA flash *decode* (TPS) test + 3-column history benchmark.
//
// Verifies the SINGLE production decode kernel (hip_gqa_flash_decode) for
// correctness against a CPU fp32 reference (incl. seqlens_k / sliding-window /
// head-sink / smooth-softmax), then reports a three-column latency comparison.
//
// IMPORTANT: all three columns run on the CURRENT v2 kernel, toggled by env.
// They measure the value of v2's per-shape autotune RELATIVE TO fixed configs
// on the SAME kernel -- they are NOT the historical legacy code. (In
// particular fix-def8 forces the v2 kernel's WMMA path at 8 splits, whose
// absolute cost differs from the real legacy WMMA kernel; do not read it as
// "PR438".) Apples-to-apples on identical inputs:
//
//   [fix-scalar8] force scalar split kernel @ 8 splits
//                 (HIPDNN_GQA_DECODE_SCALAR=1, SPLITS=8)
//   [fix-def8]    force default impl (WMMA@d64 / scalar else) @ 8 splits
//                 (HIPDNN_GQA_DECODE_SPLITS=8)
//   [v2]          autotune (impl, split-count<=64) per shape + cache (no env)
//
// Coverage spans MHA (HpG==1) and GQA (HpG in {2,4,5,8,16}) x head_dim in
// {64,128,256} x context length in {512..32768}, plus gpt-oss-20b full/sliding
// (head_sink) and a smooth-softmax variant. Pass --md to emit a Markdown table.
//
// To benchmark one build against another, add --prod-only (times just the
// autotuned config, so the fixed-config runs cannot disturb the clock state
// around it) and --only <model> (one model per process, for the same reason).
//
// Self-contained: random inputs generated in-process, no data files.
// ============================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <string>

// Launcher under test (implemented in hip/gqa_kernel.hip).
// KV-cache dtype ABI (mirrors hip_kv_dtype_t in hip_custom_kernels.h).
enum { HIP_KV_DTYPE_FP16 = 0, HIP_KV_DTYPE_INT8 = 1 };

extern "C" int hip_gqa_flash_decode(
    void* stream,
    const void* Q, const void* Kcache, const void* Vcache,
    void* O,
    void* partials_workspace,
    int B, int H, int G, int d, int skv, int max_seq, int max_splits,
    float scale,
    const void* seqlens_k,
    int local_window_size,
    const void* head_sink,
    int use_smooth_softmax,
    int kv_dtype, const void* k_scale, const void* v_scale,
    // Additive attention mask; this test exercises the unmasked path, so it
    // passes NULL. Declared to match the entry point exactly -- extern "C" is
    // unmangled, so a short declaration here would link and then read garbage.
    const void* attn_bias,
    int attn_bias_batch, int attn_bias_heads, int attn_bias_kv_stride,
    // Ring cache; linear here (0). Only the mask reads a position rather than
    // a slot, so an unmasked decode cannot tell a ring from a linear cache --
    // test_gqa_decode_bias owns that coverage.
    int ring_base, int ring_cap);

// Legacy one-block-per-head fused decode (the ORIGINAL baseline that the
// OPTIMIZATION.md 10-20x figure was measured against). No window/sink/split-K.
extern "C" int hip_gqa_fused_decode(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int H, int G, int d, int skv, int max_seq,
    float scale, const void* seqlens_k);

// Partials workspace capacity in splits. Must be >= the launcher's autotune
// ceiling (kFlashDecodeMaxSplits=64 in gqa_kernel.hip) so the autotuned path
// can pick any split count without overflowing dPart.
static constexpr int MAX_SPLITS = 64;
// Production-baseline split count (what the EP locked before this change).
static constexpr int BASELINE_SPLITS = 8;

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),        \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

struct Case {
  const char* name;
  int B, H, G, D, max_seq, total;  // total = effective KV length (seqlens_k+1)
  int window;                       // <=0 = full attention
  int sink;                         // head_sink present
  int smooth;                       // use_smooth_softmax (no explicit sink)
};

// One collected result row for the Markdown perf report. The three latency
// columns are fixed-config vs autotune on the SAME kernel (see file header);
// under --prod-only the two fixed-config columns are left at zero:
//   ms_fix_scalar8 = [fix-scalar8] force scalar split kernel @ 8 splits
//   ms_fix_def8    = [fix-def8]    force default impl @ 8 splits
//   ms_v2          = [v2]          autotuned (impl, split-count)
struct Row {
  std::string name;
  int H, G, D, len, window, sink, smooth;
  double ms_fix_scalar8, ms_fix_def8, ms_v2;
  double l2_v2;
  bool ok;
};
static std::vector<Row> g_rows;
static bool g_do_fused = false;  // --fused: also time the oldest one-block/head decode
static bool g_md = false;        // --md: suppress per-case text, emit Markdown table
static bool g_prod_only = false; // --prod-only: time only the autotuned config
static std::string g_only;       // --only: run just the models matching this

// ---- CPU fp32 reference ---------------------------------------------------
static void cpu_reference(const Case& c,
                          const std::vector<float>& Q,    // [B,H,D]
                          const std::vector<float>& K,    // [B,G,max_seq,D]
                          const std::vector<float>& V,    // [B,G,max_seq,D]
                          const std::vector<int>& seqlens,// [B]
                          const std::vector<float>& sink, // [H] (natural units)
                          float scale,
                          std::vector<float>& O) {        // [B,H,D]
  const int B = c.B, H = c.H, G = c.G, D = c.D, max_seq = c.max_seq;
  const int hpg = H / G;
  O.assign((size_t)B * H * D, 0.0f);
  for (int b = 0; b < B; ++b) {
    int raw = seqlens[b] + 1;
    int eff = raw < 0 ? 0 : (raw > max_seq ? max_seq : raw);
    int kv_lo = (c.window > 0 && eff > c.window) ? (eff - c.window) : 0;
    for (int h = 0; h < H; ++h) {
      int g = h / hpg;
      const float* q = &Q[((size_t)b * H + h) * D];
      // max
      float m = -INFINITY;
      for (int kv = kv_lo; kv < eff; ++kv) {
        const float* k = &K[(((size_t)b * G + g) * max_seq + kv) * D];
        float dot = 0.0f;
        for (int e = 0; e < D; ++e) dot += q[e] * k[e];
        float s = dot * scale;
        if (s > m) m = s;
      }
      if (!std::isfinite(m)) {
        // empty slice: with sink the answer is 0; without, also 0 here.
        continue;
      }
      float l = 0.0f;
      std::vector<float> acc(D, 0.0f);
      for (int kv = kv_lo; kv < eff; ++kv) {
        const float* k = &K[(((size_t)b * G + g) * max_seq + kv) * D];
        const float* v = &V[(((size_t)b * G + g) * max_seq + kv) * D];
        float dot = 0.0f;
        for (int e = 0; e < D; ++e) dot += q[e] * k[e];
        float p = std::exp(dot * scale - m);
        l += p;
        for (int e = 0; e < D; ++e) acc[e] += p * v[e];
      }
      if (c.sink) {
        l += std::exp(sink[h] - m);
      } else if (c.smooth) {
        l += std::exp(0.0f - m);
      }
      float inv = 1.0f / (l < 1e-6f ? 1e-6f : l);
      float* o = &O[((size_t)b * H + h) * D];
      for (int e = 0; e < D; ++e) o[e] = acc[e] * inv;
    }
  }
}

static double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return std::sqrt(num / (den + 1e-12));
}
static float max_abs(const std::vector<float>& a, const std::vector<float>& b) {
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

// Decode configurations the harness exercises.
//   AUTO       : production flow after this change -- launcher autotunes
//                (impl, split-count<=64) per shape and caches the winner.
//   BASELINE   : production flow BEFORE this change -- default impl
//                (wmma@d64 / scalar@d128) locked at K_SPLITS=8.
//   SCALAR8/WMMA8: force one impl at 8 splits (correctness A/B coverage).
enum DecodeMode { MODE_AUTO, MODE_BASELINE, MODE_SCALAR8, MODE_WMMA8 };

static void set_mode_env(DecodeMode m) {
  // Cleared ("") => atoi==0 => not forced (launcher reads getenv every call).
  switch (m) {
    case MODE_AUTO:
      _putenv_s("HIPDNN_GQA_DECODE_SCALAR", "");
      _putenv_s("HIPDNN_GQA_DECODE_WMMA", "");
      _putenv_s("HIPDNN_GQA_DECODE_SPLITS", "");
      break;
    case MODE_BASELINE:  // default impl, fixed 8 splits
      _putenv_s("HIPDNN_GQA_DECODE_SCALAR", "");
      _putenv_s("HIPDNN_GQA_DECODE_WMMA", "");
      _putenv_s("HIPDNN_GQA_DECODE_SPLITS", "8");
      break;
    case MODE_SCALAR8:
      _putenv_s("HIPDNN_GQA_DECODE_SCALAR", "1");
      _putenv_s("HIPDNN_GQA_DECODE_WMMA", "");
      _putenv_s("HIPDNN_GQA_DECODE_SPLITS", "8");
      break;
    case MODE_WMMA8:
      _putenv_s("HIPDNN_GQA_DECODE_SCALAR", "");
      _putenv_s("HIPDNN_GQA_DECODE_WMMA", "1");
      _putenv_s("HIPDNN_GQA_DECODE_SPLITS", "8");
      break;
  }
}

// ---- run one launcher config; returns avg ms over iters --------------------
static double run_kernel(DecodeMode mode, const Case& c, float scale,
                         const __half* dQ, const __half* dK, const __half* dV,
                         __half* dO, float* dPart, const int* dSeq,
                         const __half* dSink, int iters,
                         std::vector<float>& host_O) {
  set_mode_env(mode);

  const int B = c.B, H = c.H, G = c.G, D = c.D, max_seq = c.max_seq;
  const void* sinkp = c.sink ? (const void*)dSink : nullptr;

  // Warmup + correctness fetch. For AUTO this first call also runs the autotune
  // pass (timed candidates) and caches the winner; subsequent calls reuse it.
  HIP_CHECK((hipError_t)hip_gqa_flash_decode(
      nullptr, dQ, dK, dV, dO, dPart, B, H, G, D, c.total, max_seq, MAX_SPLITS,
      scale, dSeq, c.window, sinkp, c.smooth, HIP_KV_DTYPE_FP16, nullptr,
      nullptr, nullptr, 1, 1, 0, 0, 0));
  HIP_CHECK(hipDeviceSynchronize());
  host_O.resize((size_t)B * H * D);
  {
    std::vector<__half> tmp((size_t)B * H * D);
    HIP_CHECK(hipMemcpy(tmp.data(), dO, tmp.size() * sizeof(__half),
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < tmp.size(); ++i) host_O[i] = __half2float(tmp[i]);
  }

  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  HIP_CHECK(hipEventRecord(a));
  for (int it = 0; it < iters; ++it) {
    hip_gqa_flash_decode(nullptr, dQ, dK, dV, dO, dPart, B, H, G, D, c.total,
                            max_seq, MAX_SPLITS, scale, dSeq, c.window, sinkp,
                            c.smooth, HIP_KV_DTYPE_FP16, nullptr, nullptr,
                            nullptr, 1, 1, 0, 0, 0);
  }
  HIP_CHECK(hipEventRecord(b));
  HIP_CHECK(hipEventSynchronize(b));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  return ms / iters;
}

// Legacy fused decode (original baseline). Only valid without window/sink.
static double run_fused(const Case& c, float scale,
                        const __half* dQ, const __half* dK, const __half* dV,
                        __half* dO, const int* dSeq, int iters,
                        std::vector<float>& host_O) {
  const int B = c.B, H = c.H, G = c.G, D = c.D, max_seq = c.max_seq;
  HIP_CHECK((hipError_t)hip_gqa_fused_decode(
      nullptr, dQ, dK, dV, dO, B, H, G, D, c.total, max_seq, scale, dSeq));
  HIP_CHECK(hipDeviceSynchronize());
  host_O.resize((size_t)B * H * D);
  {
    std::vector<__half> tmp((size_t)B * H * D);
    HIP_CHECK(hipMemcpy(tmp.data(), dO, tmp.size() * sizeof(__half),
                        hipMemcpyDeviceToHost));
    for (size_t i = 0; i < tmp.size(); ++i) host_O[i] = __half2float(tmp[i]);
  }
  hipEvent_t a, b;
  HIP_CHECK(hipEventCreate(&a));
  HIP_CHECK(hipEventCreate(&b));
  HIP_CHECK(hipEventRecord(a));
  for (int it = 0; it < iters; ++it)
    hip_gqa_fused_decode(nullptr, dQ, dK, dV, dO, B, H, G, D, c.total, max_seq,
                         scale, dSeq);
  HIP_CHECK(hipEventRecord(b));
  HIP_CHECK(hipEventSynchronize(b));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  return ms / iters;
}

static int run_case(const Case& c, int iters, unsigned seed, bool verbose) {
  const int B = c.B, H = c.H, G = c.G, D = c.D, max_seq = c.max_seq;
  if (H % G != 0) { printf("[skip] %s: H%%G!=0\n", c.name); return 0; }
  const float scale = 1.0f / std::sqrt((float)D);

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> Q((size_t)B * H * D);
  std::vector<float> K((size_t)B * G * max_seq * D);
  std::vector<float> Vv((size_t)B * G * max_seq * D);
  std::vector<float> Sink(H);
  std::vector<int> Seq(B, c.total - 1);
  for (auto& x : Q) x = dist(rng);
  for (auto& x : K) x = dist(rng);
  for (auto& x : Vv) x = dist(rng);
  for (auto& x : Sink) x = dist(rng);  // natural-unit sink logits

  // fp16 device copies.
  auto to_half = [](const std::vector<float>& f) {
    std::vector<__half> h(f.size());
    for (size_t i = 0; i < f.size(); ++i) h[i] = __float2half(f[i]);
    return h;
  };
  auto hQ = to_half(Q), hK = to_half(K), hV = to_half(Vv), hSink = to_half(Sink);

  __half *dQ, *dK, *dV, *dO, *dSink;
  float* dPart;
  int* dSeq;
  HIP_CHECK(hipMalloc(&dQ, hQ.size() * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dK, hK.size() * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dV, hV.size() * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dO, (size_t)B * H * D * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dSink, hSink.size() * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dSeq, (size_t)B * sizeof(int)));
  HIP_CHECK(hipMalloc(&dPart, (size_t)B * H * MAX_SPLITS * (D + 2) * sizeof(float)));
  HIP_CHECK(hipMemcpy(dQ, hQ.data(), hQ.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dK, hK.data(), hK.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dV, hV.data(), hV.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dSink, hSink.data(), hSink.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dSeq, Seq.data(), Seq.size() * sizeof(int), hipMemcpyHostToDevice));

  std::vector<float> ref;
  cpu_reference(c, Q, K, Vv, Seq, Sink, scale, ref);

  // ours = autotuned (impl + dynamic split-K); ori = production baseline
  // (default impl @ 8 splits). scalar8/wmma8 give per-impl correctness coverage.
  // --prod-only drops all three: when comparing two builds, the extra configs
  // would run between the timed ones and move the clock state under them.
  std::vector<float> O_auto, O_base, O_wmma, O_scalar;
  double ms_auto   = run_kernel(MODE_AUTO,     c, scale, dQ, dK, dV, dO, dPart, dSeq, dSink, iters, O_auto);
  double ms_base = 0.0, ms_wmma = 0.0, ms_scalar = 0.0;
  double l2_base = 0.0, l2_wmma = 0.0, l2_scalar = 0.0, l2_ab = 0.0;
  if (!g_prod_only) {
    ms_base   = run_kernel(MODE_BASELINE, c, scale, dQ, dK, dV, dO, dPart, dSeq, dSink, iters, O_base);
    ms_wmma   = run_kernel(MODE_WMMA8,    c, scale, dQ, dK, dV, dO, dPart, dSeq, dSink, iters, O_wmma);
    ms_scalar = run_kernel(MODE_SCALAR8,  c, scale, dQ, dK, dV, dO, dPart, dSeq, dSink, iters, O_scalar);
    l2_base   = rel_l2(O_base, ref);
    l2_wmma   = rel_l2(O_wmma, ref);
    l2_scalar = rel_l2(O_scalar, ref);
    l2_ab     = rel_l2(O_wmma, O_scalar);
  }

  double l2_auto   = rel_l2(O_auto, ref);

  // One-block-per-head fused decode = the OLDEST baseline (pre-split-K); only
  // valid without window/sink, and very slow at long context, so it is opt-in
  // (--fused) and timed with fewer iters.
  const bool fused_ok_shape =
      g_do_fused && (c.window <= 0 && !c.sink && !c.smooth);
  double ms_fused = 0.0; double l2_fused = 0.0; bool fused_ran = false;
  if (fused_ok_shape) {
    std::vector<float> O_fused;
    int fiters = iters < 16 ? iters : 16;
    ms_fused = run_fused(c, scale, dQ, dK, dV, dO, dSeq, fiters, O_fused);
    l2_fused = rel_l2(O_fused, ref);
    fused_ran = true;
  }

  const double tol = 2e-2;  // fp16 accumulation tolerance
  bool ok = (l2_auto < tol) && (!fused_ran || l2_fused < tol);
  if (!g_prod_only)
    ok = ok && (l2_base < tol) && (l2_wmma < tol) && (l2_scalar < tol);

  // Three columns: fixed-config vs autotune on the SAME kernel, so they measure
  // the value of per-shape tuning, not the difference between two versions:
  //   [fix-scalar8] = force scalar @ 8 splits  (ms_scalar / MODE_SCALAR8)
  //   [fix-def8]    = force default @ 8 splits (ms_base   / MODE_BASELINE)
  //   [v2]          = autotuned impl + splits  (ms_auto    / MODE_AUTO)
  if (g_md) {
    // nothing per case; emit_markdown prints the collected rows at the end
  } else if (g_prod_only) {
    printf("%-24s H%d G%d(hpg%d) D%d | len=%5d win=%3d sink=%d smooth=%d | "
           "relL2=%.2e  %.4f ms  %s\n",
           c.name, H, G, H / G, D, c.total, c.window, c.sink, c.smooth,
           l2_auto, ms_auto, ok ? "PASS" : "*** FAIL ***");
  } else {
    printf("%-24s H%d G%d(hpg%d) D%d | len=%d win=%d sink=%d smooth=%d\n",
           c.name, H, G, H / G, D, c.total, c.window, c.sink, c.smooth);
    printf("   relL2(v2 vs cpu)=%.2e  A/B(wmma vs scalar)=%.2e\n", l2_auto, l2_ab);
    printf("   latency  [fix-scalar8]=%.4f  [fix-def8]=%.4f  [v2]auto=%.4f ms\n",
           ms_scalar, ms_base, ms_auto);
    if (fused_ran) {
      printf("   speedup  v2-vs-scalar8=%.2fx  v2-vs-def8=%.2fx  (v2-vs-one-block/head=%.1fx)   %s\n",
             ms_scalar / ms_auto, ms_base / ms_auto, ms_fused / ms_auto,
             ok ? "PASS" : "*** FAIL ***");
    } else {
      printf("   speedup  v2-vs-scalar8=%.2fx  v2-vs-def8=%.2fx   %s\n",
             ms_scalar / ms_auto, ms_base / ms_auto, ok ? "PASS" : "*** FAIL ***");
    }
  }
  (void)l2_base; (void)l2_wmma; (void)l2_fused;
  if (verbose && !ok) {
    for (int i = 0; i < std::min(8, B * H * D); ++i)
      printf("      [%d] ref=%.5f v2=%.5f scalar8=%.5f\n", i, ref[i], O_auto[i], O_scalar[i]);
  }

  g_rows.push_back({std::string(c.name), H, G, D, c.total, c.window, c.sink,
                    c.smooth, ms_scalar, ms_base, ms_auto, l2_auto, ok});

  hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);
  hipFree(dSink); hipFree(dSeq); hipFree(dPart);
  return ok ? 0 : 1;
}

// Markdown table for the perf report. Columns are fixed-config vs autotune on
// the SAME v2 kernel (NOT historical code): fix-scalar8 = force scalar @ 8
// splits; fix-def8 = force default impl @ 8 splits; v2 = autotune. This
// isolates the value of per-shape autotune over a fixed 8-split policy.
static void emit_markdown(int iters, const char* dev, int cus) {
  printf("<!-- device: %s | %d CUs | iters=%d | per-decode-step latency (ms) | B=1 | fixed-config vs autotune on the v2 kernel (NOT historical code) -->\n\n",
         dev, cus, iters);
  // Speedup columns "v2 vs xxx" = fixed_ms / v2_ms (unified so >1 always = v2 faster).
  printf("| # | model / geometry | HpG | D | len | win | feat | fix-scalar8 (ms) | fix-def8 (ms) | v2 (ms) | v2 vs scalar8 | v2 vs def8 | result |\n");
  printf("|--:|---|--:|--:|--:|--:|:--|--:|--:|--:|--:|--:|:--|\n");
  int i = 1;
  for (const auto& r : g_rows) {
    const char* feat = r.sink ? "sink" : (r.smooth ? "smooth" : "-");
    printf("| %d | %s | %d | %d | %d | %d | %s | %.4f | %.4f | %.4f | %.2fx | %.2fx | %s |\n",
           i++, r.name.c_str(), r.H / r.G, r.D, r.len, r.window, feat,
           r.ms_fix_scalar8, r.ms_fix_def8, r.ms_v2, r.ms_fix_scalar8 / r.ms_v2,
           r.ms_fix_def8 / r.ms_v2, r.ok ? "PASS" : "FAIL");
  }
}

int main(int argc, char** argv) {
  int iters = 200;
  unsigned seed = 1234;
  bool all = false, verbose = false;
  Case single = {"custom", 1, 64, 8, 64, 8192, 8192, 0, 0, 0};
  bool have_single = false;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](int& v) { if (i + 1 < argc) v = atoi(argv[++i]); };
    if (a == "--all") all = true;
    else if (a == "--verbose") verbose = true;
    else if (a == "--md") { g_md = true; all = true; }
    else if (a == "--fused") g_do_fused = true;
    else if (a == "--prod-only") g_prod_only = true;
    else if (a == "--only" && i + 1 < argc) { g_only = argv[++i]; all = true; }
    else if (a == "--iters") next(iters);
    else if (a == "--seed") { int s; next(s); seed = (unsigned)s; }
    else if (a == "--b") { next(single.B); have_single = true; }
    else if (a == "--h") { next(single.H); have_single = true; }
    else if (a == "--g") { next(single.G); have_single = true; }
    else if (a == "--d") { next(single.D); have_single = true; }
    else if (a == "--max-seq") { next(single.max_seq); have_single = true; }
    else if (a == "--total") { next(single.total); have_single = true; }
    else if (a == "--window") { next(single.window); have_single = true; }
    else if (a == "--sink") { next(single.sink); have_single = true; }
    else if (a == "--smooth") { next(single.smooth); have_single = true; }
  }

  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  if (!g_md)
    printf("Device: %s (%d CUs)  iters=%d  max_splits=%d (PR438 baseline=%d)\n\n",
           prop.name, prop.multiProcessorCount, iters, MAX_SPLITS, BASELINE_SPLITS);

  int fails = 0;
  if (all || !have_single) {
    // B=1 single-stream decode. Coverage: real models + a geometry sweep over
    // MHA (HpG==1) and GQA (HpG in {2,4,5,8,16}) x head_dim in {64,128,256,512}.
    const int lens[] = {512, 2048, 8192, 32768};
    auto maybe = [&](const Case& c) {
      if (!g_only.empty() &&
          std::string(c.name).find(g_only) == std::string::npos)
        return;
      fails += run_case(c, iters, seed, verbose);
    };
    for (int L : lens) {
      // ---- Real models ----
      // gpt-oss-20b carries a learnable per-head attention SINK on ALL 24
      // layers; the layer TYPE alternates between full attention and a 128
      // sliding window. So "full"/"sliding" names the attention type, and
      // head_sink is present in BOTH -- it is NOT "full=smooth, sliding=sink".
      // full attention layer: HpG8 D64 + head_sink.
      maybe({"gpt_oss-20b full",    1, 64,  8,  64, L, L,   0, 1, 0});
      // sliding-window layer: window 128 + head_sink.
      maybe({"gpt_oss-20b sliding", 1, 64,  8,  64, L, L, 128, 1, 0});
      // smooth-softmax variant (sink logit fixed to 0): keeps the smooth path
      // under correctness coverage even though real gpt-oss ships head_sink.
      maybe({"gpt_oss-20b smooth",  1, 64,  8,  64, L, L,   0, 0, 1});
      // llama-3.1-8b: H32 G8 (hpg4) D128.
      maybe({"llama-3.1-8b",        1, 32,  8, 128, L, L,   0, 0, 0});
      // llama-3.2-1b: H32 G8 (hpg4) D64.
      maybe({"llama-3.2-1b",        1, 32,  8,  64, L, L,   0, 0, 0});
      // qwen2.5-14b: H40 G8 (hpg5) D128 -- HpG=5 has no WMMA path (scalar).
      maybe({"qwen2.5-14b",         1, 40,  8, 128, L, L,   0, 0, 0});
      // ---- Geometry sweep: MHA (hpg1) x head_dim ----
      maybe({"MHA hpg1 D64",        1, 16, 16,  64, L, L,   0, 0, 0});
      maybe({"MHA hpg1 D128",       1, 16, 16, 128, L, L,   0, 0, 0});
      maybe({"MHA hpg1 D256",       1,  8,  8, 256, L, L,   0, 0, 0});
      // ---- Geometry sweep: GQA HpG {2,8,16} and D256 ----
      maybe({"GQA hpg2 D128",       1, 16,  8, 128, L, L,   0, 0, 0});
      maybe({"GQA hpg8 D128",       1, 64,  8, 128, L, L,   0, 0, 0});
      maybe({"GQA hpg16 D64",       1, 32,  2,  64, L, L,   0, 0, 0});
      maybe({"GQA hpg4 D256",       1, 32,  8, 256, L, L,   0, 0, 0});
      // ---- D512: Gemma-4's 5 global-attention layers ----
      // Twice the head width of its 25 sliding layers, which is why they are
      // the only ones the geometry gate used to reject. EPT=16 here, the widest
      // the scalar kernel templates, so this is also the register-pressure
      // boundary case. The sliding sibling is covered so both halves of the
      // same model run through this kernel.
      maybe({"gemma-4-26b global",  1, 16,  8, 512, L, L,   0, 0, 0});
      maybe({"gemma-4-26b sliding", 1, 16,  8, 256, L, L, 1024, 0, 0});
      maybe({"MHA hpg1 D512",       1,  8,  8, 512, L, L,   0, 0, 0});
      maybe({"GQA hpg4 D512",       1, 32,  8, 512, L, L,   0, 0, 0});
      if (!g_md) printf("\n");
    }
  } else {
    fails += run_case(single, iters, seed, verbose);
  }

  if (g_md) emit_markdown(iters, prop.name, prop.multiProcessorCount);
  printf("%s%s (%d failing case(s))\n", g_md ? "\n" : "",
         fails == 0 ? "ALL PASS" : "FAILURES", fails);
  return fails == 0 ? 0 : 1;
}
