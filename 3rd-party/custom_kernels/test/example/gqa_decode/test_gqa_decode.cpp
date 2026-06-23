// ============================================================
// custom_kernels GQA flash *decode* (TPS) test + A/B benchmark.
//
// Verifies the WMMA decode kernel (default path of hip_gqa_flash_decode)
// against:
//   1. a CPU fp32 reference (correctness, incl. seqlens_k / sliding-window /
//      head-sink / smooth-softmax), and
//   2. the original scalar split-K kernel (HIPDNN_GQA_DECODE_SCALAR=1) for an
//      apples-to-apples latency comparison on the SAME inputs.
//
// Both kernels emit the same [B,H,K_SPLITS,D+2] partials and share the FA-2
// reduce, so any output divergence is a real bug. The benchmark reports the
// per-decode-step latency (the quantity that bounds tokens/sec) and the WMMA
// speedup over the scalar baseline.
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
extern "C" int hip_gqa_flash_decode(
    void* stream,
    const void* Q, const void* Kcache, const void* Vcache,
    void* O,
    void* partials_workspace,
    int B, int H, int G, int d, int max_seq, int K_SPLITS,
    float scale,
    const void* seqlens_k,
    int local_window_size,
    const void* head_sink,
    int use_smooth_softmax);

// Legacy one-block-per-head fused decode (the ORIGINAL baseline that the
// OPTIMIZATION.md 10-20x figure was measured against). No window/sink/split-K.
extern "C" int hip_gqa_fused_decode(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int H, int G, int d, int skv, int max_seq,
    float scale, const void* seqlens_k);

static constexpr int K_SPLITS = 8;

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

// ---- run one launcher config; returns avg ms over iters --------------------
static double run_kernel(bool use_scalar, const Case& c, float scale,
                         const __half* dQ, const __half* dK, const __half* dV,
                         __half* dO, float* dPart, const int* dSeq,
                         const __half* dSink, int iters,
                         std::vector<float>& host_O) {
  // Force the path explicitly so the A/B exercises BOTH kernels on every shape
  // (the production default only picks WMMA at D=64).
  if (use_scalar) { _putenv_s("HIPDNN_GQA_DECODE_SCALAR", "1"); _putenv_s("HIPDNN_GQA_DECODE_WMMA", "0"); }
  else            { _putenv_s("HIPDNN_GQA_DECODE_SCALAR", "0"); _putenv_s("HIPDNN_GQA_DECODE_WMMA", "1"); }

  const int B = c.B, H = c.H, G = c.G, D = c.D, max_seq = c.max_seq;
  const void* sinkp = c.sink ? (const void*)dSink : nullptr;

  // Warmup + correctness fetch.
  HIP_CHECK((hipError_t)hip_gqa_flash_decode(
      nullptr, dQ, dK, dV, dO, dPart, B, H, G, D, max_seq, K_SPLITS, scale,
      dSeq, c.window, sinkp, c.smooth));
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
    hip_gqa_flash_decode(nullptr, dQ, dK, dV, dO, dPart, B, H, G, D, max_seq,
                         K_SPLITS, scale, dSeq, c.window, sinkp, c.smooth);
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
  HIP_CHECK(hipMalloc(&dPart, (size_t)B * H * K_SPLITS * (D + 2) * sizeof(float)));
  HIP_CHECK(hipMemcpy(dQ, hQ.data(), hQ.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dK, hK.data(), hK.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dV, hV.data(), hV.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dSink, hSink.data(), hSink.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dSeq, Seq.data(), Seq.size() * sizeof(int), hipMemcpyHostToDevice));

  std::vector<float> ref;
  cpu_reference(c, Q, K, Vv, Seq, Sink, scale, ref);

  std::vector<float> O_wmma, O_scalar;
  double ms_wmma   = run_kernel(false, c, scale, dQ, dK, dV, dO, dPart, dSeq, dSink, iters, O_wmma);
  double ms_scalar = run_kernel(true,  c, scale, dQ, dK, dV, dO, dPart, dSeq, dSink, iters, O_scalar);

  double l2_wmma   = rel_l2(O_wmma, ref);
  double l2_scalar = rel_l2(O_scalar, ref);
  double l2_ab     = rel_l2(O_wmma, O_scalar);
  float  mx_wmma   = max_abs(O_wmma, ref);

  // Legacy fused decode = the ORIGINAL baseline; only valid without window/sink.
  const bool fused_ok_shape = (c.window <= 0 && !c.sink && !c.smooth);
  double ms_fused = 0.0; double l2_fused = 0.0; bool fused_ran = false;
  if (fused_ok_shape) {
    std::vector<float> O_fused;
    ms_fused = run_fused(c, scale, dQ, dK, dV, dO, dSeq, iters, O_fused);
    l2_fused = rel_l2(O_fused, ref);
    fused_ran = true;
  }

  const double tol = 2e-2;  // fp16 accumulation tolerance
  bool ok = (l2_wmma < tol) && (l2_scalar < tol) &&
            (!fused_ran || l2_fused < tol);

  printf("%-22s B%d H%d G%d(hpg%d) D%d | len=%d win=%d sink=%d smooth=%d\n",
         c.name, B, H, G, H / G, D, c.total, c.window, c.sink, c.smooth);
  printf("   relL2  wmma=%.2e  scalar=%.2e  (wmma-vs-scalar=%.2e)  maxAbs_wmma=%.3e\n",
         l2_wmma, l2_scalar, l2_ab, mx_wmma);
  if (fused_ran) {
    printf("   latency  wmma=%.4f ms  scalar=%.4f ms  fused(legacy)=%.4f ms\n",
           ms_wmma, ms_scalar, ms_fused);
    printf("   speedup  wmma-vs-scalar=%.2fx  wmma-vs-fused=%.2fx  scalar-vs-fused=%.2fx   %s\n",
           ms_scalar / ms_wmma, ms_fused / ms_wmma, ms_fused / ms_scalar,
           ok ? "PASS" : "*** FAIL ***");
  } else {
    printf("   latency  wmma=%.4f ms  scalar=%.4f ms  speedup=%.2fx   %s\n",
           ms_wmma, ms_scalar, ms_scalar / ms_wmma, ok ? "PASS" : "*** FAIL ***");
  }
  if (verbose && !ok) {
    for (int i = 0; i < std::min(8, B * H * D); ++i)
      printf("      [%d] ref=%.5f wmma=%.5f scalar=%.5f\n", i, ref[i], O_wmma[i], O_scalar[i]);
  }

  hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);
  hipFree(dSink); hipFree(dSeq); hipFree(dPart);
  return ok ? 0 : 1;
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
  printf("Device: %s (%d CUs)  iters=%d  K_SPLITS=%d\n\n",
         prop.name, prop.multiProcessorCount, iters, K_SPLITS);

  int fails = 0;
  if (all || !have_single) {
    // Representative model decode shapes (B=1 single-stream decode).
    const int lens[] = {512, 2048, 8192, 32768};
    for (int L : lens) {
      // gpt-oss-20b: H64 G8 (hpg8) D64 -- full layer + smooth softmax sink.
      fails += run_case({"gpt_oss-20b full",  1, 64, 8, 64, L, L, 0, 0, 1}, iters, seed, verbose);
      // gpt-oss-20b sliding layer: window 128 + head_sink.
      fails += run_case({"gpt_oss-20b sliding",1, 64, 8, 64, L, L, 128, 1, 0}, iters, seed, verbose);
      // llama-3.1-8b: H32 G8 (hpg4) D128.
      fails += run_case({"llama-3.1-8b",       1, 32, 8, 128, L, L, 0, 0, 0}, iters, seed, verbose);
      // llama-3.2-1b: H32 G8 (hpg4) D64.
      fails += run_case({"llama-3.2-1b",       1, 32, 8, 64, L, L, 0, 0, 0}, iters, seed, verbose);
      printf("\n");
    }
  } else {
    fails += run_case(single, iters, seed, verbose);
  }

  printf("%s (%d failing case(s))\n", fails == 0 ? "ALL PASS" : "FAILURES", fails);
  return fails == 0 ? 0 : 1;
}
