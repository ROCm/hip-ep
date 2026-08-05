// ============================================================
// REAL legacy-vs-v2 double-check probe for the gpt-oss-20b decode regression.
//
// This links the PRE-DELETION kernel backup (hip/gqa_kernel_back.hip) and calls
// the REAL legacy FA-2 split-K launcher `hip_gqa_flash_decode` directly -- NOT
// an env-simulated stand-in on the v2 kernel. It reproduces exactly what PR#438
// shipped:
//   - K_SPLITS hard-locked to 8
//   - d==64 defaults to the WMMA split kernel  (gpt-oss full = HpG8 D64)
//   - HIPDNN_GQA_DECODE_SCALAR=1 forces the scalar split kernel (the PR#438-
//     earlier "orig" impl, still at 8 splits)
//
// The backup predates v2, so v2 is NOT available here. This probe emits the
// two REAL legacy columns (orig = forced scalar @8; PR438 = default d64-WMMA
// @8) plus the "PR438 vs orig" speedup = orig_ms/PR438_ms (unified so >1 always
// means faster). Inputs use the SAME seed/layout as
// test_gqa_decode.cpp, so these real legacy numbers are directly comparable to
// the v2 column produced by the main test (current gqa_kernel.hip) and let us
// confirm the main report's env-simulated PR438 column was faithful.
//
// Geometries: only what the legacy launcher accepts (HpG in {4,8}, d in
// {64,128}, HpG8 => d64). That covers exactly the WMMA-regression shapes:
// gpt-oss full/sliding (HpG8 D64) and llama-3.2-1b (HpG4 D64), plus a D128
// control (llama-3.1-8b HpG4 D128) that should show NO PR438 regression.
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

// REAL legacy launcher from gqa_kernel_back.hip (PR#438 production path).
extern "C" int hip_gqa_flash_decode(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, void* partials_workspace,
    int B, int H, int G, int d, int max_seq, int K_SPLITS,
    float scale, const void* seqlens_k,
    int local_window_size, const void* head_sink, int use_smooth_softmax);

static constexpr int KSPLITS = 8;    // legacy is hard-locked here

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
  int B, H, G, D, max_seq, total;
  int window;  // <=0 = full
  int sink;    // head_sink present
  int smooth;  // use_smooth_softmax
};

// ---- CPU fp32 reference (identical semantics to test_gqa_decode.cpp) -------
static void cpu_reference(const Case& c, const std::vector<float>& Q,
                          const std::vector<float>& K,
                          const std::vector<float>& V,
                          const std::vector<int>& seqlens,
                          const std::vector<float>& sink, float scale,
                          std::vector<float>& O) {
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
      float m = -INFINITY;
      for (int kv = kv_lo; kv < eff; ++kv) {
        const float* k = &K[(((size_t)b * G + g) * max_seq + kv) * D];
        float dot = 0.0f;
        for (int e = 0; e < D; ++e) dot += q[e] * k[e];
        float s = dot * scale;
        if (s > m) m = s;
      }
      if (!std::isfinite(m)) continue;
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
      if (c.sink) l += std::exp(sink[h] - m);
      else if (c.smooth) l += std::exp(0.0f - m);
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

enum LegacyImpl { LEG_DEFAULT, LEG_SCALAR };  // DEFAULT: d64->WMMA (=PR438)

static void set_legacy_env(LegacyImpl im) {
  switch (im) {
    case LEG_DEFAULT:  // PR438 production: no override, d64 auto-WMMA
      _putenv_s("HIPDNN_GQA_DECODE_SCALAR", "");
      _putenv_s("HIPDNN_GQA_DECODE_WMMA", "");
      break;
    case LEG_SCALAR:   // orig: force scalar split kernel (still 8 splits)
      _putenv_s("HIPDNN_GQA_DECODE_SCALAR", "1");
      _putenv_s("HIPDNN_GQA_DECODE_WMMA", "");
      break;
  }
}

static double time_legacy(LegacyImpl im, const Case& c, float scale,
                          const __half* dQ, const __half* dK, const __half* dV,
                          __half* dO, float* dPart, const int* dSeq,
                          const __half* dSink, int iters,
                          std::vector<float>& host_O) {
  set_legacy_env(im);
  const int B = c.B, H = c.H, G = c.G, D = c.D, max_seq = c.max_seq;
  const void* sinkp = c.sink ? (const void*)dSink : nullptr;
  HIP_CHECK((hipError_t)hip_gqa_flash_decode(
      nullptr, dQ, dK, dV, dO, dPart, B, H, G, D, max_seq, KSPLITS, scale,
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
  for (int it = 0; it < iters; ++it)
    hip_gqa_flash_decode(nullptr, dQ, dK, dV, dO, dPart, B, H, G, D, max_seq,
                         KSPLITS, scale, dSeq, c.window, sinkp, c.smooth);
  HIP_CHECK(hipEventRecord(b));
  HIP_CHECK(hipEventSynchronize(b));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, a, b));
  HIP_CHECK(hipEventDestroy(a));
  HIP_CHECK(hipEventDestroy(b));
  return ms / iters;
}

static void run_case(const Case& c, int iters, unsigned seed) {
  const int B = c.B, H = c.H, G = c.G, D = c.D, max_seq = c.max_seq;
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
  for (auto& x : Sink) x = dist(rng);

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
  HIP_CHECK(hipMalloc(&dPart, (size_t)B * H * KSPLITS * (D + 2) * sizeof(float)));
  HIP_CHECK(hipMemcpy(dQ, hQ.data(), hQ.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dK, hK.data(), hK.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dV, hV.data(), hV.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dSink, hSink.data(), hSink.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dSeq, Seq.data(), Seq.size() * sizeof(int), hipMemcpyHostToDevice));

  std::vector<float> ref;
  cpu_reference(c, Q, K, Vv, Seq, Sink, scale, ref);

  std::vector<float> O_orig, O_pr438;
  double ms_orig  = time_legacy(LEG_SCALAR,  c, scale, dQ, dK, dV, dO, dPart, dSeq, dSink, iters, O_orig);
  double ms_pr438 = time_legacy(LEG_DEFAULT, c, scale, dQ, dK, dV, dO, dPart, dSeq, dSink, iters, O_pr438);

  double l2_orig  = rel_l2(O_orig, ref);
  double l2_pr438 = rel_l2(O_pr438, ref);
  const double tol = 2e-2;
  bool ok = l2_orig < tol && l2_pr438 < tol;

  const char* feat = c.sink ? "sink" : (c.smooth ? "smooth" : "-");
  printf("| %-20s | %d | %d | %5d | %3d | %-6s | %.4f | %.4f | %.2fx | %s |\n",
         c.name, H / G, D, c.total, c.window, feat,
         ms_orig, ms_pr438,
         ms_orig / ms_pr438,       // "PR438 vs orig" speedup = orig/PR438 : >1 = PR438 faster
         ok ? "PASS" : "FAIL");
  fprintf(stderr, "  %-20s len=%d relL2 orig=%.2e pr438=%.2e\n",
          c.name, c.total, l2_orig, l2_pr438);

  hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);
  hipFree(dSink); hipFree(dSeq); hipFree(dPart);
}

int main(int argc, char** argv) {
  int iters = 120;
  unsigned seed = 1234;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--iters" && i + 1 < argc) iters = atoi(argv[++i]);
    else if (a == "--seed" && i + 1 < argc) seed = (unsigned)atoi(argv[++i]);
  }

  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  fprintf(stderr, "Device: %s (%d CUs) iters=%d  [REAL legacy kernel from gqa_kernel_back.hip]\n",
          prop.name, prop.multiProcessorCount, iters);

  printf("<!-- device: %s | %d CUs | iters=%d | REAL legacy hip_gqa_flash_decode from gqa_kernel_back.hip -->\n\n",
         prop.name, prop.multiProcessorCount, iters);
  // "PR438 vs orig" = orig_ms / PR438_ms (speedup, >1 = PR438 faster; unified with report)
  printf("| model / geometry | HpG | D | len | win | feat | orig (ms) | PR438 (ms) | PR438 vs orig | result |\n");
  printf("|---|--:|--:|--:|--:|:--|--:|--:|--:|:--|\n");

  const int lens[] = {512, 2048, 8192, 32768};
  for (int L : lens) {
    // gpt-oss-20b full attention layer: HpG8 D64 + head_sink.
    run_case({"gpt_oss-20b full",    1, 64, 8,  64, L, L,   0, 1, 0}, iters, seed);
    // gpt-oss-20b sliding-window layer: window 128 + head_sink.
    run_case({"gpt_oss-20b sliding", 1, 64, 8,  64, L, L, 128, 1, 0}, iters, seed);
    // llama-3.2-1b: HpG4 D64 (also WMMA-default => expect PR438 regression).
    run_case({"llama-3.2-1b",        1, 32, 8,  64, L, L,   0, 0, 0}, iters, seed);
    // llama-3.1-8b: HpG4 D128 control (scalar default => NO PR438 regression).
    run_case({"llama-3.1-8b",        1, 32, 8, 128, L, L,   0, 0, 0}, iters, seed);
  }
  printf("\nDONE\n");
  return 0;
}
