// ============================================================
// custom_kernels GQA flash *prefill* (TTFT) test + benchmark.
//
// Verifies the ported FA-2 WMMA prefill kernels that gqa.cpp routes to on the
// fused-prefill fast path:
//   hip_gqa_flash_prefill_v5  (d == 64, gpt-oss / llama-3.2 geometry)
//   hip_gqa_flash_prefill_v7  (d == 128, llama-3.1 geometry)
// against a CPU fp32 causal-attention reference (correctness) and reports the
// per-prefill latency (the quantity that bounds TTFT).
//
// Layout matches the EP fused-prefill call site (gqa.cpp): Q is BSHD
// [B,sq,Hq,d]; K/V cache is BNSD [B,G,max_seq,d]; O is BSHD [B,sq,Hq,d].
// Pure prefill: past_len = 0, total_seq = sq. Self-contained random inputs.
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

extern "C" int hip_gqa_flash_prefill_v5(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale);

// Unified entry the runtime (gqa.cpp) actually calls -- picks v5/v7 by head dim.
extern "C" int hip_gqa_flash_prefill_v2(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale);

extern "C" int hip_gqa_flash_prefill_v7(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale);

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
  int B, H, G, D, sq;  // pure prefill: skv = sq, past_len = 0
};

// CPU fp32 reference: causal GQA attention. Q/O BSHD, K/V cache BNSD.
static void cpu_reference(const std::vector<float>& Q,
                          const std::vector<float>& K,
                          const std::vector<float>& V, std::vector<float>& O,
                          int B, int H, int G, int D, int sq, int max_seq,
                          int past_len, float scale) {
  const int HPG = H / G;
  const int total = past_len + sq;
  std::vector<float> scores(total);
  for (int b = 0; b < B; ++b) {
    for (int hq = 0; hq < H; ++hq) {
      const int hkv = hq / HPG;
      for (int s = 0; s < sq; ++s) {
        const float* q = &Q[((size_t)(b * sq + s) * H + hq) * D];
        const int kmax = past_len + s;  // causal: attend to keys 0..kmax
        float m = -1e30f;
        for (int k = 0; k <= kmax; ++k) {
          const float* kp = &K[((size_t)(b * G + hkv) * max_seq + k) * D];
          float dot = 0.0f;
          for (int e = 0; e < D; ++e) dot += q[e] * kp[e];
          scores[k] = dot * scale;
          if (scores[k] > m) m = scores[k];
        }
        float l = 0.0f;
        for (int k = 0; k <= kmax; ++k) {
          scores[k] = std::exp(scores[k] - m);
          l += scores[k];
        }
        const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        float* o = &O[((size_t)(b * sq + s) * H + hq) * D];
        for (int e = 0; e < D; ++e) o[e] = 0.0f;
        for (int k = 0; k <= kmax; ++k) {
          const float* vp = &V[((size_t)(b * G + hkv) * max_seq + k) * D];
          const float w = scores[k] * inv;
          for (int e = 0; e < D; ++e) o[e] += w * vp[e];
        }
      }
    }
  }
}

static double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    num += d * d;
    den += (double)b[i] * b[i];
  }
  return std::sqrt(num / (den + 1e-12));
}

static bool run_case(const Case& c, int iters) {
  const int B = c.B, H = c.H, G = c.G, D = c.D, sq = c.sq;
  const int max_seq = sq;       // pure-prefill cache buffer
  const int past_len = 0, skv = sq;
  const float scale = 1.0f / std::sqrt((float)D);

  const size_t qn = (size_t)B * sq * H * D;
  const size_t kn = (size_t)B * G * max_seq * D;
  std::mt19937 rng(1234 + sq + D);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> Qf(qn), Kf(kn), Vf(kn), Oref(qn);
  for (auto& x : Qf) x = dist(rng);
  for (auto& x : Kf) x = dist(rng);
  for (auto& x : Vf) x = dist(rng);

  cpu_reference(Qf, Kf, Vf, Oref, B, H, G, D, sq, max_seq, past_len, scale);

  std::vector<__half> Qh(qn), Kh(kn), Vh(kn);
  for (size_t i = 0; i < qn; ++i) Qh[i] = __float2half(Qf[i]);
  for (size_t i = 0; i < kn; ++i) { Kh[i] = __float2half(Kf[i]); Vh[i] = __float2half(Vf[i]); }

  __half *dQ, *dK, *dV, *dO;
  HIP_CHECK(hipMalloc(&dQ, qn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dK, kn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dV, kn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dO, qn * sizeof(__half)));
  HIP_CHECK(hipMemcpy(dQ, Qh.data(), qn * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dK, Kh.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dV, Vh.data(), kn * sizeof(__half), hipMemcpyHostToDevice));

  // Route through the unified entry (same path the runtime takes); it dispatches
  // v5 (D==64) / v7 (D==128) internally.
  auto launch = [&]() {
    return hip_gqa_flash_prefill_v2(nullptr, dQ, dK, dV, dO, B, H, G, sq, skv, D,
                                 max_seq, past_len, scale);
  };

  int rc = launch();  // first call self-tunes
  HIP_CHECK(hipDeviceSynchronize());
  if (rc != 0) { fprintf(stderr, "kernel returned %d\n", rc); return false; }

  std::vector<__half> Oh(qn);
  HIP_CHECK(hipMemcpy(Oh.data(), dO, qn * sizeof(__half), hipMemcpyDeviceToHost));
  std::vector<float> Oout(qn);
  for (size_t i = 0; i < qn; ++i) Oout[i] = __half2float(Oh[i]);
  const double err = rel_l2(Oout, Oref);

  for (int i = 0; i < 10; ++i) launch();
  HIP_CHECK(hipDeviceSynchronize());
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventRecord(e0));
  for (int i = 0; i < iters; ++i) launch();
  HIP_CHECK(hipEventRecord(e1));
  HIP_CHECK(hipEventSynchronize(e1));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
  ms /= iters;

  const bool pass = err < 2e-3;
  printf("%-16s B%d H%d G%d(hpg%d) D%-3d sq=%-5d | relL2=%.2e  latency=%.4f ms  %s (v%d)\n",
         c.name, B, H, G, H / G, D, sq, err, ms, pass ? "PASS" : "FAIL",
         D == 64 ? 5 : 7);

  hipEventDestroy(e0); hipEventDestroy(e1);
  hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);
  return pass;
}

int main(int argc, char** argv) {
  int iters = 100;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);

  const Case cases[] = {
      {"gpt_oss-20b",  1, 64, 8,  64, 512},
      {"gpt_oss-20b",  1, 64, 8,  64, 2048},
      {"llama-3.2-1b", 1, 32, 8,  64, 512},
      {"llama-3.2-1b", 1, 32, 8,  64, 2048},
      {"llama-3.1-8b", 1, 32, 8, 128, 512},
      {"llama-3.1-8b", 1, 32, 8, 128, 2048},
  };
  int fails = 0;
  for (const auto& c : cases) if (!run_case(c, iters)) ++fails;
  printf("\n%s (%d failing case(s))\n", fails == 0 ? "ALL PASS" : "SOME FAILED", fails);
  return fails == 0 ? 0 : 1;
}
