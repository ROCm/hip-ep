// ============================================================
// custom_kernels MultiHeadAttention flash *prefill* test + benchmark.
//
// Verifies hip_mha_flash_prefill (fused non-causal FA-2 WMMA prefill) against a
// CPU fp32 bidirectional-attention reference (correctness) and reports the
// per-prefill latency (the quantity that bounds vision-encoder TTFT).
//
// Layout matches the runtime call site: Q is BSND [B,sq,N,d]; K/V cache is
// BNSD [B,N,max_seq,d]; O is BSND [B,sq,N,d]. Self-attention (HpG==1), pure
// prefill: past_len = 0, skv = sq. Self-contained random inputs.
// ============================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" int hip_mha_flash_prefill(void* stream, const void* Q,
                                     const void* Kcache, const void* Vcache,
                                     void* O, int B, int N, int sq, int skv,
                                     int d, int max_seq, float scale);

#define HIP_CHECK(expr)                                                         \
  do {                                                                          \
    hipError_t _e = (expr);                                                     \
    if (_e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),         \
              __FILE__, __LINE__);                                              \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

struct Case {
  const char* name;
  int B, N, d, sq;  // pure prefill: skv = sq, HpG == 1
};

// CPU fp32 reference: non-causal (bidirectional) self-attention.
// Q/O BSND [B,sq,N,d]; K/V BNSD [B,N,max_seq,d].
static void cpu_reference(const std::vector<float>& Q,
                          const std::vector<float>& K,
                          const std::vector<float>& V, std::vector<float>& O,
                          int B, int N, int d, int sq, int max_seq,
                          float scale) {
  std::vector<float> scores(sq);
  for (int b = 0; b < B; ++b) {
    for (int h = 0; h < N; ++h) {
      for (int s = 0; s < sq; ++s) {
        const float* q = &Q[((size_t)(b * sq + s) * N + h) * d];
        float m = -1e30f;
        for (int k = 0; k < sq; ++k) {  // non-causal: attend to all keys
          const float* kp = &K[((size_t)(b * N + h) * max_seq + k) * d];
          float dot = 0.0f;
          for (int e = 0; e < d; ++e) dot += q[e] * kp[e];
          scores[k] = dot * scale;
          if (scores[k] > m) m = scores[k];
        }
        float l = 0.0f;
        for (int k = 0; k < sq; ++k) {
          scores[k] = std::exp(scores[k] - m);
          l += scores[k];
        }
        const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        float* o = &O[((size_t)(b * sq + s) * N + h) * d];
        for (int e = 0; e < d; ++e) o[e] = 0.0f;
        for (int k = 0; k < sq; ++k) {
          const float* vp = &V[((size_t)(b * N + h) * max_seq + k) * d];
          const float w = scores[k] * inv;
          for (int e = 0; e < d; ++e) o[e] += w * vp[e];
        }
      }
    }
  }
}

static double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double dd = a[i] - b[i];
    num += dd * dd;
    den += (double)b[i] * b[i];
  }
  return std::sqrt(num / (den + 1e-12));
}

static bool run_case(const Case& c, int iters, bool check) {
  const int B = c.B, N = c.N, d = c.d, sq = c.sq;
  const int max_seq = sq, skv = sq;
  const float scale = 1.0f / std::sqrt((float)d);

  const size_t qn = (size_t)B * sq * N * d;
  const size_t kn = (size_t)B * N * max_seq * d;
  std::mt19937 rng(1234 + sq + d);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> Qf(qn), Kf(kn), Vf(kn);
  for (auto& x : Qf) x = dist(rng);
  for (auto& x : Kf) x = dist(rng);
  for (auto& x : Vf) x = dist(rng);

  std::vector<__half> Qh(qn), Kh(kn), Vh(kn);
  for (size_t i = 0; i < qn; ++i) Qh[i] = __float2half(Qf[i]);
  for (size_t i = 0; i < kn; ++i) {
    Kh[i] = __float2half(Kf[i]);
    Vh[i] = __float2half(Vf[i]);
  }

  __half *dQ, *dK, *dV, *dO;
  HIP_CHECK(hipMalloc(&dQ, qn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dK, kn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dV, kn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dO, qn * sizeof(__half)));
  HIP_CHECK(hipMemcpy(dQ, Qh.data(), qn * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dK, Kh.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dV, Vh.data(), kn * sizeof(__half), hipMemcpyHostToDevice));

  auto launch = [&]() {
    return hip_mha_flash_prefill(nullptr, dQ, dK, dV, dO, B, N, sq, skv, d,
                                 max_seq, scale);
  };

  int rc = launch();
  HIP_CHECK(hipDeviceSynchronize());
  if (rc != 0) {
    fprintf(stderr, "kernel returned %d\n", rc);
    return false;
  }

  double err = 0.0;
  if (check) {
    std::vector<float> Oref(qn);
    cpu_reference(Qf, Kf, Vf, Oref, B, N, d, sq, max_seq, scale);
    std::vector<__half> Oh(qn);
    HIP_CHECK(hipMemcpy(Oh.data(), dO, qn * sizeof(__half), hipMemcpyDeviceToHost));
    std::vector<float> Oout(qn);
    for (size_t i = 0; i < qn; ++i) Oout[i] = __half2float(Oh[i]);
    err = rel_l2(Oout, Oref);
  }

  for (int i = 0; i < 5; ++i) launch();
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

  const bool pass = !check || err < 3e-3;
  printf("%-14s B%d N%d d%-3d sq=%-5d | relL2=%s%.2e  latency=%.4f ms  %s\n",
         c.name, B, N, d, sq, check ? "" : "(skip)", err, ms,
         pass ? "PASS" : "FAIL");

  hipEventDestroy(e0);
  hipEventDestroy(e1);
  hipFree(dQ);
  hipFree(dK);
  hipFree(dV);
  hipFree(dO);
  return pass;
}

int main(int argc, char** argv) {
  int iters = 50;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--iters") && i + 1 < argc)
      iters = std::atoi(argv[++i]);

  // Small shapes: checked against CPU reference. Vision shape (7296): perf only
  // (CPU reference for sq=7296 would be ~10^12 flops -> minutes; skip check).
  int fails = 0;
  const Case checked[] = {
      {"tiny",      1, 2,  72, 64},
      {"tiny-pad",  1, 4,  72, 128},
      {"d64",       1, 8,  64, 256},
      {"d80",       1, 8,  80, 256},
      {"d128",      1, 8, 128, 256},
      {"mid",       1, 16, 72, 512},
  };
  for (const auto& c : checked) if (!run_case(c, iters, true)) ++fails;

  // Real vision-encoder prefill geometry (Qwen VLM ViT): B1 N16 d72 sq7296.
  const Case perf[] = {
      {"vision", 1, 16, 72, 7296},
  };
  for (const auto& c : perf) run_case(c, 10, false);

  printf("\n%s (%d failing case(s))\n", fails == 0 ? "ALL PASS" : "SOME FAILED",
         fails);
  return fails == 0 ? 0 : 1;
}
