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

// Wide entry: same as v2 plus attention sinks / smooth softmax.
extern "C" int hip_gqa_flash_prefill_v3(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale, int local_window_size, const void* head_sink,
    int num_heads, int smooth_softmax);

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

// Sink handling, matching softmax_f32_to_out_kernel exactly: the row max is
// taken over the scores only (the sink does NOT participate), and the sink
// contributes a single exp(s - max) term to the denominator. kSinkSmooth is the
// smooth_softmax case, i.e. a sink logit of 0 with no sink tensor.
enum SinkMode { kSinkNone = 0, kSinkPerHead = 1, kSinkSmooth = 2 };

struct Case {
  const char* name;
  int B, H, G, D, sq;
  int past;       // past_len; total_seq = past + sq. 0 = pure prefill.
  int sink_mode;  // SinkMode
  // Expect the kernel to decline (rc != 0) instead of computing. Used for the
  // shapes v3 must refuse so the runtime falls back to the decomposed path
  // rather than dropping the sink.
  bool expect_reject;
};

// CPU fp32 reference: causal GQA attention. Q/O BSHD, K/V cache BNSD.
static void cpu_reference(const std::vector<float>& Q,
                          const std::vector<float>& K,
                          const std::vector<float>& V, std::vector<float>& O,
                          int B, int H, int G, int D, int sq, int max_seq,
                          int past_len, float scale, int sink_mode,
                          const std::vector<float>& sink) {
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
        if (sink_mode == kSinkPerHead)
          l += std::exp(sink[hq] - m);
        else if (sink_mode == kSinkSmooth)
          l += std::exp(0.0f - m);
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
  const int past_len = c.past;
  const int skv = past_len + sq;   // total_seq
  const int max_seq = skv;         // cache buffer holds exactly total_seq
  const float scale = 1.0f / std::sqrt((float)D);

  const size_t qn = (size_t)B * sq * H * D;
  const size_t kn = (size_t)B * G * max_seq * D;
  std::mt19937 rng(1234 + sq + D + past_len + c.sink_mode);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> Qf(qn), Kf(kn), Vf(kn), Oref(qn);
  for (auto& x : Qf) x = dist(rng);
  for (auto& x : Kf) x = dist(rng);
  for (auto& x : Vf) x = dist(rng);

  // gpt-oss ships sink logits around O(1); span a wider range so a sign or
  // scaling error in the log2-space conversion cannot hide. Round-trip through
  // fp16 first, because that is what the kernel reads -- otherwise the
  // comparison would charge the kernel for the host's rounding.
  std::vector<__half> sinkh(H);
  std::vector<float> sinkf(H);
  for (int h = 0; h < H; ++h) {
    sinkh[h] = __float2half(-2.0f + 4.0f * (float)h / (float)H);
    sinkf[h] = __half2float(sinkh[h]);
  }

  cpu_reference(Qf, Kf, Vf, Oref, B, H, G, D, sq, max_seq, past_len, scale,
                c.sink_mode, sinkf);

  std::vector<__half> Qh(qn), Kh(kn), Vh(kn);
  for (size_t i = 0; i < qn; ++i) Qh[i] = __float2half(Qf[i]);
  for (size_t i = 0; i < kn; ++i) { Kh[i] = __float2half(Kf[i]); Vh[i] = __float2half(Vf[i]); }

  __half *dQ, *dK, *dV, *dO, *dSink;
  HIP_CHECK(hipMalloc(&dQ, qn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dK, kn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dV, kn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dO, qn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dSink, (size_t)H * sizeof(__half)));
  HIP_CHECK(hipMemcpy(dQ, Qh.data(), qn * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dK, Kh.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dV, Vh.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dSink, sinkh.data(), (size_t)H * sizeof(__half), hipMemcpyHostToDevice));

  // Route through the unified entry (same path the runtime takes); it dispatches
  // v5 (D==64) / v7 (D==128) internally.
  const void* sink_arg = (c.sink_mode == kSinkPerHead) ? (const void*)dSink : nullptr;
  const int smooth_arg = (c.sink_mode == kSinkSmooth) ? 1 : 0;
  auto launch = [&]() {
    return hip_gqa_flash_prefill_v3(nullptr, dQ, dK, dV, dO, B, H, G, sq, skv, D,
                                    max_seq, past_len, scale,
                                    /*local_window_size=*/-1, sink_arg, H,
                                    smooth_arg);
  };

  int rc = launch();  // first call self-tunes
  HIP_CHECK(hipDeviceSynchronize());
  if (c.expect_reject) {
    const bool ok = (rc != 0);
    printf("%-16s B%d H%d G%d(hpg%d) D%-3d sq=%-5d past=%-5d %-6s | rc=%d (expected decline)  %s\n",
           c.name, B, H, G, H / G, D, sq, past_len,
           c.sink_mode == kSinkPerHead ? "sink" : "-", rc, ok ? "PASS" : "FAIL");
    hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO); hipFree(dSink);
    return ok;
  }
  if (rc != 0) { fprintf(stderr, "%s: kernel returned %d\n", c.name, rc); return false; }

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

  const char* sink_tag = (c.sink_mode == kSinkPerHead) ? "sink"
                       : (c.sink_mode == kSinkSmooth)  ? "smooth"
                                                       : "-";
  const bool pass = err < 2e-3;
  printf("%-16s B%d H%d G%d(hpg%d) D%-3d sq=%-5d past=%-5d %-6s | relL2=%.2e  latency=%.4f ms  %s (v%d)\n",
         c.name, B, H, G, H / G, D, sq, past_len, sink_tag, err, ms,
         pass ? "PASS" : "FAIL", D == 64 ? 5 : 7);

  hipEventDestroy(e0); hipEventDestroy(e1);
  hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO); hipFree(dSink);
  return pass;
}

int main(int argc, char** argv) {
  int iters = 100;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);

  const Case cases[] = {
      // No-sink regression set (must stay as accurate as before).
      {"gpt_oss-20b",  1, 64, 8,  64, 512,  0,    kSinkNone,    false},
      {"gpt_oss-20b",  1, 64, 8,  64, 2048, 0,    kSinkNone,    false},
      {"llama-3.2-1b", 1, 32, 8,  64, 512,  0,    kSinkNone,    false},
      {"llama-3.2-1b", 1, 32, 8,  64, 2048, 0,    kSinkNone,    false},
      {"llama-3.1-8b", 1, 32, 8, 128, 512,  0,    kSinkNone,    false},
      {"llama-3.1-8b", 1, 32, 8, 128, 2048, 0,    kSinkNone,    false},
      // Sink set at the real gpt-oss geometry (H=64, G=8, d=64), including
      // chunked prefill (past > 0), which is what a 16k prompt actually runs.
      {"gpt_oss-sink",  1, 64, 8,  64, 512,  0,    kSinkPerHead, false},
      {"gpt_oss-sink",  1, 64, 8,  64, 2048, 0,    kSinkPerHead, false},
      {"gpt_oss-sink",  1, 64, 8,  64, 512,  512,  kSinkPerHead, false},
      {"gpt_oss-sink",  1, 64, 8,  64, 512,  8192, kSinkPerHead, false},
      {"gpt_oss-smooth",1, 64, 8,  64, 512,  0,    kSinkSmooth,  false},
      {"gpt_oss-smooth",1, 64, 8,  64, 512,  512,  kSinkSmooth,  false},
      // A sink must not silently apply at d == 128: v3 declines so the runtime
      // falls back to the decomposed path, which does implement it.
      {"llama-sink-d128",1, 32, 8, 128, 512, 0,    kSinkPerHead, true},
  };
  int fails = 0;
  for (const auto& c : cases) if (!run_case(c, iters)) ++fails;
  printf("\n%s (%d failing case(s))\n", fails == 0 ? "ALL PASS" : "SOME FAILED", fails);
  return fails == 0 ? 0 : 1;
}
