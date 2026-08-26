// ============================================================
// custom_kernels GQA flash *decode* with an additive attention mask.
//
// Covers the path branch 2 opens up: hip_gqa_flash_decode with a dense
// [bias_batch, bias_heads, 1, total_seq] fp16 bias folded into the online
// softmax. Verified against a CPU fp32 reference over the mask shapes the
// Gemma-4 26B graph actually emits:
//
//   zero      - all-zero mask; must reproduce the unmasked result exactly.
//   window    - sliding window, masked entries at the fp16 minimum (-65504),
//               which is what an HF export writes for a finite dtype min.
//   window_inf- sliding window, masked entries at -inf. This is the dangerous
//               one: a split whose keys are ALL masked leaves the FA-2 running
//               max at -inf, and (-inf) - (-inf) is NaN in the rescale.
//   random    - dense small biases, no masking, to catch plain indexing slips.
//   headwise  - per-head mask (bias_heads == H) rather than the broadcast one.
//
// Self-contained: random inputs generated in-process, no data files.
// ============================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

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
    const void* attn_bias,
    int attn_bias_batch, int attn_bias_heads, int attn_bias_kv_stride);

static constexpr int MAX_SPLITS = 64;

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),        \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

enum MaskKind { MASK_ZERO, MASK_WINDOW, MASK_WINDOW_INF, MASK_RANDOM };

struct Case {
  const char* name;
  int B, H, G, D, max_seq, total;
  MaskKind kind;
  int window;      // for the window kinds
  int per_head;    // 1 => bias_heads == H, 0 => bias_heads == 1
};

// Build the [bias_batch, bias_heads, 1, total] mask in fp32 (host reference
// units); the device copy is the fp16 narrowing of exactly these values.
static void build_bias(const Case& c, int bias_heads, std::mt19937& rng,
                       std::vector<float>& bias) {
  std::uniform_real_distribution<float> small(-2.0f, 2.0f);
  bias.assign((size_t)c.B * bias_heads * c.total, 0.0f);
  const float kHalfMin = -65504.0f;
  for (int b = 0; b < c.B; ++b) {
    for (int h = 0; h < bias_heads; ++h) {
      float* row = &bias[((size_t)b * bias_heads + h) * c.total];
      for (int kv = 0; kv < c.total; ++kv) {
        switch (c.kind) {
          case MASK_ZERO:
            row[kv] = 0.0f;
            break;
          case MASK_RANDOM:
            row[kv] = small(rng);
            break;
          case MASK_WINDOW:
            row[kv] = (kv < c.total - c.window) ? kHalfMin : 0.0f;
            break;
          case MASK_WINDOW_INF:
            row[kv] = (kv < c.total - c.window) ? -INFINITY : 0.0f;
            break;
        }
      }
    }
  }
}

// CPU fp32 reference: plain softmax over [0, eff) with the bias added to the
// pre-softmax score, matching the decomposed pipeline's Step 8b + Step 9.
static void cpu_reference(const Case& c, int bias_heads,
                          const std::vector<float>& Q,
                          const std::vector<float>& K,
                          const std::vector<float>& V,
                          const std::vector<float>& bias,
                          const std::vector<int>& seqlens, float scale,
                          std::vector<float>& O) {
  const int B = c.B, H = c.H, G = c.G, D = c.D, max_seq = c.max_seq;
  const int hpg = H / G;
  O.assign((size_t)B * H * D, 0.0f);
  for (int b = 0; b < B; ++b) {
    int raw = seqlens[b] + 1;
    int eff = raw < 0 ? 0 : (raw > max_seq ? max_seq : raw);
    for (int h = 0; h < H; ++h) {
      int g = h / hpg;
      const float* q = &Q[((size_t)b * H + h) * D];
      const float* brow =
          &bias[((size_t)b * bias_heads + (bias_heads == 1 ? 0 : h)) * c.total];

      float m = -INFINITY;
      std::vector<float> s(eff);
      for (int kv = 0; kv < eff; ++kv) {
        const float* k = &K[(((size_t)b * G + g) * max_seq + kv) * D];
        float dot = 0.0f;
        for (int e = 0; e < D; ++e) dot += q[e] * k[e];
        s[kv] = dot * scale + brow[kv];
        if (s[kv] > m) m = s[kv];
      }
      float* o = &O[((size_t)b * H + h) * D];
      if (!std::isfinite(m)) continue;  // every key masked out

      float l = 0.0f;
      std::vector<float> acc(D, 0.0f);
      for (int kv = 0; kv < eff; ++kv) {
        const float* v = &V[(((size_t)b * G + g) * max_seq + kv) * D];
        float p = std::exp(s[kv] - m);
        l += p;
        for (int e = 0; e < D; ++e) acc[e] += p * v[e];
      }
      float inv = 1.0f / (l < 1e-6f ? 1e-6f : l);
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

static bool has_nonfinite(const std::vector<float>& a) {
  for (float v : a)
    if (!std::isfinite(v)) return true;
  return false;
}

int main() {
  // Gemma-4 26B-A4B local-attention geometry (25 of 30 layers) plus two
  // smaller shapes that exercise other instantiated (D, HpG) kernels.
  std::vector<Case> cases = {
      {"gemma4-local zero      ", 1, 16, 8, 256, 16384, 2051, MASK_ZERO, 0, 0},
      {"gemma4-local random    ", 1, 16, 8, 256, 16384, 2051, MASK_RANDOM, 0, 0},
      {"gemma4-local window1024", 1, 16, 8, 256, 16384, 2051, MASK_WINDOW, 1024, 0},
      {"gemma4-local window-inf", 1, 16, 8, 256, 16384, 2051, MASK_WINDOW_INF, 1024, 0},
      {"gemma4-local 12k wininf", 1, 16, 8, 256, 16384, 12279, MASK_WINDOW_INF, 1024, 0},
      {"gemma4-local 12k window", 1, 16, 8, 256, 16384, 12279, MASK_WINDOW, 1024, 0},
      {"gemma4-local per-head  ", 1, 16, 8, 256, 16384, 2051, MASK_RANDOM, 0, 1},
      // Gemma-4's other 5 layers: global attention at twice the head width.
      // These carry the mask too -- the model exports is_causal=0 and folds
      // causal and padding into the bias on every layer, not just the sliding
      // ones -- so d=512 has to be correct WITH a bias, not merely reachable.
      {"gemma4-global zero     ", 1, 16, 8, 512, 16384, 2051, MASK_ZERO, 0, 0},
      {"gemma4-global random   ", 1, 16, 8, 512, 16384, 2051, MASK_RANDOM, 0, 0},
      {"gemma4-global 12k rand ", 1, 16, 8, 512, 16384, 12279, MASK_RANDOM, 0, 0},
      {"gemma4-global per-head ", 1, 16, 8, 512, 16384, 2051, MASK_RANDOM, 0, 1},
      // The window/bias agreement check at d=512: the same window expressed
      // twice must agree, which is what pins the bias indexing to absolute kv
      // position rather than split-relative at the widest EPT.
      {"gemma4-global window-inf", 1, 16, 8, 512, 16384, 2051, MASK_WINDOW_INF, 1024, 0},
      {"llama-3.1-8b random    ", 1, 32, 8, 128, 16384, 4096, MASK_RANDOM, 0, 0},
      {"llama-3.1-8b window-inf", 1, 32, 8, 128, 16384, 4096, MASK_WINDOW_INF, 512, 0},
      {"d64 hpg8 window-inf    ", 1, 64, 8, 64, 16384, 8192, MASK_WINDOW_INF, 256, 0},
  };

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  int failures = 0;

  for (const Case& c : cases) {
    const int B = c.B, H = c.H, G = c.G, D = c.D, max_seq = c.max_seq;
    const int bias_heads = c.per_head ? H : 1;
    const float scale = 1.0f / std::sqrt((float)D);

    std::vector<float> Q((size_t)B * H * D);
    std::vector<float> K((size_t)B * G * max_seq * D);
    std::vector<float> V((size_t)B * G * max_seq * D);
    for (auto& x : Q) x = dist(rng);
    // Only [0, total) is ever read; leave the tail zeroed to keep the host
    // allocation cheap at max_seq = 16384.
    for (int b = 0; b < B; ++b)
      for (int g = 0; g < G; ++g)
        for (int kv = 0; kv < c.total; ++kv)
          for (int e = 0; e < D; ++e) {
            size_t i = (((size_t)b * G + g) * max_seq + kv) * D + e;
            K[i] = dist(rng);
            V[i] = dist(rng);
          }

    std::vector<float> bias;
    build_bias(c, bias_heads, rng, bias);

    std::vector<int> seqlens(B, c.total - 1);

    // Narrow to fp16 on the host so the reference sees exactly the values the
    // kernel reads (otherwise fp16 rounding shows up as a false mismatch).
    auto to_half = [](const std::vector<float>& src) {
      std::vector<__half> dst(src.size());
      for (size_t i = 0; i < src.size(); ++i) dst[i] = __float2half(src[i]);
      return dst;
    };
    std::vector<__half> Qh = to_half(Q), Kh = to_half(K), Vh = to_half(V),
                        Bh = to_half(bias);
    for (size_t i = 0; i < Q.size(); ++i) Q[i] = __half2float(Qh[i]);
    for (size_t i = 0; i < K.size(); ++i) K[i] = __half2float(Kh[i]);
    for (size_t i = 0; i < V.size(); ++i) V[i] = __half2float(Vh[i]);
    for (size_t i = 0; i < bias.size(); ++i) bias[i] = __half2float(Bh[i]);

    std::vector<float> ref;
    cpu_reference(c, bias_heads, Q, K, V, bias, seqlens, scale, ref);

    __half *dQ, *dK, *dV, *dO, *dBias;
    float* dPart;
    int* dSeq;
    HIP_CHECK(hipMalloc(&dQ, Qh.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dK, Kh.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dV, Vh.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dBias, Bh.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dO, (size_t)B * H * D * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dPart,
                        (size_t)B * H * MAX_SPLITS * (D + 2) * sizeof(float)));
    HIP_CHECK(hipMalloc(&dSeq, B * sizeof(int)));
    HIP_CHECK(hipMemcpy(dQ, Qh.data(), Qh.size() * sizeof(__half),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dK, Kh.data(), Kh.size() * sizeof(__half),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV, Vh.data(), Vh.size() * sizeof(__half),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dBias, Bh.data(), Bh.size() * sizeof(__half),
                        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dSeq, seqlens.data(), B * sizeof(int),
                        hipMemcpyHostToDevice));

    // window is carried entirely by the mask, so the kernel runs unwindowed.
    HIP_CHECK((hipError_t)hip_gqa_flash_decode(
        nullptr, dQ, dK, dV, dO, dPart, B, H, G, D, c.total, max_seq,
        MAX_SPLITS, scale, dSeq, 0, nullptr, 0, HIP_KV_DTYPE_FP16, nullptr,
        nullptr, dBias, /*bias_batch=*/B, bias_heads, /*bias_kv_stride=*/c.total));
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> got((size_t)B * H * D);
    {
      std::vector<__half> tmp(got.size());
      HIP_CHECK(hipMemcpy(tmp.data(), dO, tmp.size() * sizeof(__half),
                          hipMemcpyDeviceToHost));
      for (size_t i = 0; i < tmp.size(); ++i) got[i] = __half2float(tmp[i]);
    }

    double err = rel_l2(got, ref);
    bool nan_out = has_nonfinite(got);
    bool ok = !nan_out && err < 5e-3;
    if (!ok) ++failures;
    printf("%s B%d H%d G%d(hpg%d) D%-3d total=%-6d bias_heads=%d | relL2=%.2e%s  %s\n",
           c.name, B, H, G, H / G, D, c.total, bias_heads, err,
           nan_out ? "  NON-FINITE OUTPUT" : "", ok ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(dQ));
    HIP_CHECK(hipFree(dK));
    HIP_CHECK(hipFree(dV));
    HIP_CHECK(hipFree(dBias));
    HIP_CHECK(hipFree(dO));
    HIP_CHECK(hipFree(dPart));
    HIP_CHECK(hipFree(dSeq));
  }

  printf("\n%s (%d failing case(s))\n", failures ? "FAILURES" : "ALL PASS",
         failures);
  return failures ? 1 : 0;
}
