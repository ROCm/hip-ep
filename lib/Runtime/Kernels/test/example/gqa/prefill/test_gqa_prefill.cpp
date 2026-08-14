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

// Wide entry: same as v2 plus attention sinks / smooth softmax and a sliding
// window.
extern "C" int hip_gqa_flash_prefill_v3(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale, int local_window_size, const void* head_sink,
    int num_heads, int smooth_softmax);

extern "C" int hip_gqa_flash_prefill_v7(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale);

// Widest entry: v3 plus the additive attention mask an is_causal=0 export
// carries. no_causal must be set with a mask -- the mask replaces the implicit
// causal triangle rather than adding to it.
extern "C" int hip_gqa_flash_prefill_v4(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale, int local_window_size, const void* head_sink,
    int num_heads, int smooth_softmax, const void* attn_bias, int bias_batch,
    int bias_heads, int no_causal);

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
//
// kSinkBoth sends a sink tensor AND smooth_softmax=1, which is the only
// combination the runtime ever produces: gqa.cpp derives smooth from
// (head_sink != nullptr || smooth_softmax == 1) and then passes both. head_sink
// takes precedence, so the reference folds in the per-head logit alone; a kernel
// that also added the smooth term would double-count the denominator and fail.
enum SinkMode {
  kSinkNone = 0,
  kSinkPerHead = 1,
  kSinkSmooth = 2,
  kSinkBoth = 3
};

// How the additive attention mask is built, i.e. what an is_causal=0 export
// puts in its `attn_mask` input. All non-None modes send no_causal=1, so the
// mask is the ONLY masking the kernel applies -- which is exactly what makes
// these cases load-bearing: if the kernel kept its own causal triangle, the
// Causal cases would still pass while Free would fail, and if it dropped the
// mask entirely, Causal would fail against the same reference the no-mask cases
// already use.
enum MaskMode {
  kMaskNone = 0,
  // Causality expressed as 0 / -65504 in the mask instead of by the kernel.
  // Must reproduce the plain causal reference exactly.
  kMaskCausal = 1,
  // Causality AND a sliding window in the mask, which is what Gemma-4 ships:
  // its 25 sliding layers export is_causal=0 and put a 1024-token window in
  // the mask, with no window attribute anywhere.
  kMaskCausalWindow = 2,
  // Smoothly varying finite mask over the full square, no -inf anywhere. Only
  // this mode can catch a mask that is scaled wrongly (by `scale`, or by
  // kLog2e twice): a saturating -65504 entry maps to zero weight under any
  // positive scaling, so the Causal modes above cannot see such a bug.
  kMaskFree = 3
};

struct Case {
  const char* name;
  int B, H, G, D, sq;
  int past;       // past_len; total_seq = past + sq. 0 = pure prefill.
  int sink_mode;  // SinkMode
  // Expect the kernel to decline (rc != 0) instead of computing. Used for the
  // shapes v3/v4 must refuse so the runtime falls back to the decomposed path
  // rather than dropping the sink, the window or the mask.
  bool expect_reject;
  // Sliding window; <= 0 is full attention. Convention matches
  // causal_mask_kernel_impl: key k is masked when k < past_len + q - window + 1.
  int window;
  int mask_mode = kMaskNone;  // MaskMode
  // Send the mask as [1, H, sq, skv] rather than the head-broadcast
  // [1, 1, sq, skv] every real export uses, so the non-broadcast indexing is
  // covered too. Per-head values differ, so a kernel that ignored bias_heads
  // and always read plane 0 would fail this and pass every broadcast case.
  bool mask_per_head = false;
  // Send a mask with no_causal=0, i.e. ask for the mask AND the implicit
  // triangle. Only valid with expect_reject: serving it would apply just one of
  // the two, and which one is not observable from the output.
  bool mask_keeps_causal = false;
};

// CPU fp32 reference: causal GQA attention. Q/O BSHD, K/V cache BNSD.
//
// With a mask (`mask` non-empty) the causal triangle and the window are NOT
// applied here: the mask is the only masking, mirroring what the kernel does
// under EXT_MASK. The row span therefore opens to the whole KV extent and the
// mask value is added in natural units, right after alpha=scale, which is the
// order the decomposed runtime path uses (GEMM, then add_attention_bias, then
// softmax). `mask` is [mask_heads, sq, total], mask_heads being 1 or H.
static void cpu_reference(const std::vector<float>& Q,
                          const std::vector<float>& K,
                          const std::vector<float>& V, std::vector<float>& O,
                          int B, int H, int G, int D, int sq, int max_seq,
                          int past_len, float scale, int sink_mode,
                          const std::vector<float>& sink, int window,
                          const std::vector<float>& mask, int mask_heads) {
  const int HPG = H / G;
  const int total = past_len + sq;
  const bool has_mask = !mask.empty();
  std::vector<float> scores(total);
  for (int b = 0; b < B; ++b) {
    for (int hq = 0; hq < H; ++hq) {
      const int hkv = hq / HPG;
      const size_t mask_plane =
          has_mask ? (size_t)((mask_heads == 1) ? 0 : hq) * sq * total : 0;
      for (int s = 0; s < sq; ++s) {
        const float* q = &Q[((size_t)(b * sq + s) * H + hq) * D];
        const int kmax = has_mask ? (total - 1) : (past_len + s);
        const int kmin = (!has_mask && window > 0 && kmax - window + 1 > 0)
                             ? (kmax - window + 1)
                             : 0;
        float m = -1e30f;
        for (int k = kmin; k <= kmax; ++k) {
          const float* kp = &K[((size_t)(b * G + hkv) * max_seq + k) * D];
          float dot = 0.0f;
          for (int e = 0; e < D; ++e) dot += q[e] * kp[e];
          scores[k] = dot * scale;
          if (has_mask) scores[k] += mask[mask_plane + (size_t)s * total + k];
          if (scores[k] > m) m = scores[k];
        }
        float l = 0.0f;
        for (int k = kmin; k <= kmax; ++k) {
          scores[k] = std::exp(scores[k] - m);
          l += scores[k];
        }
        if (sink_mode == kSinkPerHead || sink_mode == kSinkBoth)
          l += std::exp(sink[hq] - m);
        else if (sink_mode == kSinkSmooth)
          l += std::exp(0.0f - m);
        const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        float* o = &O[((size_t)(b * sq + s) * H + hq) * D];
        for (int e = 0; e < D; ++e) o[e] = 0.0f;
        for (int k = kmin; k <= kmax; ++k) {
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

  // Build the mask in fp16 first and read it back for the reference, so the
  // comparison is not charged for the host's rounding -- fp16 is what the
  // kernel loads. -65504 is the fp16 minimum and is exactly representable,
  // which is what the runtime's own masks use.
  const int mask_heads = c.mask_per_head ? H : 1;
  const size_t mask_n = (size_t)mask_heads * sq * skv;
  std::vector<__half> maskh;
  std::vector<float> maskf;
  if (c.mask_mode != kMaskNone) {
    maskh.resize(mask_n);
    maskf.resize(mask_n);
    const int win = (c.window > 0) ? c.window : 0;
    for (int h = 0; h < mask_heads; ++h)
      for (int s = 0; s < sq; ++s)
        for (int k = 0; k < skv; ++k) {
          const int kmax = past_len + s;
          float v;
          if (c.mask_mode == kMaskFree) {
            // Finite everywhere and O(1), so every entry actually reaches the
            // softmax and a mis-scaled mask changes the result.
            v = 0.75f * std::sin(0.05f * (float)k + 0.11f * (float)s +
                                 0.7f * (float)h);
          } else {
            const bool keep =
                (k <= kmax) && !(c.mask_mode == kMaskCausalWindow && win > 0 &&
                                 k < kmax - win + 1);
            v = keep ? 0.0f : -65504.0f;
          }
          maskh[((size_t)h * sq + s) * skv + k] = __float2half(v);
        }
    for (size_t i = 0; i < mask_n; ++i) maskf[i] = __half2float(maskh[i]);
  }

  cpu_reference(Qf, Kf, Vf, Oref, B, H, G, D, sq, max_seq, past_len, scale,
                c.sink_mode, sinkf, c.window, maskf, mask_heads);

  std::vector<__half> Qh(qn), Kh(kn), Vh(kn);
  for (size_t i = 0; i < qn; ++i) Qh[i] = __float2half(Qf[i]);
  for (size_t i = 0; i < kn; ++i) { Kh[i] = __float2half(Kf[i]); Vh[i] = __float2half(Vf[i]); }

  __half *dQ, *dK, *dV, *dO, *dSink, *dMask = nullptr;
  HIP_CHECK(hipMalloc(&dQ, qn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dK, kn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dV, kn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dO, qn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dSink, (size_t)H * sizeof(__half)));
  HIP_CHECK(hipMemcpy(dQ, Qh.data(), qn * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dK, Kh.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dV, Vh.data(), kn * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dSink, sinkh.data(), (size_t)H * sizeof(__half), hipMemcpyHostToDevice));
  if (!maskh.empty()) {
    HIP_CHECK(hipMalloc(&dMask, mask_n * sizeof(__half)));
    HIP_CHECK(hipMemcpy(dMask, maskh.data(), mask_n * sizeof(__half),
                        hipMemcpyHostToDevice));
  }

  auto free_all = [&]() {
    hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO); hipFree(dSink);
    if (dMask) hipFree(dMask);
  };

  // Route through the widest entry (the one the runtime calls); it dispatches
  // v5 (D==64) / v7 (D==128) / v8 (D==256) internally.
  const void* sink_arg =
      (c.sink_mode == kSinkPerHead || c.sink_mode == kSinkBoth)
          ? (const void*)dSink
          : nullptr;
  const int smooth_arg =
      (c.sink_mode == kSinkSmooth || c.sink_mode == kSinkBoth) ? 1 : 0;
  const int window_arg = (c.window > 0) ? c.window : -1;
  const char* sink_tag = (c.sink_mode == kSinkPerHead) ? "sink"
                       : (c.sink_mode == kSinkSmooth)  ? "smooth"
                       : (c.sink_mode == kSinkBoth)    ? "both"
                                                       : "-";
  const char* mask_tag = (c.mask_mode == kMaskCausal)       ? "causal"
                       : (c.mask_mode == kMaskCausalWindow) ? "cauwin"
                       : (c.mask_mode == kMaskFree)         ? "free"
                                                            : "-";
  // A mask goes with no_causal=1 and suppresses the separate window argument,
  // because the window is inside the mask -- passing both would be a
  // combination v4 declines by design.
  const bool has_mask = (c.mask_mode != kMaskNone);
  auto launch = [&]() {
    return hip_gqa_flash_prefill_v4(nullptr, dQ, dK, dV, dO, B, H, G, sq, skv, D,
                                    max_seq, past_len, scale,
                                    has_mask ? -1 : window_arg, sink_arg, H,
                                    smooth_arg, dMask, 1, mask_heads,
                                    (has_mask && !c.mask_keeps_causal) ? 1 : 0);
  };

  int rc = launch();  // first call self-tunes
  HIP_CHECK(hipDeviceSynchronize());
  if (c.expect_reject) {
    const bool ok = (rc != 0);
    printf("%-16s B%d H%d G%d(hpg%d) D%-3d sq=%-5d past=%-5d %-6s w=%-5d m=%-6s | rc=%d (expected decline)  %s\n",
           c.name, B, H, G, H / G, D, sq, past_len, sink_tag, c.window, mask_tag,
           rc, ok ? "PASS" : "FAIL");
    free_all();
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

  const bool pass = err < 2e-3;
  printf("%-16s B%d H%d G%d(hpg%d) D%-3d sq=%-5d past=%-5d %-6s w=%-5d m=%-6s | relL2=%.2e  latency=%.4f ms  %s (v%d)\n",
         c.name, B, H, G, H / G, D, sq, past_len, sink_tag, c.window, mask_tag,
         err, ms, pass ? "PASS" : "FAIL",
         D == 64 ? 5 : (D == 256 ? 8 : 7));

  hipEventDestroy(e0); hipEventDestroy(e1);
  free_all();
  return pass;
}

int main(int argc, char** argv) {
  int iters = 100;
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);

  const Case cases[] = {
      // No-sink regression set (must stay as accurate as before).
      {"gpt_oss-20b",  1, 64, 8,  64, 512,  0,    kSinkNone,    false, 0},
      {"gpt_oss-20b",  1, 64, 8,  64, 2048, 0,    kSinkNone,    false, 0},
      {"llama-3.2-1b", 1, 32, 8,  64, 512,  0,    kSinkNone,    false, 0},
      {"llama-3.2-1b", 1, 32, 8,  64, 2048, 0,    kSinkNone,    false, 0},
      {"llama-3.1-8b", 1, 32, 8, 128, 512,  0,    kSinkNone,    false, 0},
      {"llama-3.1-8b", 1, 32, 8, 128, 2048, 0,    kSinkNone,    false, 0},
      // Sink set at the real gpt-oss geometry (H=64, G=8, d=64), including
      // chunked prefill (past > 0), which is what a 16k prompt actually runs.
      {"gpt_oss-sink",  1, 64, 8,  64, 512,  0,    kSinkPerHead, false, 0},
      {"gpt_oss-sink",  1, 64, 8,  64, 2048, 0,    kSinkPerHead, false, 0},
      {"gpt_oss-sink",  1, 64, 8,  64, 512,  512,  kSinkPerHead, false, 0},
      {"gpt_oss-sink",  1, 64, 8,  64, 512,  8192, kSinkPerHead, false, 0},
      {"gpt_oss-smooth",1, 64, 8,  64, 512,  0,    kSinkSmooth,  false, 0},
      {"gpt_oss-smooth",1, 64, 8,  64, 512,  512,  kSinkSmooth,  false, 0},
      // Sink tensor AND smooth_softmax=1 together. The cases above each set one
      // argument, but gqa.cpp only ever sets both at once, so without this the
      // exact combination the runtime sends is untested.
      {"gpt_oss-both",  1, 64, 8,  64, 512,  0,    kSinkBoth,    false, 0},
      {"gpt_oss-both",  1, 64, 8,  64, 512,  512,  kSinkBoth,    false, 0},
      // A sink must not silently apply at d == 128: v3 declines so the runtime
      // falls back to the decomposed path, which does implement it.
      {"llama-sink-d128",1, 32, 8, 128, 512, 0,    kSinkPerHead, true,  0},

      // Sliding window, gpt-oss geometry, window=128 as the model ships.
      // Window alone first, so a window bug cannot hide behind the sink.
      {"gpt_oss-win",   1, 64, 8,  64, 512,  0,    kSinkNone,    false, 128},
      {"gpt_oss-win",   1, 64, 8,  64, 2048, 0,    kSinkNone,    false, 128},
      // Chunked: past deeper than the window is the case where whole KV tiles
      // must be skipped rather than merely masked.
      {"gpt_oss-win",   1, 64, 8,  64, 512,  512,  kSinkNone,    false, 128},
      {"gpt_oss-win",   1, 64, 8,  64, 512,  8192, kSinkNone,    false, 128},
      // Window not aligned to any BKV (32/64), to catch an off-by-one in the
      // start-tile clamp.
      {"gpt_oss-win",   1, 64, 8,  64, 512,  1000, kSinkNone,    false, 100},
      // Window wider than the whole sequence must equal full attention.
      {"gpt_oss-win-big",1, 64, 8, 64, 512,  0,    kSinkNone,    false, 4096},
      // Window == 1 is the degenerate case: each query sees only itself.
      {"gpt_oss-win1",  1, 64, 8,  64, 512,  512,  kSinkNone,    false, 1},
      // Window together with the sink, which is what gpt-oss actually runs on
      // its 12 sliding layers.
      {"gpt_oss-win+sk",1, 64, 8,  64, 512,  0,    kSinkPerHead, false, 128},
      {"gpt_oss-win+sk",1, 64, 8,  64, 512,  8192, kSinkPerHead, false, 128},
      // The full production configuration of a gpt-oss sliding layer: window,
      // sink tensor and smooth together, deep enough to skip whole KV tiles.
      {"gpt_oss-win+bo",1, 64, 8,  64, 512,  8192, kSinkBoth,    false, 128},
      // A window must not silently apply at d == 128 either.
      {"llama-win-d128",1, 32, 8, 128, 512,  0,    kSinkNone,    true,  128},

      // ---- Additive attention mask (is_causal=0 exports), Gemma-4 geometry:
      // H=16, G=8, d=256 on the 25 sliding layers. Before this the mask forced
      // every such layer onto the decomposed pipeline.
      //
      // Causality via the mask must reproduce the plain causal reference. This
      // is the load-bearing pair: it fails if the kernel drops the mask, and it
      // also fails if the kernel keeps its own triangle *and* adds the mask,
      // because kMaskFree below then disagrees.
      {"gemma4-mask",   1, 16, 8, 256, 512,  0,    kSinkNone,    false, 0, kMaskCausal},
      {"gemma4-mask",   1, 16, 8, 256, 2048, 0,    kSinkNone,    false, 0, kMaskCausal},
      // Chunked prefill: past_len > 0 shifts the diagonal, so a mask indexed by
      // absolute rather than in-chunk query position fails here and nowhere else.
      {"gemma4-mask",   1, 16, 8, 256, 512,  512,  kSinkNone,    false, 0, kMaskCausal},
      {"gemma4-mask",   1, 16, 8, 256, 512,  2048, kSinkNone,    false, 0, kMaskCausal},
      // sq not a multiple of any ROWS (MT*16), so the last query tile is partial
      // and the mask must not be read past its end.
      {"gemma4-mask-odd",1, 16, 8, 256, 500, 300,  kSinkNone,    false, 0, kMaskCausal},
      // The production sliding layer: causality and a 1024-token window both
      // inside the mask, deep enough that most of the square is masked out.
      {"gemma4-win",    1, 16, 8, 256, 512,  2048, kSinkNone,    false, 1024, kMaskCausalWindow},
      {"gemma4-win",    1, 16, 8, 256, 2048, 0,    kSinkNone,    false, 1024, kMaskCausalWindow},
      // Window unaligned to BKV (32/64) and narrower than a tile.
      {"gemma4-win",    1, 16, 8, 256, 512,  1000, kSinkNone,    false, 100,  kMaskCausalWindow},
      // Finite mask over the whole square: the only case that can catch a mask
      // scaled by `scale`, or by kLog2e twice, since saturating -65504 entries
      // give zero weight under any positive scaling.
      {"gemma4-free",   1, 16, 8, 256, 512,  0,    kSinkNone,    false, 0, kMaskFree},
      {"gemma4-free",   1, 16, 8, 256, 512,  512,  kSinkNone,    false, 0, kMaskFree},
      // Per-head mask planes ([1,H,sq,skv] instead of head-broadcast): a kernel
      // that ignored bias_heads and always read plane 0 passes every case above.
      {"gemma4-free-ph",1, 16, 8, 256, 512,  0,    kSinkNone,    false, 0, kMaskFree, true},
      {"gemma4-mask-ph",1, 16, 8, 256, 512,  256,  kSinkNone,    false, 0, kMaskCausal, true},

      // Declines. Each is a combination the decomposed pipeline implements and
      // this kernel does not; serving any of them would drop a term silently.
      // A mask at d == 128 has no instantiation (EXT_MASK lives in v8, d=256).
      {"mask-d128",     1, 32, 8, 128, 512,  0,    kSinkNone,    true,  0, kMaskCausal},
      {"mask-d64",      1, 64, 8,  64, 512,  0,    kSinkNone,    true,  0, kMaskCausal},
      // A mask together with a sink: the masked instantiation has no sink term.
      {"mask+sink",     1, 16, 8, 256, 512,  0,    kSinkPerHead, true,  0, kMaskCausal},
      {"mask+smooth",   1, 16, 8, 256, 512,  0,    kSinkSmooth,  true,  0, kMaskCausal},
      // A mask asked for alongside the implicit triangle (is_causal=1 with a
      // mask). The kernel applies one masking rule, not two, so it must decline
      // rather than pick one -- from the output you cannot tell which it picked.
      {"mask+causal",   1, 16, 8, 256, 512,  0,    kSinkNone,    true,  0, kMaskCausal, false, true},
  };
  int fails = 0;
  for (const auto& c : cases) if (!run_case(c, iters)) ++fails;
  printf("\n%s (%d failing case(s))\n", fails == 0 ? "ALL PASS" : "SOME FAILED", fails);
  return fails == 0 ? 0 : 1;
}
