// ============================================================
// GQA dispatch A/B harness: NEW (production gqa.cpp) vs BACK (archived gqa_back.cpp).
//
// Drives the REAL runtime dispatch entry of each flow -- loaded from its own
// isolated DLL so the two kernel sets don't collide -- on identical GQA
// problems. For every case it:
//   * verifies each flow's output against a CPU fp32 reference,
//   * cross-checks NEW vs BACK output (should agree),
//   * reports per-dispatch latency (prefill ~ TTFT, decode ~ per-token),
//   * labels which code path NEW takes: our_* (optimized fused/flash kernels)
//     or ori_* (legacy decomposed hipBLASLt fallback in gqa_back.cpp). BACK is
//     always ori_* (it is the legacy implementation).
//
// The NEW DLL is built in PRODUCTION mode (gqa.cpp -DGQA_SLIM_PRODUCTION linked
// with gqa_back.cpp -DGQA_LEGACY_FALLBACK_BUILD), so cases the fused kernels do
// not implement (fp32, sliding window, smooth softmax, other head_dim,
// untemplated decode geometry) are routed to the legacy fallback instead of
// being rejected -- this is what lets us exercise "all scenarios".
//
// Problem setup mirrors the EP call site: separate Q/K/V, BNSD KV cache,
// in-place append driven by seqlens_k, causal mask, scale = 1/sqrt(d), no RoPE.
// (no_causal/bidirectional attention uses a different data model -- full-length
// key, no append -- and is validated by the e2e Whisper test, not here.)
// ============================================================

#define NOMINMAX // keep std::max/std::min usable (windows.h defines max/min macros)
#include <windows.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include "runtime_state_internal.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),        \
              __FILE__, __LINE__);                                             \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

using gqa_dispatch_fn = int (*)(
    RuntimeState *, int, void *, void *, void *, void *, void *, void *, void *,
    void *, void *, void *, void *, void *, void *, void *, void *, void *,
    void *, void *, int64_t, int64_t, float, int64_t, int64_t, float, int64_t,
    int64_t, int64_t, int64_t, int64_t, int64_t, int32_t, int64_t, int64_t,
    int64_t, int64_t, int64_t, int64_t);
using gqa_construct_fn = int8_t (*)(RuntimeState *, int32_t);

struct Flow {
  const char *name;
  HMODULE mod = nullptr;
  gqa_dispatch_fn dispatch = nullptr;
  gqa_construct_fn construct = nullptr;
  hipblasLtHandle_t lt = nullptr;
};

struct Case {
  const char *name;
  int B, H, G, D, sq, past;
  int elem;   // element_size_bytes: 2 = fp16, 4 = fp32
  int window; // local_window_size: -1/<=0 = none
  int smooth; // smooth_softmax attribute (0/1); sink factor = 0
  int sink;   // head_sink: 1 => provide a per-head [H] fp16 sink-logit buffer
};

// Mirror of gqa.cpp's decode-geometry gate (flash_decode_geometry_ok): the
// optimized flash_decode kernel is templated for HpG in {1,2,3,4,5,8,16} and
// head_dim in {64,128}.
static bool geometry_ok(int H, int G, int D) {
  if (D != 64 && D != 128)
    return false;
  if (G <= 0 || H % G != 0)
    return false;
  const int hpg = H / G;
  return hpg == 1 || hpg == 2 || hpg == 3 || hpg == 4 || hpg == 5 || hpg == 8 ||
         hpg == 16;
}

// Mirror of gqa.cpp::wrap_gqa_flash path selection: true => the optimized
// fused/flash kernels handle it (our_*); false => routed to the legacy
// decomposed fallback (ori_*).
static bool fused_supported(const Case &c) {
  const bool decode_geometry_ok = (c.sq != 1) || geometry_ok(c.H, c.G, c.D);
  return c.elem == 2 && c.window <= 0 && c.smooth != 1 && c.sink == 0 &&
         (c.D == 64 || c.D == 128) && decode_geometry_ok;
}

static double rel_l2(const std::vector<float> &a, const std::vector<float> &b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    num += d * d;
    den += (double)b[i] * b[i];
  }
  return std::sqrt(num / (den + 1e-12));
}

// CPU fp32 causal GQA reference over the FULL cache (BNSD K/V, length `total`).
// Q is BSHD [B,sq,H,D]; query s sits at global position past+s and attends to
// keys [kmin, past+s] where kmin = max(0, past+s-window+1) when window>0.
// Attention sink: when `sink` is non-empty (head_sink) the softmax denominator
// gets an extra term exp(sink[h] - m) with no V contribution; smooth_softmax is
// the same with sink[h] = 0. (ORT GQA attention-sink / smooth-softmax.)
static void cpu_reference(const std::vector<float> &Q,
                          const std::vector<float> &K,
                          const std::vector<float> &V, std::vector<float> &O,
                          int B, int H, int G, int D, int sq, int total,
                          int past, float scale, int window, int smooth,
                          const std::vector<float> &sink) {
  const int HPG = H / G;
  std::vector<float> scores(total);
  for (int b = 0; b < B; ++b)
    for (int hq = 0; hq < H; ++hq) {
      const int hkv = hq / HPG;
      for (int s = 0; s < sq; ++s) {
        const float *q = &Q[((size_t)(b * sq + s) * H + hq) * D];
        const int kmax = past + s;
        const int kmin = (window > 0) ? std::max(0, kmax - window + 1) : 0;
        float m = -1e30f;
        for (int k = kmin; k <= kmax; ++k) {
          const float *kp = &K[((size_t)(b * G + hkv) * total + k) * D];
          float dot = 0.0f;
          for (int e = 0; e < D; ++e)
            dot += q[e] * kp[e];
          scores[k] = dot * scale;
          if (scores[k] > m)
            m = scores[k];
        }
        float l = 0.0f;
        for (int k = kmin; k <= kmax; ++k) {
          scores[k] = std::exp(scores[k] - m);
          l += scores[k];
        }
        if (!sink.empty() || smooth) {
          const float sh = sink.empty() ? 0.0f : sink[hq];
          l += std::exp(sh - m); // sink term: denominator only, no V
        }
        const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        float *o = &O[((size_t)(b * sq + s) * H + hq) * D];
        for (int e = 0; e < D; ++e)
          o[e] = 0.0f;
        for (int k = kmin; k <= kmax; ++k) {
          const float *vp = &V[((size_t)(b * G + hkv) * total + k) * D];
          const float w = scores[k] * inv;
          for (int e = 0; e < D; ++e)
            o[e] += w * vp[e];
        }
      }
    }
}

// Upload a host fp32 buffer to a device buffer of `elem` bytes/element
// (fp16 down-convert or raw fp32 copy). Returns the device pointer.
static void *up_device(const std::vector<float> &src, int elem) {
  void *d = nullptr;
  const size_t n = src.size();
  HIP_CHECK(hipMalloc(&d, n * (size_t)elem));
  if (elem == 2) {
    std::vector<__half> h(n);
    for (size_t i = 0; i < n; ++i)
      h[i] = __float2half(src[i]);
    HIP_CHECK(hipMemcpy(d, h.data(), n * sizeof(__half), hipMemcpyHostToDevice));
  } else {
    HIP_CHECK(hipMemcpy(d, src.data(), n * sizeof(float), hipMemcpyHostToDevice));
  }
  return d;
}

// Download a device buffer of `elem` bytes/element into a host fp32 buffer.
static void down_device(void *d, std::vector<float> &dst, int elem) {
  const size_t n = dst.size();
  if (elem == 2) {
    std::vector<__half> h(n);
    HIP_CHECK(hipMemcpy(h.data(), d, n * sizeof(__half), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i)
      dst[i] = __half2float(h[i]);
  } else {
    HIP_CHECK(hipMemcpy(dst.data(), d, n * sizeof(float), hipMemcpyDeviceToHost));
  }
}

// Run one flow on one case. Returns latency (ms/dispatch); fills `outHost`.
static double run_flow(Flow &flow, const Case &c,
                       const std::vector<float> &Qf,
                       const std::vector<float> &Kf,
                       const std::vector<float> &Vf,
                       const std::vector<float> &sinkVals, hipStream_t stream,
                       int iters, std::vector<float> &outHost) {
  const int B = c.B, H = c.H, G = c.G, D = c.D, sq = c.sq, past = c.past;
  const int total = past + sq;
  const int elem = c.elem;

  const size_t qn = (size_t)B * sq * H * D;        // BSHD Q / O
  const size_t newn = (size_t)B * sq * G * D;      // BSHD new K/V (sq tokens)
  const size_t cachen = (size_t)B * G * total * D; // BNSD present cache

  // Host fp32 staging (converted to the requested element type on upload).
  std::vector<float> Qs(qn), newKs(newn), newVs(newn), PKs(cachen, 0.0f),
      PVs(cachen, 0.0f);
  for (size_t i = 0; i < qn; ++i)
    Qs[i] = Qf[i];
  // new tokens = cache positions [past, total)
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < sq; ++s)
      for (int g = 0; g < G; ++g)
        for (int e = 0; e < D; ++e) {
          const int t = past + s;
          const size_t src = ((size_t)(b * G + g) * total + t) * D + e;
          const size_t dst = ((size_t)(b * sq + s) * G + g) * D + e;
          newKs[dst] = Kf[src];
          newVs[dst] = Vf[src];
        }
  // Pre-load present cache [0, past) with the past tokens (decode only).
  for (int b = 0; b < B; ++b)
    for (int g = 0; g < G; ++g)
      for (int t = 0; t < past; ++t)
        for (int e = 0; e < D; ++e) {
          const size_t idx = ((size_t)(b * G + g) * total + t) * D + e;
          PKs[idx] = Kf[idx];
          PVs[idx] = Vf[idx];
        }

  void *dQ = up_device(Qs, elem);
  void *dNewK = up_device(newKs, elem);
  void *dNewV = up_device(newVs, elem);
  void *dPK = up_device(PKs, elem);
  void *dPV = up_device(PVs, elem);
  void *dO = nullptr;
  HIP_CHECK(hipMalloc(&dO, qn * (size_t)elem));
  int32_t *dSeq = nullptr;
  HIP_CHECK(hipMalloc(&dSeq, (size_t)B * sizeof(int32_t)));
  std::vector<int32_t> seq(B, total - 1); // ORT: seqlens_k = total_seq - 1
  HIP_CHECK(hipMemcpy(dSeq, seq.data(), (size_t)B * sizeof(int32_t),
                      hipMemcpyHostToDevice));

  // head_sink: optional per-head [H] fp16 buffer (null when not exercised).
  void *dSink = sinkVals.empty() ? nullptr : up_device(sinkVals, 2);

  // Per-flow RuntimeState (own workspace + op-state slot; shared stream/handle).
  OpState *slots[1] = {nullptr};
  RuntimeState st{};
  st.stream = stream;
  st.hipblas_handle = flow.lt;
  st.op_states = slots;
  st.num_op_states = 1;

  flow.construct(&st, 0);

  auto launch = [&]() {
    return flow.dispatch(
        &st, /*op_state_slot=*/0, dQ, dNewK, dNewV, /*past_key=*/nullptr,
        /*past_value=*/nullptr, dSeq, /*total_seq_len=*/nullptr,
        /*cos=*/nullptr, /*sin=*/nullptr, /*position_ids=*/nullptr,
        /*attention_bias=*/nullptr, /*head_sink=*/dSink, /*k_scale=*/nullptr,
        /*v_scale=*/nullptr, dO, dPK, dPV, /*output_qk=*/nullptr,
        /*num_heads=*/H, /*kv_num_heads=*/G, /*scale=*/0.0f, /*do_rotary=*/0,
        /*rotary_interleaved=*/0, /*softcap=*/0.0f,
        /*local_window_size=*/c.window,
        /*smooth_softmax=*/c.smooth, /*qk_output=*/0, /*k_quant_type=*/0,
        /*v_quant_type=*/0, /*kv_cache_bit_width=*/8, /*no_causal=*/0,
        /*batch_size=*/B, /*seq_len_q=*/sq, /*seq_len_kv=*/total,
        /*past_buf_seq=*/total, /*head_dim=*/D, /*element_size_bytes=*/elem);
  };

  int rc = launch();
  HIP_CHECK(hipDeviceSynchronize());
  outHost.assign(qn, 0.0f);
  if (rc != 0) {
    fprintf(stderr, "  [%s] dispatch returned %d\n", flow.name, rc);
  } else {
    down_device(dO, outHost, elem);
  }

  double ms = 0.0;
  if (rc == 0) {
    for (int i = 0; i < 10; ++i)
      launch();
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    HIP_CHECK(hipEventRecord(e0, stream));
    for (int i = 0; i < iters; ++i)
      launch();
    HIP_CHECK(hipEventRecord(e1, stream));
    HIP_CHECK(hipEventSynchronize(e1));
    float t = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&t, e0, e1));
    ms = t / iters;
    hipEventDestroy(e0);
    hipEventDestroy(e1);
  }

  if (st.workspace)
    hipFree(st.workspace);
  // op-state slot object is leaked intentionally (process-lifetime bench).
  hipFree(dQ);
  hipFree(dNewK);
  hipFree(dNewV);
  hipFree(dPK);
  hipFree(dPV);
  hipFree(dO);
  hipFree(dSeq);
  if (dSink)
    hipFree(dSink);
  return ms;
}

static bool load_flow(Flow &flow, const char *dll) {
  flow.mod = LoadLibraryA(dll);
  if (!flow.mod) {
    fprintf(stderr, "FAILED to load %s (err=%lu)\n", dll, GetLastError());
    return false;
  }
  flow.dispatch =
      reinterpret_cast<gqa_dispatch_fn>(GetProcAddress(flow.mod, "gqa_dispatch"));
  flow.construct = reinterpret_cast<gqa_construct_fn>(
      GetProcAddress(flow.mod, "gqa_construct"));
  if (!flow.dispatch || !flow.construct) {
    fprintf(stderr, "FAILED to resolve gqa_dispatch/gqa_construct in %s\n", dll);
    return false;
  }
  if (hipblasLtCreate(&flow.lt) != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "hipblasLtCreate failed for %s\n", flow.name);
    return false;
  }
  return true;
}

int main(int argc, char **argv) {
  int iters = 50;
  const char *only = nullptr; // run only cases whose name contains this substr
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--iters") && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--only") && i + 1 < argc)
      only = argv[++i];
  }

  Flow neu{"NEW"}, bak{"BACK"};
  if (!load_flow(neu, "gqa_new.dll") || !load_flow(bak, "gqa_back.dll"))
    return 2;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Fields: name,B,H,G,D,sq,past, elem, window, smooth, sink.
  // Defaults that mean "off": elem=2 (fp16), window=-1, smooth=0, sink=0.
  const Case cases[] = {
      // ===== FUSED-SUPPORTED (our_*): fp16 causal, d in {64,128} =====
      // ---- prefill (TTFT): NEW flash_prefill (v5/v7), BACK hipBLASLt. ----
      {"prefill llama-3.2-1b  ", 1, 32, 8, 64, 512, 0, 2, -1, 0, 0},
      {"prefill llama-3.2-1b  ", 1, 32, 8, 64, 2048, 0, 2, -1, 0, 0},
      {"prefill gpt-oss-20b   ", 1, 64, 8, 64, 2048, 0, 2, -1, 0, 0}, // HpG8
      {"prefill llama-3.1-8b  ", 1, 32, 8, 128, 2048, 0, 2, -1, 0, 0}, // HpG4 d128
      {"prefill MHA d128 HpG1 ", 1, 32, 32, 128, 512, 0, 2, -1, 0, 0}, // HpG1
      {"prefill chunked past>0", 1, 32, 8, 64, 256, 256, 2, -1, 0, 0}, // past>0
      {"prefill batch B2      ", 2, 32, 8, 64, 512, 0, 2, -1, 0, 0}, // B>1
      // ---- decode (per-token): NEW flash_decode, BACK fused/flash. ----
      // Cover every templated HpG in {1,2,3,4,8,16} and both d in {64,128}.
      {"decode  MHA d64 HpG1  ", 1, 16, 16, 64, 1, 2048, 2, -1, 0, 0},
      {"decode  llama2-7b MHA ", 1, 32, 32, 128, 1, 2048, 2, -1, 0, 0}, // HpG1 d128
      {"decode  HpG2 d64      ", 1, 16, 8, 64, 1, 2048, 2, -1, 0, 0},
      {"decode  HpG3 d64      ", 1, 24, 8, 64, 1, 2048, 2, -1, 0, 0},
      {"decode  llama-3.2-1b  ", 1, 32, 8, 64, 1, 128, 2, -1, 0, 0}, // HpG4 small skv
      {"decode  llama-3.2-1b  ", 1, 32, 8, 64, 1, 2048, 2, -1, 0, 0}, // HpG4 d64 WMMA
      {"decode  llama-3.1-8b  ", 1, 32, 8, 128, 1, 2048, 2, -1, 0, 0}, // HpG4 d128 WMMA
      {"decode  gpt_oss HpG8  ", 1, 64, 8, 64, 1, 512, 2, -1, 0, 0}, // HpG8 d64 WMMA
      {"decode  gpt_oss big-kv", 1, 64, 8, 64, 1, 4096, 2, -1, 0, 0}, // large skv bucket
      {"decode  HpG16 d64     ", 1, 16, 1, 64, 1, 2048, 2, -1, 0, 0},
      {"decode  batch B2      ", 2, 32, 8, 64, 1, 2048, 2, -1, 0, 0}, // B>1

      // ===== FALLBACK (ori_*): NEW routes to the legacy decomposed pipeline =====
      // ---- sliding window (gpt-oss sliding-attn layers): window>0 ----
      {"decode  gptoss WIN d64", 1, 64, 8, 64, 1, 2048, 2, 128, 0, 0},
      {"decode  gptoss WIN d128", 1, 32, 8, 128, 1, 2048, 2, 128, 0, 0},
      {"prefill gptoss WINDOW ", 1, 64, 8, 64, 1024, 0, 2, 128, 0, 0},
      // ---- smooth softmax (sink logit = 0) ----
      {"decode  gptoss SMOOTH ", 1, 64, 8, 64, 1, 2048, 2, -1, 1, 0},
      {"prefill gptoss SMOOTH ", 1, 64, 8, 64, 512, 0, 2, -1, 1, 0},
      // ---- head_sink (per-head learned sink logit; gpt-oss attention sinks) ----
      {"decode  gptoss SINK   ", 1, 64, 8, 64, 1, 2048, 2, -1, 0, 1},
      {"prefill gptoss SINK   ", 1, 64, 8, 64, 512, 0, 2, -1, 0, 1},
      // ---- window + sink combined (gpt-oss sliding layer with sink) ----
      {"decode  gptoss WIN+SNK", 1, 64, 8, 64, 1, 2048, 2, 128, 0, 1},
      // ---- other head_dim (d=96): not in {64,128} -> decomposed ----
      {"decode  d96 odd-dim   ", 1, 32, 8, 96, 1, 2048, 2, -1, 0, 0},
      {"prefill d96 odd-dim   ", 1, 32, 8, 96, 512, 0, 2, -1, 0, 0},
      // ---- untemplated DECODE geometry (HpG=7): decode -> decomposed ----
      {"decode  HpG7 odd-geom ", 1, 28, 4, 64, 1, 2048, 2, -1, 0, 0},
      // ---- fp32 (Whisper decoder self-attn analogue, causal) -> decomposed ----
      {"decode  fp32 whisper  ", 1, 20, 20, 64, 1, 1500, 4, -1, 0, 0},
      {"prefill fp32 whisper  ", 1, 20, 20, 64, 448, 0, 4, -1, 0, 0},

      // ===== EXTENDED SWEEP: more shapes per scenario (no gaps) =====
      // ---- fused prefill: seq-len sweep + extra geometries ----
      {"prefill seq128 d64    ", 1, 32, 8, 64, 128, 0, 2, -1, 0, 0},
      {"prefill seq1024 d64   ", 1, 32, 8, 64, 1024, 0, 2, -1, 0, 0},
      {"prefill seq4096 d64   ", 1, 32, 8, 64, 4096, 0, 2, -1, 0, 0},
      {"prefill HpG2 d128     ", 1, 16, 8, 128, 1024, 0, 2, -1, 0, 0},
      {"prefill HpG16 d64     ", 1, 16, 1, 64, 1024, 0, 2, -1, 0, 0},
      {"prefill big chunk past", 1, 32, 8, 128, 512, 1536, 2, -1, 0, 0},
      {"prefill batch B4      ", 4, 32, 8, 64, 512, 0, 2, -1, 0, 0},
      // ---- fused decode: skv bucket sweep (small->huge) ----
      {"decode  skv65 d64     ", 1, 32, 8, 64, 1, 64, 2, -1, 0, 0},
      {"decode  skv257 d64    ", 1, 32, 8, 64, 1, 256, 2, -1, 0, 0},
      {"decode  skv1025 d64   ", 1, 32, 8, 64, 1, 1024, 2, -1, 0, 0},
      {"decode  skv8193 d64   ", 1, 32, 8, 64, 1, 8192, 2, -1, 0, 0},
      {"decode  skv1025 d128  ", 1, 32, 8, 128, 1, 1024, 2, -1, 0, 0},
      {"decode  HpG8 d128     ", 1, 64, 8, 128, 1, 2048, 2, -1, 0, 0},
      {"decode  HpG16 d128    ", 1, 32, 2, 128, 1, 2048, 2, -1, 0, 0},
      {"decode  batch B4 d64  ", 4, 32, 8, 64, 1, 2048, 2, -1, 0, 0},
      // ---- llama-3.1-8b d128 skv threshold sweep (min_skv=256 boundary) ----
      // BACK: skv<256 -> fused_decode (serial), skv>=256 -> flash_decode.
      // NEW : always flash_decode_v2.  Expect NEW win only for skv<=255.
      {"decode  l31-8b skv128 ", 1, 32, 8, 128, 1, 127, 2, -1, 0, 0},
      {"decode  l31-8b skv192 ", 1, 32, 8, 128, 1, 191, 2, -1, 0, 0},
      {"decode  l31-8b skv255 ", 1, 32, 8, 128, 1, 254, 2, -1, 0, 0},
      {"decode  l31-8b skv256 ", 1, 32, 8, 128, 1, 255, 2, -1, 0, 0},
      {"decode  l31-8b skv257 ", 1, 32, 8, 128, 1, 256, 2, -1, 0, 0},
      {"decode  l31-8b skv512 ", 1, 32, 8, 128, 1, 511, 2, -1, 0, 0},
      // ---- qwen2.5-14b 40:8 (HpG=5) d128: now fused flash_decode_v2 ----
      // BACK has no HpG=5 flash decode (legacy gate = HpG 4/8 only) -> it runs
      // the slow decomposed hipBLASLt pipeline, so NEW should win big here.
      {"decode  qwen14b skv256", 1, 40, 8, 128, 1, 255, 2, -1, 0, 0},
      {"decode  qwen14b skv1k ", 1, 40, 8, 128, 1, 1023, 2, -1, 0, 0},
      {"decode  qwen14b skv2k ", 1, 40, 8, 128, 1, 2047, 2, -1, 0, 0},
      {"prefill qwen14b s512  ", 1, 40, 8, 128, 512, 0, 2, -1, 0, 0},
      // ---- fallback: sliding-window size sweep ----
      {"decode  WIN64 d64     ", 1, 64, 8, 64, 1, 2048, 2, 64, 0, 0},
      {"decode  WIN256 d64    ", 1, 64, 8, 64, 1, 2048, 2, 256, 0, 0},
      {"decode  WIN512 d128   ", 1, 32, 8, 128, 1, 2048, 2, 512, 0, 0},
      {"prefill WIN256 d128   ", 1, 32, 8, 128, 1024, 0, 2, 256, 0, 0},
      // ---- fallback: fp32 extra (d128, HpG4, prefill+decode) ----
      {"decode  fp32 d128     ", 1, 32, 8, 128, 1, 2048, 4, -1, 0, 0},
      {"prefill fp32 d128     ", 1, 32, 8, 128, 512, 0, 4, -1, 0, 0},
      {"decode  fp32 HpG8     ", 1, 64, 8, 64, 1, 2048, 4, -1, 0, 0},
      // ---- fallback: odd head_dim sweep (d=80 phi-style, d=128 ok but HpG odd) ----
      {"decode  d80 odd-dim   ", 1, 32, 8, 80, 1, 2048, 2, -1, 0, 0},
      {"prefill d80 odd-dim   ", 1, 32, 8, 80, 512, 0, 2, -1, 0, 0},
      {"decode  d256 big-dim  ", 1, 16, 4, 256, 1, 2048, 2, -1, 0, 0},
      // ---- untemplated decode geometry sweep (HpG 6,12 still fallback; HpG5
      //      is now fused via flash_decode_v2, kept here as a d64 cross-check) --
      {"decode  HpG5 d64      ", 1, 20, 4, 64, 1, 2048, 2, -1, 0, 0},
      {"decode  HpG6 odd-geom ", 1, 24, 4, 64, 1, 2048, 2, -1, 0, 0},
      {"decode  HpG12 odd-geom", 1, 24, 2, 64, 1, 2048, 2, -1, 0, 0},
      // ---- fallback: smooth/sink extra geometries ----
      {"decode  SMOOTH d128   ", 1, 32, 8, 128, 1, 2048, 2, -1, 1, 0},
      {"decode  SINK d128     ", 1, 32, 8, 128, 1, 2048, 2, -1, 0, 1},
      {"decode  WIN+SNK d128  ", 1, 32, 8, 128, 1, 2048, 2, 256, 0, 1},

      // ===== CI REPRO: gpt-oss-20b prefill TTFT @ 2048 (H64 G8 d64) =====
      // gpt-oss alternates sliding-window(128) and full-attn layers; sink on all.
      {"CI gptoss 2048 WIN+SNK", 1, 64, 8, 64, 2048, 0, 2, 128, 0, 1},
      {"CI gptoss 2048 SINK   ", 1, 64, 8, 64, 2048, 0, 2, -1, 0, 1},
      {"CI gptoss 2048 WINonly", 1, 64, 8, 64, 2048, 0, 2, 128, 0, 0},
      {"CI gptoss 2048 plain  ", 1, 64, 8, 64, 2048, 0, 2, -1, 0, 0},
  };

  printf("GQA dispatch A/B  (iters=%d)  rel-L2 vs CPU fp32 ref; thr=5e-3\n",
         iters);
  printf("speedup = BACK_ms / NEW_ms (>1 => NEW faster); NB = rel-L2(NEW,BACK)\n");
  printf("%-24s %4s %5s %3s %3s | %-12s %-9s %-8s %-4s | %-11s %-9s %-8s %-4s | "
         "%-9s %-7s\n",
         "case", "sq", "skv", "D", "hpg", "NEW path", "NEW L2", "NEW ms", "ok",
         "BACK path", "BACK L2", "BACK ms", "ok", "NB-L2", "speedup");
  printf("%s\n", std::string(134, '-').c_str());

  int fails = 0;
  for (const auto &c : cases) {
    if (only && !std::strstr(c.name, only))
      continue;
    const int total = c.past + c.sq;
    const float scale = 1.0f / std::sqrt((float)c.D);
    const size_t qn = (size_t)c.B * c.sq * c.H * c.D;
    const size_t cachen = (size_t)c.B * c.G * total * c.D;

    std::mt19937 rng(1234u + c.sq + c.D + c.past + c.H + c.window + c.smooth +
                     c.sink * 7u + c.B * 13u);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> Qf(qn), Kf(cachen), Vf(cachen), Oref(qn);
    for (auto &x : Qf)
      x = dist(rng);
    for (auto &x : Kf)
      x = dist(rng);
    for (auto &x : Vf)
      x = dist(rng);
    // head_sink: per-head [H] sink logits (empty when not exercised). Modest
    // range keeps the exp(sink-m) term well-conditioned in fp16.
    std::vector<float> sinkVals;
    if (c.sink) {
      sinkVals.resize(c.H);
      for (auto &x : sinkVals)
        x = dist(rng);
    }
    cpu_reference(Qf, Kf, Vf, Oref, c.B, c.H, c.G, c.D, c.sq, total, c.past,
                  scale, c.window, c.smooth, sinkVals);

    std::vector<float> oNew, oBak;
    double msNew = run_flow(neu, c, Qf, Kf, Vf, sinkVals, stream, iters, oNew);
    double msBak = run_flow(bak, c, Qf, Kf, Vf, sinkVals, stream, iters, oBak);

    const double errNew = rel_l2(oNew, Oref);
    const double errBak = rel_l2(oBak, Oref);
    const double errNB = rel_l2(oNew, oBak);
    const bool okNew = errNew < 5e-3;
    const bool okBak = errBak < 5e-3;
    if (!okNew || !okBak)
      ++fails;
    const double speedup = (msNew > 0.0) ? msBak / msNew : 0.0;

    const bool ours = fused_supported(c);
    const char *newPath = ours ? (c.sq == 1 ? "our_flashdec" : "our_flashpre")
                               : "ori_decomp";
    const char *bakPath = (c.sq == 1 ? "ori_decode" : "ori_prefill");

    printf("%-24s %4d %5d %3d %3d | %-12s %.1e %.4f  %-4s | %-11s %.1e %.4f  "
           "%-4s | %.1e  %.2fx\n",
           c.name, c.sq, total, c.D, c.H / c.G, newPath, errNew, msNew,
           okNew ? "PASS" : "FAIL", bakPath, errBak, msBak,
           okBak ? "PASS" : "FAIL", errNB, speedup);
  }

  printf("%s\n", std::string(134, '-').c_str());
  printf("%s (%d failing case(s)). our_* = optimized fused/flash kernels; "
         "ori_* = legacy decomposed fallback (gqa_back.cpp).\n",
         fails == 0 ? "ALL PASS" : "SOME FAILED", fails);

  hipStreamDestroy(stream);
  return fails == 0 ? 0 : 1;
}
