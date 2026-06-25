// ============================================================
// GQA dispatch A/B harness: NEW (slim, gqa.cpp) vs BACK (decomposed, gqa_back.cpp).
//
// Drives the REAL runtime dispatch entry (wrap_gqa_flash / wrap_group_query_attention) of each
// flow -- loaded from its own isolated DLL so the two kernel sets don't collide
// -- on identical GQA problems. For every case it:
//   * verifies each flow's output against a CPU fp32 causal-GQA reference,
//   * cross-checks NEW vs BACK output (should agree),
//   * reports per-dispatch latency (prefill ~ TTFT, decode ~ per-token).
//
// Problem setup mirrors the EP call site: separate Q/K/V (fp16), BNSD KV cache,
// in-place append driven by seqlens_k, causal mask, scale = 1/sqrt(d), no RoPE.
//   prefill: past=0, sq>1  -> NEW hip_gqa_flash_prefill  vs BACK hipBLASLt GEMMs
//   decode : sq=1, past>0   -> both -> hip_gqa_flash_decode / hip_gqa_fused_decode
// ============================================================

#include <windows.h>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>

#include "runtime_state_internal.h"

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
};

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
// keys [0, past+s].
static void cpu_reference(const std::vector<float> &Q,
                          const std::vector<float> &K,
                          const std::vector<float> &V, std::vector<float> &O,
                          int B, int H, int G, int D, int sq, int total,
                          int past, float scale) {
  const int HPG = H / G;
  std::vector<float> scores(total);
  for (int b = 0; b < B; ++b)
    for (int hq = 0; hq < H; ++hq) {
      const int hkv = hq / HPG;
      for (int s = 0; s < sq; ++s) {
        const float *q = &Q[((size_t)(b * sq + s) * H + hq) * D];
        const int kmax = past + s;
        float m = -1e30f;
        for (int k = 0; k <= kmax; ++k) {
          const float *kp = &K[((size_t)(b * G + hkv) * total + k) * D];
          float dot = 0.0f;
          for (int e = 0; e < D; ++e)
            dot += q[e] * kp[e];
          scores[k] = dot * scale;
          if (scores[k] > m)
            m = scores[k];
        }
        float l = 0.0f;
        for (int k = 0; k <= kmax; ++k) {
          scores[k] = std::exp(scores[k] - m);
          l += scores[k];
        }
        const float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        float *o = &O[((size_t)(b * sq + s) * H + hq) * D];
        for (int e = 0; e < D; ++e)
          o[e] = 0.0f;
        for (int k = 0; k <= kmax; ++k) {
          const float *vp = &V[((size_t)(b * G + hkv) * total + k) * D];
          const float w = scores[k] * inv;
          for (int e = 0; e < D; ++e)
            o[e] += w * vp[e];
        }
      }
    }
}

// Run one flow on one case. Returns latency (ms/dispatch); fills `outHost`.
static double run_flow(Flow &flow, const Case &c,
                       const std::vector<float> &Qf,
                       const std::vector<float> &Kf,
                       const std::vector<float> &Vf, hipStream_t stream,
                       int iters, std::vector<float> &outHost) {
  const int B = c.B, H = c.H, G = c.G, D = c.D, sq = c.sq, past = c.past;
  const int total = past + sq;

  const size_t qn = (size_t)B * sq * H * D;     // BSHD Q / O
  const size_t newn = (size_t)B * sq * G * D;   // BSHD new K/V (sq tokens)
  const size_t cachen = (size_t)B * G * total * D; // BNSD present cache

  // Host fp16 staging.
  std::vector<__half> Qh(qn), newKh(newn), newVh(newn), PKh(cachen, __half(0)),
      PVh(cachen, __half(0));
  for (size_t i = 0; i < qn; ++i)
    Qh[i] = __float2half(Qf[i]);
  // new tokens = cache positions [past, total)
  for (int b = 0; b < B; ++b)
    for (int s = 0; s < sq; ++s)
      for (int g = 0; g < G; ++g)
        for (int e = 0; e < D; ++e) {
          const int t = past + s;
          const size_t src = ((size_t)(b * G + g) * total + t) * D + e;
          const size_t dst = ((size_t)(b * sq + s) * G + g) * D + e;
          newKh[dst] = __float2half(Kf[src]);
          newVh[dst] = __float2half(Vf[src]);
        }
  // Pre-load present cache [0, past) with the past tokens (decode only).
  for (int b = 0; b < B; ++b)
    for (int g = 0; g < G; ++g)
      for (int t = 0; t < past; ++t)
        for (int e = 0; e < D; ++e) {
          const size_t idx = ((size_t)(b * G + g) * total + t) * D + e;
          PKh[idx] = __float2half(Kf[idx]);
          PVh[idx] = __float2half(Vf[idx]);
        }

  __half *dQ, *dNewK, *dNewV, *dPK, *dPV, *dO;
  int32_t *dSeq;
  HIP_CHECK(hipMalloc(&dQ, qn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dNewK, newn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dNewV, newn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dPK, cachen * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dPV, cachen * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dO, qn * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dSeq, (size_t)B * sizeof(int32_t)));
  HIP_CHECK(hipMemcpy(dQ, Qh.data(), qn * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dNewK, newKh.data(), newn * sizeof(__half),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dNewV, newVh.data(), newn * sizeof(__half),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dPK, PKh.data(), cachen * sizeof(__half),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dPV, PVh.data(), cachen * sizeof(__half),
                      hipMemcpyHostToDevice));
  std::vector<int32_t> seq(B, total - 1); // ORT: seqlens_k = total_seq - 1
  HIP_CHECK(hipMemcpy(dSeq, seq.data(), (size_t)B * sizeof(int32_t),
                      hipMemcpyHostToDevice));

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
        /*attention_bias=*/nullptr, /*head_sink=*/nullptr, /*k_scale=*/nullptr,
        /*v_scale=*/nullptr, dO, dPK, dPV, /*output_qk=*/nullptr,
        /*num_heads=*/H, /*kv_num_heads=*/G, /*scale=*/0.0f, /*do_rotary=*/0,
        /*rotary_interleaved=*/0, /*softcap=*/0.0f, /*local_window_size=*/-1,
        /*smooth_softmax=*/0, /*qk_output=*/0, /*k_quant_type=*/0,
        /*v_quant_type=*/0, /*kv_cache_bit_width=*/8, /*no_causal=*/0,
        /*batch_size=*/B, /*seq_len_q=*/sq, /*seq_len_kv=*/total,
        /*past_buf_seq=*/total, /*head_dim=*/D, /*element_size_bytes=*/2);
  };

  int rc = launch();
  HIP_CHECK(hipDeviceSynchronize());
  if (rc != 0) {
    fprintf(stderr, "  [%s] dispatch returned %d\n", flow.name, rc);
    outHost.assign(qn, 0.0f);
  } else {
    std::vector<__half> Oh(qn);
    HIP_CHECK(hipMemcpy(Oh.data(), dO, qn * sizeof(__half),
                        hipMemcpyDeviceToHost));
    outHost.resize(qn);
    for (size_t i = 0; i < qn; ++i)
      outHost[i] = __half2float(Oh[i]);
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
  for (int i = 1; i < argc; ++i)
    if (!std::strcmp(argv[i], "--iters") && i + 1 < argc)
      iters = std::atoi(argv[++i]);

  Flow neu{"NEW"}, bak{"BACK"};
  if (!load_flow(neu, "gqa_new.dll") || !load_flow(bak, "gqa_back.dll"))
    return 2;

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  const Case cases[] = {
      // --- prefill (TTFT): past=0, sq>1.  NEW=flash_prefill  BACK=hipBLASLt ---
      {"prefill llama-3.2-1b", 1, 32, 8, 64, 512, 0},
      {"prefill llama-3.2-1b", 1, 32, 8, 64, 2048, 0},
      {"prefill gpt-oss-20b", 1, 64, 8, 64, 512, 0},
      {"prefill gpt-oss-20b", 1, 64, 8, 64, 2048, 0},
      {"prefill llama-3.1-8b", 1, 32, 8, 128, 512, 0},
      {"prefill llama-3.1-8b", 1, 32, 8, 128, 2048, 0},
      // --- decode (per-token): sq=1.  NEW = single flash_decode path (scalar|
      //     WMMA by geometry, no KV-depth branch); BACK = legacy fused/decomposed.
      {"decode  llama-3.2-1b", 1, 32, 8, 64, 1, 128},   // HpG=4 d64  small KV
      {"decode  llama-3.2-1b", 1, 32, 8, 64, 1, 2048},  // HpG=4 d64  long KV
      {"decode  llama-3.1-8b", 1, 32, 8, 128, 1, 512},  // HpG=4 d128
      {"decode  llama-3.1-8b", 1, 32, 8, 128, 1, 2048},
      // gpt_oss D64/HpG8 small-Sk: probe wmma-vs-scalar autotune crossover.
      {"decode  gpt_oss     ", 1, 64, 8, 64, 1, 256},
      {"decode  gpt_oss     ", 1, 64, 8, 64, 1, 512},
      // MHA (HpG=1): exercises the newly-templated scalar flash_decode path that
      // replaced the legacy fused_decode for one-head-per-group geometries.
      {"decode  llama2-7b MHA", 1, 32, 32, 128, 1, 128},
      {"decode  llama2-7b MHA", 1, 32, 32, 128, 1, 2048},
  };

  printf("GQA dispatch A/B  (iters=%d)  rel-L2 vs CPU fp32 ref; thr=5e-3\n",
         iters);
  printf("%-24s %6s %4s %4s %5s | %-22s | %-22s | NEWvsBACK\n", "case", "sq",
         "skv", "D", "hpg", "NEW (slim flash)", "BACK (decomposed)");
  printf("%s\n", std::string(120, '-').c_str());

  int fails = 0;
  for (const auto &c : cases) {
    const int total = c.past + c.sq;
    const float scale = 1.0f / std::sqrt((float)c.D);
    const size_t qn = (size_t)c.B * c.sq * c.H * c.D;
    const size_t cachen = (size_t)c.B * c.G * total * c.D;

    std::mt19937 rng(1234u + c.sq + c.D + c.past);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> Qf(qn), Kf(cachen), Vf(cachen), Oref(qn);
    for (auto &x : Qf)
      x = dist(rng);
    for (auto &x : Kf)
      x = dist(rng);
    for (auto &x : Vf)
      x = dist(rng);
    cpu_reference(Qf, Kf, Vf, Oref, c.B, c.H, c.G, c.D, c.sq, total, c.past,
                  scale);

    std::vector<float> oNew, oBak;
    double msNew = run_flow(neu, c, Qf, Kf, Vf, stream, iters, oNew);
    double msBak = run_flow(bak, c, Qf, Kf, Vf, stream, iters, oBak);

    const double errNew = rel_l2(oNew, Oref);
    const double errBak = rel_l2(oBak, Oref);
    const double errNB = rel_l2(oNew, oBak);
    const bool okNew = errNew < 5e-3;
    const bool okBak = errBak < 5e-3;
    if (!okNew || !okBak)
      ++fails;
    const double speedup = (msNew > 0.0) ? msBak / msNew : 0.0;

    printf("%-24s %6d %4d %4d %5d | L2=%.2e %.4fms %-4s | L2=%.2e %.4fms %-4s "
           "| %.2e (%.2fx)\n",
           c.name, c.sq, total, c.D, c.H / c.G, errNew, msNew,
           okNew ? "PASS" : "FAIL", errBak, msBak, okBak ? "PASS" : "FAIL",
           errNB, speedup);
  }

  printf("%s\n", std::string(120, '-').c_str());
  printf("%s (%d failing case(s)). speedup = BACK_ms / NEW_ms (>1 means NEW "
         "faster)\n",
         fails == 0 ? "ALL PASS" : "SOME FAILED", fails);

  hipStreamDestroy(stream);
  return fails == 0 ? 0 : 1;
}
