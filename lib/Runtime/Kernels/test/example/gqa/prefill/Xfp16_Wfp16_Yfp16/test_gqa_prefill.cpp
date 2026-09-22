/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ============================================================
// custom_kernels GQA flash *prefill* (TTFT) test + benchmark.
//
// Verifies the ported FA-2 WMMA prefill kernels that gqa.cpp routes to on the
// fused-prefill fast path:
//   hip_gqa_flash_prefill_v5  (d == 64, gpt-oss / llama-3.2 geometry)
//   hip_gqa_flash_prefill_v7  (d == 128, llama-3.1 geometry)
//   hip_gqa_flash_prefill_v8  (d == 256, Qwen3.6 geometry)
// against a CPU fp32 causal-attention reference (correctness) and reports the
// per-prefill latency (the quantity that bounds TTFT).
//
// LUT (`make lut`, -DHIPDNN_LUT_LINKED_EXTERNALLY=1): each case resolves the
// offline table then dispatches hip_gqa_flash_prefill_v3_configured — the
// same host resolve + configured-kernel path as production real/gqa.cpp.
// Default `make test` is unchanged (unified hip_gqa_flash_prefill, online tune).
//
// Layout matches the EP fused-prefill call site (gqa.cpp): Q is BSHD
// [B,sq,Hq,d]; K/V cache is BNSD [B,G,max_seq,d]; O is BSHD [B,sq,Hq,d].
// Pure prefill: past_len = 0, total_seq = sq. Self-contained random inputs.
// ============================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <thread>
#include <vector>
#include <string>

#include <functional>

// ---- Tiny inline coverage-tier resolver + CSV writer (replaces
// example/common/coverage.h + csv_writer.h) ----
// Model: categorical situations (D dispatch v5/v7/v8, sink modes, window
// on/off, chunked-prefill past>0, the d128-must-decline case) are covered by
// every case in `cases[]` below regardless of tier; COVERAGE only picks how
// many of those (categorical, typical-shape) rows run, via a fixed index
// subset per tier (not a blind cross product).
namespace hipdnn_ep_test {
inline int resolveCoverageTier(int argc, char** argv, int default_tier = 3) {
  int tier = default_tier;
  if (const char* env = std::getenv("HIPDNN_UT_COVERAGE")) tier = std::atoi(env);
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--coverage") == 0 && i + 1 < argc)
      tier = std::atoi(argv[++i]);
  return tier < 1 ? 1 : (tier > 3 ? 3 : tier);
}

struct CsvRow {
  std::string shape, config;
  double time_ms = 0.0;
  double rel_l2 = 0.0;
  std::string verdict;
};
class CsvWriter {
 public:
  CsvWriter() {
    const char* path = std::getenv("HIPDNN_RESULTS_CSV");
    if (!path || !path[0]) return;
    path_ = path;
    bool need_header = true;
    if (FILE* probe = std::fopen(path_.c_str(), "rb")) {
      std::fseek(probe, 0, SEEK_END);
      need_header = std::ftell(probe) == 0;
      std::fclose(probe);
    }
    f_ = std::fopen(path_.c_str(), "a");
    if (f_ && need_header) {
      std::fprintf(f_, "op,leaf,arch,mode,shape,config,time_ms,relL2,verdict\n");
      std::fflush(f_);
    }
  }
  ~CsvWriter() { if (f_) std::fclose(f_); }
  void write(const CsvRow& r) {
    if (!f_) return;
    std::fprintf(f_, "%s,%s,%s,%s,%s,%s,%.6f,%.6e,%s\n",
                 env_or("HIPDNN_RESULTS_OP", "unknown"),
                 env_or("HIPDNN_RESULTS_LEAF", "unknown"),
                 env_or("HIPDNN_RESULTS_ARCH", "unknown"),
                 env_or("HIPDNN_RESULTS_MODE", "unknown"), r.shape.c_str(),
                 r.config.c_str(), r.time_ms, r.rel_l2, r.verdict.c_str());
    std::fflush(f_);
  }
 private:
  static const char* env_or(const char* name, const char* dflt) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? v : dflt;
  }
  std::string path_;
  FILE* f_ = nullptr;
};

inline std::vector<std::string> captureLogLines(
    const char* env_var, const std::string& tmp_path,
    const std::function<void()>& fn) {
  std::vector<std::string> lines;
  _putenv_s(env_var, "1");
  std::fflush(stderr);
  FILE* redirected = std::freopen(tmp_path.c_str(), "w", stderr);
  if (!redirected) { _putenv_s(env_var, ""); return lines; }
  fn();
  std::fflush(stderr);
  std::freopen("CON", "w", stderr);
  _putenv_s(env_var, "");
  std::ifstream in(tmp_path);
  std::string line;
  while (std::getline(in, line)) lines.push_back(line);
  in.close();
  std::remove(tmp_path.c_str());
  return lines;
}
}  // namespace hipdnn_ep_test

static bool kernel_ut_lookup_mode() {
  const char* mode = std::getenv("HIPDNN_KERNEL_UT_MODE");
  return !mode || std::strcmp(mode, "autotune") != 0;
}

#ifdef HIPDNN_LUT_LINKED_EXTERNALLY
#include "gqa_autotune.h"
#ifndef HIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS
// One-line C23 #embed of the real LUT .fb -- HIPDNN_LUT_FB is defined by a
// tiny Makefile-generated header (a plain #define, not a data file) so the
// path never has to survive hipcc's Windows -D quoting (which mangles
// embedded quote characters).
#include "lut_fb_path.h"
static const unsigned char kLutBlob0[] = {
#embed HIPDNN_LUT_FB
};
extern "C" const unsigned char* const kGqaLutBlobs[1]   = { kLutBlob0 };
extern "C" const size_t               kGqaLutBlobSizes[1] = { sizeof(kLutBlob0) };
extern "C" const size_t               kGqaLutBlobCount    = 1;
#endif  // HIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS

// Lookup-only prefill entry: same ABI as hip_gqa_flash_prefill plus the
// resolved v5/v7/v8 knobs (see gqa_kernel.hip).
extern "C" int hip_gqa_flash_prefill_v3_configured(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale, int local_window_size, const void* head_sink,
    int num_heads, int smooth_softmax,
    int m_tiles, int bkv, int nw, int mt, int nd);
#endif

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
extern "C" int hip_gqa_flash_prefill(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale, int local_window_size, const void* head_sink,
    int num_heads, int smooth_softmax);

extern "C" int hip_gqa_flash_prefill_v7(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, int B, int Hq, int G, int sq, int skv, int d, int max_seq,
    int past_len, float scale);

#ifdef HIPDNN_LUT_LINKED_EXTERNALLY
#include "gqa_autotune.h"

extern "C" int hip_gqa_flash_prefill_v3_configured(
    void* stream_ptr,
    const void* Q, const void* Kcache, const void* Vcache, void* O,
    int B, int Hq, int G, int sq, int skv, int d, int max_seq, int past_len,
    float scale, int local_window_size, const void* head_sink,
    int num_heads, int smooth_softmax,
    int m_tiles, int bkv, int nw, int mt, int nd);

static void* gqa_policy() {
  static void* p = hip_gqa_autotune_create(nullptr);
  return p;
}

static hipdnn_ep::GqaPrefillVariant prefill_variant(int d) {
  if (d == 64) return hipdnn_ep::GqaPrefillVariant::V5;
  if (d == 256) return hipdnn_ep::GqaPrefillVariant::V8;
  return hipdnn_ep::GqaPrefillVariant::V7;
}
#endif

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

struct Case {
  const char* name;
  int B, H, G, D, sq;
  int past;       // past_len; total_seq = past + sq. 0 = pure prefill.
  int sink_mode;  // SinkMode
  // Expect the kernel to decline (rc != 0) instead of computing. Used for the
  // shapes v3 must refuse so the runtime falls back to the decomposed path
  // rather than dropping the sink or the window.
  bool expect_reject;
  // Sliding window; <= 0 is full attention. Convention matches
  // causal_mask_kernel_impl: key k is masked when k < past_len + q - window + 1.
  int window;
};

// CPU fp32 reference: causal GQA attention. Q/O BSHD, K/V cache BNSD. O(sq^2)
// per (b,hq) -- threaded over the B*H (batch, query-head) pairs (each is
// fully independent) so the wider sq/past additions below (including the new
// sq=8192 rows) stay inside the shared ~30 min tier3 budget across all 9 UT
// leaves. Each thread gets its own `scores` scratch buffer.
static void cpu_reference(const std::vector<float>& Q,
                          const std::vector<float>& K,
                          const std::vector<float>& V, std::vector<float>& O,
                          int B, int H, int G, int D, int sq, int max_seq,
                          int past_len, float scale, int sink_mode,
                          const std::vector<float>& sink, int window) {
  const int HPG = H / G;
  const int total = past_len + sq;
  auto bhRange = [&](int idx0, int idx1) {
    std::vector<float> scores(total);
    for (int idx = idx0; idx < idx1; ++idx) {
      const int b = idx / H;
      const int hq = idx % H;
      const int hkv = hq / HPG;
      for (int s = 0; s < sq; ++s) {
        const float* q = &Q[((size_t)(b * sq + s) * H + hq) * D];
        const int kmax = past_len + s;  // causal: attend to keys 0..kmax
        const int kmin = (window > 0 && kmax - window + 1 > 0)
                             ? (kmax - window + 1)
                             : 0;
        float m = -1e30f;
        for (int k = kmin; k <= kmax; ++k) {
          const float* kp = &K[((size_t)(b * G + hkv) * max_seq + k) * D];
          float dot = 0.0f;
          for (int e = 0; e < D; ++e) dot += q[e] * kp[e];
          scores[k] = dot * scale;
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
  };
  const int total_bh = B * H;
  const unsigned hw = std::thread::hardware_concurrency();
  const unsigned nthreads = std::min<unsigned>(hw ? hw : 4u,
                                                total_bh > 0 ? static_cast<unsigned>(total_bh) : 1u);
  if (nthreads <= 1 || total_bh < 2) {
    bhRange(0, total_bh);
  } else {
    std::vector<std::thread> pool;
    const int chunk = (total_bh + static_cast<int>(nthreads) - 1) / static_cast<int>(nthreads);
    for (unsigned t = 0; t < nthreads; ++t) {
      const int i0 = static_cast<int>(t) * chunk;
      const int i1 = std::min(total_bh, i0 + chunk);
      if (i0 >= i1) break;
      pool.emplace_back(bhRange, i0, i1);
    }
    for (auto& th : pool) th.join();
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
                c.sink_mode, sinkf, c.window);

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
  bool use_lookup = false;
#ifdef HIPDNN_LUT_LINKED_EXTERNALLY
  hipdnn_ep::GqaPrefillResult lut_res{};
  use_lookup = kernel_ut_lookup_mode() && hip_gqa_autotune_table_loaded();
  if (use_lookup) {
    // This exactly mirrors real/gqa.cpp: resolve against the LUT embedded in
    // custom_kernels_<arch>.dll, then call the lookup-only kernel entry.
    hipdnn_ep::GqaPrefillRequest lut_req{};
    lut_req.variant = (D == 64)    ? hipdnn_ep::GqaPrefillVariant::V5
                      : (D == 128) ? hipdnn_ep::GqaPrefillVariant::V7
                                   : hipdnn_ep::GqaPrefillVariant::V8;
    lut_req.batch = B;
    lut_req.num_heads = H;
    lut_req.kv_num_heads = G;
    lut_req.head_dim = D;
    lut_req.seq_q = sq;
    lut_req.seq_kv = skv;
    lut_req.local_window = (c.window > 0) ? c.window : 0;
    void* lut_policy = hip_gqa_autotune_create(nullptr);
    hip_gqa_autotune_resolve_prefill(lut_policy, &lut_req, &lut_res);
    hip_gqa_autotune_destroy(lut_policy);
  }
  auto launch = [&]() {
    if (use_lookup) {
      return hip_gqa_flash_prefill_v3_configured(
          nullptr, dQ, dK, dV, dO, B, H, G, sq, skv, D, max_seq, past_len,
          scale, window_arg, sink_arg, H, smooth_arg, lut_res.config.m_tiles,
          lut_res.config.bkv, lut_res.config.nw, lut_res.config.mt,
          lut_res.config.nd);
    }
    return hip_gqa_flash_prefill(nullptr, dQ, dK, dV, dO, B, H, G, sq, skv, D,
                                  max_seq, past_len, scale, window_arg,
                                  sink_arg, H, smooth_arg);
  };
#else
  auto launch = [&]() {
    return hip_gqa_flash_prefill(nullptr, dQ, dK, dV, dO, B, H, G, sq, skv, D,
                                    max_seq, past_len, scale, window_arg,
                                    sink_arg, H, smooth_arg);
  };
#endif

  // The op's [gqa-cfg] diagnostic names the config it dispatched. It latches on
  // at its first read and then prints once per dispatch, so every launch below
  // runs inside a capture window to keep those lines off the console.
  std::string launched_cfg;
  int rc = 0;
  {
    const std::vector<std::string> lines = hipdnn_ep_test::captureLogLines(
        "HIPDNN_GQA_AUTOTUNE_LOG", "out/_cfg_capture.tmp", [&]() {
          rc = launch();  // first call self-tunes (or resolves from the LUT)
          HIP_CHECK(hipDeviceSynchronize());
        });
    for (const std::string& line : lines) {
      const size_t knobs = line.find("-> ");
      if (line.find("[gqa-cfg]") != std::string::npos &&
          knobs != std::string::npos)
        launched_cfg = line.substr(knobs + 3);
    }
  }
  if (c.expect_reject) {
    const bool ok = (rc != 0);
    printf("%-16s B%d H%d G%d(hpg%d) D%-3d sq=%-5d past=%-5d %-6s w=%-5d | rc=%d (expected decline)  %s\n",
           c.name, B, H, G, H / G, D, sq, past_len, sink_tag, c.window, rc,
           ok ? "PASS" : "FAIL");
    {
      // A declined combination still gets a row: results.csv is the record of
      // every case that ran, and "no config" is the expected outcome here.
      using namespace hipdnn_ep_test;
      char shape_buf[96];
      std::snprintf(shape_buf, sizeof(shape_buf), "%s_H%d_G%d_D%d_sq%d_past%d",
                    c.name, H, G, D, sq, past_len);
      CsvWriter csv;
      CsvRow row;
      row.shape = shape_buf;
      row.config = "declined";
      row.verdict = ok ? "PASS" : "FAIL";
      csv.write(row);
    }
    hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO); hipFree(dSink);
    return ok;
  }
  if (rc != 0) { fprintf(stderr, "%s: kernel returned %d\n", c.name, rc); return false; }

  std::vector<__half> Oh(qn);
  HIP_CHECK(hipMemcpy(Oh.data(), dO, qn * sizeof(__half), hipMemcpyDeviceToHost));
  std::vector<float> Oout(qn);
  for (size_t i = 0; i < qn; ++i) Oout[i] = __half2float(Oh[i]);
  const double err = rel_l2(Oout, Oref);

  float ms = 0.0f;
  hipdnn_ep_test::captureLogLines(
      "HIPDNN_GQA_AUTOTUNE_LOG", "out/_cfg_capture.tmp", [&]() {
        for (int i = 0; i < 10; ++i) launch();
        HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t e0, e1;
        HIP_CHECK(hipEventCreate(&e0));
        HIP_CHECK(hipEventCreate(&e1));
        HIP_CHECK(hipEventRecord(e0));
        for (int i = 0; i < iters; ++i) launch();
        HIP_CHECK(hipEventRecord(e1));
        HIP_CHECK(hipEventSynchronize(e1));
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
        HIP_CHECK(hipEventDestroy(e0));
        HIP_CHECK(hipEventDestroy(e1));
      });
  ms /= iters;

  const bool pass = err < 2e-3;
  printf("%-16s B%d H%d G%d(hpg%d) D%-3d sq=%-5d past=%-5d %-6s w=%-5d | relL2=%.2e  latency=%.4f ms  %s (v%d)\n",
         c.name, B, H, G, H / G, D, sq, past_len, sink_tag, c.window, err, ms,
         pass ? "PASS" : "FAIL", D == 64 ? 5 : (D == 256 ? 8 : 7));

  {
    using namespace hipdnn_ep_test;
    char shape_buf[96];
    std::snprintf(shape_buf, sizeof(shape_buf), "%s_H%d_G%d_D%d_sq%d_past%d",
                  c.name, H, G, D, sq, past_len);
    CsvWriter csv;
#ifdef HIPDNN_LUT_LINKED_EXTERNALLY
    CsvRow row;
    row.shape = shape_buf;
    if (use_lookup) {
      char config_buf[96];
      std::snprintf(config_buf, sizeof(config_buf),
                    "lookup:m_tiles=%d bkv=%d nw=%d mt=%d nd=%d",
                    lut_res.config.m_tiles, lut_res.config.bkv,
                    lut_res.config.nw, lut_res.config.mt, lut_res.config.nd);
      row.config = config_buf;
    } else {
      row.config =
          launched_cfg.empty() ? "unlogged" : "autotune:" + launched_cfg;
    }
    row.time_ms = ms;
    row.rel_l2 = err;
    row.verdict = pass ? "PASS" : "FAIL";
    csv.write(row);
#else
    CsvRow row;
    row.shape = shape_buf;
    row.config = launched_cfg.empty() ? "unlogged" : "autotune:" + launched_cfg;
    row.time_ms = ms;
    row.rel_l2 = err;
    row.verdict = pass ? "PASS" : "FAIL";
    csv.write(row);
#endif
  }

  hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO); hipFree(dSink);
  return pass;
}

int main(int argc, char** argv) {
  const int coverage_tier = hipdnn_ep_test::resolveCoverageTier(argc, argv);

  int iters = 100;
  // "custom" single-shape mode, mirroring the i8 prefill test's --h/--g/--d/--sq
  // (plus --b/--past/--window, which the i8 variant doesn't expose).
  Case single = {"custom", 1, 32, 8, 128, 512, 0, kSinkNone, false, 0};
  bool have_single = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](int& v) { if (i + 1 < argc) v = std::atoi(argv[++i]); };
    if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters = std::atoi(argv[++i]);
    else if (a == "--b") { next(single.B); have_single = true; }
    else if (a == "--h") { next(single.H); have_single = true; }
    else if (a == "--g") { next(single.G); have_single = true; }
    else if (a == "--d") { next(single.D); have_single = true; }
    else if (a == "--sq") { next(single.sq); have_single = true; }
    else if (a == "--past") { next(single.past); have_single = true; }
    else if (a == "--window") { next(single.window); have_single = true; }
  }

  if (have_single) {
    bool ok = run_case(single, iters);
    printf("\n%s (%d failing case(s))\n", ok ? "ALL PASS" : "SOME FAILED", ok ? 0 : 1);
    return ok ? 0 : 1;
  }

  const Case cases[] = {
      // Qwen3.6-35B-A3B text decoder: d=256 routes to the v8 kernel, which no
      // other case covered. sq=1000 is deliberately not a multiple of the 16-row
      // Q tile or the 16-key KV tile, so it exercises the partial tiles that 512
      // and 2048 both skip.
      {"qwen3.6-d256", 1, 16, 2, 256, 512,  0,    kSinkNone,    false, 0},
      {"qwen3.6-d256", 1, 16, 2, 256, 1000, 0,    kSinkNone,    false, 0},
      {"qwen3.6-d256", 1, 16, 2, 256, 2048, 0,    kSinkNone,    false, 0},
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
      // Window at d==128 is implemented (prefill v5). Check relL2 like gpt_oss-win.
      {"llama-win-d128",1, 32, 8, 128, 512,  0,    kSinkNone,    false, 128},

      // ---- Widened shape coverage: short (sq=128) and long (sq=8192, pure
      // prefill) prompts, appended so the existing 0-28 indices above are
      // unchanged. sq=128 is cheap (O(sq^2) reference) so it is spread across
      // most scenario types; sq=8192 pure-prefill (past=0, so this is the
      // "long single prompt" case, distinct from the existing past=8192
      // chunked-prefill rows above which keep sq short) is expensive even
      // threaded, so only 2 representative (D=64, D=128) rows are added. ----
      {"qwen3.6-d256",   1, 16, 2, 256, 128,  0,    kSinkNone,    false, 0},
      {"gpt_oss-20b",    1, 64, 8,  64, 128,  0,    kSinkNone,    false, 0},
      {"llama-3.1-8b",   1, 32, 8, 128, 128,  0,    kSinkNone,    false, 0},
      {"gpt_oss-sink",   1, 64, 8,  64, 128,  0,    kSinkPerHead, false, 0},
      {"gpt_oss-win",    1, 64, 8,  64, 128,  0,    kSinkNone,    false, 128},
      {"gpt_oss-both",   1, 64, 8,  64, 128,  0,    kSinkBoth,    false, 0},
      {"llama-win-d128", 1, 32, 8, 128, 128,  0,    kSinkNone,    false, 128},
      {"llama-sink-d128",1, 32, 8, 128, 128,  0,    kSinkPerHead, true,  0},
      {"gpt_oss-20b",    1, 64, 8,  64, 8192, 0,    kSinkNone,    false, 0},
      {"llama-3.1-8b",   1, 32, 8, 128, 8192, 0,    kSinkNone,    false, 0},
  };
  const size_t kNumCases = sizeof(cases) / sizeof(cases[0]);

  // tier1 (12 rows) still touches D{64,128,256}, every sink_mode, window
  // on/off, chunked-prefill (past>0), and the d128-must-decline case; now
  // also includes 2 of the new sq=128 rows (the cheapest shape) so tier1's
  // "smallest shape per scenario" rule covers the widened range too.
  static const size_t kTier1[] = {0, 3, 7, 9, 11, 13, 15, 17, 18, 25, 29, 30};
  // tier2 (~65% by case count, but includes ALL the cheap sq=128 additions
  // since they cost almost nothing -- see kTier2 vs tier3 wall time in
  // RESULT.md) still excludes the 2 new expensive sq=8192 rows (kept
  // tier3-only, a deliberate budget trade-off documented in RESULT.md).
  static const size_t kTier2[] = {0,  1,  3,  4,  6,  7,  9,  10, 11, 12,
                                  13, 14, 15, 16, 17, 18, 19, 20, 21, 25,
                                  26, 28, 29, 30, 31, 32, 33, 34, 35, 36};
  const size_t* idxs = coverage_tier == 1 ? kTier1
                     : coverage_tier == 2 ? kTier2
                                          : nullptr;
  const size_t n_idxs = coverage_tier == 1 ? sizeof(kTier1) / sizeof(kTier1[0])
                      : coverage_tier == 2 ? sizeof(kTier2) / sizeof(kTier2[0])
                                           : kNumCases;
  printf("coverage=%d -> running %zu/%zu gqa_prefill cases\n", coverage_tier,
         n_idxs, kNumCases);

  int fails = 0;
  for (size_t i = 0; i < n_idxs; ++i) {
    const Case& c = idxs ? cases[idxs[i]] : cases[i];
    if (!run_case(c, iters)) ++fails;
  }
  printf("\n%s (%d failing case(s))\n", fails == 0 ? "ALL PASS" : "SOME FAILED", fails);
  return fails == 0 ? 0 : 1;
}
