/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// ============================================================
// Single-dispatch GQA decode (hip_gqa_flash_decode_fused) test + benchmark.
//
// The fused kernel rotates Q and the new K (RoPE at position_ids[b], or at
// seqlens_k[b] without position_ids), appends K/V at row seqlens_k[b] of an
// in-place BNSD cache, runs split-K attention and merges the splits in the
// same launch. Each case checks against an fp64 CPU reference:
//   - the output O (relative L2),
//   - the appended cache rows (the rotated K and the copied V, bit-exact to
//     the fp16 rounding hip_gqa_rope applies).
// Timing compares one fused launch against the multi-dispatch sequence it
// replaces (rope Q, rope K, append K, append V, split-K decode + reduce),
// both rotating over cache copies so every launch reads DRAM.
//
//   test_gqa_decode_fused.exe                   correctness suite + timing
//   test_gqa_decode_fused.exe --h 40 --g 10 --d 128 --len 2048 [--iters N]
//                                               one shape
//   --splits N                                  fused launch split count
//                                               (default: derived)
//   --cap N                                     cache capacity (default: len)
//   --qscale X                                  scale Q by X to sharpen the
//                                               softmax (default: 1)
// One shape checks both the fused and the multi-dispatch path.
// Self-contained: random inputs generated in-process, no data files.
// ============================================================

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

extern "C" int hip_gqa_flash_decode_fused_supported(int H, int G, int d);
extern "C" int hip_gqa_flash_decode_fused(
    void* stream, const void* Q, const void* K_new, const void* V_new,
    void* Kcache, void* Vcache, void* O, void* partials_workspace,
    int* tickets, const void* cos_cache, const void* sin_cache,
    const void* position_ids, int B, int H, int G, int d, int max_seq,
    int splits, float scale, const void* seqlens_k, int local_window_size,
    const void* head_sink, int use_smooth_softmax);
extern "C" int hip_gqa_flash_decode(
    void* stream, const void* Q, const void* Kcache, const void* Vcache,
    void* O, void* partials_workspace, int B, int H, int G, int d, int skv,
    int max_seq, int max_splits, float scale, const void* seqlens_k,
    int local_window_size, const void* head_sink, int use_smooth_softmax,
    int kv_dtype, const void* k_scale, const void* v_scale);
extern "C" int hip_gqa_rope(
    void* stream, const void* input, void* output, const void* cos_cache,
    const void* sin_cache, int batch_size, int seq_len, int num_heads,
    int head_dim, int half_rot, int past_len, const void* seqlens_k,
    const void* position_ids, int element_size_bytes);
extern "C" int hip_gqa_kv_cache_append(
    void* stream, const void* src, void* cache, int batch_size, int sq, int G,
    int d, int present_seq, int past_len, const void* seqlens_k,
    int element_size_bytes, int kv_dtype, const void* scale);

#define HIP_CHECK(x)                                                          \
  do {                                                                        \
    hipError_t e_ = (x);                                                      \
    if (e_ != hipSuccess) {                                                   \
      std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_),  \
                   __FILE__, __LINE__);                                       \
      std::exit(2);                                                           \
    }                                                                         \
  } while (0)

namespace {

constexpr int kMaxSplits = 64;
// Fused launch split count; 0 lets the kernel derive one (--splits).
int g_splits = 0;
constexpr int kMaxPos = 32768 + 64;
constexpr size_t kColdBytes = size_t(256) << 20;
constexpr int kMaxCopies = 512;

struct Case {
  int B = 1, H = 32, G = 8, D = 128;
  int len = 2048;       // total keys including the decoded token
  int window = -1;
  bool sink = false, smooth = false;
  bool rope = true;
  bool pos_ids = true;  // position_ids given (offset from the cache row)
  bool append = true;   // K_new / V_new given (else already in the cache)
  int cap = 0;          // cache capacity (max_seq); 0 = len
  float qscale = 1.0f;  // Q magnitude; > 1 sharpens the softmax
};

int capacityOf(const Case& c) { return c.cap > c.len ? c.cap : c.len; }

float h2f(__half h) { return __half2float(h); }
__half f2h(float f) { return __float2half(f); }
float rnd16(float f) { return h2f(f2h(f)); }

std::string describe(const Case& c) {
  char b[160];
  std::snprintf(b, sizeof(b),
                "B=%d H=%d G=%d d=%d len=%d cap=%d win=%d sink=%d smooth=%d "
                "rope=%d pos=%d append=%d qscale=%g",
                c.B, c.H, c.G, c.D, c.len, capacityOf(c), c.window, c.sink,
                c.smooth, c.rope, c.pos_ids, c.append, c.qscale);
  return b;
}

struct HostData {
  std::vector<__half> q, knew, vnew, k, v, cos, sin, sinkv;
  std::vector<int> seqlens;
  std::vector<int64_t> pos;
};

HostData makeData(const Case& c, int max_seq, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  HostData h;
  auto fill = [&](std::vector<__half>& v, size_t n) {
    v.resize(n);
    for (auto& x : v) x = f2h(u(rng));
  };
  fill(h.q, (size_t)c.B * c.H * c.D);
  for (auto& x : h.q) x = f2h(h2f(x) * c.qscale);
  fill(h.knew, (size_t)c.B * c.G * c.D);
  fill(h.vnew, (size_t)c.B * c.G * c.D);
  fill(h.k, (size_t)c.B * c.G * max_seq * c.D);
  fill(h.v, (size_t)c.B * c.G * max_seq * c.D);
  const int half = c.D / 2;
  h.cos.resize((size_t)kMaxPos * half);
  h.sin.resize((size_t)kMaxPos * half);
  for (int p = 0; p < kMaxPos; ++p)
    for (int i = 0; i < half; ++i) {
      const double inv = std::pow(10000.0, -2.0 * i / c.D);
      h.cos[(size_t)p * half + i] = f2h((float)std::cos(p * inv));
      h.sin[(size_t)p * half + i] = f2h((float)std::sin(p * inv));
    }
  h.sinkv.resize(c.H);
  for (auto& x : h.sinkv) x = f2h(u(rng));
  h.seqlens.resize(c.B);
  h.pos.resize(c.B);
  for (int b = 0; b < c.B; ++b) {
    const int L = std::max(1, c.len - 7 * b);  // batches differ in length
    h.seqlens[b] = L - 1;
    h.pos[b] = c.pos_ids ? (L - 1) + 13 * (b + 1) : (L - 1);
  }
  return h;
}

// Half-rotated RoPE of one D-row, rounded to fp16 like hip_gqa_rope.
std::vector<float> ropeRow(const __half* row, int D, const HostData& h,
                           int pos) {
  const int half = D / 2;
  std::vector<float> out(D);
  for (int i = 0; i < half; ++i) {
    const float c = h2f(h.cos[(size_t)pos * half + i]);
    const float s = h2f(h.sin[(size_t)pos * half + i]);
    const float x1 = h2f(row[i]), x2 = h2f(row[i + half]);
    out[i] = rnd16(x1 * c - x2 * s);
    out[i + half] = rnd16(x2 * c + x1 * s);
  }
  return out;
}

// Reference O [B,H,D] and the expected cache after the append.
void reference(const Case& c, int max_seq, const HostData& h,
               std::vector<float>& O, std::vector<__half>& k_exp,
               std::vector<__half>& v_exp) {
  const int D = c.D, hpg = c.H / c.G;
  k_exp = h.k;
  v_exp = h.v;
  O.assign((size_t)c.B * c.H * D, 0.0f);
  const double scale = 1.0 / std::sqrt((double)D);
  for (int b = 0; b < c.B; ++b) {
    const int L = h.seqlens[b] + 1;
    const int pos = (int)h.pos[b];
    if (c.append)
      for (int g = 0; g < c.G; ++g) {
        const __half* kn = &h.knew[((size_t)b * c.G + g) * D];
        const __half* vn = &h.vnew[((size_t)b * c.G + g) * D];
        const size_t row = (((size_t)b * c.G + g) * max_seq + (L - 1)) * D;
        std::vector<float> kr(D);
        if (c.rope) kr = ropeRow(kn, D, h, pos);
        else for (int i = 0; i < D; ++i) kr[i] = h2f(kn[i]);
        for (int i = 0; i < D; ++i) {
          k_exp[row + i] = f2h(kr[i]);
          v_exp[row + i] = vn[i];
        }
      }
    const int lo = (c.window > 0 && L > c.window) ? L - c.window : 0;
    for (int hq = 0; hq < c.H; ++hq) {
      const int g = hq / hpg;
      const __half* qrow = &h.q[((size_t)b * c.H + hq) * D];
      std::vector<float> q(D);
      if (c.rope) q = ropeRow(qrow, D, h, pos);
      else for (int i = 0; i < D; ++i) q[i] = h2f(qrow[i]);
      std::vector<double> s(L, 0.0);
      double m = -INFINITY;
      for (int t = lo; t < L; ++t) {
        const __half* kr = &k_exp[(((size_t)b * c.G + g) * max_seq + t) * D];
        double dot = 0;
        for (int i = 0; i < D; ++i) dot += (double)q[i] * h2f(kr[i]);
        s[t] = dot * scale;
        m = std::max(m, s[t]);
      }
      double sink_s = -INFINITY;
      if (c.sink) sink_s = h2f(h.sinkv[hq]);
      else if (c.smooth) sink_s = 0.0;
      const double mm = std::max(m, sink_s);
      double l = 0;
      std::vector<double> acc(D, 0.0);
      for (int t = lo; t < L; ++t) {
        const double p = std::exp(s[t] - mm);
        l += p;
        const __half* vr = &v_exp[(((size_t)b * c.G + g) * max_seq + t) * D];
        for (int i = 0; i < D; ++i) acc[i] += p * h2f(vr[i]);
      }
      if (std::isfinite(sink_s)) l += std::exp(sink_s - mm);
      for (int i = 0; i < D; ++i)
        O[((size_t)b * c.H + hq) * D + i] = (float)(acc[i] / l);
    }
  }
}

struct DevBufs {
  __half *q = nullptr, *knew = nullptr, *vnew = nullptr, *o = nullptr;
  __half *cos = nullptr, *sin = nullptr, *sink = nullptr;
  __half *qrot = nullptr, *krot = nullptr;
  int *seqlens = nullptr, *tickets = nullptr;
  int64_t* pos = nullptr;
  float* partials = nullptr;
  std::vector<__half*> kc, vc;  // cache copies
};

template <typename T>
T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  HIP_CHECK(hipMalloc(&d, v.size() * sizeof(T)));
  HIP_CHECK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}

DevBufs makeDev(const Case& c, const HostData& h, int copies) {
  DevBufs d;
  d.q = upload(h.q);
  d.knew = upload(h.knew);
  d.vnew = upload(h.vnew);
  d.cos = upload(h.cos);
  d.sin = upload(h.sin);
  d.sink = upload(h.sinkv);
  d.seqlens = upload(h.seqlens);
  d.pos = upload(h.pos);
  HIP_CHECK(hipMalloc(&d.o, (size_t)c.B * c.H * c.D * sizeof(__half)));
  HIP_CHECK(hipMalloc(&d.qrot, (size_t)c.B * c.H * c.D * sizeof(__half)));
  HIP_CHECK(hipMalloc(&d.krot, (size_t)c.B * c.G * c.D * sizeof(__half)));
  HIP_CHECK(hipMalloc(&d.partials,
                      (size_t)c.B * c.H * kMaxSplits * (c.D + 2) * sizeof(float)));
  HIP_CHECK(hipMalloc(&d.tickets, (size_t)c.B * c.G * sizeof(int)));
  HIP_CHECK(hipMemset(d.tickets, 0, (size_t)c.B * c.G * sizeof(int)));
  for (int i = 0; i < copies; ++i) {
    d.kc.push_back(upload(h.k));
    d.vc.push_back(upload(h.v));
  }
  return d;
}

void freeDev(DevBufs& d) {
  for (void* p : {(void*)d.q, (void*)d.knew, (void*)d.vnew, (void*)d.o,
                  (void*)d.cos, (void*)d.sin, (void*)d.sink, (void*)d.qrot,
                  (void*)d.krot, (void*)d.seqlens, (void*)d.tickets,
                  (void*)d.pos, (void*)d.partials})
    if (p) (void)hipFree(p);
  for (auto* p : d.kc) (void)hipFree(p);
  for (auto* p : d.vc) (void)hipFree(p);
}

int launchFused(const Case& c, DevBufs& d, int copy, int max_seq) {
  return hip_gqa_flash_decode_fused(
      nullptr, d.q, c.append ? d.knew : nullptr, c.append ? d.vnew : nullptr,
      d.kc[copy], d.vc[copy], d.o, d.partials, d.tickets,
      c.rope ? d.cos : nullptr, c.rope ? d.sin : nullptr,
      (c.rope && c.pos_ids) ? d.pos : nullptr, c.B, c.H, c.G, c.D, max_seq,
      g_splits, 1.0f / std::sqrt((float)c.D), d.seqlens, c.window,
      c.sink ? d.sink : nullptr, c.smooth ? 1 : 0);
}

// The sequence the fused launch replaces.
int launchUnfused(const Case& c, DevBufs& d, int copy, int max_seq) {
  const void* q = d.q;
  const void* k = d.knew;
  if (c.rope) {
    const void* pos = c.pos_ids ? d.pos : nullptr;
    if (hip_gqa_rope(nullptr, d.q, d.qrot, d.cos, d.sin, c.B, 1, c.H, c.D,
                     c.D / 2, 0, d.seqlens, pos, 2) != 0 ||
        hip_gqa_rope(nullptr, d.knew, d.krot, d.cos, d.sin, c.B, 1, c.G, c.D,
                     c.D / 2, 0, d.seqlens, pos, 2) != 0)
      return -1;
    q = d.qrot;
    k = d.krot;
  }
  if (c.append &&
      (hip_gqa_kv_cache_append(nullptr, k, d.kc[copy], c.B, 1, c.G, c.D,
                               max_seq, 0, d.seqlens, 2, 0, nullptr) != 0 ||
       hip_gqa_kv_cache_append(nullptr, d.vnew, d.vc[copy], c.B, 1, c.G, c.D,
                               max_seq, 0, d.seqlens, 2, 0, nullptr) != 0))
    return -1;
  return hip_gqa_flash_decode(
      nullptr, q, d.kc[copy], d.vc[copy], d.o, d.partials, c.B, c.H, c.G, c.D,
      c.len, max_seq, kMaxSplits, 1.0f / std::sqrt((float)c.D), d.seqlens,
      c.window, c.sink ? d.sink : nullptr, c.smooth ? 1 : 0, 0, nullptr,
      nullptr);
}

bool checkCase(const Case& c, bool use_fused, double tol) {
  const int max_seq = capacityOf(c);
  HostData h = makeData(c, max_seq, 1234u + c.len + 7u * c.H);
  std::vector<float> ref;
  std::vector<__half> k_exp, v_exp;
  reference(c, max_seq, h, ref, k_exp, v_exp);
  DevBufs d = makeDev(c, h, 1);
  // Twice: the second launch must find the tickets reset by the first.
  for (int rep = 0; rep < 2; ++rep) {
    const int rc = use_fused ? launchFused(c, d, 0, max_seq)
                             : launchUnfused(c, d, 0, max_seq);
    if (rc != 0) {
      std::printf("  LAUNCH FAIL %s\n", describe(c).c_str());
      freeDev(d);
      return false;
    }
  }
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<__half> o((size_t)c.B * c.H * c.D), kc(k_exp.size()),
      vc(v_exp.size());
  std::vector<int> tk(c.B * c.G);
  HIP_CHECK(hipMemcpy(o.data(), d.o, o.size() * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(kc.data(), d.kc[0], kc.size() * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(vc.data(), d.vc[0], vc.size() * 2, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(tk.data(), d.tickets, tk.size() * 4, hipMemcpyDeviceToHost));
  double num = 0, den = 0;
  for (size_t i = 0; i < o.size(); ++i) {
    const double e = (double)h2f(o[i]) - ref[i];
    num += e * e;
    den += (double)ref[i] * ref[i];
  }
  const double rel = std::sqrt(num / std::max(den, 1e-30));
  size_t kbad = 0, vbad = 0;
  for (size_t i = 0; i < kc.size(); ++i) {
    // fp32 contraction may differ from the reference by one fp16 ulp.
    if (std::fabs(h2f(kc[i]) - h2f(k_exp[i])) >
        std::fabs(h2f(k_exp[i])) * 1e-3f + 1e-4f)
      ++kbad;
    if (__half_as_ushort(vc[i]) != __half_as_ushort(v_exp[i])) ++vbad;
  }
  int tbad = 0;
  for (int t : tk) tbad += t != 0;
  const bool ok = rel <= tol && kbad == 0 && vbad == 0 && tbad == 0;
  std::printf("  %-4s %-6s relL2=%.2e kbad=%zu vbad=%zu tickets=%d  %s\n",
              ok ? "PASS" : "FAIL", use_fused ? "fused" : "multi", rel, kbad,
              vbad, tbad, describe(c).c_str());
  freeDev(d);
  return ok;
}

// Holds the stream until every timed launch is queued, so a reading is GPU
// time rather than the host's submission rate (the multi-dispatch decode
// issues six launches per step).
__global__ void holdStreamKernel(const volatile int* release) {
  for (int spin = 0; spin < (1 << 16) && *release == 0; ++spin)
    __builtin_amdgcn_s_sleep(127);
}

__global__ void streamReadKernel(const uint4* p, size_t n, uint4* sink) {
  uint4 acc{0, 0, 0, 0};
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = p[i];
    acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
  }
  if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u) sink[0] = acc;
}

// An APU's memory clock reaches its high state only after about a second of
// continuous DRAM traffic and drops again when idle, so the timed rounds start
// right after this much streaming.
void rampMemoryClock() {
  constexpr size_t kBytes = size_t(512) << 20;
  uint4 *buf = nullptr, *sink = nullptr;
  HIP_CHECK(hipMalloc(&buf, kBytes));
  HIP_CHECK(hipMalloc(&sink, sizeof(uint4)));
  HIP_CHECK(hipMemset(buf, 1, kBytes));
  const auto t0 = std::chrono::steady_clock::now();
  while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
             .count() < 2.0) {
    streamReadKernel<<<2048, 256>>>(buf, kBytes / sizeof(uint4), sink);
    HIP_CHECK(hipDeviceSynchronize());
  }
  HIP_CHECK(hipFree(buf));
  HIP_CHECK(hipFree(sink));
}

int* holdReleaseFlag() {
  static int* flag = [] {
    void* p = nullptr;
    return hipHostMalloc(&p, sizeof(int),
                         hipHostMallocMapped | hipHostMallocCoherent) == hipSuccess
               ? static_cast<int*>(p)
               : nullptr;
  }();
  return flag;
}

template <typename Fn>
double timeRotating(int copies, int iters, int& turn, Fn&& launch) {
  HIP_CHECK(hipDeviceSynchronize());
  volatile int* release = holdReleaseFlag();
  void* release_dev = nullptr;
  if (release &&
      hipHostGetDevicePointer(&release_dev, const_cast<int*>(release), 0) ==
          hipSuccess) {
    *release = 0;
    holdStreamKernel<<<1, 1>>>(static_cast<const int*>(release_dev));
  }
  hipEvent_t e0, e1;
  HIP_CHECK(hipEventCreate(&e0));
  HIP_CHECK(hipEventCreate(&e1));
  HIP_CHECK(hipEventRecord(e0, nullptr));
  for (int i = 0; i < iters; ++i) launch(turn++ % copies);
  HIP_CHECK(hipEventRecord(e1, nullptr));
  if (release) *release = 1;
  HIP_CHECK(hipEventSynchronize(e1));
  float ms = 0;
  HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
  (void)hipEventDestroy(e0);
  (void)hipEventDestroy(e1);
  return ms / iters;
}

void benchCase(const Case& c, int iters) {
  const int max_seq = capacityOf(c);
  const size_t cache_bytes = (size_t)2 * c.B * c.G * max_seq * c.D * 2;
  const int copies = (int)std::max<size_t>(
      1, std::min<size_t>(kMaxCopies, (kColdBytes + cache_bytes - 1) / cache_bytes));
  HostData h = makeData(c, max_seq, 99u);
  DevBufs d = makeDev(c, h, copies);
  // The multi-dispatch decode autotunes on first use; keep that out of timing.
  (void)launchUnfused(c, d, 0, max_seq);
  int turn = 0;
  for (int i = 0; i < copies; ++i) (void)launchFused(c, d, turn++ % copies, max_seq);
  rampMemoryClock();
  // Alternating rounds put any clock drift on both paths alike.
  double ms_multi = 1e30, ms_fused = 1e30;
  for (int round = 0; round < 5; ++round) {
    ms_multi = std::min(ms_multi, timeRotating(copies, iters, turn, [&](int i) {
                          (void)launchUnfused(c, d, i, max_seq);
                        }));
    ms_fused = std::min(ms_fused, timeRotating(copies, iters, turn, [&](int i) {
                          (void)launchFused(c, d, i, max_seq);
                        }));
  }
  const double gb = (double)2 * c.B * c.G * c.len * c.D * 2 / 1e9;
  std::printf("  BENCH H=%d G=%d d=%d len=%-6d multi=%.4f ms fused=%.4f ms "
              "(%.2fx)  fused %.0f GB/s  copies=%d\n",
              c.H, c.G, c.D, c.len, ms_multi, ms_fused, ms_multi / ms_fused,
              gb / (ms_fused / 1e3), copies);
  freeDev(d);
}

}  // namespace

int main(int argc, char** argv) {
  Case one;
  bool single = false;
  int iters = 200;
  bool bench = true;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&](int& v) {
      if (i + 1 < argc) v = std::atoi(argv[++i]);
      single = true;
    };
    if (a == "--b") next(one.B);
    else if (a == "--h") next(one.H);
    else if (a == "--g") next(one.G);
    else if (a == "--d") next(one.D);
    else if (a == "--len") next(one.len);
    else if (a == "--window") next(one.window);
    else if (a == "--iters") { if (i + 1 < argc) iters = std::atoi(argv[++i]); }
    else if (a == "--no-bench") bench = false;
    else if (a == "--splits") { if (i + 1 < argc) g_splits = std::atoi(argv[++i]); }
    else if (a == "--cap") next(one.cap);
    else if (a == "--qscale") {
      if (i + 1 < argc) one.qscale = (float)std::atof(argv[++i]);
      single = true;
    }
  }
  if (single) {
    if (!hip_gqa_flash_decode_fused_supported(one.H, one.G, one.D)) {
      std::printf("geometry not supported by the fused kernel\n");
      return 1;
    }
    const bool ok = checkCase(one, true, 5e-3) & checkCase(one, false, 5e-3);
    if (bench) benchCase(one, iters);
    return ok ? 0 : 1;
  }

  struct Geo { int H, G, D; };
  const Geo geos[] = {{32, 32, 96}, {32, 8, 128}, {40, 40, 128}, {40, 10, 128}};
  int fails = 0, total = 0;
  for (const Geo& g : geos) {
    for (int len : {1, 2, 17, 129, 1000, 2048}) {
      Case c;
      c.H = g.H; c.G = g.G; c.D = g.D; c.len = len;
      Case variants[8] = {c, c, c, c, c, c, c, c};
      variants[1].pos_ids = false;
      variants[2].rope = false;
      variants[3].append = false; variants[3].rope = false;
      variants[4].window = 64; variants[4].sink = true;
      variants[5].smooth = true;
      variants[6].B = 2;
      // Spare cache capacity past the live rows, and a peaked softmax.
      variants[7].cap = 2 * len + 3; variants[7].qscale = 8.0f;
      for (const Case& v : variants) {
        ++total;
        fails += !checkCase(v, true, 5e-3);
      }
      // The multi-dispatch path through the new hip_gqa_rope position_ids.
      ++total;
      fails += !checkCase(c, false, 5e-3);
    }
  }
  std::printf("%d / %d cases passed\n", total - fails, total);
  if (bench)
    for (const Geo& g : geos)
      for (int len : {128, 2048, 8192}) {
        Case c;
        c.H = g.H; c.G = g.G; c.D = g.D; c.len = len;
        benchCase(c, iters);
      }
  return fails == 0 ? 0 : 1;
}
