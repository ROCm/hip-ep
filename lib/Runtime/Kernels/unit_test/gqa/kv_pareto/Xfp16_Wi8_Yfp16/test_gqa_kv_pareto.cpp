/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Direct calls into the packaged KV-Pareto kernels, per-token codec
// k8v8_g32_planar_v1 (INT8 K/V, one BF16 scale and INT8 zero point per token,
// KV head and 32 channels). No model compilation and no runtime opt-in: the
// HIPDNN_EP_USE_KV_PARETO flag only gates model inference, so CI exercises the
// kernels that every build ships. Tier 1 is the short PR suite; tiers 2/3 add
// long-context decode and prefill through the 262,272-row capacity, including
// a single-call 256K prompt.
//
// Checks: the encoder is bit-exact against a host reference quantizer; every
// GQA call matches an FP64 causal reference over the dequantized packed
// history plus the direct FP16 new K/V; afterwards the whole cache buffer is
// byte-exact (new rows encoded, history and unused rows untouched).
#include "gqa_autotune.h"
#include "hip_custom_kernels.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int D = 256, G = 4, GROUPS = 8;
constexpr size_t ROW_BYTES = HIP_EP_KV_PARETO_K8V8_WIDTH * 2; // 280

void check(hipError_t rc) {
  if (rc != hipSuccess) throw std::runtime_error(hipGetErrorString(rc));
}

struct Buffer {
  void* data = nullptr;
  explicit Buffer(size_t bytes) {
    if (bytes) check(hipMalloc(&data, bytes));
  }
  ~Buffer() {
    if (data) (void)hipFree(data);
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

const char* env(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value ? value : fallback;
}

// Round-to-nearest-even FP32 -> BF16, as the kernel does.
uint16_t to_bf16(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return uint16_t((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return uint16_t(u >> 16);
}
float from_bf16(uint16_t h) {
  const uint32_t u = uint32_t(h) << 16;
  float x;
  std::memcpy(&x, &u, 4);
  return x;
}
float bf(float x) { return from_bf16(to_bf16(x)); }

// Planar cache block [payload 4*C*256 | BF16 scales 4*C*8 | INT8 zeros 4*C*8].
struct HostCache {
  int capacity;
  std::vector<uint8_t> bytes;
  explicit HostCache(int c) : capacity(c), bytes(size_t(G) * c * ROW_BYTES, 0x5a) {}
  int8_t* payload() { return reinterpret_cast<int8_t*>(bytes.data()); }
  uint16_t* scales() {
    return reinterpret_cast<uint16_t*>(bytes.data() + size_t(G) * capacity * D);
  }
  int8_t* zeros() {
    return reinterpret_cast<int8_t*>(bytes.data() + size_t(G) * capacity * (D + 16));
  }
  size_t row(int g, int token) const { return size_t(g) * capacity + token; }
  float dequant(int g, int token, int d) {
    const size_t r = row(g, token), m = r * GROUPS + d / 32;
    return (float(payload()[r * D + d]) - float(zeros()[m])) * from_bf16(scales()[m]);
  }
  // Reference quantizer for one FP16 row (BF16-rounded intermediates).
  void encode(int g, int token, const __half* x) {
    const size_t r = row(g, token);
    for (int group = 0; group < GROUPS; ++group) {
      float lo = INFINITY, hi = -INFINITY;
      for (int i = 0; i < 32; ++i) {
        const float v = __half2float(x[group * 32 + i]);
        lo = std::min(lo, v);
        hi = std::max(hi, v);
      }
      const float range = std::max(bf(hi - lo), bf(1e-6f));
      const uint16_t scale_bits = to_bf16(range / 255.0f);
      const float scale = from_bf16(scale_bits);
      const float zero = std::min(std::max(-128.0f - std::rint(bf(lo / scale)), -128.0f), 127.0f);
      for (int i = 0; i < 32; ++i) {
        const float v = __half2float(x[group * 32 + i]);
        const float code = std::min(std::max(std::rint(bf(v / scale)) + zero, -128.0f), 127.0f);
        payload()[r * D + group * 32 + i] = int8_t(code);
      }
      scales()[r * GROUPS + group] = scale_bits;
      zeros()[r * GROUPS + group] = int8_t(zero);
    }
  }
};

// Synthetic packed history: random codes, zero points and BF16 scales. Long
// contexts need no FP16 history on the host; the encoder is checked separately.
void fill_history(HostCache& cache, int past, std::mt19937& rng) {
  std::uniform_int_distribution<int> code(-128, 127), zero(-16, 16);
  std::uniform_real_distribution<float> scale(0.004f, 0.012f);
  for (int g = 0; g < G; ++g)
    for (int t = 0; t < past; ++t) {
      const size_t r = cache.row(g, t);
      for (int d = 0; d < D; ++d) cache.payload()[r * D + d] = int8_t(code(rng));
      for (int m = 0; m < GROUPS; ++m) {
        cache.scales()[r * GROUPS + m] = to_bf16(scale(rng));
        cache.zeros()[r * GROUPS + m] = int8_t(zero(rng));
      }
    }
}

size_t mismatches(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
  return n;
}

void record(const std::string& shape, double ms, double rel, bool passed) {
  if (const char* path = std::getenv("HIPDNN_RESULTS_CSV")) {
    FILE* f = std::fopen(path, "a");
    if (!f) throw std::runtime_error("cannot append kernel test results");
    std::fprintf(f, "gqa/kv_pareto,Xfp16_Wi8_Yfp16,%s,%s,%s,k8v8_g32,%.6f,%.6e,%s\n",
                 env("HIPDNN_RESULTS_ARCH", "gfx1151"), env("HIPDNN_RESULTS_MODE", "lookup"),
                 shape.c_str(), ms, rel, passed ? "PASS" : "FAIL");
    std::fclose(f);
  }
}

// Prefill launch config resolved exactly as real/gqa.cpp does (offline LUT or
// heuristic); online mode lets the kernel tune itself (nullptr).
const int* prefill_config(int heads, int nq, int total, int (&config)[5]) {
  static void* policy = hip_gqa_autotune_create(nullptr);
  if (!policy || hip_gqa_autotune_mode(policy) ==
                     static_cast<int>(hipdnn_ep::GqaAutotuneMode::Online))
    return nullptr;
  hipdnn_ep::GqaPrefillRequest request{hipdnn_ep::GqaPrefillVariant::V8x,
                                       1, heads, G, D, nq, total, -1};
  hipdnn_ep::GqaPrefillResult selected{};
  hip_gqa_autotune_resolve_prefill(policy, &request, &selected);
  config[0] = selected.config.hb;
  config[1] = selected.config.qw;
  config[2] = selected.config.mt;
  config[3] = selected.config.bkv;
  config[4] = selected.config.nd;
  return config;
}

// Encode rows [start, start + count) and compare the whole buffer.
bool encode_case(int count, int start, int capacity) {
  std::mt19937 rng(17 + count + start);
  std::normal_distribution<float> normal(0.f, 1.f);
  std::vector<__half> k(size_t(count) * G * D), v(k.size());
  for (auto* t : {&k, &v})
    for (auto& x : *t) x = __float2half(normal(rng));
  // One constant group exercises the minimum range.
  std::fill(k.begin(), k.begin() + 32, __float2half(0.25f));
  HostCache want_k(capacity), want_v(capacity);
  const size_t cache_bytes = want_k.bytes.size();
  Buffer dk(k.size() * 2), dv(v.size() * 2), kc(cache_bytes), vc(cache_bytes);
  check(hipMemcpy(dk.data, k.data(), k.size() * 2, hipMemcpyHostToDevice));
  check(hipMemcpy(dv.data, v.data(), v.size() * 2, hipMemcpyHostToDevice));
  check(hipMemcpy(kc.data, want_k.bytes.data(), cache_bytes, hipMemcpyHostToDevice));
  check(hipMemcpy(vc.data, want_v.bytes.data(), cache_bytes, hipMemcpyHostToDevice));
  if (hip_kv_pareto_k8v8_encode(nullptr, dk.data, dv.data, kc.data, vc.data, start,
                                count, capacity) != 0)
    throw std::runtime_error("encode launch failed");
  check(hipDeviceSynchronize());
  for (int t = 0; t < count; ++t)
    for (int g = 0; g < G; ++g) {
      want_k.encode(g, start + t, &k[(size_t(t) * G + g) * D]);
      want_v.encode(g, start + t, &v[(size_t(t) * G + g) * D]);
    }
  std::vector<uint8_t> got(cache_bytes);
  size_t bad = 0;
  check(hipMemcpy(got.data(), kc.data, cache_bytes, hipMemcpyDeviceToHost));
  bad += mismatches(got, want_k.bytes);
  check(hipMemcpy(got.data(), vc.data, cache_bytes, hipMemcpyDeviceToHost));
  bad += mismatches(got, want_v.bytes);
  const bool passed = bad == 0;
  const std::string shape = "encode_n" + std::to_string(count) + "_at" + std::to_string(start) +
                            "_cap" + std::to_string(capacity);
  std::printf("KV-Pareto %s: %s (%zu mismatched bytes)\n", shape.c_str(),
              passed ? "PASS" : "FAIL", bad);
  record(shape, 0.0, double(bad), passed);
  return passed;
}

// One hip_kv_pareto_k8v8_gqa call: nq new tokens after `past` packed rows.
// Reference rows: all of them for short queries, else first/middle/last.
bool gqa_case(int heads, int nq, int past) {
  // Spare capacity checks that unused rows stay untouched (capped at the limit).
  const int hpg = heads / G, total = past + nq,
            capacity = std::min(total + 37, HIP_EP_KV_PARETO_MAX_CAPACITY);
  std::mt19937 rng(911 + heads * 7 + nq * 131 + past);
  std::normal_distribution<float> normal(0.f, 1.f);
  HostCache k_cache(capacity), v_cache(capacity);
  fill_history(k_cache, past, rng);
  fill_history(v_cache, past, rng);
  std::vector<__half> q(size_t(nq) * heads * D), k(size_t(nq) * G * D), v(k.size());
  for (auto* t : {&q, &k, &v})
    for (auto& x : *t) x = __float2half(normal(rng));

  const size_t cache_bytes = k_cache.bytes.size();
  const size_t qbytes = q.size() * 2, kvbytes = k.size() * 2;
  const size_t scratch = hip_kv_pareto_k8v8_workspace(heads, nq, past);
  if (!scratch) throw std::runtime_error("workspace rejected a supported call");
  Buffer dq(qbytes), dk(kvbytes), dv(kvbytes), dout(qbytes), kc(cache_bytes),
      vc(cache_bytes), workspace(scratch);
  check(hipMemcpy(dq.data, q.data(), qbytes, hipMemcpyHostToDevice));
  check(hipMemcpy(dk.data, k.data(), kvbytes, hipMemcpyHostToDevice));
  check(hipMemcpy(dv.data, v.data(), kvbytes, hipMemcpyHostToDevice));
  check(hipMemcpy(kc.data, k_cache.bytes.data(), cache_bytes, hipMemcpyHostToDevice));
  check(hipMemcpy(vc.data, v_cache.bytes.data(), cache_bytes, hipMemcpyHostToDevice));
  check(hipMemset(dout.data, 0xff, qbytes));
  int config[5];
  const int* tuned = nq > 1 || past == 0 ? prefill_config(heads, nq, total, config) : nullptr;
  if (hip_kv_pareto_k8v8_gqa(nullptr, dq.data, dk.data, dv.data, kc.data, vc.data,
                             dout.data, heads, nq, past, capacity, 0.0625f,
                             workspace.data, scratch - 1, tuned) == 0)
    throw std::runtime_error("undersized workspace accepted");

  hipEvent_t start, stop;
  check(hipEventCreate(&start));
  check(hipEventCreate(&stop));
  check(hipEventRecord(start));
  const int rc = hip_kv_pareto_k8v8_gqa(nullptr, dq.data, dk.data, dv.data, kc.data,
                                        vc.data, dout.data, heads, nq, past, capacity,
                                        0.0625f, workspace.data, scratch, tuned);
  check(hipEventRecord(stop));
  check(hipEventSynchronize(stop));
  float ms = 0.f;
  check(hipEventElapsedTime(&ms, start, stop));
  check(hipEventDestroy(start));
  check(hipEventDestroy(stop));
  if (rc != 0) throw std::runtime_error("hip_kv_pareto_k8v8_gqa failed");
  std::vector<__half> actual(q.size());
  check(hipMemcpy(actual.data(), dout.data, qbytes, hipMemcpyDeviceToHost));

  std::vector<int> rows;
  if (nq <= 64) {
    for (int t = 0; t < nq; ++t) rows.push_back(t);
  } else {
    rows = {0, nq / 2, nq - 1};
  }
  // Relative L2 per checked query row (over all heads); the worst row counts,
  // so a short-history row cannot mask a long-history one.
  double worst = 0;
  bool passed = true;
  std::vector<double> scores(total), acc(D);
  for (int t : rows) {
    double err2 = 0, ref2 = 0;
    for (int h = 0; h < heads; ++h) {
      const int g = h / hpg, valid = past + t + 1;
      const __half* qrow = &q[(size_t(t) * heads + h) * D];
      double maximum = -INFINITY;
      for (int s = 0; s < valid; ++s) {
        double dot = 0;
        if (s < past) {
          for (int d = 0; d < D; ++d) dot += double(__half2float(qrow[d])) * k_cache.dequant(g, s, d);
        } else {
          const __half* krow = &k[(size_t(s - past) * G + g) * D];
          for (int d = 0; d < D; ++d) dot += double(__half2float(qrow[d])) * __half2float(krow[d]);
        }
        scores[s] = dot * 0.0625;
        maximum = std::max(maximum, scores[s]);
      }
      double denominator = 0;
      std::fill(acc.begin(), acc.end(), 0.0);
      for (int s = 0; s < valid; ++s) {
        const double w = std::exp(scores[s] - maximum);
        denominator += w;
        if (s < past) {
          for (int d = 0; d < D; ++d) acc[d] += w * v_cache.dequant(g, s, d);
        } else {
          const __half* vrow = &v[(size_t(s - past) * G + g) * D];
          for (int d = 0; d < D; ++d) acc[d] += w * __half2float(vrow[d]);
        }
      }
      for (int d = 0; d < D; ++d) {
        const double want = acc[d] / denominator;
        const double got = __half2float(actual[(size_t(t) * heads + h) * D + d]);
        const double error = got - want;
        passed &= std::isfinite(got) && std::abs(error) <= 0.002 + 0.03 * std::abs(want);
        err2 += error * error;
        ref2 += want * want;
      }
    }
    worst = std::max(worst, std::sqrt(err2 / std::max(ref2, 1e-30)));
  }
  const double rel = worst;
  passed &= rel < 5e-3;

  // New rows are encoded; history and unused capacity are never rewritten.
  for (int t = 0; t < nq; ++t)
    for (int g = 0; g < G; ++g) {
      k_cache.encode(g, past + t, &k[(size_t(t) * G + g) * D]);
      v_cache.encode(g, past + t, &v[(size_t(t) * G + g) * D]);
    }
  std::vector<uint8_t> got(cache_bytes);
  size_t bad = 0;
  check(hipMemcpy(got.data(), kc.data, cache_bytes, hipMemcpyDeviceToHost));
  bad += mismatches(got, k_cache.bytes);
  check(hipMemcpy(got.data(), vc.data, cache_bytes, hipMemcpyDeviceToHost));
  bad += mismatches(got, v_cache.bytes);
  passed &= bad == 0;

  const std::string shape = "h" + std::to_string(heads) + "_q" + std::to_string(nq) +
                            "_past" + std::to_string(past);
  std::printf("KV-Pareto %s %s: %s (%.4f ms; worst-row relL2 %.3g; cache mismatches %zu)\n",
              nq == 1 && past > 0 ? "decode" : "prefill", shape.c_str(),
              passed ? "PASS" : "FAIL", ms, rel, bad);
  record(shape, ms, rel, passed);
  return passed;
}

bool contract_case() {
  bool ok = hip_kv_pareto_k8v8_supported(24, 4, 256) && hip_kv_pareto_k8v8_supported(16, 4, 256) &&
            !hip_kv_pareto_k8v8_supported(32, 4, 256) && !hip_kv_pareto_k8v8_supported(24, 8, 256) &&
            !hip_kv_pareto_k8v8_supported(24, 4, 128);
  ok &= hip_kv_pareto_k8v8_workspace(32, 1, 8) == 0;
  ok &= hip_kv_pareto_k8v8_workspace(24, 0, 8) == 0;
  ok &= hip_kv_pareto_k8v8_workspace(24, HIP_EP_KV_PARETO_MAX_NEW_TOKENS + 1, 0) == 0;
  ok &= hip_kv_pareto_k8v8_workspace(24, 1, HIP_EP_KV_PARETO_MAX_CAPACITY) == 0;
  ok &= hip_kv_pareto_k8v8_workspace(24, 1, HIP_EP_KV_PARETO_MAX_CAPACITY - 1) > 0;
  ok &= hip_kv_pareto_k8v8_workspace(24, HIP_EP_KV_PARETO_MAX_NEW_TOKENS, 0) > 0;
  Buffer cache(size_t(G) * 8 * ROW_BYTES), kv(size_t(G) * D * 2);
  ok &= hip_kv_pareto_k8v8_encode(nullptr, kv.data, kv.data, cache.data, cache.data, 0, 1, 8) != 0;
  ok &= hip_kv_pareto_k8v8_encode(nullptr, kv.data, kv.data, cache.data, kv.data, 8, 1, 8) != 0;
  std::printf("KV-Pareto contract: %s\n", ok ? "PASS" : "FAIL");
  record("contract", 0.0, 0.0, ok);
  return ok;
}
} // namespace

int main(int argc, char** argv) {
  int coverage = std::atoi(env("HIPDNN_UT_COVERAGE", "1"));
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--coverage") == 0 && i + 1 < argc)
      coverage = std::atoi(argv[++i]);
    else
      return 2;
  }
  if (coverage < 1 || coverage > 3) return 2;
  try {
    if (!hip_kv_pareto_device_supported())
      throw std::runtime_error(
          "KV-Pareto kernel test requires a gfx1151 device and a kernel library built "
          "with HIP_EP_ENABLE_KV_PARETO=ON");
    bool passed = contract_case();
    passed &= encode_case(1, 0, 8);
    passed &= encode_case(37, 5, 64);
    for (int heads : {24, 16}) {
      // Fresh prefill (including a 1-token prompt) and append prefill.
      for (int nq : {1, 17}) passed &= gqa_case(heads, nq, 0);
      passed &= gqa_case(heads, 33, 129);
      passed &= gqa_case(heads, 300, 1000);
      // Decode around tile (32) and one-wave partition boundaries.
      for (int past : {1, 31, 32, 33, 480, 481, 4095, 8191}) passed &= gqa_case(heads, 1, past);
    }
    if (coverage >= 2) {
      for (int heads : {24, 16}) {
        passed &= gqa_case(heads, 1, 65535);
        passed &= gqa_case(heads, 1, 131071);
        passed &= gqa_case(heads, 4096, 28672);
      }
    }
    if (coverage >= 3) {
      for (int heads : {24, 16}) passed &= gqa_case(heads, 1, HIP_EP_KV_PARETO_MAX_CAPACITY - 1);
      passed &= gqa_case(24, 2048, HIP_EP_KV_PARETO_MAX_CAPACITY - 2048);
      passed &= gqa_case(24, HIP_EP_KV_PARETO_MAX_NEW_TOKENS, 0); // 256K prompt, one call
    }
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "KV-Pareto kernel test failed: %s\n", error.what());
    return 1;
  }
}
