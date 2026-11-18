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
#include "../../../common/hip_adaptive_timing.h"
#include "gqa_autotune.h"
#include "hip_custom_kernels.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <fstream>
#include <functional>
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
namespace timing = hipdnn_ep_test::timing;
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

// Repeated calls use one operand/cache allocation. New rows are encoded
// idempotently, so past and unused rows stay unchanged. These correctness UT
// timings are explicitly warm-cache measurements, not model DRAM throughput.
struct WarmOperands {
  size_t operand_bytes;
  int count() const { return 1; }
  size_t bytes() const { return operand_bytes; }
};

__global__ void holdStreamKernel(const volatile int* release) {
  for (int spin = 0; spin < (1 << 16) && *release == 0; ++spin)
    __builtin_amdgcn_s_sleep(127);
}

struct StreamGate {
  int* release = nullptr;
  StreamGate() {
    check(hipHostMalloc(reinterpret_cast<void**>(&release), sizeof(int),
                        hipHostMallocMapped | hipHostMallocCoherent));
  }
  ~StreamGate() { if (release) (void)hipHostFree(release); }
  StreamGate(const StreamGate&) = delete;
  StreamGate& operator=(const StreamGate&) = delete;
};

template <class Launch>
double measure(const Launch& launch, size_t operand_bytes, const char* entry) {
  StreamGate gate;
  return timing::adaptiveRotatedMeanMs(
      nullptr, WarmOperands{operand_bytes}, launch, gate.release,
      [](const int* flag) { holdStreamKernel<<<1, 1>>>(flag); },
      true, 0, entry, entry);
}

void record(const std::string& shape, double ms, double rel, bool passed,
            const std::string& config = "fixed:k8v8_g32_planar_v1") {
  timing::emitCsvBinding(shape, config, passed ? "PASS" : "FAIL", rel, ms);
  if (const char* path = std::getenv("HIPDNN_RESULTS_CSV")) {
    bool header = true;
    if (FILE* probe = std::fopen(path, "rb")) {
      std::fseek(probe, 0, SEEK_END);
      header = std::ftell(probe) == 0;
      std::fclose(probe);
    }
    FILE* f = std::fopen(path, "a");
    if (!f) throw std::runtime_error("cannot append kernel test results");
    if (header) std::fprintf(f, "op,leaf,arch,mode,shape,config,time_ms,relL2,verdict\n");
    std::fprintf(f, "gqa/kv_pareto,Xfp16_Wi8_Yfp16,%s,%s,%s,%s,%.6f,%.6e,%s\n",
                 env("HIPDNN_RESULTS_ARCH", "gfx1151"), env("HIPDNN_RESULTS_MODE", "lookup"),
                 shape.c_str(), config.c_str(), ms, rel, passed ? "PASS" : "FAIL");
    std::fclose(f);
  }
}

// Online prefill reports its actual selected configuration through the
// production [gqa-cfg] diagnostic. The controller and Makefile capture stderr
// from process startup, including the Windows kernel DLL's static CRT.
std::string online_config(const std::function<int()>& launch) {
  const char* path = std::getenv("HIPDNN_KERNEL_UT_STDERR");
  if (!path || !path[0])
    throw std::runtime_error("online prefill needs stderr capture: use the packaged controller or make test_custom MODE=autotune");
  std::fflush(stderr);
  std::streamoff offset = 0;
  { std::ifstream before(path, std::ios::binary | std::ios::ate);
    if (before) offset = before.tellg(); }
  if (launch() != 0) throw std::runtime_error("online prefill launch failed");
  check(hipDeviceSynchronize());
  std::fflush(stderr);
  std::ifstream log(path, std::ios::binary);
  log.seekg(offset);
  std::string config;
  for (std::string line; std::getline(log, line);) {
    const size_t knobs = line.find("-> ");
    if (line.find("[gqa-cfg]") != std::string::npos && knobs != std::string::npos) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      config = "autotune:" + line.substr(knobs + 3);
    }
  }
  if (config.empty()) throw std::runtime_error("online prefill did not report its dispatched configuration");
  return config;
}

// Prefill launch config resolved exactly as real/gqa.cpp does (offline LUT or
// heuristic); online mode lets the kernel tune itself (nullptr).
const int* prefill_config(int heads, int nq, int total, int (&config)[5],
                          std::string& identity) {
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
  identity = "lookup:hb=" + std::to_string(config[0]) + " qw=" + std::to_string(config[1]) +
             " mt=" + std::to_string(config[2]) + " bkv=" + std::to_string(config[3]) +
             " nd=" + std::to_string(config[4]) + " (" +
             hipdnn_ep::gqa_tune_source_name(selected.source) + ")";
  return config;
}

// Encode rows [start, start + count) and compare the whole buffer.
bool encode_case(int count, int start, int capacity) {
  const std::string shape = "encode_n" + std::to_string(count) + "_at" + std::to_string(start) +
                            "_cap" + std::to_string(capacity);
  timing::CaseScope timing_case("gqa/kv_pareto/Xfp16_Wi8_Yfp16", shape,
                                "hip_kv_pareto_k8v8_encode");
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
  const double ms = measure([&](int) {
    return hip_kv_pareto_k8v8_encode(nullptr, dk.data, dv.data, kc.data, vc.data,
                                    start, count, capacity);
  }, k.size() * 4 + cache_bytes * 2, "hip_kv_pareto_k8v8_encode");
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
  std::printf("KV-Pareto %s: %s (%zu mismatched bytes)\n", shape.c_str(),
              passed ? "PASS" : "FAIL", bad);
  record(shape, ms, double(bad), passed);
  return passed;
}

// One hip_kv_pareto_k8v8_gqa call: nq new tokens after `past` packed rows.
// Reference rows: all of them for short queries, else first/middle/last.
bool gqa_case(int heads, int nq, int past, int requested_capacity = -1,
              unsigned seed = 0, float scale = 0.0625f) {
  // Spare capacity checks that unused rows stay untouched (capped at the limit).
  const int hpg = heads / G, total = past + nq,
            capacity = requested_capacity > 0 ? requested_capacity :
                std::min(total + 37, HIP_EP_KV_PARETO_MAX_CAPACITY);
  if (seed == 0) seed = 911 + heads * 7 + nq * 131 + past;
  const std::string shape = "h" + std::to_string(heads) + "_q" + std::to_string(nq) +
                            "_past" + std::to_string(past) + "_cap" + std::to_string(capacity) +
                            "_seed" + std::to_string(seed) + "_scale" + std::to_string(scale);
  timing::CaseScope timing_case("gqa/kv_pareto/Xfp16_Wi8_Yfp16", shape,
                                "hip_kv_pareto_k8v8_gqa");
  std::mt19937 rng(seed);
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
  std::string identity = "fixed:k8v8_g32_planar_v1";
  const int* tuned = nq > 1 || past == 0 ? prefill_config(heads, nq, total, config, identity) : nullptr;
  if (hip_kv_pareto_k8v8_gqa(nullptr, dq.data, dk.data, dv.data, kc.data, vc.data,
                             dout.data, heads, nq, past, capacity, scale,
                             workspace.data, scratch - 1, tuned) == 0)
    throw std::runtime_error("undersized workspace accepted");

  auto launch = [&](int) {
    return hip_kv_pareto_k8v8_gqa(nullptr, dq.data, dk.data, dv.data, kc.data,
                                 vc.data, dout.data, heads, nq, past, capacity,
                                 scale, workspace.data, scratch, tuned);
  };
  if ((nq > 1 || past == 0) && !tuned)
    identity = online_config([&] { return launch(0); });
  else {
    if (launch(0) != 0) throw std::runtime_error("hip_kv_pareto_k8v8_gqa failed");
    check(hipDeviceSynchronize());
  }
  const double ms = measure(launch, qbytes + 2 * kvbytes + 2 * cache_bytes,
                            "hip_kv_pareto_k8v8_gqa");
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
        scores[s] = dot * double(scale);
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

  std::printf("KV-Pareto %s %s: %s (%.4f ms; worst-row relL2 %.3g; cache mismatches %zu)\n",
              nq == 1 && past > 0 ? "decode" : "prefill", shape.c_str(),
              passed ? "PASS" : "FAIL", ms, rel, bad);
  record(shape, ms, rel, passed, identity);
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
  int coverage = 1;
  int batch = 1, heads = 24, kv_heads = 4, dim = 256, nq = 17, past = 129, capacity = 4096;
  unsigned seed = 1234;
  float scale = 0.0625f;
  bool custom = false;
  auto integer = [](const char* text) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(text, &end, 10);
    if (errno || end == text || *end || value < 0 || value > INT_MAX)
      throw std::invalid_argument("expected a nonnegative integer within INT_MAX");
    return int(value);
  };
  try {
    coverage = integer(env("HIPDNN_UT_COVERAGE", "1"));
    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (flag == "--help") {
        std::printf("Usage: test_gqa_kv_pareto [--coverage 1|2|3] or "
                    "[--batch 1 --heads 16|24 --kv-heads 4 --head-dim 256 "
                    "--new-tokens N --past N --capacity N --seed N --scale F]\n");
        return 0;
      }
      if (i + 1 >= argc) throw std::invalid_argument("missing option value");
      const char* value = argv[++i];
      if (flag == "--coverage") coverage = integer(value);
      else {
        custom = true;
        if (flag == "--batch") batch = integer(value);
        else if (flag == "--heads") heads = integer(value);
        else if (flag == "--kv-heads") kv_heads = integer(value);
        else if (flag == "--head-dim") dim = integer(value);
        else if (flag == "--new-tokens") nq = integer(value);
        else if (flag == "--past") past = integer(value);
        else if (flag == "--capacity") capacity = integer(value);
        else if (flag == "--seed") seed = integer(value);
        else if (flag == "--scale") {
          char* end = nullptr;
          errno = 0;
          scale = std::strtof(value, &end);
          if (errno || end == value || *end || !std::isfinite(scale) || scale <= 0)
            throw std::invalid_argument("scale must be finite and positive");
        } else throw std::invalid_argument("unrecognized option: " + flag);
      }
    }
    if (coverage < 1 || coverage > 3) throw std::invalid_argument("coverage must be 1, 2 or 3");
    if (custom && (batch != 1 || (heads != 16 && heads != 24) || kv_heads != G || dim != D ||
                   nq < 1 || nq > HIP_EP_KV_PARETO_MAX_NEW_TOKENS ||
                   capacity < 1 || capacity > HIP_EP_KV_PARETO_MAX_CAPACITY ||
                   int64_t(past) + nq > capacity || seed == 0))
      throw std::invalid_argument("unsupported KV-Pareto geometry, cache capacity or seed");
#ifdef _WIN32
    _putenv_s("HIPDNN_GQA_AUTOTUNE_LOG", "1");
#else
    if (setenv("HIPDNN_GQA_AUTOTUNE_LOG", "1", 1) != 0)
      throw std::runtime_error("cannot enable production GQA config diagnostics");
#endif
    hipDeviceProp_t props{};
    check(hipGetDeviceProperties(&props, 0));
    const std::string arch = std::string(props.gcnArchName).substr(0, std::string(props.gcnArchName).find(':'));
    if (arch != "gfx1151") {
      std::printf("KV-Pareto SKIP: device %s is unsupported (requires gfx1151)\n", arch.c_str());
      return 0;
    }
    if (!hip_kv_pareto_device_supported())
      throw std::runtime_error("KV-Pareto kernel library must be built with HIP_EP_ENABLE_KV_PARETO=ON");
    timing::printTimingMode();
    if (custom) return gqa_case(heads, nq, past, capacity, seed, scale) ? 0 : 1;
    std::printf("KV-Pareto coverage=%d, H16/H24, B1/G4/D256, warm-cache correctness timings\n", coverage);
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
