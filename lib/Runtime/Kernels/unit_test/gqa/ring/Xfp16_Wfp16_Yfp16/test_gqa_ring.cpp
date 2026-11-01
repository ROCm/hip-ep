/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Direct calls into the packaged Ring kernels: no model compilation, runtime
// opt-in, autotuning or duplicate kernel build. Tier 1 is the short PR suite;
// tiers 2/3 add long-context coverage through 256K plus the decode tail.
#include "hip_custom_kernels.h"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void check(hipError_t rc) {
  if (rc != hipSuccess) throw std::runtime_error(hipGetErrorString(rc));
}

struct Buffer {
  void* data = nullptr;
  explicit Buffer(size_t bytes) {
    if (bytes) check(hipMalloc(&data, bytes));
  }
  ~Buffer() { if (data) hipFree(data); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

// Full FP32 causal reference from FP16-rounded inputs, including all heads and
// all valid keys. Capacity padding is deliberately excluded from attention.
static std::vector<float> reference(const std::vector<__half>& q,
                                    const std::vector<__half>& k,
                                    const std::vector<__half>& v,
                                    int nq, int past, int capacity) {
  std::vector<float> out(q.size(), 0.f), scores(past + nq);
  for (int t = 0; t < nq; ++t) {
    for (int h = 0; h < 24; ++h) {
      const size_t qi = (size_t(t) * 24 + h) * 256;
      const size_t ki = size_t(h / 6) * capacity * 256;
      const int valid = past + t + 1;
      float maximum = -INFINITY, denominator = 0.f;
      for (int s = 0; s < valid; ++s) {
        float dot = 0.f;
        for (int d = 0; d < 256; ++d)
          dot += __half2float(q[qi + d]) * __half2float(k[ki + size_t(s) * 256 + d]);
        scores[s] = dot * 0.0625f;
        maximum = std::max(maximum, scores[s]);
      }
      for (int s = 0; s < valid; ++s) {
        const float weight = std::exp(scores[s] - maximum);
        denominator += weight;
        for (int d = 0; d < 256; ++d)
          out[qi + d] += weight * __half2float(v[ki + size_t(s) * 256 + d]);
      }
      for (int d = 0; d < 256; ++d) out[qi + d] /= denominator;
    }
  }
  return out;
}

static const char* env(const char* name, const char* fallback) {
  const char* value = std::getenv(name);
  return value ? value : fallback;
}

static bool run_case(int nq, int past, int delta = 0) {
  const int total = past + nq, capacity = total + 19;
  std::vector<__half> q(size_t(nq) * 24 * 256);
  std::vector<__half> k(size_t(4) * capacity * 256), v(k.size()), actual(q.size());
  std::vector<float> expected;
  if (delta) {
    // exp(-18) underflows in unscaled FP16; exp(-32) also exercises the
    // prefill residual path. Large V makes the lost contribution observable.
    std::fill(q.begin(), q.end(), __float2half(0.f));
    std::fill(k.begin(), k.end(), __float2half(0.f));
    std::fill(v.begin(), v.end(), __float2half(60000.f));
    for (size_t i = 0; i < q.size(); i += 256) q[i] = __float2half(16.f);
    for (int g = 0; g < 4; ++g) {
      const size_t base = size_t(g) * capacity * 256;
      std::fill(v.begin() + base, v.begin() + base + 256, __float2half(0.f));
      for (int s = 1; s < total; ++s) k[base + size_t(s) * 256] = __float2half(-float(delta));
    }
    expected.resize(q.size());
    for (int t = 0; t < nq; ++t) {
      const double mass = (past + t) * std::exp(-double(delta));
      std::fill(expected.begin() + size_t(t) * 6144,
                expected.begin() + size_t(t + 1) * 6144,
                float(60000.0 * mass / (1.0 + mass)));
    }
  } else {
    std::mt19937 rng(984 + nq + past);
    std::normal_distribution<float> normal(0.f, 0.5f);
    for (auto* tensor : {&q, &k, &v})
      for (auto& x : *tensor) x = __float2half(normal(rng));
    expected = reference(q, k, v, nq, past, capacity);
  }

  const size_t qbytes = q.size() * sizeof(__half), kvbytes = k.size() * sizeof(__half);
  const size_t scratch = hip_gqa_ring_workspace(nq, total);
  Buffer dq(qbytes), dk(kvbytes), dv(kvbytes), dout(qbytes), workspace(scratch);
  check(hipMemcpy(dq.data, q.data(), qbytes, hipMemcpyHostToDevice));
  check(hipMemcpy(dk.data, k.data(), kvbytes, hipMemcpyHostToDevice));
  check(hipMemcpy(dv.data, v.data(), kvbytes, hipMemcpyHostToDevice));
  check(hipMemset(dout.data, 0xff, qbytes));
  if (scratch && hip_gqa_ring_attention(nullptr, dq.data, dk.data, dv.data,
                                       dout.data, nq, past, capacity,
                                       workspace.data, scratch - 1) == 0)
    throw std::runtime_error("undersized workspace accepted");

  hipEvent_t start, stop;
  check(hipEventCreate(&start));
  check(hipEventCreate(&stop));
  check(hipEventRecord(start));
  check(static_cast<hipError_t>(hip_gqa_ring_attention(
      nullptr, dq.data, dk.data, dv.data, dout.data, nq, past, capacity,
      workspace.data, scratch)));
  check(hipEventRecord(stop));
  check(hipEventSynchronize(stop));
  float ms = 0.f;
  check(hipEventElapsedTime(&ms, start, stop));
  check(hipEventDestroy(start));
  check(hipEventDestroy(stop));
  check(hipMemcpy(actual.data(), dout.data, qbytes, hipMemcpyDeviceToHost));
  double err2 = 0., ref2 = 0.;
  bool passed = true;
  for (size_t i = 0; i < actual.size(); ++i) {
    const float got = __half2float(actual[i]);
    const double error = double(got) - expected[i];
    const double tolerance = delta ? 0.000002 + 0.001 * std::abs(expected[i])
                                   : 0.005 + 0.005 * std::abs(expected[i]);
    passed &= std::isfinite(got) && std::abs(error) <= tolerance;
    err2 += error * error;
    ref2 += double(expected[i]) * expected[i];
  }
  const double rel = std::sqrt(err2 / std::max(ref2, 1e-30));
  const std::string shape = "q" + std::to_string(nq) + "_kv" + std::to_string(total) +
                            "_delta" + std::to_string(delta);
  std::printf("Ring %s: %s (%.4f ms; relL2 %.6g)\n", shape.c_str(),
              passed ? "PASS" : "FAIL", ms, rel);
  if (const char* path = std::getenv("HIPDNN_RESULTS_CSV")) {
    FILE* f = std::fopen(path, "a");
    if (!f) throw std::runtime_error("cannot append kernel test results");
    std::fprintf(f, "gqa/ring,Xfp16_Wfp16_Yfp16,%s,%s,%s,ring,%.6f,%.6e,%s\n",
                 env("HIPDNN_RESULTS_ARCH", "gfx1151"), env("HIPDNN_RESULTS_MODE", "lookup"),
                 shape.c_str(), ms, rel, passed ? "PASS" : "FAIL");
    std::fclose(f);
  }
  return passed;
}

int main(int argc, char** argv) {
  int coverage = std::atoi(env("HIPDNN_UT_COVERAGE", "1"));
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--coverage") == 0 && i + 1 < argc)
      coverage = std::atoi(argv[++i]);
    else return 2;
  }
  if (coverage < 1 || coverage > 3) return 2;
  try {
    if (!hip_gqa_ring_device_supported())
      throw std::runtime_error("Ring kernel test requires a gfx1151 device and enabled kernel library");
    bool passed = true;
    for (int total : {1, 32, 33, 128, 129, 1024, 1025, 8192, 8193})
      passed &= run_case(1, total - 1);
    passed &= run_case(17, 0);
    passed &= run_case(33, 129);
    passed &= run_case(1, 128, 18);
    passed &= run_case(17, 0, 18);
    passed &= run_case(17, 0, 24);
    for (int total : {65664, 65665}) passed &= run_case(1, total - 1, 18);
    if (coverage >= 2) {
      passed &= run_case(1, 131071, 18);
      passed &= run_case(17, 65536, 32);
    }
    if (coverage >= 3) {
      passed &= run_case(1, 262143, 18);
      passed &= run_case(1, 262271, 18);
      passed &= run_case(17, 262255, 32);
    }
    return passed ? 0 : 1;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "Ring test failed: %s\n", error.what());
    return 1;
  }
}
