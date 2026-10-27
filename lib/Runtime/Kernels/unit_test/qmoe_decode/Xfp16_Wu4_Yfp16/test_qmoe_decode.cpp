/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for the fused single-token QMoE decode
 * (hip_qmoe_decode_fused and its dp4a variant): fp16 activations, 4-bit
 * weights, fp16 output.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    const hipError_t error_ = (call);                                          \
    if (error_ != hipSuccess) {                                                \
      std::cerr << "HIP failure " << hipGetErrorString(error_) << " at "       \
                << __FILE__ << ':' << __LINE__ << '\n';                        \
      std::exit(2);                                                            \
    }                                                                          \
  } while (false)

constexpr float kSwigluAlpha = 1.702f;
constexpr float kSwigluBeta = 1.0f;
constexpr float kSwigluLimit = 7.0f;
constexpr int kDefaultZeroPoint = 8;
constexpr int kGuardElements = 32;
// The first call of a shape picks its GEMV tiling. If the choice were made by
// sampling candidates on real calls, it would take 12 candidates x 9 rounds of
// calls, with results that differ in the low bits; these many calls must all
// match the first bit for bit.
constexpr int kRepeatCalls = 129;
constexpr int kTimedCalls = 50;
// The kernel accumulates in fp32 in its own order; the reference in double,
// rounding to fp16 where the kernel stores (SwiGLU output, weighted slot,
// output). Each element may then differ by a few fp16 ulps of itself, and,
// since the k slots are stored as fp16 before they are summed, by an ulp of
// the slots where they cancel: hence a term scaled by the largest output.
constexpr double kRelTolerance = 4e-3;
constexpr double kScaleTolerance = 2e-3;
constexpr double kRelL2Tolerance = 1e-3;

struct Shape {
  int64_t hidden;
  int64_t inter;
  int64_t experts;
  int64_t k;
  int min_coverage;
};

constexpr Shape kShapes[] = {
    {64, 64, 4, 2, 1},
    {2880, 2880, 32, 4, 2},   // GPT-OSS-20B
    {2048, 768, 128, 8, 3},
};

constexpr int64_t kBlockSizes[] = {32, 64, 128};

struct Category {
  bool dp4a;
  bool zero_points;
  bool bias;
  int64_t block_size;
};

class Rng {
public:
  explicit Rng(uint64_t seed) : state_(seed) {}
  uint64_t next() {
    uint64_t z = (state_ += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }
  float uniform(float lo, float hi) {
    return lo + (hi - lo) * static_cast<float>(next() >> 40) /
                    static_cast<float>(1ull << 24);
  }

private:
  uint64_t state_;
};

float to_float(__half h) { return __half2float(h); }
__half to_half(double v) { return __float2half(static_cast<float>(v)); }

// One expert matrix [E, N, K] of 4-bit weights with fp16 group scales, packed
// 4-bit zero points and an fp16 bias.
struct Matrix {
  int64_t n = 0, k = 0, blocks = 0, zp_stride = 0;
  std::vector<uint8_t> weights;    // [E, N, K/2], element k in nibble k&1
  std::vector<__half> scales;      // [E, N, blocks]
  std::vector<uint8_t> zero_points;  // [E, N, ceil(blocks/2)], group g in nibble g&1
  std::vector<__half> bias;        // [E, N]

  int q(int64_t e, int64_t row, int64_t col) const {
    const uint8_t b = weights[((e * n + row) * k + col) / 2];
    return (col & 1) ? (b >> 4) : (b & 0x0f);
  }
  int zp(int64_t e, int64_t row, int64_t group, bool on) const {
    if (!on) {
      return kDefaultZeroPoint;
    }
    const uint8_t b = zero_points[(e * n + row) * zp_stride + group / 2];
    return (group & 1) ? (b >> 4) : (b & 0x0f);
  }
  float scale(int64_t e, int64_t row, int64_t group) const {
    return to_float(scales[(e * n + row) * blocks + group]);
  }
};

Matrix make_matrix(int64_t experts, int64_t n, int64_t k, int64_t block_size,
                   float scale_lo, float scale_hi, Rng& rng) {
  Matrix m;
  m.n = n;
  m.k = k;
  m.blocks = (k + block_size - 1) / block_size;
  m.zp_stride = (m.blocks + 1) / 2;
  m.weights.resize(static_cast<size_t>(experts * n * k / 2));
  for (uint8_t& b : m.weights) {
    b = static_cast<uint8_t>(rng.next());
  }
  m.scales.resize(static_cast<size_t>(experts * n * m.blocks));
  for (__half& s : m.scales) {
    s = to_half(rng.uniform(scale_lo, scale_hi));
  }
  m.zero_points.resize(static_cast<size_t>(experts * n * m.zp_stride));
  for (uint8_t& b : m.zero_points) {
    b = static_cast<uint8_t>(rng.next());
  }
  m.bias.resize(static_cast<size_t>(experts * n));
  for (__half& b : m.bias) {
    b = to_half(rng.uniform(-0.5f, 0.5f));
  }
  return m;
}

// Per-group int8 quantization of an activation row, as the dp4a path does it:
// scale = max|a| / 127 per group, q = round-to-nearest-even(a * 127 / max|a|)
// clamped to +-127.
void quantize_row(const std::vector<float>& a, int64_t block_size,
                  std::vector<int>& q, std::vector<float>& scale) {
  const int64_t blocks = (static_cast<int64_t>(a.size()) + block_size - 1) / block_size;
  q.assign(a.size(), 0);
  scale.assign(static_cast<size_t>(blocks), 1.0f);
  for (int64_t g = 0; g < blocks; ++g) {
    const int64_t end = std::min<int64_t>((g + 1) * block_size, a.size());
    float gmax = 0.0f;
    for (int64_t i = g * block_size; i < end; ++i) {
      gmax = std::max(gmax, std::fabs(a[i]));
    }
    scale[g] = gmax > 0.0f ? gmax * (1.0f / 127.0f) : 1.0f;
    const float inv = gmax > 0.0f ? 127.0f / gmax : 0.0f;
    for (int64_t i = g * block_size; i < end; ++i) {
      q[i] = std::max(-127, std::min(127, static_cast<int>(std::nearbyint(a[i] * inv))));
    }
  }
}

// One output row of an expert GEMV: sum over k of a[k] * (w - zp) * scale.
double gemv(const Matrix& m, int64_t e, int64_t row,
            const std::vector<float>& a, const std::vector<int>& aq,
            const std::vector<float>& a_scale, const Category& c) {
  double acc = 0.0;
  for (int64_t g = 0; g < m.blocks; ++g) {
    const int zp = m.zp(e, row, g, c.zero_points);
    const float s = m.scale(e, row, g);
    const int64_t end = std::min((g + 1) * c.block_size, m.k);
    if (c.dp4a) {
      int64_t dot = 0;
      for (int64_t i = g * c.block_size; i < end; ++i) {
        dot += static_cast<int64_t>(aq[i]) * (m.q(e, row, i) - zp);
      }
      acc += static_cast<double>(a_scale[g] * s) * static_cast<double>(dot);
    } else {
      double dot = 0.0;
      for (int64_t i = g * c.block_size; i < end; ++i) {
        dot += static_cast<double>(a[i]) * (m.q(e, row, i) - zp);
      }
      acc += dot * s;
    }
  }
  return acc;
}

struct Problem {
  Shape shape;
  Matrix fc1, fc2;
  std::vector<__half> input;
  std::vector<int32_t> indices;
  std::vector<__half> weights;
};

Problem make_problem(const Shape& shape, int64_t block_size, uint64_t seed) {
  Rng rng(seed);
  Problem p;
  p.shape = shape;
  p.fc1 = make_matrix(shape.experts, 2 * shape.inter, shape.hidden, block_size,
                      0.002f, 0.02f, rng);
  p.fc2 = make_matrix(shape.experts, shape.hidden, shape.inter, block_size,
                      0.002f, 0.02f, rng);
  p.input.resize(static_cast<size_t>(shape.hidden));
  for (__half& v : p.input) {
    v = to_half(rng.uniform(-1.0f, 1.0f));
  }
  // k distinct experts, as top-k routing gives.
  std::vector<int32_t> pool(static_cast<size_t>(shape.experts));
  for (int64_t e = 0; e < shape.experts; ++e) {
    pool[e] = static_cast<int32_t>(e);
  }
  for (int64_t s = 0; s < shape.k; ++s) {
    const size_t pick = s + rng.next() % (pool.size() - s);
    std::swap(pool[s], pool[pick]);
    p.indices.push_back(pool[s]);
    p.weights.push_back(to_half(rng.uniform(0.1f, 0.5f)));
  }
  return p;
}

std::vector<__half> reference(const Problem& p, const Category& c) {
  const Shape& sh = p.shape;
  std::vector<float> x(static_cast<size_t>(sh.hidden));
  for (int64_t i = 0; i < sh.hidden; ++i) {
    x[i] = to_float(p.input[i]);
  }
  std::vector<int> xq;
  std::vector<float> x_scale;
  if (c.dp4a) {
    quantize_row(x, c.block_size, xq, x_scale);
  }
  std::vector<double> out(static_cast<size_t>(sh.hidden), 0.0);
  for (int64_t s = 0; s < sh.k; ++s) {
    const int64_t e = p.indices[s];
    // FC1 outputs pair (gate, linear) at (2j, 2j+1); SwiGLU, stored as fp16.
    std::vector<float> act(static_cast<size_t>(sh.inter));
    for (int64_t j = 0; j < sh.inter; ++j) {
      double g = gemv(p.fc1, e, 2 * j, x, xq, x_scale, c);
      double l = gemv(p.fc1, e, 2 * j + 1, x, xq, x_scale, c);
      if (c.bias) {
        g += to_float(p.fc1.bias[e * p.fc1.n + 2 * j]);
        l += to_float(p.fc1.bias[e * p.fc1.n + 2 * j + 1]);
      }
      const double G = std::min(g, static_cast<double>(kSwigluLimit));
      const double L = std::max(std::min(l, static_cast<double>(kSwigluLimit)),
                                -static_cast<double>(kSwigluLimit));
      const double sig = 1.0 / (1.0 + std::exp(-kSwigluAlpha * G));
      act[j] = to_float(to_half(G * sig * (L + kSwigluBeta)));
    }
    std::vector<int> aq;
    std::vector<float> a_scale;
    if (c.dp4a) {
      quantize_row(act, c.block_size, aq, a_scale);
    }
    const double w = to_float(p.weights[s]);
    // FC2, scaled by the routing weight and stored as fp16 per slot.
    for (int64_t n = 0; n < sh.hidden; ++n) {
      double v = gemv(p.fc2, e, n, act, aq, a_scale, c);
      if (c.bias) {
        v += to_float(p.fc2.bias[e * p.fc2.n + n]);
      }
      out[n] += to_float(to_half(w * v));
    }
  }
  std::vector<__half> result(out.size());
  for (size_t i = 0; i < out.size(); ++i) {
    result[i] = to_half(out[i]);
  }
  return result;
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count) {
    if (count != 0) {
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
  }
  ~DeviceBuffer() {
    if (data_) {
      (void)hipFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() { return data_; }

private:
  T* data_ = nullptr;
};

template <typename T> void upload(DeviceBuffer<T>& d, const std::vector<T>& h) {
  HIP_CHECK(hipMemcpy(d.get(), h.data(), h.size() * sizeof(T),
                      hipMemcpyHostToDevice));
}

// Device copies of a problem plus the call's scratch and a guarded output.
struct Device {
  explicit Device(const Problem& p, int64_t block_size)
      : input(p.input.size()), indices(p.indices.size()),
        weights(p.weights.size()), fc1_w(p.fc1.weights.size()),
        fc1_s(p.fc1.scales.size()), fc1_z(p.fc1.zero_points.size()),
        fc1_b(p.fc1.bias.size()), fc2_w(p.fc2.weights.size()),
        fc2_s(p.fc2.scales.size()), fc2_z(p.fc2.zero_points.size()),
        fc2_b(p.fc2.bias.size()),
        slot(static_cast<size_t>(p.shape.k * p.shape.hidden)),
        act(static_cast<size_t>(p.shape.k * p.shape.inter)),
        output(static_cast<size_t>(p.shape.hidden + 2 * kGuardElements)),
        a_qb_in(static_cast<size_t>(p.shape.hidden)),
        a_scale_in(static_cast<size_t>(p.shape.hidden / block_size + 1)),
        a_qb_mid(static_cast<size_t>(p.shape.k * p.shape.inter)),
        a_scale_mid(static_cast<size_t>(p.shape.k * (p.shape.inter / block_size + 1))) {
    upload(input, p.input);
    upload(indices, p.indices);
    upload(weights, p.weights);
    upload(fc1_w, p.fc1.weights);
    upload(fc1_s, p.fc1.scales);
    upload(fc1_z, p.fc1.zero_points);
    upload(fc1_b, p.fc1.bias);
    upload(fc2_w, p.fc2.weights);
    upload(fc2_s, p.fc2.scales);
    upload(fc2_z, p.fc2.zero_points);
    upload(fc2_b, p.fc2.bias);
  }
  DeviceBuffer<__half> input;
  DeviceBuffer<int32_t> indices;
  DeviceBuffer<__half> weights;
  DeviceBuffer<uint8_t> fc1_w;
  DeviceBuffer<__half> fc1_s;
  DeviceBuffer<uint8_t> fc1_z;
  DeviceBuffer<__half> fc1_b;
  DeviceBuffer<uint8_t> fc2_w;
  DeviceBuffer<__half> fc2_s;
  DeviceBuffer<uint8_t> fc2_z;
  DeviceBuffer<__half> fc2_b;
  DeviceBuffer<__half> slot;
  DeviceBuffer<__half> act;
  DeviceBuffer<__half> output;
  DeviceBuffer<int8_t> a_qb_in;
  DeviceBuffer<float> a_scale_in;
  DeviceBuffer<int8_t> a_qb_mid;
  DeviceBuffer<float> a_scale_mid;
};

int launch(Device& d, const Shape& sh, const Category& c, hipStream_t stream,
           int64_t element_size = 2, int64_t hidden = -1, int64_t k = -1) {
  hidden = hidden < 0 ? sh.hidden : hidden;
  k = k < 0 ? sh.k : k;
  const void* z1 = c.zero_points ? d.fc1_z.get() : nullptr;
  const void* z2 = c.zero_points ? d.fc2_z.get() : nullptr;
  const void* b1 = c.bias ? d.fc1_b.get() : nullptr;
  const void* b2 = c.bias ? d.fc2_b.get() : nullptr;
  __half* out = d.output.get() + kGuardElements;
  if (c.dp4a) {
    return hip_qmoe_decode_fused_dp4a(
        reinterpret_cast<void*>(stream), d.input.get(), d.indices.get(),
        d.weights.get(), d.fc1_w.get(), d.fc1_s.get(), z1, b1, d.fc2_w.get(),
        d.fc2_s.get(), z2, b2, d.slot.get(), d.act.get(), out, d.a_qb_in.get(),
        d.a_scale_in.get(), d.a_qb_mid.get(), d.a_scale_mid.get(), hidden,
        sh.inter, k, c.block_size, kSwigluAlpha, kSwigluBeta, kSwigluLimit,
        element_size, sh.experts);
  }
  return hip_qmoe_decode_fused(
      reinterpret_cast<void*>(stream), d.input.get(), d.indices.get(),
      d.weights.get(), d.fc1_w.get(), d.fc1_s.get(), z1, b1, d.fc2_w.get(),
      d.fc2_s.get(), z2, b2, d.slot.get(), d.act.get(), out, hidden, sh.inter,
      k, c.block_size, kSwigluAlpha, kSwigluBeta, kSwigluLimit, element_size,
      sh.experts);
}

std::vector<uint16_t> read_output(Device& d, size_t count) {
  std::vector<uint16_t> bits(count + 2 * kGuardElements);
  HIP_CHECK(hipMemcpy(bits.data(), d.output.get(), bits.size() * sizeof(uint16_t),
                      hipMemcpyDeviceToHost));
  return bits;
}

// A NaN no case produces, so an element left unwritten fails the compare.
constexpr uint16_t kFillBits = 0x7fff;

void fill_output(Device& d, size_t count) {
  const std::vector<uint16_t> fill(count + 2 * kGuardElements, kFillBits);
  HIP_CHECK(hipMemcpy(d.output.get(), fill.data(), fill.size() * sizeof(uint16_t),
                      hipMemcpyHostToDevice));
}

bool guards_intact(const std::vector<uint16_t>& bits, size_t count) {
  for (int i = 0; i < kGuardElements; ++i) {
    if (bits[i] != kFillBits || bits[kGuardElements + count + i] != kFillBits) {
      return false;
    }
  }
  return true;
}

// Tiny out/results.csv writer (op,leaf,arch,mode,shape,config,time_ms,relL2,
// verdict), opt-in via HIPDNN_RESULTS_CSV (set by the Makefile).
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
    if (!path || !path[0]) {
      return;
    }
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
  ~CsvWriter() {
    if (f_) {
      std::fclose(f_);
    }
  }
  void write(const CsvRow& r) {
    if (!f_) {
      return;
    }
    std::fprintf(f_, "%s,%s,%s,%s,%s,%s,%.6f,%.6e,%s\n",
                 env_or("HIPDNN_RESULTS_OP", "qmoe_decode"),
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

// One case's results.csv row, written when it goes out of scope so that a
// case returning early still reports FAIL.
struct ResultRow {
  CsvRow row;
  ResultRow(std::string shape, std::string config) {
    row.shape = std::move(shape);
    row.config = std::move(config);
    row.rel_l2 = std::numeric_limits<double>::infinity();
    row.verdict = "FAIL";
  }
  ResultRow(const ResultRow&) = delete;
  ResultRow& operator=(const ResultRow&) = delete;
  ~ResultRow() { CsvWriter().write(row); }
  void finish(double time_ms, double rel_l2, bool pass) {
    row.time_ms = time_ms;
    row.rel_l2 = rel_l2;
    row.verdict = pass ? "PASS" : "FAIL";
  }
};

std::string shape_name(const Shape& sh) {
  return "H" + std::to_string(sh.hidden) + "_I" + std::to_string(sh.inter) +
         "_E" + std::to_string(sh.experts) + "_k" + std::to_string(sh.k);
}

std::string category_name(const Category& c) {
  return std::string(c.dp4a ? "dp4a" : "fp") + "_bs" +
         std::to_string(c.block_size) + (c.zero_points ? "_zp" : "") +
         (c.bias ? "_bias" : "");
}

double mean_call_ms(Device& d, const Shape& sh, const Category& c,
                    hipStream_t stream) {
  hipEvent_t start = nullptr, stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, stream));
  for (int i = 0; i < kTimedCalls; ++i) {
    (void)launch(d, sh, c, stream);
  }
  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return static_cast<double>(ms) / kTimedCalls;
}

// First call against the reference, then kRepeatCalls - 1 more that must match
// it bit for bit, then the mean steady-state call time.
bool run_case(const Shape& sh, const Category& c, uint64_t seed,
              hipStream_t stream, const char* label = nullptr) {
  const std::string name = (label ? std::string(label) + "_" : std::string()) +
                           shape_name(sh);
  const std::string config = category_name(c);
  ResultRow result(name, config);
  const Problem p = make_problem(sh, c.block_size, seed);
  const std::vector<__half> expected = reference(p, c);
  Device d(p, c.block_size);
  const size_t count = static_cast<size_t>(sh.hidden);
  fill_output(d, count);

  const int status = launch(d, sh, c, stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  const std::vector<uint16_t> first = read_output(d, count);
  const bool guards = guards_intact(first, count);

  double err_sq = 0.0, ref_sq = 0.0, max_abs = 0.0, ref_max = 0.0;
  int64_t mismatches = 0;
  for (const __half& h : expected) {
    ref_max = std::max(ref_max, std::abs(static_cast<double>(to_float(h))));
  }
  for (size_t i = 0; i < count; ++i) {
    uint16_t bits = first[kGuardElements + i];
    __half h;
    std::memcpy(&h, &bits, sizeof(h));
    const double got = to_float(h);
    const double want = to_float(expected[i]);
    const double delta = std::isnan(got) ? std::numeric_limits<double>::infinity()
                                         : std::abs(got - want);
    max_abs = std::max(max_abs, delta);
    mismatches +=
        delta <= kRelTolerance * std::abs(want) + kScaleTolerance * ref_max ? 0 : 1;
    err_sq += delta * delta;
    ref_sq += want * want;
  }
  const double rel_l2 = std::sqrt(err_sq / std::max(ref_sq, 1e-30));
  const bool accurate = rel_l2 <= kRelL2Tolerance && mismatches == 0;

  int differing_call = -1;
  for (int call = 1; call < kRepeatCalls && status == 0; ++call) {
    (void)launch(d, sh, c, stream);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (read_output(d, count) != first) {
      differing_call = call;
      break;
    }
  }
  const bool pass = status == 0 && guards && accurate && differing_call < 0;
  const double ms = pass ? mean_call_ms(d, sh, c, stream) : 0.0;

  std::cout << (pass ? "PASS" : "FAIL") << " shape=" << name
            << " config=" << config << " status=" << status
            << " relL2=" << rel_l2 << " maxAbs=" << max_abs
            << " mismatches=" << mismatches << " guards=" << (guards ? "PASS" : "FAIL") << " repeat="
            << (differing_call < 0 ? std::string("identical")
                                   : "call " + std::to_string(differing_call) +
                                         " differs")
            << " time_ms=" << ms << '\n';
  result.finish(ms, rel_l2, pass);
  return pass;
}

// A failed HIP call earlier on this thread leaves its error pending. The first
// call of a shape times its tiling candidates before computing; a stale error
// must neither fail that timing nor be reported as the call's own.
bool run_pending_error(bool dp4a, uint64_t seed, hipStream_t stream) {
  // A shape no other case uses, so this call is the one that picks the tiling.
  const Shape sh{96, 160, 6, 2, 1};
  (void)hipSetDevice(-1);
  if (hipPeekAtLastError() == hipSuccess) {
    std::cout << "FAIL pending_error: hipSetDevice(-1) left no pending error\n";
    ResultRow result("pending_error_" + shape_name(sh), dp4a ? "dp4a" : "fp");
    return false;
  }
  return run_case(sh, {dp4a, true, true, 32}, seed, stream, "pending_error");
}

// Arguments the launchers must refuse with -1 and without writing anything.
bool run_rejected(const char* label, const Category& c, int64_t element_size,
                  int64_t hidden, int64_t k, hipStream_t stream) {
  ResultRow result(std::string("rejected_") + label, category_name(c));
  const Shape sh{64, 64, 4, 2, 1};
  const Problem p = make_problem(sh, 32, 1);
  Device d(p, 32);
  fill_output(d, static_cast<size_t>(sh.hidden));
  const int status = launch(d, sh, c, stream, element_size, hidden, k);
  HIP_CHECK(hipStreamSynchronize(stream));
  const std::vector<uint16_t> bits = read_output(d, static_cast<size_t>(sh.hidden));
  const bool untouched =
      std::all_of(bits.begin(), bits.end(), [](uint16_t b) { return b == kFillBits; });
  const bool pass = status == -1 && untouched;
  std::cout << (pass ? "PASS" : "FAIL") << " rejected=" << label
            << " config=" << category_name(c) << " status=" << status
            << " output=" << (untouched ? "untouched" : "WRITTEN") << '\n';
  result.finish(0.0, pass ? 0.0 : 1.0, pass);
  return pass;
}

bool parse_options(int argc, char** argv, int* coverage) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--coverage" && i + 1 < argc) {
      *coverage = std::atoi(argv[++i]);
    } else {
      std::cerr << "usage: " << argv[0] << " [--coverage 1|2|3]\n";
      return false;
    }
  }
  return *coverage >= 1 && *coverage <= 3;
}

} // namespace

int main(int argc, char** argv) {
  int coverage = 3;
  if (!parse_options(argc, argv, &coverage)) {
    return 2;
  }

  int shape_count = 0;
  for (const Shape& sh : kShapes) {
    shape_count += sh.min_coverage <= coverage ? 1 : 0;
  }
  constexpr int kCategoryCount = 2 * 2 * 2 * 3;
  constexpr int kExtraCount = 2 + 6;
  std::cout << "qmoe_decode coverage=" << coverage
            << " cases=" << shape_count * kCategoryCount + kExtraCount << '\n';

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  uint64_t seed = 0;
  // The pending-error cases first, while their shape is still untimed.
  all_pass = run_pending_error(false, ++seed, stream) && all_pass;
  all_pass = run_pending_error(true, ++seed, stream) && all_pass;
  for (const Shape& sh : kShapes) {
    ++seed;
    if (sh.min_coverage > coverage) {
      continue;
    }
    for (bool dp4a : {false, true}) {
      for (int64_t block_size : kBlockSizes) {
        for (bool zero_points : {false, true}) {
          for (bool bias : {false, true}) {
            all_pass = run_case(sh, {dp4a, zero_points, bias, block_size}, seed,
                                stream) &&
                       all_pass;
          }
        }
      }
    }
  }
  all_pass = run_rejected("fp32", {false, false, false, 32}, 4, -1, -1, stream) && all_pass;
  all_pass = run_rejected("odd_block", {false, false, false, 15}, 2, -1, -1, stream) && all_pass;
  all_pass = run_rejected("dp4a_block48", {true, false, false, 48}, 2, -1, -1, stream) && all_pass;
  all_pass = run_rejected("hidden48", {false, false, false, 32}, 2, 48, -1, stream) && all_pass;
  all_pass = run_rejected("k0", {false, false, false, 32}, 2, -1, 0, stream) && all_pass;
  all_pass = run_rejected("dp4a_k0", {true, false, false, 32}, 2, -1, 0, stream) && all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
