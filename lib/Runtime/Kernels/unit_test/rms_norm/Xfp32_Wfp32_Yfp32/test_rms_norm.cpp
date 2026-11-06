/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for the fp32 RMS normalization.
 */

#include "hip_custom_kernels.h"

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

constexpr int kGuardElements = 32;
constexpr float kEpsilon = 1e-5f;
// The kernel accumulates in fp32 in its own order; the reference in double.
constexpr double kRelTolerance = 2e-5;
constexpr double kAbsTolerance = 1e-6;
constexpr int kWarmupLaunches = 3;
constexpr int kTimedLaunches = 50;

// fp32 rows take the float4 body when the row is whole float4s with 16-byte
// aligned pointers, else the scalar one, and pick their block by width: from
// 2048 a 1024-thread block, below that a 256-thread block, and below 512 a
// 32-thread block when the row takes the float4 body. These widths take each
// block x body pair; 511/512 and 2047/2048 sit on the width boundaries and
// 5120 loops more than once per thread.
constexpr int64_t kPathWidths[] = {64, 511, 512, 1030, 2047, 2048, 4099, 5120};

struct Shape {
  int64_t rows;
  int64_t width;
  int min_coverage;
};

// Prefill-sized row counts at real hidden sizes; COVERAGE thins these.
constexpr Shape kTypicalShapes[] = {
    {64, 4096, 1},
    {512, 2048, 2},
    {128, 5120, 2},
    {2048, 4096, 3},
    {512, 8192, 3},
};

struct Case {
  int64_t rows;
  int64_t width;
  int64_t scale_rows;
  // Offset every pointer by one element, so a row of whole float4s still
  // takes the scalar body.
  bool misaligned;
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

std::vector<float> reference(const Case& c, const std::vector<float>& x,
                             const std::vector<float>& g) {
  std::vector<float> y(x.size());
  for (int64_t r = 0; r < c.rows; ++r) {
    const float* xr = x.data() + r * c.width;
    const float* gr = g.data() + (r % c.scale_rows) * c.width;
    double sumsq = 0.0;
    for (int64_t i = 0; i < c.width; ++i) {
      sumsq += static_cast<double>(xr[i]) * xr[i];
    }
    const double inv_rms =
        1.0 / std::sqrt(sumsq / static_cast<double>(c.width) + kEpsilon);
    for (int64_t i = 0; i < c.width; ++i) {
      y[r * c.width + i] = static_cast<float>(xr[i] * inv_rms * gr[i]);
    }
  }
  return y;
}

class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count) {
    if (count != 0) {
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), count * sizeof(float)));
    }
  }
  ~DeviceBuffer() {
    if (data_) {
      (void)hipFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  float* get() { return data_; }

private:
  float* data_ = nullptr;
};

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
                 env_or("HIPDNN_RESULTS_OP", "rms_norm"),
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

// The fp32 block and body the launcher picks for this case, for the label.
std::string config_label(const Case& c) {
  const bool vec4 = c.width % 4 == 0 && !c.misaligned;
  const int threads = c.width >= 2048 ? 1024 : (c.width >= 512 || !vec4) ? 256 : 32;
  return "block" + std::to_string(threads) + (vec4 ? "_vec4" : "_scalar");
}

// Mean time of back-to-back launches on warm caches, in ms.
double time_launches(const Case& c, const float* x, const float* g, float* y,
                     hipStream_t stream) {
  for (int i = 0; i < kWarmupLaunches; ++i) {
    (void)hip_rms_norm(reinterpret_cast<void*>(stream), x, g, y, c.rows,
                       c.width, c.scale_rows, kEpsilon, HIP_DTYPE_FLOAT32);
  }
  hipEvent_t start = nullptr, stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, stream));
  for (int i = 0; i < kTimedLaunches; ++i) {
    (void)hip_rms_norm(reinterpret_cast<void*>(stream), x, g, y, c.rows,
                       c.width, c.scale_rows, kEpsilon, HIP_DTYPE_FLOAT32);
  }
  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return static_cast<double>(ms) / kTimedLaunches;
}

bool run_case(const Case& c, uint64_t seed, hipStream_t stream) {
  char name[96];
  std::snprintf(name, sizeof(name), "R%lld_D%lld_G%lld%s",
                static_cast<long long>(c.rows), static_cast<long long>(c.width),
                static_cast<long long>(c.scale_rows),
                c.misaligned ? "_misaligned" : "");
  const std::string config = config_label(c);
  ResultRow result(name, config);

  Rng rng(seed);
  const int64_t count = c.rows * c.width;
  std::vector<float> x(static_cast<size_t>(count));
  for (float& v : x) {
    v = rng.uniform(-2.0f, 2.0f);
  }
  std::vector<float> g(static_cast<size_t>(c.scale_rows * c.width));
  for (float& v : g) {
    v = rng.uniform(0.5f, 1.5f);
  }
  const std::vector<float> expected = reference(c, x, g);

  // One extra element in front of every buffer moves its pointer off 16-byte
  // alignment when the case asks for it.
  const size_t shift = c.misaligned ? 1 : 0;
  DeviceBuffer d_x(x.size() + 1);
  DeviceBuffer d_g(g.size() + 1);
  DeviceBuffer d_y(static_cast<size_t>(count) + 2 * kGuardElements + 1);
  float* x_ptr = d_x.get() + shift;
  float* g_ptr = d_g.get() + shift;
  float* y_base = d_y.get() + shift;
  HIP_CHECK(hipMemcpy(x_ptr, x.data(), x.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(g_ptr, g.data(), g.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  // A NaN fill fails any element the kernel leaves unwritten or writes past.
  const std::vector<float> fill(static_cast<size_t>(count) + 2 * kGuardElements,
                                std::numeric_limits<float>::quiet_NaN());
  HIP_CHECK(hipMemcpy(y_base, fill.data(), fill.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  const int status =
      hip_rms_norm(reinterpret_cast<void*>(stream), x_ptr, g_ptr,
                   y_base + kGuardElements, c.rows, c.width, c.scale_rows,
                   kEpsilon, HIP_DTYPE_FLOAT32);
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<float> actual(fill.size());
  HIP_CHECK(hipMemcpy(actual.data(), y_base, actual.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  bool guards = true;
  for (int i = 0; i < kGuardElements; ++i) {
    guards = guards && std::isnan(actual[i]) &&
             std::isnan(actual[kGuardElements + count + i]);
  }
  int64_t mismatches = 0;
  int64_t first = -1;
  double max_rel = 0.0, err_sq = 0.0, ref_sq = 0.0;
  for (int64_t i = 0; i < count; ++i) {
    const double got = actual[kGuardElements + i];
    const double want = expected[i];
    const double delta = std::isnan(got)
                             ? std::numeric_limits<double>::infinity()
                             : std::abs(got - want);
    if (!(delta <= kRelTolerance * std::abs(want) + kAbsTolerance)) {
      if (first < 0) {
        first = i;
      }
      ++mismatches;
    }
    max_rel = std::max(max_rel, delta / std::max(std::abs(want), 1e-30));
    err_sq += delta * delta;
    ref_sq += want * want;
  }
  const double rel_l2 = std::sqrt(err_sq / std::max(ref_sq, 1e-30));
  const bool pass = status == 0 && guards && mismatches == 0;
  const double ms =
      pass ? time_launches(c, x_ptr, g_ptr, y_base + kGuardElements, stream)
           : 0.0;

  std::cout << (pass ? "PASS" : "FAIL") << " shape=" << name
            << " config=" << config << " status=" << status
            << " relL2=" << rel_l2 << " maxRel=" << max_rel
            << " mismatches=" << mismatches
            << " guards=" << (guards ? "PASS" : "FAIL") << " time_ms=" << ms;
  if (first >= 0) {
    std::cout << " first=" << first << " got=" << actual[kGuardElements + first]
              << " want=" << expected[first];
  }
  std::cout << '\n';
  result.finish(ms, rel_l2, pass);
  return pass;
}

// Arguments hip_rms_norm must refuse (-1) or treat as empty (0), writing
// nothing either way.
bool run_edge(const char* label, int64_t outer, int64_t scale_rows, int dtype,
              int expected_status, hipStream_t stream) {
  ResultRow result(std::string("edge_") + label, "-");
  constexpr int64_t kWidth = 64;
  DeviceBuffer d_x(kWidth * 4);
  DeviceBuffer d_g(kWidth * 4);
  const std::vector<float> fill(kWidth * 4,
                                std::numeric_limits<float>::quiet_NaN());
  DeviceBuffer d_y(fill.size());
  HIP_CHECK(hipMemcpy(d_y.get(), fill.data(), fill.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  const int status =
      hip_rms_norm(reinterpret_cast<void*>(stream), d_x.get(), d_g.get(),
                   d_y.get(), outer, kWidth, scale_rows, kEpsilon, dtype);
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<float> actual(fill.size());
  HIP_CHECK(hipMemcpy(actual.data(), d_y.get(), actual.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  bool untouched = true;
  for (float v : actual) {
    untouched = untouched && std::isnan(v);
  }
  const bool pass = status == expected_status && untouched;
  std::cout << (pass ? "PASS" : "FAIL") << " edge=" << label
            << " status=" << status
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

  // Every block x body path, aligned and not, shared and grouped gains, one
  // row (a decode step: the whole grid is one block) and several, at every
  // tier; then a one-element row; then the typical sizes COVERAGE selects.
  std::vector<Case> cases;
  for (int64_t width : kPathWidths) {
    for (bool misaligned : {false, true}) {
      for (int64_t scale_rows : {1, 3}) {
        for (int64_t rows : {1, 6}) {
          if (scale_rows <= rows) {
            cases.push_back({rows, width, scale_rows, misaligned});
          }
        }
      }
    }
  }
  cases.push_back({4, 1, 1, false});
  for (const Shape& shape : kTypicalShapes) {
    if (shape.min_coverage <= coverage) {
      cases.push_back({shape.rows, shape.width, 1, false});
    }
  }
  constexpr int kEdgeCount = 3;
  std::cout << "rms_norm fp32 coverage=" << coverage
            << " cases=" << cases.size() + kEdgeCount << '\n';

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  uint64_t seed = 0;
  for (const Case& c : cases) {
    all_pass = run_case(c, ++seed, stream) && all_pass;
  }
  all_pass = run_edge("outer0", 0, 1, HIP_DTYPE_FLOAT32, 0, stream) && all_pass;
  all_pass = run_edge("scale_rows0", 2, 0, HIP_DTYPE_FLOAT32, -1, stream) &&
             all_pass;
  all_pass = run_edge("dtype", 2, 1, HIP_DTYPE_INT32, -1, stream) && all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
