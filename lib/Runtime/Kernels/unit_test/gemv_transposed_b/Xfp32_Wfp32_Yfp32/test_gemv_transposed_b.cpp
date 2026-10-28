/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for the fp32 transposed-B GEMVs
 * (C[1,N] = A[1,K] * transpose(B[N,K])).
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
constexpr int kWarmupLaunches = 3;
constexpr int kTimedLaunches = 50;
// The multi-wave launcher takes at most this many outputs.
constexpr int kMultiwaveMaxN = 256;

// The two launchers: one wave per output (hip_gemv_f32_transposed_b), and a
// 256-thread block per output (hip_gemv_f32_transposed_b_multiwave_gfx1151),
// which the runtime runs for every one-row F32 projection with N <= 256.
enum class Launcher { kOneWave, kMultiwave };

const char* launcher_name(Launcher launcher) {
  return launcher == Launcher::kOneWave ? "one_wave" : "multiwave";
}

int launch(Launcher launcher, hipStream_t stream, const float* a,
           const float* b, float* c, int n, int k) {
  return launcher == Launcher::kOneWave
             ? hip_gemv_f32_transposed_b(stream, a, b, c, n, k)
             : hip_gemv_f32_transposed_b_multiwave_gfx1151(stream, a, b, c, n,
                                                           k);
}

struct Shape {
  int n;
  int k;
  int min_coverage;
};

// Every tier: the outputs and reduction lengths the runtime sends to a GEMV
// (1 <= N <= 256, K >= 256), K a multiple of 4 and not, and reductions
// shorter than one wave, which both launchers also accept.
constexpr Shape kPathShapes[] = {
    {1, 256, 1},    {7, 256, 1},   {128, 1000, 1}, {256, 1024, 1},
    {256, 4096, 1}, {64, 4097, 1}, {3, 1, 1},      {5, 31, 1},
};

// Real decode projection sizes; COVERAGE thins these. N above 256 runs on the
// one-wave launcher only.
constexpr Shape kTypicalShapes[] = {
    {128, 2048, 1},
    {256, 5120, 2},
    {32, 14336, 2},
    {256, 14336, 3},
    {1024, 4096, 3},
};

struct Case {
  Launcher launcher;
  int n;
  int k;
  // Offset every pointer by one element off 16-byte alignment.
  bool misaligned;
};

// The kernel the launcher picks for this case, for the label.
std::string config_label(const Case& c) {
  if (c.launcher == Launcher::kOneWave) {
    return "one_wave";
  }
  return c.k % 4 == 0 && !c.misaligned ? "multiwave_vec4" : "multiwave_scalar";
}

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
                 env_or("HIPDNN_RESULTS_OP", "gemv_transposed_b"),
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

// Mean time of back-to-back launches on warm caches, in ms.
double time_launches(const Case& c, const float* a, const float* b, float* y,
                     hipStream_t stream) {
  for (int i = 0; i < kWarmupLaunches; ++i) {
    (void)launch(c.launcher, stream, a, b, y, c.n, c.k);
  }
  hipEvent_t start = nullptr, stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, stream));
  for (int i = 0; i < kTimedLaunches; ++i) {
    (void)launch(c.launcher, stream, a, b, y, c.n, c.k);
  }
  HIP_CHECK(hipEventRecord(stop, stream));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return static_cast<double>(ms) / kTimedLaunches;
}

// Leaves a failed HIP call's error as the thread's last error, the state an
// earlier runtime call can leave behind before the matmul runs.
void leave_stale_error() {
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  (void)hipSetDevice(count);
}

bool run_case(const Case& c, uint64_t seed, hipStream_t stream,
              bool stale_error = false) {
  char name[96];
  std::snprintf(name, sizeof(name), "%s_N%d_K%d%s%s",
                launcher_name(c.launcher), c.n, c.k,
                c.misaligned ? "_misaligned" : "",
                stale_error ? "_stale_error" : "");
  const std::string config = config_label(c);
  ResultRow result(name, config);

  Rng rng(seed);
  std::vector<float> a(static_cast<size_t>(c.k));
  for (float& v : a) {
    v = rng.uniform(-1.0f, 1.0f);
  }
  std::vector<float> b(static_cast<size_t>(c.n) * c.k);
  for (float& v : b) {
    v = rng.uniform(-1.0f, 1.0f);
  }
  // Double-precision reference. Every kernel sums at most about K / 32
  // products per thread in fp32 and then combines at most 16 partial sums
  // (accumulators, wave lanes, waves), so every output is within
  // (K / 32 + 16) units of fp32 rounding of the sum of the products'
  // magnitudes.
  std::vector<double> expected(static_cast<size_t>(c.n));
  std::vector<double> tolerance(static_cast<size_t>(c.n));
  const double steps = std::ceil(c.k / 32.0) + 16.0;
  for (int n = 0; n < c.n; ++n) {
    double sum = 0.0, magnitude = 0.0;
    for (int k = 0; k < c.k; ++k) {
      const double product =
          static_cast<double>(a[k]) * b[static_cast<size_t>(n) * c.k + k];
      sum += product;
      magnitude += std::abs(product);
    }
    expected[n] = sum;
    tolerance[n] = steps * std::ldexp(1.0, -24) * magnitude + 1e-30;
  }

  const size_t shift = c.misaligned ? 1 : 0;
  DeviceBuffer d_a(a.size() + 1);
  DeviceBuffer d_b(b.size() + 1);
  DeviceBuffer d_y(static_cast<size_t>(c.n) + 2 * kGuardElements + 1);
  float* a_ptr = d_a.get() + shift;
  float* b_ptr = d_b.get() + shift;
  float* y_base = d_y.get() + shift;
  HIP_CHECK(hipMemcpy(a_ptr, a.data(), a.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b_ptr, b.data(), b.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  // A NaN fill fails any output the kernel leaves unwritten or writes past.
  const std::vector<float> fill(static_cast<size_t>(c.n) + 2 * kGuardElements,
                                std::numeric_limits<float>::quiet_NaN());
  HIP_CHECK(hipMemcpy(y_base, fill.data(), fill.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  if (stale_error) {
    leave_stale_error();
  }
  const int status = launch(c.launcher, stream, a_ptr, b_ptr,
                            y_base + kGuardElements, c.n, c.k);
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<float> actual(fill.size());
  HIP_CHECK(hipMemcpy(actual.data(), y_base, actual.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  bool guards = true;
  for (int i = 0; i < kGuardElements; ++i) {
    guards = guards && std::isnan(actual[i]) &&
             std::isnan(actual[kGuardElements + c.n + i]);
  }
  int mismatches = 0;
  int first = -1;
  double err_sq = 0.0, ref_sq = 0.0;
  for (int n = 0; n < c.n; ++n) {
    const double got = actual[kGuardElements + n];
    const double delta = std::isnan(got)
                             ? std::numeric_limits<double>::infinity()
                             : std::abs(got - expected[n]);
    if (!(delta <= tolerance[n])) {
      if (first < 0) {
        first = n;
      }
      ++mismatches;
    }
    err_sq += delta * delta;
    ref_sq += expected[n] * expected[n];
  }
  const double rel_l2 = std::sqrt(err_sq / std::max(ref_sq, 1e-30));
  const bool pass = status == 0 && guards && mismatches == 0;
  const double ms =
      pass && !stale_error
          ? time_launches(c, a_ptr, b_ptr, y_base + kGuardElements, stream)
          : 0.0;

  std::cout << (pass ? "PASS" : "FAIL") << " shape=" << name
            << " config=" << config << " status=" << status
            << " relL2=" << rel_l2 << " mismatches=" << mismatches
            << " guards=" << (guards ? "PASS" : "FAIL") << " time_ms=" << ms;
  if (first >= 0) {
    std::cout << " first=" << first << " got=" << actual[kGuardElements + first]
              << " want=" << expected[first] << " tol=" << tolerance[first];
  }
  std::cout << '\n';
  result.finish(ms, rel_l2, pass);
  return pass;
}

// Arguments a launcher must refuse with -1, writing nothing.
struct Edge {
  const char* label;
  bool null_stream, null_a, null_b, null_c;
  int n, k;
};

constexpr Edge kEdges[] = {
    {"n0", false, false, false, false, 0, 256},
    {"k0", false, false, false, false, 4, 0},
    {"negative_n", false, false, false, false, -1, 256},
    {"null_stream", true, false, false, false, 4, 64},
    {"null_a", false, true, false, false, 4, 64},
    {"null_b", false, false, true, false, 4, 64},
    {"null_c", false, false, false, true, 4, 64},
};

// Past the multi-wave launcher's output limit.
constexpr Edge kMultiwaveEdge = {"n257", false, false, false, false,
                                 kMultiwaveMaxN + 1, 64};

bool run_edge(Launcher launcher, const Edge& e, hipStream_t stream) {
  const std::string label =
      std::string(launcher_name(launcher)) + "_edge_" + e.label;
  ResultRow result(label, "-");
  constexpr int kCount = 512;
  DeviceBuffer d_a(kCount);
  DeviceBuffer d_b(static_cast<size_t>(kCount) * 64);
  const std::vector<float> fill(kCount, std::numeric_limits<float>::quiet_NaN());
  DeviceBuffer d_y(fill.size());
  HIP_CHECK(hipMemcpy(d_y.get(), fill.data(), fill.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  const int status = launch(launcher, e.null_stream ? nullptr : stream,
                            e.null_a ? nullptr : d_a.get(),
                            e.null_b ? nullptr : d_b.get(),
                            e.null_c ? nullptr : d_y.get(), e.n, e.k);
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<float> actual(fill.size());
  HIP_CHECK(hipMemcpy(actual.data(), d_y.get(), actual.size() * sizeof(float),
                      hipMemcpyDeviceToHost));
  bool untouched = true;
  for (float v : actual) {
    untouched = untouched && std::isnan(v);
  }
  const bool pass = status == -1 && untouched;
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
  if (const char* env = std::getenv("HIPDNN_UT_COVERAGE")) {
    coverage = std::atoi(env);
  }
  if (!parse_options(argc, argv, &coverage)) {
    return 2;
  }

  // Both launchers on the dispatch-range shapes, aligned and not, at every
  // tier; then the typical sizes COVERAGE selects.
  std::vector<Case> cases;
  for (Launcher launcher : {Launcher::kOneWave, Launcher::kMultiwave}) {
    for (const Shape& shape : kPathShapes) {
      for (bool misaligned : {false, true}) {
        cases.push_back({launcher, shape.n, shape.k, misaligned});
      }
    }
    for (const Shape& shape : kTypicalShapes) {
      if (shape.min_coverage <= coverage &&
          (launcher == Launcher::kOneWave || shape.n <= kMultiwaveMaxN)) {
        cases.push_back({launcher, shape.n, shape.k, false});
      }
    }
  }
  constexpr size_t kEdgeCount = 2 * (sizeof(kEdges) / sizeof(kEdges[0])) + 1;
  constexpr size_t kStaleErrorCount = 2;
  std::cout << "gemv_transposed_b fp32 coverage=" << coverage
            << " cases=" << cases.size() + kStaleErrorCount + kEdgeCount
            << '\n';

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  uint64_t seed = 0;
  for (const Case& c : cases) {
    all_pass = run_case(c, ++seed, stream) && all_pass;
  }
  // A failed HIP call earlier on the thread must not fail either launch.
  for (Launcher launcher : {Launcher::kOneWave, Launcher::kMultiwave}) {
    all_pass = run_case({launcher, 128, 1024, false}, ++seed, stream,
                        /*stale_error=*/true) &&
               all_pass;
  }
  for (Launcher launcher : {Launcher::kOneWave, Launcher::kMultiwave}) {
    for (const Edge& edge : kEdges) {
      all_pass = run_edge(launcher, edge, stream) && all_pass;
    }
  }
  all_pass = run_edge(Launcher::kMultiwave, kMultiwaveEdge, stream) && all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
