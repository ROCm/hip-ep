/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for the bf16-input, fp32-output transposed-B WMMA GEMM
 * (C[M,N] = A[M,K] * transpose(B[N,K])).
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
constexpr int kTimedLaunches = 20;
// The launcher runs 128x64 tiles from this M on, 64x64 tiles below.
constexpr int kTallTileMinM = 128;
// The kernel stages K in slices of this many elements.
constexpr int kSliceK = 64;
// Cases with more multiply-adds than this check a sample of the rows (every
// output of each sampled row) against the CPU reference instead of all rows.
constexpr double kFullCheckMacs = 4e8;

struct Shape {
  int m;
  int n;
  int k;
  int min_coverage;
};

// Every tier: both tile shapes, the smallest M the runtime sends to this GEMM
// (2), outputs that do not fill a whole number of tiles in M or N, K a
// multiple of the 64-wide slice and not, and the smallest K (8).
constexpr Shape kPathShapes[] = {
    {2, 64, 64, 1},   {37, 77, 136, 1},  {5, 70, 1032, 1},
    {127, 64, 8, 1},  {128, 64, 64, 1},  {200, 130, 520, 1},
    {129, 1, 72, 1},  {2, 1024, 3072, 1},
};

// Prefill projection sizes (Llama 3.2 3B and Llama 3.1 8B attention and MLP);
// COVERAGE thins these.
constexpr Shape kTypicalShapes[] = {
    {512, 3072, 3072, 1},
    {512, 1024, 3072, 2},
    {128, 8192, 3072, 2},
    {512, 3072, 8192, 3},
    {512, 14336, 4096, 3},
};

struct Case {
  int m;
  int n;
  int k;
};

// The tile shape the launcher picks for this case, for the label.
const char* config_label(int m) {
  return m >= kTallTileMinM ? "tile128x64" : "tile64x64";
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

// fp32 -> bf16 with round-to-nearest-even (inputs here are finite).
uint16_t to_bf16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

float from_bf16(uint16_t value) {
  const uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result = 0.0f;
  std::memcpy(&result, &bits, sizeof(result));
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
                 env_or("HIPDNN_RESULTS_OP", "gemm_transposed_b"),
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
double time_launches(const Case& c, const uint16_t* a, const uint16_t* b,
                     float* y, hipStream_t stream) {
  for (int i = 0; i < kWarmupLaunches; ++i) {
    (void)hip_gemm_bf16_f32_transposed_b(stream, a, b, y, c.m, c.n, c.k);
  }
  hipEvent_t start = nullptr, stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, stream));
  for (int i = 0; i < kTimedLaunches; ++i) {
    (void)hip_gemm_bf16_f32_transposed_b(stream, a, b, y, c.m, c.n, c.k);
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

// The rows checked against the CPU reference: all of them for small cases;
// otherwise the first and last row of every tile row plus every 7th row.
std::vector<int> checked_rows(const Case& c) {
  std::vector<int> rows;
  const bool all = static_cast<double>(c.m) * c.n * c.k <= kFullCheckMacs;
  for (int m = 0; m < c.m; ++m) {
    if (all || m % 7 == 0 || m % 64 == 0 || m % 64 == 63 || m == c.m - 1) {
      rows.push_back(m);
    }
  }
  return rows;
}

bool run_case(const Case& c, uint64_t seed, hipStream_t stream,
              bool stale_error = false) {
  char name[96];
  std::snprintf(name, sizeof(name), "M%d_N%d_K%d%s", c.m, c.n, c.k,
                stale_error ? "_stale_error" : "");
  const std::string config = config_label(c.m);
  ResultRow result(name, config);

  Rng rng(seed);
  std::vector<uint16_t> a(static_cast<size_t>(c.m) * c.k);
  std::vector<uint16_t> b(static_cast<size_t>(c.n) * c.k);
  std::vector<float> af(a.size()), bf(b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    a[i] = to_bf16(rng.uniform(-1.0f, 1.0f));
    af[i] = from_bf16(a[i]);
  }
  for (size_t i = 0; i < b.size(); ++i) {
    b[i] = to_bf16(rng.uniform(-1.0f, 1.0f));
    bf[i] = from_bf16(b[i]);
  }

  DeviceBuffer<uint16_t> d_a(a.size());
  DeviceBuffer<uint16_t> d_b(b.size());
  const size_t out_count = static_cast<size_t>(c.m) * c.n;
  DeviceBuffer<float> d_y(out_count + 2 * kGuardElements);
  HIP_CHECK(hipMemcpy(d_a.get(), a.data(), a.size() * sizeof(uint16_t),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b.get(), b.data(), b.size() * sizeof(uint16_t),
                      hipMemcpyHostToDevice));
  // A NaN fill fails any output the kernel leaves unwritten or writes past.
  const std::vector<float> fill(out_count + 2 * kGuardElements,
                                std::numeric_limits<float>::quiet_NaN());
  HIP_CHECK(hipMemcpy(d_y.get(), fill.data(), fill.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  float* y = d_y.get() + kGuardElements;

  if (stale_error) {
    leave_stale_error();
  }
  const int status = hip_gemm_bf16_f32_transposed_b(stream, d_a.get(),
                                                    d_b.get(), y, c.m, c.n, c.k);
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<float> actual(fill.size());
  HIP_CHECK(hipMemcpy(actual.data(), d_y.get(), actual.size() * sizeof(float),
                      hipMemcpyDeviceToHost));

  bool guards = true;
  for (int i = 0; i < kGuardElements; ++i) {
    guards = guards && std::isnan(actual[i]) &&
             std::isnan(actual[kGuardElements + out_count + i]);
  }
  // Rows not checked against the reference must still be written.
  int unwritten = 0;
  for (size_t i = 0; i < out_count; ++i) {
    unwritten += std::isnan(actual[kGuardElements + i]) ? 1 : 0;
  }
  // Double-precision reference on the bf16 values. Each product of two bf16
  // values is exact in fp32; the kernel accumulates them in fp32 WMMA tiles,
  // 16 products per instruction in an order the ISA does not specify. Every
  // output must be within (K / 16 + 32) units of fp32 rounding of the sum of
  // the products' magnitudes (on gfx1151 the largest error over this test's
  // cases is 24 units).
  const double steps = c.k / 16.0 + 32.0;
  int mismatches = 0;
  long first = -1;
  double first_want = 0.0, first_tol = 0.0;
  double err_sq = 0.0, ref_sq = 0.0;
  for (int m : checked_rows(c)) {
    const float* arow = &af[static_cast<size_t>(m) * c.k];
    for (int n = 0; n < c.n; ++n) {
      const float* brow = &bf[static_cast<size_t>(n) * c.k];
      double sum = 0.0, magnitude = 0.0;
      for (int k = 0; k < c.k; ++k) {
        const double product = static_cast<double>(arow[k]) * brow[k];
        sum += product;
        magnitude += std::abs(product);
      }
      const double tolerance =
          steps * std::ldexp(1.0, -24) * magnitude + 1e-30;
      const size_t index = static_cast<size_t>(m) * c.n + n;
      const double got = actual[kGuardElements + index];
      const double delta = std::isnan(got)
                               ? std::numeric_limits<double>::infinity()
                               : std::abs(got - sum);
      if (!(delta <= tolerance)) {
        if (first < 0) {
          first = static_cast<long>(index);
          first_want = sum;
          first_tol = tolerance;
        }
        ++mismatches;
      }
      err_sq += delta * delta;
      ref_sq += sum * sum;
    }
  }
  const double rel_l2 = std::sqrt(err_sq / std::max(ref_sq, 1e-30));
  const bool pass = status == 0 && guards && unwritten == 0 && mismatches == 0;
  const double ms = pass && !stale_error
                        ? time_launches(c, d_a.get(), d_b.get(), y, stream)
                        : 0.0;

  std::cout << (pass ? "PASS" : "FAIL") << " shape=" << name
            << " config=" << config << " status=" << status
            << " relL2=" << rel_l2 << " mismatches=" << mismatches
            << " unwritten=" << unwritten
            << " guards=" << (guards ? "PASS" : "FAIL") << " time_ms=" << ms;
  if (first >= 0) {
    std::cout << " first=" << first << " got=" << actual[kGuardElements + first]
              << " want=" << first_want << " tol=" << first_tol;
  }
  std::cout << '\n';
  result.finish(ms, rel_l2, pass);
  return pass;
}

// Arguments the launcher must refuse with -1, writing nothing.
struct Edge {
  const char* label;
  bool null_stream, null_a, null_b, null_c;
  // Offset A or B by four bf16 elements (8 bytes) off 16-byte alignment; the
  // kernel loads eight bf16 values at a time.
  bool misaligned_a, misaligned_b;
  int m, n, k;
};

constexpr Edge kEdges[] = {
    {"m0", false, false, false, false, false, false, 0, 64, 64},
    {"n0", false, false, false, false, false, false, 4, 0, 64},
    {"k0", false, false, false, false, false, false, 4, 64, 0},
    {"negative_m", false, false, false, false, false, false, -1, 64, 64},
    {"negative_n", false, false, false, false, false, false, 4, -1, 64},
    {"k_not_multiple_of_8", false, false, false, false, false, false, 4, 64, 60},
    {"null_stream", true, false, false, false, false, false, 4, 64, 64},
    {"null_a", false, true, false, false, false, false, 4, 64, 64},
    {"null_b", false, false, true, false, false, false, 4, 64, 64},
    {"null_c", false, false, false, true, false, false, 4, 64, 64},
    {"misaligned_a", false, false, false, false, true, false, 4, 64, 64},
    {"misaligned_b", false, false, false, false, false, true, 4, 64, 64},
};

bool run_edge(const Edge& e, hipStream_t stream) {
  const std::string label = std::string("edge_") + e.label;
  ResultRow result(label, "-");
  constexpr int kCount = 512;
  DeviceBuffer<uint16_t> d_a(static_cast<size_t>(kCount) * 64 + 8);
  DeviceBuffer<uint16_t> d_b(static_cast<size_t>(kCount) * 64 + 8);
  const std::vector<float> fill(kCount, std::numeric_limits<float>::quiet_NaN());
  DeviceBuffer<float> d_y(fill.size());
  HIP_CHECK(hipMemcpy(d_y.get(), fill.data(), fill.size() * sizeof(float),
                      hipMemcpyHostToDevice));
  const uint16_t* a = d_a.get() + (e.misaligned_a ? 4 : 0);
  const uint16_t* b = d_b.get() + (e.misaligned_b ? 4 : 0);
  const int status = hip_gemm_bf16_f32_transposed_b(
      e.null_stream ? nullptr : stream, e.null_a ? nullptr : a,
      e.null_b ? nullptr : b, e.null_c ? nullptr : d_y.get(), e.m, e.n, e.k);
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

// The kernel implements only the gfx11 WMMA encoding. gfx1170 and gfx12
// (wave32, gfx12-style WMMA only) would trap, and the launcher refuses wave64
// devices, so the test runs on the other gfx11 parts only.
bool device_has_gfx11_wmma(std::string* arch) {
  int device = 0;
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&props, device));
  *arch = props.gcnArchName;
  return arch->rfind("gfx11", 0) == 0 && arch->rfind("gfx1170", 0) != 0 &&
         props.warpSize == 32;
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
  std::string arch;
  if (!device_has_gfx11_wmma(&arch)) {
    std::cout << "gemm_transposed_b bf16: SKIP, " << arch
              << " has no gfx11 WMMA\n";
    return 0;
  }

  // The path shapes at every tier; then the typical sizes COVERAGE selects.
  std::vector<Case> cases;
  for (const Shape& shape : kPathShapes) {
    cases.push_back({shape.m, shape.n, shape.k});
  }
  for (const Shape& shape : kTypicalShapes) {
    if (shape.min_coverage <= coverage) {
      cases.push_back({shape.m, shape.n, shape.k});
    }
  }
  constexpr size_t kEdgeCount = sizeof(kEdges) / sizeof(kEdges[0]);
  constexpr size_t kStaleErrorCount = 2;
  std::cout << "gemm_transposed_b bf16 coverage=" << coverage
            << " cases=" << cases.size() + kStaleErrorCount + kEdgeCount
            << '\n';

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  uint64_t seed = 0;
  for (const Case& c : cases) {
    all_pass = run_case(c, ++seed, stream) && all_pass;
  }
  // A failed HIP call earlier on the thread must not fail either tile shape.
  for (int m : {64, kTallTileMinM}) {
    all_pass = run_case({m, 256, 2 * kSliceK}, ++seed, stream,
                        /*stale_error=*/true) &&
               all_pass;
  }
  for (const Edge& edge : kEdges) {
    all_pass = run_edge(edge, stream) && all_pass;
  }
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
