/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for hip_cast from bf16 to fp32.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <algorithm>
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
// Every output and guard element starts as this fp32 NaN pattern, which no
// bf16 input widens to (its low 16 bits are not zero), so an output left
// unwritten fails.
constexpr uint32_t kFill = 0xFFEEDDCCu;
constexpr int kWarmupLaunches = 3;
constexpr int kTimedLaunches = 50;

struct Size {
  int64_t n;
  int min_coverage;
};

// Element counts around the 256-thread block; COVERAGE adds larger tensors.
constexpr Size kSizes[] = {
    {1, 1},   {255, 1},          {256, 1},
    {257, 1}, {1 << 16, 1},      {(1 << 22) + 3, 2},
    {(1 << 24) + 1, 3},
};

// bf16 -> fp32 is exact: the bf16 bits become the upper half of the fp32,
// NaN payloads included.
uint32_t reference(uint16_t bits) { return static_cast<uint32_t>(bits) << 16; }

class Rng {
public:
  explicit Rng(uint64_t seed) : state_(seed) {}
  uint64_t next() {
    uint64_t z = (state_ += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
  }

private:
  uint64_t state_;
};

class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t bytes) {
    if (bytes != 0) {
      HIP_CHECK(hipMalloc(&data_, bytes));
    }
  }
  ~DeviceBuffer() {
    if (data_) {
      (void)hipFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  char* get() { return static_cast<char*>(data_); }

private:
  void* data_ = nullptr;
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
                 env_or("HIPDNN_RESULTS_OP", "cast"),
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

int cast(hipStream_t stream, const void* input, void* output, int64_t n) {
  return hip_cast(stream, input, output, n, HIP_DTYPE_BFLOAT16,
                  HIP_DTYPE_FLOAT32);
}

// Mean time of back-to-back launches on warm caches, in ms.
double time_launches(const void* input, void* output, int64_t n,
                     hipStream_t stream) {
  for (int i = 0; i < kWarmupLaunches; ++i) {
    (void)cast(stream, input, output, n);
  }
  hipEvent_t start = nullptr, stop = nullptr;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, stream));
  for (int i = 0; i < kTimedLaunches; ++i) {
    (void)cast(stream, input, output, n);
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
// earlier runtime call can leave behind before the cast runs.
void leave_stale_error() {
  int count = 0;
  HIP_CHECK(hipGetDeviceCount(&count));
  (void)hipSetDevice(count);
}

// Casts `input` (bf16 bit patterns) and compares every output bit-exactly
// with the widened reference.
bool run_case(const std::string& label, const std::vector<uint16_t>& input,
              bool misaligned, hipStream_t stream, bool stale_error = false) {
  const std::string name = label + (misaligned ? "_misaligned" : "") +
                           (stale_error ? "_stale_error" : "");
  ResultRow result(name, "bf16_to_f32");
  const int64_t n = static_cast<int64_t>(input.size());

  // One extra element in front moves each pointer off its natural 16-byte
  // alignment when the case asks for it.
  const size_t shift = misaligned ? 1 : 0;
  DeviceBuffer d_in((input.size() + 1) * sizeof(uint16_t));
  DeviceBuffer d_out((input.size() + 2 * kGuardElements + 1) *
                     sizeof(uint32_t));
  uint16_t* in_ptr = reinterpret_cast<uint16_t*>(d_in.get()) + shift;
  uint32_t* out_base = reinterpret_cast<uint32_t*>(d_out.get()) + shift;
  HIP_CHECK(hipMemcpy(in_ptr, input.data(), input.size() * sizeof(uint16_t),
                      hipMemcpyHostToDevice));
  const std::vector<uint32_t> fill(input.size() + 2 * kGuardElements, kFill);
  HIP_CHECK(hipMemcpy(out_base, fill.data(), fill.size() * sizeof(uint32_t),
                      hipMemcpyHostToDevice));

  if (stale_error) {
    leave_stale_error();
  }
  const int status = cast(stream, in_ptr, out_base + kGuardElements, n);
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<uint32_t> actual(fill.size());
  HIP_CHECK(hipMemcpy(actual.data(), out_base,
                      actual.size() * sizeof(uint32_t),
                      hipMemcpyDeviceToHost));

  bool guards = true;
  for (int i = 0; i < kGuardElements; ++i) {
    guards = guards && actual[i] == kFill &&
             actual[kGuardElements + input.size() + i] == kFill;
  }
  int64_t mismatches = 0;
  int64_t first = -1;
  for (int64_t i = 0; i < n; ++i) {
    if (actual[kGuardElements + i] != reference(input[i])) {
      if (first < 0) {
        first = i;
      }
      ++mismatches;
    }
  }
  const bool pass = status == 0 && guards && mismatches == 0;
  const double ms =
      pass && !stale_error
          ? time_launches(in_ptr, out_base + kGuardElements, n, stream)
          : 0.0;

  std::cout << (pass ? "PASS" : "FAIL") << " shape=" << name << " n=" << n
            << " status=" << status << " mismatches=" << mismatches
            << " guards=" << (guards ? "PASS" : "FAIL") << " time_ms=" << ms;
  if (first >= 0) {
    char detail[96];
    std::snprintf(detail, sizeof(detail),
                  " first=%lld in=0x%04x got=0x%08x want=0x%08x",
                  static_cast<long long>(first), input[first],
                  actual[kGuardElements + first], reference(input[first]));
    std::cout << detail;
  }
  std::cout << '\n';
  result.finish(ms, pass ? 0.0 : 1.0, pass);
  return pass;
}

// A call hip_cast must refuse (-1) or treat as empty (0), writing nothing.
bool run_edge(const char* label, int64_t n, int output_dtype,
              int expected_status, hipStream_t stream) {
  ResultRow result(std::string("edge_") + label, "-");
  constexpr int kCount = 64;
  DeviceBuffer d_in(kCount * sizeof(uint16_t));
  const std::vector<uint32_t> fill(kCount * 2, kFill);
  DeviceBuffer d_out(fill.size() * sizeof(uint32_t));
  HIP_CHECK(hipMemcpy(d_out.get(), fill.data(), fill.size() * sizeof(uint32_t),
                      hipMemcpyHostToDevice));
  const int status = hip_cast(stream, d_in.get(), d_out.get(), n,
                              HIP_DTYPE_BFLOAT16, output_dtype);
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<uint32_t> actual(fill.size());
  HIP_CHECK(hipMemcpy(actual.data(), d_out.get(),
                      actual.size() * sizeof(uint32_t),
                      hipMemcpyDeviceToHost));
  const bool untouched = actual == fill;
  const bool pass = status == expected_status && untouched;
  std::cout << (pass ? "PASS" : "FAIL") << " edge=" << label
            << " status=" << status
            << " output=" << (untouched ? "untouched" : "WRITTEN") << '\n';
  result.finish(0.0, pass ? 0.0 : 1.0, pass);
  return pass;
}

// Every bf16 bit pattern: zeros, subnormals, normals, infinities and NaNs of
// both signs.
std::vector<uint16_t> every_pattern() {
  std::vector<uint16_t> values(65536);
  for (uint32_t bits = 0; bits < 65536; ++bits) {
    values[bits] = static_cast<uint16_t>(bits);
  }
  return values;
}

std::vector<uint16_t> random_values(int64_t n, uint64_t seed) {
  Rng rng(seed);
  std::vector<uint16_t> values(static_cast<size_t>(n));
  for (uint16_t& v : values) {
    v = static_cast<uint16_t>(rng.next());
  }
  return values;
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

  std::vector<std::pair<std::string, std::vector<uint16_t>>> inputs;
  inputs.emplace_back("every_pattern", every_pattern());
  uint64_t seed = 0;
  for (const Size& size : kSizes) {
    if (size.min_coverage <= coverage) {
      inputs.emplace_back("random_n" + std::to_string(size.n),
                          random_values(size.n, ++seed));
    }
  }
  constexpr int kStaleErrorCount = 1;
  constexpr int kEdgeCount = 4;
  std::cout << "cast bf16->f32 coverage=" << coverage
            << " cases=" << 2 * inputs.size() + kStaleErrorCount + kEdgeCount
            << '\n';

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  // Every input set with aligned and misaligned pointers.
  for (const auto& [label, values] : inputs) {
    for (bool misaligned : {false, true}) {
      all_pass = run_case(label, values, misaligned, stream) && all_pass;
    }
  }
  // A failed HIP call earlier on the thread must not fail the cast.
  all_pass = run_case("every_pattern", every_pattern(), false, stream,
                      /*stale_error=*/true) &&
             all_pass;
  all_pass = run_edge("n0", 0, HIP_DTYPE_FLOAT32, 0, stream) && all_pass;
  all_pass = run_edge("negative_n", -1, HIP_DTYPE_FLOAT32, 0, stream) &&
             all_pass;
  all_pass = run_edge("f64_output", 64, HIP_DTYPE_FLOAT64, -1, stream) &&
             all_pass;
  all_pass = run_edge("f16_output", 64, HIP_DTYPE_FLOAT16, -1, stream) &&
             all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
