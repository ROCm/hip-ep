/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for the runtime Slice over 64-bit elements.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
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

// hip_slice moves elements without looking at them, so every dtype of this
// size runs the same instantiation. Each one is still launched, to check that
// its dtype id reaches it.
using Element = uint64_t;
struct Dtype {
  int id;
  const char* name;
};
constexpr Dtype kDtypes[] = {
    {HIP_DTYPE_INT64, "i64"},
    {HIP_DTYPE_FLOAT64, "f64"},
};
constexpr int kDtypeCount = static_cast<int>(sizeof(kDtypes) / sizeof(kDtypes[0]));

constexpr int kMaxRank = 8;
constexpr int kGuardElements = 32;
// Every byte of an element the kernel did not write. No input element has this
// low byte and the over-allocated tail is zero, so no expected element is the
// fill.
constexpr uint8_t kFillByte = 0xa5;

struct Case {
  const char* name;
  int rank;
  int64_t input[kMaxRank];
  int64_t output[kMaxRank];
  // The slice's own extent per axis when the output is over-allocated past it;
  // all zero means it equals `output` and hip_slice gets a null pointer.
  int64_t logical[kMaxRank];
  int64_t starts[kMaxRank];
  int64_t steps[kMaxRank];
  int min_coverage;
};

// The first group is the categorical situations, run at every tier; the rest
// are typical sizes that COVERAGE thins.
constexpr Case kCases[] = {
    {"contiguous", 1, {1000}, {500}, {}, {37}, {1}, 1},
    {"strided", 2, {33, 70}, {16, 34}, {}, {1, 3}, {2, 2}, 1},
    {"reversed", 3, {4, 50, 7}, {4, 50, 7}, {}, {0, 49, 0}, {1, -1, 1}, 1},
    {"mixed", 4, {3, 9, 10, 6}, {2, 3, 5, 6}, {}, {1, 2, 9, 0},
     {1, 3, -2, 1}, 1},
    {"overallocated", 2, {8, 40}, {8, 48}, {8, 30}, {0, 5}, {1, 1}, 1},
    {"rank8", 8, {2, 3, 2, 3, 2, 3, 2, 5}, {2, 2, 2, 3, 1, 3, 2, 5}, {},
     {0, 2, 0, 0, 1, 0, 1, 4}, {1, -1, 1, 1, 1, 1, -1, -1}, 1},
    {"empty", 2, {6, 8}, {0, 8}, {}, {0, 0}, {1, 1}, 1},
    {"rows", 2, {512, 1024}, {256, 512}, {}, {100, 1}, {1, 2}, 1},
    {"reversed_large", 3, {64, 128, 96}, {64, 64, 96}, {}, {63, 0, 0},
     {-1, 2, 1}, 2},
    {"overallocated_large", 2, {1024, 768}, {512, 768}, {500, 700}, {0, 10},
     {2, 1}, 2},
    {"large", 2, {4096, 2048}, {2048, 1024}, {}, {0, 1}, {2, 2}, 3},
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

private:
  uint64_t state_;
};

int64_t product(const int64_t* dims, int rank) {
  int64_t n = 1;
  for (int d = 0; d < rank; ++d) {
    n *= dims[d];
  }
  return n;
}

bool has_logical(const Case& c) {
  for (int d = 0; d < c.rank; ++d) {
    if (c.logical[d] != 0) {
      return true;
    }
  }
  return false;
}

std::string dims_string(const int64_t* dims, int rank) {
  std::string s;
  for (int d = 0; d < rank; ++d) {
    s += (d ? "x" : "") + std::to_string(dims[d]);
  }
  return s;
}

// Any bit pattern, NaNs and denormals of the float dtypes included, except
// a low byte equal to the fill.
std::vector<Element> make_input(const Case& c, Rng& rng) {
  std::vector<Element> input(static_cast<size_t>(product(c.input, c.rank)));
  for (Element& value : input) {
    value = static_cast<Element>(rng.next());
    if ((value & 0xff) == kFillByte) {
      value ^= 1;
    }
  }
  return input;
}

// ONNX Slice on resolved per-axis (start, step), with positions past the
// logical extent of an over-allocated output set to zero.
std::vector<Element> reference(const Case& c, const std::vector<Element>& in) {
  int64_t in_strides[kMaxRank];
  int64_t stride = 1;
  for (int d = c.rank - 1; d >= 0; --d) {
    in_strides[d] = stride;
    stride *= c.input[d];
  }
  const int64_t count = product(c.output, c.rank);
  std::vector<Element> out(static_cast<size_t>(count));
  const bool logical = has_logical(c);
  for (int64_t o = 0; o < count; ++o) {
    int64_t remaining = o;
    int64_t offset = 0;
    bool inside = true;
    for (int d = c.rank - 1; d >= 0; --d) {
      const int64_t coord = remaining % c.output[d];
      remaining /= c.output[d];
      inside = inside && (!logical || coord < c.logical[d]);
      offset += (c.starts[d] + coord * c.steps[d]) * in_strides[d];
    }
    out[o] = inside ? in[offset] : Element(0);
  }
  return out;
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
                 env_or("HIPDNN_RESULTS_OP", "slice"),
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
// case returning early still reports FAIL. A copy is exact or wrong, so relL2
// is 0 on a pass and 1 otherwise.
struct ResultRow {
  CsvRow row;
  ResultRow(std::string shape, std::string config) {
    row.shape = std::move(shape);
    row.config = std::move(config);
    row.rel_l2 = 1.0;
    row.verdict = "FAIL";
  }
  ResultRow(const ResultRow&) = delete;
  ResultRow& operator=(const ResultRow&) = delete;
  ~ResultRow() { CsvWriter().write(row); }
  void pass() {
    row.rel_l2 = 0.0;
    row.verdict = "PASS";
  }
};

// The output buffer with kGuardElements of fill on each side.
struct Output {
  explicit Output(int64_t count)
      : count(count), host(static_cast<size_t>(count + 2 * kGuardElements)),
        device(host.size()) {
    std::memset(host.data(), kFillByte, host.size() * sizeof(Element));
    HIP_CHECK(hipMemcpy(device.get(), host.data(), host.size() * sizeof(Element),
                        hipMemcpyHostToDevice));
  }
  Element* payload() { return device.get() + kGuardElements; }
  void read_back() {
    HIP_CHECK(hipMemcpy(host.data(), device.get(), host.size() * sizeof(Element),
                        hipMemcpyDeviceToHost));
  }
  bool is_fill(size_t i) const {
    Element fill;
    std::memset(&fill, kFillByte, sizeof(fill));
    return host[i] == fill;
  }
  bool guards_intact() const {
    for (int i = 0; i < kGuardElements; ++i) {
      if (!is_fill(i) || !is_fill(static_cast<size_t>(kGuardElements + count + i))) {
        return false;
      }
    }
    return true;
  }
  bool untouched() const {
    for (size_t i = 0; i < host.size(); ++i) {
      if (!is_fill(i)) {
        return false;
      }
    }
    return true;
  }
  int64_t count;
  std::vector<Element> host;
  DeviceBuffer<Element> device;
};

int launch(const Case& c, const Element* input, Element* output, int dtype,
           hipStream_t stream) {
  return hip_slice(reinterpret_cast<void*>(stream), input, output, c.input,
                   c.output, has_logical(c) ? c.logical : nullptr, c.starts,
                   c.steps, c.rank, dtype);
}

bool run_case(const Case& c, uint64_t seed, const Dtype& dtype,
              hipStream_t stream) {
  const std::string name = std::string(c.name) + "_in" +
                           dims_string(c.input, c.rank) + "_out" +
                           dims_string(c.output, c.rank);
  ResultRow result(name, dtype.name);

  Rng rng(seed);
  const std::vector<Element> input = make_input(c, rng);
  const std::vector<Element> expected = reference(c, input);

  DeviceBuffer<Element> d_input(input.size());
  if (!input.empty()) {
    HIP_CHECK(hipMemcpy(d_input.get(), input.data(),
                        input.size() * sizeof(Element), hipMemcpyHostToDevice));
  }
  Output out(static_cast<int64_t>(expected.size()));

  const int status = launch(c, d_input.get(), out.payload(), dtype.id, stream);
  HIP_CHECK(hipStreamSynchronize(stream));
  out.read_back();

  int64_t mismatches = 0;
  int64_t first = -1;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (out.host[kGuardElements + i] != expected[i]) {
      if (first < 0) {
        first = static_cast<int64_t>(i);
      }
      ++mismatches;
    }
  }
  const bool guards = out.guards_intact();
  const bool pass = status == 0 && mismatches == 0 && guards;

  std::cout << (pass ? "PASS" : "FAIL") << " shape=" << name
            << " dtype=" << dtype.name << " n=" << expected.size()
            << " status=" << status << " mismatches=" << mismatches
            << " guards=" << (guards ? "PASS" : "FAIL");
  if (first >= 0) {
    std::cout << " first=" << first << " got=0x" << std::hex
              << static_cast<uint64_t>(out.host[kGuardElements + first])
              << " want=0x" << static_cast<uint64_t>(expected[first])
              << std::dec;
  }
  std::cout << '\n';
  if (pass) {
    result.pass();
  }
  return pass;
}

// Arguments hip_slice must refuse with -1 and without writing anything.
bool run_rejected(const char* label, int rank, int dtype, hipStream_t stream) {
  ResultRow result(std::string("rejected_") + label, "-");
  // One more slot than the largest rank, so rank 9 reads no garbage if the
  // launcher ever looks at the arrays before checking the rank.
  int64_t dims[kMaxRank + 1];
  int64_t starts[kMaxRank + 1] = {};
  int64_t steps[kMaxRank + 1];
  for (int d = 0; d <= kMaxRank; ++d) {
    dims[d] = 2;
    steps[d] = 1;
  }
  DeviceBuffer<Element> d_input(1 << (kMaxRank + 1));
  Output out(1 << (kMaxRank + 1));
  const int status =
      hip_slice(reinterpret_cast<void*>(stream), d_input.get(), out.payload(),
                dims, dims, nullptr, starts, steps, rank, dtype);
  HIP_CHECK(hipStreamSynchronize(stream));
  out.read_back();
  const bool untouched = out.untouched();
  const bool pass = status == -1 && untouched;
  std::cout << (pass ? "PASS" : "FAIL") << " rejected=" << label
            << " status=" << status
            << " output=" << (untouched ? "untouched" : "WRITTEN") << '\n';
  if (pass) {
    result.pass();
  }
  return pass;
}

// A failed HIP call earlier on this thread leaves its error pending. hip_slice
// reports hipGetLastError() after its launch, so it must clear that error
// first rather than report it as its own.
bool run_stale_error(uint64_t seed, hipStream_t stream) {
  (void)hipSetDevice(-1);
  if (hipPeekAtLastError() == hipSuccess) {
    std::cout << "FAIL stale_error: hipSetDevice(-1) left no pending error\n";
    ResultRow result("stale_error", kDtypes[0].name);
    return false;
  }
  Case c = kCases[1];
  c.name = "stale_error";
  return run_case(c, seed, kDtypes[0], stream);
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

  int case_count = 0;
  for (const Case& c : kCases) {
    case_count += c.min_coverage <= coverage ? kDtypeCount : 0;
  }
  constexpr int kRejectedCount = 3;
  std::cout << "slice u64 coverage=" << coverage
            << " cases=" << case_count + kRejectedCount + 1 << '\n';

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  uint64_t seed = 0;
  for (const Case& c : kCases) {
    ++seed;
    if (c.min_coverage > coverage) {
      continue;
    }
    for (const Dtype& dtype : kDtypes) {
      all_pass = run_case(c, seed, dtype, stream) && all_pass;
    }
  }
  all_pass = run_rejected("rank0", 0, kDtypes[0].id, stream) && all_pass;
  all_pass = run_rejected("rank9", kMaxRank + 1, kDtypes[0].id, stream) &&
             all_pass;
  all_pass = run_rejected("dtype", 2, -1, stream) && all_pass;
  all_pass = run_stale_error(seed + 1, stream) && all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
