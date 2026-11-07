/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for hip_gather_nd with uint8 data and output (the
 * one-byte storage of a bool tensor) and int64 indices.
 *
 * Every output is compared bit-exactly with a CPU GatherND reference written
 * in this file (ONNX semantics, including batch_dims and negative indices).
 * Data holds 0/1 bool rows followed by arbitrary bytes, so a kernel that
 * reads or writes the wrong element width shows up as a mismatch.
 *
 * Each case runs twice with two different fill bytes in the output buffer,
 * which has guard bytes on both sides: an element the kernel did not write or
 * a write outside the output fails at least one of the two runs.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <algorithm>
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

constexpr int64_t kGuardBytes = 64;
constexpr uint8_t kFillBytes[2] = {0xa5, 0x5a};

int64_t product(const std::vector<int64_t>& v, size_t begin, size_t end) {
  int64_t p = 1;
  for (size_t i = begin; i < end; ++i) {
    p *= v[i];
  }
  return p;
}

uint64_t next_random(uint64_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  return *state;
}

// ---- one GatherND problem --------------------------------------------------

struct Problem {
  std::string name;
  std::vector<int64_t> data_shape;
  std::vector<int64_t> indices_shape; // last dim is K
  int batch_dims = 0;
};

// Output shape: indices.shape[:-1] + data.shape[batch_dims + K:].
int64_t output_elements(const Problem& p) {
  const size_t k = static_cast<size_t>(p.indices_shape.back());
  return product(p.indices_shape, 0, p.indices_shape.size() - 1) *
         product(p.data_shape, p.batch_dims + k, p.data_shape.size());
}

// Data: the first half holds bool values (0/1), the rest arbitrary bytes.
std::vector<uint8_t> make_data(const Problem& p, uint64_t seed) {
  const int64_t n = product(p.data_shape, 0, p.data_shape.size());
  std::vector<uint8_t> data(static_cast<size_t>(n));
  uint64_t state = seed | 1u;
  for (int64_t i = 0; i < n; ++i) {
    const uint64_t r = next_random(&state);
    data[i] = i < n / 2 ? static_cast<uint8_t>(r & 1u)
                        : static_cast<uint8_t>(r >> 24);
  }
  return data;
}

// Indices: every row picks valid positions. The first rows pick the first
// and the last position of each indexed dim, once positive and once as the
// equivalent negative index; the rest are random, half of them negative.
std::vector<int64_t> make_indices(const Problem& p, uint64_t seed) {
  const size_t k = static_cast<size_t>(p.indices_shape.back());
  const int64_t rows = product(p.indices_shape, 0, p.indices_shape.size() - 1);
  std::vector<int64_t> indices(static_cast<size_t>(rows * k));
  uint64_t state = seed | 1u;
  for (int64_t r = 0; r < rows; ++r) {
    for (size_t j = 0; j < k; ++j) {
      const int64_t dim = p.data_shape[p.batch_dims + j];
      int64_t idx;
      switch (r) {
      case 0: idx = 0; break;
      case 1: idx = dim - 1; break;
      case 2: idx = -dim; break;
      case 3: idx = -1; break;
      default:
        idx = static_cast<int64_t>(next_random(&state) %
                                   static_cast<uint64_t>(dim));
        if (next_random(&state) & 1u) {
          idx -= dim;
        }
      }
      indices[r * k + j] = idx;
    }
  }
  return indices;
}

// CPU GatherND (ONNX), one slice per index row.
std::vector<uint8_t> reference(const Problem& p,
                               const std::vector<uint8_t>& data,
                               const std::vector<int64_t>& indices) {
  const size_t k = static_cast<size_t>(p.indices_shape.back());
  const size_t bd = static_cast<size_t>(p.batch_dims);
  const int64_t slice = product(p.data_shape, bd + k, p.data_shape.size());
  const int64_t batch_stride = product(p.data_shape, bd, p.data_shape.size());
  const int64_t slices_per_batch =
      product(p.indices_shape, bd, p.indices_shape.size() - 1);
  const int64_t rows = product(p.indices_shape, 0, p.indices_shape.size() - 1);
  std::vector<uint8_t> out(static_cast<size_t>(rows * slice));
  for (int64_t r = 0; r < rows; ++r) {
    int64_t offset = (r / slices_per_batch) * batch_stride;
    for (size_t j = 0; j < k; ++j) {
      const int64_t dim = p.data_shape[bd + j];
      int64_t idx = indices[r * k + j];
      if (idx < 0) {
        idx += dim;
      }
      offset += idx * product(p.data_shape, bd + j + 1, p.data_shape.size());
    }
    std::memcpy(out.data() + r * slice, data.data() + offset,
                static_cast<size_t>(slice));
  }
  return out;
}

// ---- device helpers --------------------------------------------------------

class DeviceBytes {
public:
  explicit DeviceBytes(size_t bytes) {
    HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_),
                        std::max<size_t>(bytes, 1)));
  }
  ~DeviceBytes() { (void)hipFree(data_); }
  DeviceBytes(const DeviceBytes&) = delete;
  DeviceBytes& operator=(const DeviceBytes&) = delete;
  uint8_t* get() { return data_; }

private:
  uint8_t* data_ = nullptr;
};

void upload(uint8_t* device, const void* host, size_t bytes) {
  HIP_CHECK(hipMemcpy(device, host, bytes, hipMemcpyHostToDevice));
}

std::vector<uint8_t> download(uint8_t* device, size_t bytes) {
  std::vector<uint8_t> host(bytes);
  HIP_CHECK(hipMemcpy(host.data(), device, bytes, hipMemcpyDeviceToHost));
  return host;
}

// ---- results.csv -----------------------------------------------------------

// Tiny out/results.csv writer (op,leaf,arch,mode,shape,config,time_ms,relL2,
// verdict), opt-in via HIPDNN_RESULTS_CSV (set by the Makefile/controller).
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
                 env_or("HIPDNN_RESULTS_OP", "gather_nd"),
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
// case returning early still reports FAIL. relL2 is 0 on PASS (bit-exact)
// and 1 on FAIL; this leaf only counts mismatching elements.
struct ResultRow {
  CsvRow row;
  explicit ResultRow(std::string shape) {
    row.shape = std::move(shape);
    row.config = "fixed:gather_nd";
    row.rel_l2 = 1.0;
    row.verdict = "FAIL";
  }
  ResultRow(const ResultRow&) = delete;
  ResultRow& operator=(const ResultRow&) = delete;
  ~ResultRow() { CsvWriter().write(row); }
  void finish(bool pass) {
    row.rel_l2 = pass ? 0.0 : 1.0;
    row.verdict = pass ? "PASS" : "FAIL";
  }
};

// ---- cases -----------------------------------------------------------------

bool run_case(const Problem& p, uint64_t seed, hipStream_t stream) {
  ResultRow result(p.name);
  const std::vector<uint8_t> data = make_data(p, seed);
  const std::vector<int64_t> indices = make_indices(p, seed * 31 + 7);
  const std::vector<uint8_t> expected = reference(p, data, indices);
  const int64_t n = output_elements(p);

  DeviceBytes d_data(data.size());
  upload(d_data.get(), data.data(), data.size());
  DeviceBytes d_indices(indices.size() * sizeof(int64_t));
  upload(d_indices.get(), indices.data(), indices.size() * sizeof(int64_t));
  const size_t out_bytes = static_cast<size_t>(n + 2 * kGuardBytes);
  DeviceBytes d_out(out_bytes);

  bool pass = true;
  for (uint8_t fill : kFillBytes) {
    const std::vector<uint8_t> filled(out_bytes, fill);
    upload(d_out.get(), filled.data(), out_bytes);
    const int status = hip_gather_nd(
        reinterpret_cast<void*>(stream), d_data.get(), d_indices.get(),
        d_out.get() + kGuardBytes, p.data_shape.data(),
        static_cast<int>(p.data_shape.size()), p.indices_shape.data(),
        static_cast<int>(p.indices_shape.size()), p.batch_dims,
        HIP_DTYPE_UINT8);
    if (status != 0) {
      std::cout << "FAIL " << p.name << " hip_gather_nd returned " << status
                << '\n';
      return false;
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    const std::vector<uint8_t> out = download(d_out.get(), out_bytes);

    bool guards_ok = true;
    for (int64_t b = 0; b < kGuardBytes; ++b) {
      guards_ok = guards_ok && out[b] == fill &&
                  out[out_bytes - kGuardBytes + b] == fill;
    }
    int64_t mismatches = 0;
    int64_t first = -1;
    for (int64_t i = 0; i < n; ++i) {
      if (out[kGuardBytes + i] != expected[i]) {
        if (first < 0) {
          first = i;
        }
        ++mismatches;
      }
    }
    const bool ok = guards_ok && mismatches == 0;
    if (!ok) {
      std::printf("FAIL %s fill=0x%02x mismatches=%lld guards=%s",
                  p.name.c_str(), fill, static_cast<long long>(mismatches),
                  guards_ok ? "PASS" : "FAIL");
      if (first >= 0) {
        std::printf(" first=%lld got=0x%02x want=0x%02x",
                    static_cast<long long>(first), out[kGuardBytes + first],
                    expected[first]);
      }
      std::printf("\n");
    }
    pass = pass && ok;
  }
  if (pass) {
    std::cout << "PASS " << p.name << '\n';
  }
  result.finish(pass);
  return pass;
}

// A dtype the kernel has no GatherND for must return -1 and leave the output
// untouched.
bool run_unsupported_case(hipStream_t stream) {
  const std::string name = "unsupported_u16_gather_nd";
  ResultRow result(name);
  const std::vector<int64_t> data_shape = {4, 8};
  const std::vector<int64_t> indices_shape = {2, 1};
  const std::vector<int64_t> indices = {0, 3};
  DeviceBytes d_data(4 * 8 * sizeof(uint16_t));
  HIP_CHECK(hipMemset(d_data.get(), 1, 4 * 8 * sizeof(uint16_t)));
  DeviceBytes d_indices(indices.size() * sizeof(int64_t));
  upload(d_indices.get(), indices.data(), indices.size() * sizeof(int64_t));
  const std::vector<uint8_t> filled(2 * 8 * sizeof(uint16_t), kFillBytes[0]);
  DeviceBytes d_out(filled.size());
  upload(d_out.get(), filled.data(), filled.size());
  const int status = hip_gather_nd(
      reinterpret_cast<void*>(stream), d_data.get(), d_indices.get(),
      d_out.get(), data_shape.data(), 2, indices_shape.data(), 2, 0,
      HIP_DTYPE_UINT16);
  HIP_CHECK(hipStreamSynchronize(stream));
  const std::vector<uint8_t> out = download(d_out.get(), filled.size());
  const bool untouched = out == filled;
  const bool pass = status == -1 && untouched;
  std::cout << (pass ? "PASS " : "FAIL ") << name << " status=" << status
            << " output_untouched=" << (untouched ? "yes" : "no")
            << " (an 'unsupported dtype' message above is expected)\n";
  result.finish(pass);
  return pass;
}

// Categorical situations, each built for a typical size (rows x cols of the
// gathered tensor, and the number of index rows):
//   rows      K = 1 on [R, C]: pick whole rows (e.g. mask rows by position)
//   elements  K = 2 on [R, C]: pick single elements
//   batched   batch_dims = 1 on [2, R, C], K = 1: rows per batch
//   planes    K = 2 on [2, R, C]: pick rows of a rank-3 tensor
std::vector<Problem> make_problems(int64_t r, int64_t c, int64_t m) {
  const std::string dims = std::to_string(r) + "x" + std::to_string(c) +
                           "_m" + std::to_string(m);
  return {
      {"rows_" + dims, {r, c}, {m, 1}, 0},
      {"elements_" + dims, {r, c}, {m, 2}, 0},
      {"batched_" + dims, {2, r, c}, {2, m, 1}, 1},
      {"planes_" + dims, {2, r, c}, {m, 2}, 0},
  };
}

struct Size {
  int64_t rows, cols, index_rows;
  int min_coverage;
};

// Small odd sizes end in a partial block; 512 x 512 is the causal mask of a
// 512-token window; the last one is a large tensor.
constexpr Size kSizes[] = {
    {7, 33, 5, 1}, {512, 512, 300, 1}, {64, 256, 64, 2}, {2048, 2048, 2048, 3},
};

bool parse_options(int argc, char** argv, int* coverage) {
  if (const char* env = std::getenv("HIPDNN_UT_COVERAGE")) {
    *coverage = std::atoi(env);
  }
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

  int size_count = 0;
  for (const Size& size : kSizes) {
    size_count += size.min_coverage <= coverage ? 1 : 0;
  }
  std::cout << "gather_nd coverage=" << coverage << " -> running "
            << size_count << " size(s) x 4 categorical case(s) + 1 "
            << "unsupported case = " << size_count * 4 + 1
            << " gather_nd cases\n";

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  uint64_t seed = 1;
  for (const Size& size : kSizes) {
    if (size.min_coverage > coverage) {
      continue;
    }
    for (const Problem& p :
         make_problems(size.rows, size.cols, size.index_rows)) {
      all_pass = run_case(p, seed++, stream) && all_pass;
    }
  }
  all_pass = run_unsupported_case(stream) && all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
