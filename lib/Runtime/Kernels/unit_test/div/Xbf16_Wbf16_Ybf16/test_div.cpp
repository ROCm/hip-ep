/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for hip_elementwise_div with bf16 lhs, rhs and output.
 *
 * Every output is compared bit-exactly with a CPU reference computed in this
 * file: both operands are widened to fp32 (exact), divided in fp32 (IEEE,
 * correctly rounded) and rounded once to bf16 (round to nearest even). The
 * bf16 <-> fp32 conversions are plain integer bit operations; the test does
 * not use the HIP conversion intrinsics the kernel uses. A NaN result matches
 * any NaN (payloads may differ).
 *
 * Inputs: every pair of a list of bf16 special values (signed zeros,
 * infinities, NaN, +-1, the largest finite and the smallest normal value,
 * values that need rounding), then seeded random values. Quotients that are
 * fp32 subnormals are left out: whether the GPU keeps them is a property of
 * the device's denormal mode, not of this kernel.
 *
 * Each case runs twice with two different fill bytes in the output buffer,
 * which has guard elements on both sides: an element the kernel did not write
 * or a write outside the output fails at least one of the two runs.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
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

constexpr int64_t kGuardElements = 64;
constexpr uint8_t kFillBytes[2] = {0xa5, 0x5a};

// ---- bf16 <-> fp32 (integer bit operations) --------------------------------

float bits_float(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

uint32_t float_bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  return bits;
}

float bf16_to_float(uint16_t h) { return bits_float(uint32_t{h} << 16); }

bool bf16_is_nan(uint16_t h) {
  return (h & 0x7f80u) == 0x7f80u && (h & 0x007fu) != 0;
}

// fp32 -> bf16, round to nearest even. A NaN stays a NaN.
uint16_t float_to_bf16(float f) {
  const uint32_t bits = float_bits(f);
  if ((bits & 0x7f800000u) == 0x7f800000u && (bits & 0x007fffffu) != 0) {
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);
  }
  const uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

// Reference quotient, or false when it is an fp32 subnormal (left out).
bool reference_div(uint16_t a, uint16_t b, uint16_t* out) {
  const float q = bf16_to_float(a) / bf16_to_float(b);
  if (q != 0.0f && std::fabs(q) < FLT_MIN) {
    return false;
  }
  *out = float_to_bf16(q);
  return true;
}

// ---- inputs ----------------------------------------------------------------

// bf16 bit patterns: +-0, +-inf, NaN, +-1, 2, 3, 7, 10, 0.1, the largest
// finite value and the smallest normal value (both signs).
constexpr uint16_t kSpecials[] = {
    0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc0, 0x3f80, 0xbf80, 0x4000,
    0x4040, 0x40e0, 0x4120, 0x3dcd, 0x7f7f, 0xff7f, 0x0080, 0x8080,
};

struct Operands {
  std::vector<uint16_t> lhs, rhs, want;
};

uint64_t next_random(uint64_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  return *state;
}

// A finite, normal bf16 whose exponent keeps every quotient of two such
// values a normal fp32: unbiased exponent in [-60, 60], any mantissa.
uint16_t random_bf16(uint64_t* state) {
  const uint64_t r = next_random(state);
  const uint16_t sign = static_cast<uint16_t>((r & 1u) << 15);
  const uint16_t exponent = static_cast<uint16_t>(127 - 60 + (r >> 1) % 121);
  const uint16_t mantissa = static_cast<uint16_t>((r >> 8) & 0x7fu);
  return static_cast<uint16_t>(sign | (exponent << 7) | mantissa);
}

// The first elements are the special pairs whose quotient is not an fp32
// subnormal (about 230); the rest are random. A smaller n keeps a prefix.
Operands make_operands(int64_t n, uint64_t seed) {
  Operands o;
  o.lhs.reserve(n);
  o.rhs.reserve(n);
  o.want.reserve(n);
  for (uint16_t a : kSpecials) {
    for (uint16_t b : kSpecials) {
      uint16_t q;
      if (static_cast<int64_t>(o.lhs.size()) < n && reference_div(a, b, &q)) {
        o.lhs.push_back(a);
        o.rhs.push_back(b);
        o.want.push_back(q);
      }
    }
  }
  uint64_t state = seed | 1u;
  while (static_cast<int64_t>(o.lhs.size()) < n) {
    const uint16_t a = random_bf16(&state);
    const uint16_t b = random_bf16(&state);
    uint16_t q;
    if (reference_div(a, b, &q)) {
      o.lhs.push_back(a);
      o.rhs.push_back(b);
      o.want.push_back(q);
    }
  }
  return o;
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
                 env_or("HIPDNN_RESULTS_OP", "div"),
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
    row.config = "fixed:elementwise";
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

// bf16 / bf16 -> bf16: every element must match the reference bit for bit
// (any NaN for a NaN reference) and nothing outside the output is written.
bool run_div_case(int64_t n, uint64_t seed, hipStream_t stream) {
  const std::string name = "bf16_div_n" + std::to_string(n);
  ResultRow result(name);
  const Operands o = make_operands(n, seed);
  const size_t in_bytes = static_cast<size_t>(n) * sizeof(uint16_t);

  DeviceBytes d_lhs(in_bytes);
  DeviceBytes d_rhs(in_bytes);
  upload(d_lhs.get(), o.lhs.data(), in_bytes);
  upload(d_rhs.get(), o.rhs.data(), in_bytes);
  const size_t guard_bytes = kGuardElements * sizeof(uint16_t);
  const size_t out_bytes = in_bytes + 2 * guard_bytes;
  DeviceBytes d_out(out_bytes);

  bool pass = true;
  for (uint8_t fill : kFillBytes) {
    const std::vector<uint8_t> filled(out_bytes, fill);
    upload(d_out.get(), filled.data(), out_bytes);
    const int status = hip_elementwise_div(
        reinterpret_cast<void*>(stream), d_lhs.get(), d_rhs.get(),
        d_out.get() + guard_bytes, n, HIP_DTYPE_BFLOAT16);
    if (status != 0) {
      std::cout << "FAIL " << name << " hip_elementwise_div returned "
                << status << '\n';
      return false;
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    const std::vector<uint8_t> out = download(d_out.get(), out_bytes);

    bool guards_ok = true;
    for (size_t b = 0; b < guard_bytes; ++b) {
      guards_ok = guards_ok && out[b] == fill &&
                  out[out_bytes - guard_bytes + b] == fill;
    }
    int64_t mismatches = 0;
    int64_t first = -1;
    for (int64_t i = 0; i < n; ++i) {
      uint16_t got;
      std::memcpy(&got, out.data() + guard_bytes + i * sizeof(uint16_t),
                  sizeof(got));
      const uint16_t want = o.want[i];
      if (got == want || (bf16_is_nan(want) && bf16_is_nan(got))) {
        continue;
      }
      if (first < 0) {
        first = i;
      }
      ++mismatches;
    }
    const bool ok = guards_ok && mismatches == 0;
    if (!ok) {
      std::printf("FAIL %s fill=0x%02x mismatches=%lld guards=%s", name.c_str(),
                  fill, static_cast<long long>(mismatches),
                  guards_ok ? "PASS" : "FAIL");
      if (first >= 0) {
        uint16_t got;
        std::memcpy(&got, out.data() + guard_bytes + first * sizeof(uint16_t),
                    sizeof(got));
        std::printf(" first=%lld lhs=0x%04x rhs=0x%04x got=0x%04x want=0x%04x",
                    static_cast<long long>(first), o.lhs[first], o.rhs[first],
                    got, o.want[first]);
      }
      std::printf("\n");
    }
    pass = pass && ok;
  }
  if (pass) {
    std::cout << "PASS " << name << '\n';
  }
  result.finish(pass);
  return pass;
}

// A dtype the kernel has no Div for must return -1 and leave the output
// untouched.
bool run_unsupported_case(hipStream_t stream) {
  const int64_t n = 257;
  const std::string name = "unsupported_u8_div";
  ResultRow result(name);
  DeviceBytes d_lhs(static_cast<size_t>(n));
  DeviceBytes d_rhs(static_cast<size_t>(n));
  HIP_CHECK(hipMemset(d_lhs.get(), 1, static_cast<size_t>(n)));
  HIP_CHECK(hipMemset(d_rhs.get(), 1, static_cast<size_t>(n)));
  DeviceBytes d_out(static_cast<size_t>(n));
  const std::vector<uint8_t> filled(static_cast<size_t>(n), kFillBytes[0]);
  upload(d_out.get(), filled.data(), filled.size());
  const int status =
      hip_elementwise_div(reinterpret_cast<void*>(stream), d_lhs.get(),
                          d_rhs.get(), d_out.get(), n, HIP_DTYPE_UINT8);
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

struct Size {
  int64_t n;
  int min_coverage;
};

// 257 ends in a partial 256-thread block and holds every special pair; 256
// is exactly one block; the last one is a large tensor.
constexpr Size kSizes[] = {
    {257, 1}, {70001, 1}, {256, 2}, {1048579, 3},
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
  std::cout << "div coverage=" << coverage << " -> running " << size_count
            << " size(s) x 1 dtype (bf16) + 1 unsupported case = "
            << size_count + 1 << " div cases\n";

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  for (const Size& size : kSizes) {
    if (size.min_coverage > coverage) {
      continue;
    }
    all_pass = run_div_case(size.n, static_cast<uint64_t>(size.n), stream) &&
               all_pass;
  }
  all_pass = run_unsupported_case(stream) && all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
