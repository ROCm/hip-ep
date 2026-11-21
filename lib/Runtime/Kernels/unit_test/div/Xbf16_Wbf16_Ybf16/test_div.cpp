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
 * or a write outside the output fails at least one of the two runs. A case
 * that passes is then timed with the shared UT timer
 * (common/hip_adaptive_timing.h).
 *
 * Usage:
 *   test_div.exe [--coverage 1|2|3]      built-in sizes (default 3)
 *   test_div.exe --n <elements> [--seed <seed>]
 *                                        one case of that size; seed defaults to n
 */

#include "hip_custom_kernels.h"

#include <hip/hip_runtime.h>
#include "../../common/hip_adaptive_timing.h"
#include "../../common/ut_case_input.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// ---- Cold-cache timing ----
// The timed launches rotate over copies of the two inputs, so that each launch
// reads them from DRAM as in a model rather than from L2 / the Infinity Cache;
// each copy is read again only after kColdRotationBytes of other copies.
// HIPDNN_KERNEL_UT_WARM=1 times one copy back to back instead. Same scheme as
// the other leaves (e.g. qmoe_ragged_mm/Xfp16_Wu4_Yfp16).
namespace hipdnn_ep_test {
// Several times the largest last-level cache among the targets.
constexpr size_t kColdRotationBytes = size_t(256) << 20;
// Untimed reads before the timed launches, so that the copies' initial fill
// has left the caches.
constexpr size_t kColdSettleBytes = size_t(128) << 20;
// Bounds the copies of inputs of a few KB.
constexpr int kColdMaxCopies = 4096;

inline bool warmTiming() {
  const char* v = std::getenv("HIPDNN_KERNEL_UT_WARM");
  return v && v[0] == '1';
}

// Copies of the buffers one launch reads. Copy 0 is the buffers themselves;
// the others share one allocation.
class ColdCopies {
public:
  struct Part {
    const void* src;
    size_t bytes;
  };

  explicit ColdCopies(std::initializer_list<Part> parts) : parts_(parts) {
    size_t stride = 0;
    for (const Part& p : parts_) {
      offsets_.push_back(stride);
      stride += (p.bytes + 255) & ~size_t(255);
      bytes_ += p.bytes;
    }
    if (warmTiming() || bytes_ == 0) {
      return;
    }
    count_ = 1 + static_cast<int>(std::min<size_t>(
                     kColdMaxCopies - 1,
                     (kColdRotationBytes + bytes_ - 1) / bytes_));
    stride_ = stride;
    if (hipMalloc(&slab_, stride_ * (count_ - 1)) != hipSuccess) {
      std::fprintf(stderr,
                   "cold-cache copies do not fit; timing on warm caches\n");
      slab_ = nullptr;
      count_ = 1;
      return;
    }
    // Copy 1 from the buffers, then double the run of filled copies.
    for (size_t i = 0; i < parts_.size(); ++i) {
      if (parts_[i].bytes) {
        (void)hipMemcpy(part(1, i), parts_[i].src, parts_[i].bytes,
                        hipMemcpyDeviceToDevice);
      }
    }
    for (int filled = 1; filled < count_ - 1;) {
      const int more = std::min(filled, count_ - 1 - filled);
      (void)hipMemcpy(static_cast<char*>(slab_) + stride_ * filled, slab_,
                      stride_ * more, hipMemcpyDeviceToDevice);
      filled += more;
    }
  }
  ~ColdCopies() {
    if (slab_) {
      (void)hipFree(slab_);
    }
  }
  ColdCopies(const ColdCopies&) = delete;
  ColdCopies& operator=(const ColdCopies&) = delete;

  int count() const { return count_; }
  size_t bytes() const { return bytes_; }
  // Part `i` of copy `c`.
  void* part(int c, size_t i) const {
    if (c == 0 || parts_[i].bytes == 0) {
      return const_cast<void*>(parts_[i].src);
    }
    return static_cast<char*>(slab_) + stride_ * (c - 1) + offsets_[i];
  }

private:
  std::vector<Part> parts_;
  std::vector<size_t> offsets_;
  size_t bytes_ = 0;
  size_t stride_ = 0;
  int count_ = 1;
  void* slab_ = nullptr;
};

// Holds the stream until the host sets *release, so that every timed launch is
// queued before the first one runs. Gives up after about 0.25 s.
__global__ void holdStreamKernel(const volatile int* release) {
  for (int spin = 0; spin < (1 << 16) && *release == 0; ++spin) {
    __builtin_amdgcn_s_sleep(127);
  }
}

// Host-mapped, coherent release flag for holdStreamKernel, or null if it is
// unavailable.
inline int* holdReleaseFlag() {
  static int* flag = [] {
    void* p = nullptr;
    return hipHostMalloc(&p, sizeof(int),
                         hipHostMallocMapped | hipHostMallocCoherent) ==
                   hipSuccess
               ? static_cast<int*>(p)
               : nullptr;
  }();
  return flag;
}

// Mean time of one complete hip_elementwise_div call; launch(c) uses input
// copy c. The shared UT policy selects the call count.
template <class Launch>
double rotatedMeanMs(hipStream_t stream, const ColdCopies& copies,
                     Launch&& launch) {
  return timing::adaptiveRotatedMeanMs(
      stream, copies, launch, holdReleaseFlag(),
      [&](const int* flag) { holdStreamKernel<<<1, 1, 0, stream>>>(flag); },
      warmTiming(), kColdSettleBytes, "div_xbf16_wbf16_ybf16",
      "hip_elementwise_div");
}
} // namespace hipdnn_ep_test

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
// and 1 on FAIL; this leaf only counts mismatching elements. finish() binds a
// timed PASS to its UT_TIMING record, so the case's timing::CaseScope must be
// alive when it is called.
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
  void finish(bool pass, double time_ms) {
    row.rel_l2 = pass ? 0.0 : 1.0;
    row.verdict = pass ? "PASS" : "FAIL";
    row.time_ms = time_ms;
    hipdnn_ep_test::timing::emitCsvBinding(row.shape, row.config, row.verdict,
                                           row.rel_l2, row.time_ms);
  }
};

// ---- cases -----------------------------------------------------------------

// bf16 / bf16 -> bf16: every element must match the reference bit for bit
// (any NaN for a NaN reference) and nothing outside the output is written.
// A passing case is then timed.
bool run_div_case(int64_t n, uint64_t seed, hipStream_t stream) {
  std::string name = "bf16_div_n" + std::to_string(n);
  if (seed != static_cast<uint64_t>(n)) {
    name += "_seed" + std::to_string(seed);
  }
  hipdnn_ep_test::timing::CaseScope timing_case("div/Xbf16_Wbf16_Ybf16", name,
                                                "hip_elementwise_div");
  ResultRow result(name);
  const Operands o = make_operands(n, seed);
  const size_t in_bytes = hipdnn_ep_test::input::bytes(
      {static_cast<uint64_t>(n)}, sizeof(uint16_t));

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

  double time_ms = 0.0;
  if (pass) {
    const hipdnn_ep_test::ColdCopies inputs(
        {{d_lhs.get(), in_bytes}, {d_rhs.get(), in_bytes}});
    time_ms = hipdnn_ep_test::rotatedMeanMs(stream, inputs, [&](int c) {
      return hip_elementwise_div(reinterpret_cast<void*>(stream),
                                 inputs.part(c, 0), inputs.part(c, 1),
                                 d_out.get() + guard_bytes, n,
                                 HIP_DTYPE_BFLOAT16);
    });
    std::printf("PASS %s time_ms=%.6f\n", name.c_str(), time_ms);
  }
  result.finish(pass, time_ms);
  return pass;
}

// A dtype the kernel has no Div for must return -1 and leave the output
// untouched. Not timed.
bool run_unsupported_case(hipStream_t stream) {
  const int64_t n = 257;
  const std::string name = "unsupported_u8_div";
  hipdnn_ep_test::timing::CaseScope timing_case("div/Xbf16_Wbf16_Ybf16", name,
                                                "hip_elementwise_div");
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
  result.finish(pass, 0.0);
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

struct Options {
  int coverage = 3;
  bool custom = false;
  int64_t n = 257;
  uint64_t seed = 0;
};

// Throws std::invalid_argument on an unknown, incomplete or out-of-range
// argument.
Options parse_options(int argc, char** argv) {
  using hipdnn_ep_test::input::integer;
  using hipdnn_ep_test::input::value;
  Options o;
  const char* env = std::getenv("HIPDNN_UT_COVERAGE");
  if (env && env[0]) {
    o.coverage = static_cast<int>(integer(env, "HIPDNN_UT_COVERAGE", 1, 3));
  }
  bool seed_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--coverage") {
      o.coverage = static_cast<int>(integer(value(argc, argv, i), "coverage", 1, 3));
    } else if (argument == "--n") {
      o.n = integer(value(argc, argv, i), "n");
      o.custom = true;
    } else if (argument == "--seed") {
      o.seed = static_cast<uint64_t>(integer(value(argc, argv, i), "seed", 0));
      seed_set = true;
      o.custom = true;
    } else {
      throw std::invalid_argument(
          "unknown argument " + argument +
          "; usage: [--coverage 1|2|3] | [--n <elements>] [--seed <seed>]");
    }
  }
  if (!seed_set) {
    o.seed = static_cast<uint64_t>(o.n);
  }
  return o;
}

int run(int argc, char** argv) {
  const Options options = parse_options(argc, argv);
  hipdnn_ep_test::timing::printTimingMode();

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  if (options.custom) {
    std::cout << "div custom case: n=" << options.n
              << " seed=" << options.seed << '\n';
    all_pass = run_div_case(options.n, options.seed, stream);
  } else {
    int size_count = 0;
    for (const Size& size : kSizes) {
      size_count += size.min_coverage <= options.coverage ? 1 : 0;
    }
    std::cout << "div coverage=" << options.coverage << " -> running "
              << size_count
              << " size(s) x 1 dtype (bf16) + 1 unsupported case = "
              << size_count + 1 << " div cases\n";
    for (const Size& size : kSizes) {
      if (size.min_coverage > options.coverage) {
        continue;
      }
      all_pass = run_div_case(size.n, static_cast<uint64_t>(size.n), stream) &&
                 all_pass;
    }
    all_pass = run_unsupported_case(stream) && all_pass;
  }
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
  try {
    return run(argc, argv);
  } catch (const std::invalid_argument& error) {
    std::cerr << "invalid UT input: " << error.what() << '\n';
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 1;
  }
}
