/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for the decomposed-GQA softmax entries that produce
 * bf16 probabilities: hip_gqa_softmax_f32_to_bf16 and
 * hip_gqa_softmax_f32_to_out_biased with bias_dtype / output_dtype
 * HIP_DTYPE_BFLOAT16. The same biased entry is also run with fp16 and fp32
 * dtypes, because its dtype arguments replaced the element-size arguments the
 * fp16 and fp32 GQA paths used.
 *
 * Layout (as real/gqa.cpp calls it): fp32 scores [heads, cols, rows] where
 * heads = B*H, cols = query rows and rows = keys; each (head, col) is one
 * softmax over `rows` values. The bias is [bias_batch, bias_heads, bias_sq,
 * bias_total] and a block of it starting at (bias_row_offset,
 * bias_col_offset) is added before the max.
 *
 * The reference is computed in double and rounded once to the output type;
 * the kernel uses fp32 exp2f, so an output may be one unit in the last place
 * of the output type away from the reference, plus a few fp32 ulps of compute
 * error that matter only for fp32 output. The bf16 <-> fp32 conversions
 * are integer bit operations, not the HIP intrinsics the kernel uses.
 *
 * The output has a padded per-head stride and guard elements on both sides;
 * each case runs twice with two fill bytes, so an element the kernel did not
 * write, or a write into the padding or the guards, fails at least one run.
 * A case that passes is then timed with the shared UT timer
 * (common/hip_adaptive_timing.h).
 *
 * Usage:
 *   test_gqa_softmax.exe [--coverage 1|2|3]      built-in cases (default 3)
 *   test_gqa_softmax.exe --batch B --heads H --queries Q --keys K
 *       [--mask none|full|key_row|chunk] [--mask-dtype bf16|f16|f32]
 *       [--out-dtype bf16|f16|f32] [--smooth 0|1] [--seed S]
 *                                                one case; any flag selects it
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include "../../../common/hip_adaptive_timing.h"
#include "../../../common/ut_case_input.h"

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
// The timed launches rotate over copies of the scores and the mask, so that
// each launch reads them from DRAM as in a model rather than from L2 / the
// Infinity Cache; each copy is read again only after kColdRotationBytes of
// other copies. HIPDNN_KERNEL_UT_WARM=1 times one copy back to back instead.
// Same scheme as the other leaves (e.g. div/Xbf16_Wbf16_Ybf16).
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

// Mean time of one complete softmax call; launch(c) uses input copy c. The
// shared UT policy selects the call count.
template <class Launch>
double rotatedMeanMs(hipStream_t stream, const ColdCopies& copies,
                     const char* kernel, Launch&& launch) {
  return timing::adaptiveRotatedMeanMs(
      stream, copies, launch, holdReleaseFlag(),
      [&](const int* flag) { holdStreamKernel<<<1, 1, 0, stream>>>(flag); },
      warmTiming(), kColdSettleBytes, "gqa_softmax_xfp32_wbf16_ybf16", kernel);
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

constexpr const char* kLeaf = "gqa/softmax/Xfp32_Wbf16_Ybf16";
constexpr int64_t kGuardElements = 64;
constexpr int64_t kStridePad = 5;
constexpr uint8_t kFillBytes[2] = {0xa5, 0x5a};
// Largest custom case, in score elements: the double reference holds 8 bytes
// per element.
constexpr int64_t kMaxScoreElements = int64_t(1) << 26;

// ---- element types -----------------------------------------------------------

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

// fp32 -> bf16, round to nearest even (inputs here are finite).
uint16_t float_to_bf16(float f) {
  const uint32_t bits = float_bits(f);
  const uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<uint16_t>((bits + 0x7fffu + lsb) >> 16);
}

float f16_to_float(uint16_t h) {
  __half v;
  std::memcpy(&v, &h, sizeof(h));
  return __half2float(v);
}

uint16_t float_to_f16(float f) {
  const __half v = __float2half(f);
  uint16_t h;
  std::memcpy(&h, &v, sizeof(h));
  return h;
}

int dtype_bytes(int dtype) { return dtype == HIP_DTYPE_FLOAT32 ? 4 : 2; }

const char* dtype_name(int dtype) {
  switch (dtype) {
  case HIP_DTYPE_FLOAT32:
    return "f32";
  case HIP_DTYPE_FLOAT16:
    return "f16";
  case HIP_DTYPE_BFLOAT16:
    return "bf16";
  default:
    return "other";
  }
}

// Store `value` as `dtype` (rounded) at element i of a byte buffer.
void store(std::vector<uint8_t>* buf, int dtype, int64_t i, float value) {
  uint8_t* p = buf->data() + i * dtype_bytes(dtype);
  if (dtype == HIP_DTYPE_FLOAT32) {
    std::memcpy(p, &value, 4);
    return;
  }
  const uint16_t h = dtype == HIP_DTYPE_BFLOAT16 ? float_to_bf16(value)
                                                 : float_to_f16(value);
  std::memcpy(p, &h, 2);
}

float load(const uint8_t* buf, int dtype, int64_t i) {
  const uint8_t* p = buf + i * dtype_bytes(dtype);
  if (dtype == HIP_DTYPE_FLOAT32) {
    float f;
    std::memcpy(&f, p, 4);
    return f;
  }
  uint16_t h;
  std::memcpy(&h, p, 2);
  return dtype == HIP_DTYPE_BFLOAT16 ? bf16_to_float(h) : f16_to_float(h);
}

// One unit in the last place of the output type at |v|, plus the type's
// smallest subnormal step so values near zero have a floor.
double ulp(int dtype, double v) {
  const int mantissa_bits =
      dtype == HIP_DTYPE_FLOAT32 ? 23 : (dtype == HIP_DTYPE_BFLOAT16 ? 7 : 10);
  const double floor_step =
      dtype == HIP_DTYPE_FLOAT32
          ? 1.4e-45
          : (dtype == HIP_DTYPE_BFLOAT16 ? 9.2e-41 : 5.96e-8);
  if (v == 0.0) {
    return floor_step;
  }
  int exponent;
  std::frexp(std::fabs(v), &exponent);
  return std::ldexp(1.0, exponent - 1 - mantissa_bits) + floor_step;
}

// ---- inputs ------------------------------------------------------------------

uint64_t next_random(uint64_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 7;
  *state ^= *state << 17;
  return *state;
}

// Uniform in [lo, hi).
float random_float(uint64_t* state, float lo, float hi) {
  const double u =
      static_cast<double>(next_random(state) >> 11) / 9007199254740992.0;
  return static_cast<float>(lo + (hi - lo) * u);
}

// ---- device helpers ----------------------------------------------------------

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

// ---- results.csv -------------------------------------------------------------

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
                 env_or("HIPDNN_RESULTS_OP", "gqa/softmax"),
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
// case returning early still reports FAIL. finish() binds a timed PASS to its
// UT_TIMING record, so the case's timing::CaseScope must be alive when it is
// called.
struct ResultRow {
  CsvRow row;
  explicit ResultRow(std::string shape) {
    row.shape = std::move(shape);
    row.config = "fixed:gqa_softmax";
    row.rel_l2 = 1.0;
    row.verdict = "FAIL";
  }
  ResultRow(const ResultRow&) = delete;
  ResultRow& operator=(const ResultRow&) = delete;
  ~ResultRow() { CsvWriter().write(row); }
  void finish(bool pass, double rel_l2, double time_ms) {
    row.rel_l2 = rel_l2;
    row.verdict = pass ? "PASS" : "FAIL";
    row.time_ms = time_ms;
    hipdnn_ep_test::timing::emitCsvBinding(row.shape, row.config, row.verdict,
                                           row.rel_l2, row.time_ms);
  }
};

// ---- cases -------------------------------------------------------------------

struct Shape {
  int batch, heads, cols, rows;
  int min_coverage;
};

// Bias extents and the block of it each call reads. sq == 1 is the one-row
// (per-key) bias that serves every query row.
struct BiasShape {
  int batch, heads, sq, total, row_offset, col_offset;
};

// The three bias layouts real/gqa.cpp passes: a full [B, H, q, k] mask; a
// broadcast one-row mask [1, H, 1, k + 3] read at column offset 3 (a sliding
// window); and a [B, 1, 2q, k] mask read from row offset q (a later prefill
// chunk).
enum class Mask { None, Full, KeyRow, Chunk };

BiasShape bias_shape(Mask mask, const Shape& s) {
  switch (mask) {
  case Mask::KeyRow:
    return {1, s.heads, 1, s.rows + 3, 0, 3};
  case Mask::Chunk:
    return {s.batch, 1, 2 * s.cols, s.rows, s.cols, 0};
  default:
    return {s.batch, s.heads, s.cols, s.rows, 0, 0};
  }
}

struct Case {
  Shape shape;
  bool smooth;
  bool biased;
  BiasShape bias;
  int bias_dtype, out_dtype;
};

Case make_case(const Shape& shape, Mask mask, int bias_dtype, int out_dtype,
               bool smooth) {
  const bool biased = mask != Mask::None;
  return {shape,
          smooth,
          biased,
          biased ? bias_shape(mask, shape) : BiasShape{},
          biased ? bias_dtype : 0,
          out_dtype};
}

std::string case_name(const Case& c) {
  std::string name = std::string(c.biased ? "biased_" : "softmax_") +
                     (c.biased ? std::string(dtype_name(c.bias_dtype)) + "_"
                               : std::string()) +
                     dtype_name(c.out_dtype) + "_b" +
                     std::to_string(c.shape.batch) + "h" +
                     std::to_string(c.shape.heads) + "_q" +
                     std::to_string(c.shape.cols) + "_k" +
                     std::to_string(c.shape.rows);
  if (c.biased) {
    name += "_bias" + std::to_string(c.bias.batch) + "x" +
            std::to_string(c.bias.heads) + "x" + std::to_string(c.bias.sq) +
            "x" + std::to_string(c.bias.total) + "@" +
            std::to_string(c.bias.row_offset) + "," +
            std::to_string(c.bias.col_offset);
  }
  if (c.smooth) {
    name += "_smooth";
  }
  return name;
}

// Checks one case against the reference and, when it passes, times it.
bool run_case(const Case& c, uint64_t seed, hipStream_t stream) {
  std::string name = case_name(c);
  if (seed != 1) {
    name += "_seed" + std::to_string(seed);
  }
  const char* kernel = c.biased ? "hip_gqa_softmax_f32_to_out_biased"
                                : "hip_gqa_softmax_f32_to_bf16";
  hipdnn_ep_test::timing::CaseScope timing_case(kLeaf, name, kernel);
  ResultRow result(name);
  const Shape& s = c.shape;
  const int64_t heads = static_cast<int64_t>(s.batch) * s.heads;
  const int64_t plane = static_cast<int64_t>(s.cols) * s.rows;
  const int64_t out_stride = plane + kStridePad;
  const int out_bytes = dtype_bytes(c.out_dtype);

  uint64_t state = (seed * 0x9e3779b97f4a7c15ull) | 1u;
  std::vector<float> scores(static_cast<size_t>(heads * plane));
  for (float& v : scores) {
    v = random_float(&state, -12.0f, 12.0f);
  }

  // The bias is stored in its own dtype; the reference reads it back from the
  // rounded buffer so both sides add the same value. Every 7th element is a
  // mask value (a large negative number) so masked keys are covered too.
  std::vector<uint8_t> bias_host;
  if (c.biased) {
    const int64_t bias_elems = static_cast<int64_t>(c.bias.batch) *
                               c.bias.heads * c.bias.sq * c.bias.total;
    bias_host.assign(
        static_cast<size_t>(bias_elems * dtype_bytes(c.bias_dtype)), 0);
    const float mask =
        c.bias_dtype == HIP_DTYPE_FLOAT16 ? -65504.0f : -1.0e30f;
    for (int64_t i = 0; i < bias_elems; ++i) {
      store(&bias_host, c.bias_dtype, i,
            i % 7 == 3 ? mask : random_float(&state, -4.0f, 4.0f));
    }
  }

  // Reference probabilities, in double, per (head, col) row.
  std::vector<double> want(static_cast<size_t>(heads * plane));
  for (int64_t h = 0; h < heads; ++h) {
    const int b = static_cast<int>(h / s.heads);
    const int hh = static_cast<int>(h % s.heads);
    for (int col = 0; col < s.cols; ++col) {
      std::vector<double> row(static_cast<size_t>(s.rows));
      for (int k = 0; k < s.rows; ++k) {
        double v = scores[static_cast<size_t>(h * plane + col * s.rows + k)];
        if (c.biased) {
          const int bb = c.bias.batch == 1 ? 0 : b;
          const int bh = c.bias.heads == 1 ? 0 : hh;
          const int bq = c.bias.sq == 1 ? 0 : c.bias.row_offset + col;
          const int64_t idx =
              ((static_cast<int64_t>(bb) * c.bias.heads + bh) * c.bias.sq +
               bq) *
                  c.bias.total +
              c.bias.col_offset + k;
          v += load(bias_host.data(), c.bias_dtype, idx);
        }
        row[static_cast<size_t>(k)] = v;
      }
      const double mx = *std::max_element(row.begin(), row.end());
      double sum = 0.0;
      for (double v : row) {
        sum += std::exp(v - mx);
      }
      if (c.smooth) {
        sum += std::exp(0.0 - mx);
      }
      for (int k = 0; k < s.rows; ++k) {
        want[static_cast<size_t>(h * plane + col * s.rows + k)] =
            std::exp(row[static_cast<size_t>(k)] - mx) / sum;
      }
    }
  }

  const size_t scores_bytes = scores.size() * sizeof(float);
  DeviceBytes d_scores(scores_bytes);
  upload(d_scores.get(), scores.data(), scores_bytes);
  DeviceBytes d_bias(bias_host.size());
  if (c.biased) {
    upload(d_bias.get(), bias_host.data(), bias_host.size());
  }
  const size_t guard_bytes = static_cast<size_t>(kGuardElements * out_bytes);
  const size_t body_bytes = static_cast<size_t>(heads * out_stride * out_bytes);
  const size_t total_bytes = body_bytes + 2 * guard_bytes;
  DeviceBytes d_out(total_bytes);
  void* out_ptr = d_out.get() + guard_bytes;

  const int total_head_queries = static_cast<int>(heads * s.cols);
  auto launch = [&](const void* scores_ptr, const void* bias_ptr) {
    if (c.biased) {
      return hip_gqa_softmax_f32_to_out_biased(
          reinterpret_cast<void*>(stream), scores_ptr, out_ptr,
          total_head_queries, s.rows, s.cols, static_cast<int>(plane),
          static_cast<int>(out_stride), nullptr, s.heads, c.smooth ? 1 : 0,
          bias_ptr, c.bias.batch, c.bias.heads, c.bias_dtype, c.out_dtype,
          c.bias.sq, c.bias.row_offset, c.bias.total, c.bias.col_offset);
    }
    return hip_gqa_softmax_f32_to_bf16(
        reinterpret_cast<void*>(stream), scores_ptr, out_ptr,
        total_head_queries, s.rows, s.cols, static_cast<int>(plane),
        static_cast<int>(out_stride), nullptr, s.heads, c.smooth ? 1 : 0);
  };

  bool pass = true;
  double err2 = 0.0;
  double ref2 = 0.0;
  for (uint8_t fill : kFillBytes) {
    const std::vector<uint8_t> filled(total_bytes, fill);
    upload(d_out.get(), filled.data(), total_bytes);
    const int status = launch(d_scores.get(), d_bias.get());
    if (status != 0) {
      std::cout << "FAIL " << name << " returned " << status << '\n';
      return false;
    }
    HIP_CHECK(hipStreamSynchronize(stream));
    const std::vector<uint8_t> out = download(d_out.get(), total_bytes);
    const uint8_t* body = out.data() + guard_bytes;

    // Guards and the per-head padding must keep the fill byte.
    bool untouched_ok = true;
    for (size_t i = 0; i < guard_bytes; ++i) {
      untouched_ok = untouched_ok && out[i] == fill &&
                     out[total_bytes - guard_bytes + i] == fill;
    }
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t p = plane; p < out_stride; ++p) {
        for (int byte = 0; byte < out_bytes; ++byte) {
          untouched_ok = untouched_ok &&
                         body[(h * out_stride + p) * out_bytes + byte] == fill;
        }
      }
    }

    int64_t mismatches = 0;
    int64_t first = -1;
    double first_got = 0.0;
    for (int64_t h = 0; h < heads; ++h) {
      for (int64_t i = 0; i < plane; ++i) {
        const double ref = want[static_cast<size_t>(h * plane + i)];
        const double got = load(body, c.out_dtype, h * out_stride + i);
        const double err = std::fabs(got - ref);
        err2 += err * err;
        ref2 += ref * ref;
        // One ulp of the output type covers the final rounding; the 4e-6
        // relative term covers the kernel's fp32 exp2f and sum, which only
        // matters for fp32 output (bf16/fp16 ulps are 1000x larger).
        if (!(err <= ulp(c.out_dtype, ref) + 4e-6 * ref)) {
          if (first < 0) {
            first = h * plane + i;
            first_got = got;
          }
          ++mismatches;
        }
      }
    }
    const bool ok = untouched_ok && mismatches == 0;
    if (!ok) {
      std::printf("FAIL %s fill=0x%02x mismatches=%lld untouched=%s",
                  name.c_str(), fill, static_cast<long long>(mismatches),
                  untouched_ok ? "PASS" : "FAIL");
      if (first >= 0) {
        std::printf(" first=%lld got=%.9g want=%.9g",
                    static_cast<long long>(first), first_got,
                    want[static_cast<size_t>(first)]);
      }
      std::printf("\n");
    }
    pass = pass && ok;
  }
  const double rel_l2 = ref2 > 0.0 ? std::sqrt(err2 / ref2) : 0.0;

  double time_ms = 0.0;
  if (pass) {
    const hipdnn_ep_test::ColdCopies inputs(
        {{d_scores.get(), scores_bytes}, {d_bias.get(), bias_host.size()}});
    time_ms = hipdnn_ep_test::rotatedMeanMs(stream, inputs, kernel, [&](int k) {
      return launch(inputs.part(k, 0), inputs.part(k, 1));
    });
    std::printf("PASS %s relL2=%.3e time_ms=%.6f\n", name.c_str(), rel_l2,
                time_ms);
  }
  result.finish(pass, rel_l2, time_ms);
  return pass;
}

// An unsupported dtype or a null bias must return -1 and leave the output
// untouched. Not timed.
bool run_rejected_case(const char* name, int bias_dtype, int out_dtype,
                       bool null_bias, hipStream_t stream) {
  hipdnn_ep_test::timing::CaseScope timing_case(
      kLeaf, name, "hip_gqa_softmax_f32_to_out_biased");
  ResultRow result(name);
  const int rows = 33;
  const int cols = 2;
  const size_t n = static_cast<size_t>(rows) * cols;
  DeviceBytes d_scores(n * sizeof(float));
  HIP_CHECK(hipMemset(d_scores.get(), 0, n * sizeof(float)));
  DeviceBytes d_bias(n * sizeof(float));
  HIP_CHECK(hipMemset(d_bias.get(), 0, n * sizeof(float)));
  DeviceBytes d_out(n * sizeof(float));
  const std::vector<uint8_t> filled(n * sizeof(float), kFillBytes[0]);
  upload(d_out.get(), filled.data(), filled.size());
  const int status = hip_gqa_softmax_f32_to_out_biased(
      reinterpret_cast<void*>(stream), d_scores.get(), d_out.get(), cols, rows,
      cols, rows * cols, rows * cols, nullptr, 1, 0,
      null_bias ? nullptr : d_bias.get(), 1, 1, bias_dtype, out_dtype, 0, 0, 0,
      0);
  HIP_CHECK(hipStreamSynchronize(stream));
  const bool untouched = download(d_out.get(), filled.size()) == filled;
  const bool pass = status == -1 && untouched;
  std::cout << (pass ? "PASS " : "FAIL ") << name << " status=" << status
            << " output_untouched=" << (untouched ? "yes" : "no") << '\n';
  result.finish(pass, pass ? 0.0 : 1.0, 0.0);
  return pass;
}

// rows (keys) 1 and 7 are under one wave; 33 a partial second wave; 256 a full
// 256-thread block; 1000 makes each thread loop over several keys.
constexpr Shape kShapes[] = {
    {1, 2, 3, 7, 1},   {2, 3, 4, 33, 1},  {1, 4, 2, 1000, 1},
    {1, 1, 1, 1, 2},   {2, 2, 5, 256, 2}, {1, 8, 16, 1000, 3},
};

// Every shape runs the bf16 cases (unbiased with and without smooth softmax,
// and the three mask layouts) plus the fp16 and fp32 regression cases of the
// biased entry.
std::vector<Case> suite_cases(int coverage) {
  std::vector<Case> cases;
  for (const Shape& shape : kShapes) {
    if (shape.min_coverage > coverage) {
      continue;
    }
    for (bool smooth : {false, true}) {
      cases.push_back(
          make_case(shape, Mask::None, 0, HIP_DTYPE_BFLOAT16, smooth));
    }
    cases.push_back(make_case(shape, Mask::Full, HIP_DTYPE_BFLOAT16,
                              HIP_DTYPE_BFLOAT16, false));
    cases.push_back(make_case(shape, Mask::KeyRow, HIP_DTYPE_BFLOAT16,
                              HIP_DTYPE_BFLOAT16, true));
    cases.push_back(make_case(shape, Mask::Chunk, HIP_DTYPE_BFLOAT16,
                              HIP_DTYPE_BFLOAT16, false));
    cases.push_back(make_case(shape, Mask::Full, HIP_DTYPE_FLOAT16,
                              HIP_DTYPE_FLOAT16, false));
    cases.push_back(make_case(shape, Mask::Chunk, HIP_DTYPE_FLOAT32,
                              HIP_DTYPE_FLOAT32, false));
  }
  return cases;
}

struct Options {
  int coverage = 3;
  bool custom = false;
  Shape shape{1, 8, 16, 1000, 0};
  Mask mask = Mask::Full;
  int mask_dtype = HIP_DTYPE_BFLOAT16;
  int out_dtype = HIP_DTYPE_BFLOAT16;
  bool smooth = false;
  uint64_t seed = 1;
};

int parse_dtype(const std::string& text, const char* name) {
  if (text == "bf16")
    return HIP_DTYPE_BFLOAT16;
  if (text == "f16")
    return HIP_DTYPE_FLOAT16;
  if (text == "f32")
    return HIP_DTYPE_FLOAT32;
  throw std::invalid_argument(std::string(name) + " must be bf16, f16 or f32");
}

Mask parse_mask(const std::string& text) {
  if (text == "none")
    return Mask::None;
  if (text == "full")
    return Mask::Full;
  if (text == "key_row")
    return Mask::KeyRow;
  if (text == "chunk")
    return Mask::Chunk;
  throw std::invalid_argument("mask must be none, full, key_row or chunk");
}

// Throws std::invalid_argument on an unknown, incomplete or out-of-range
// argument.
Options parse_options(int argc, char** argv) {
  using hipdnn_ep_test::input::boolean;
  using hipdnn_ep_test::input::integer;
  using hipdnn_ep_test::input::value;
  Options o;
  const char* env = std::getenv("HIPDNN_UT_COVERAGE");
  if (env && env[0]) {
    o.coverage = static_cast<int>(integer(env, "HIPDNN_UT_COVERAGE", 1, 3));
  }
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--coverage") {
      o.coverage =
          static_cast<int>(integer(value(argc, argv, i), "coverage", 1, 3));
      continue;
    }
    o.custom = true;
    if (argument == "--batch") {
      o.shape.batch = static_cast<int>(integer(value(argc, argv, i), "batch"));
    } else if (argument == "--heads") {
      o.shape.heads = static_cast<int>(integer(value(argc, argv, i), "heads"));
    } else if (argument == "--queries") {
      o.shape.cols = static_cast<int>(integer(value(argc, argv, i), "queries"));
    } else if (argument == "--keys") {
      o.shape.rows = static_cast<int>(integer(value(argc, argv, i), "keys"));
    } else if (argument == "--mask") {
      o.mask = parse_mask(value(argc, argv, i));
    } else if (argument == "--mask-dtype") {
      o.mask_dtype = parse_dtype(value(argc, argv, i), "mask_dtype");
    } else if (argument == "--out-dtype") {
      o.out_dtype = parse_dtype(value(argc, argv, i), "out_dtype");
    } else if (argument == "--smooth") {
      o.smooth = boolean(value(argc, argv, i), "smooth");
    } else if (argument == "--seed") {
      o.seed = static_cast<uint64_t>(integer(value(argc, argv, i), "seed", 0));
    } else {
      throw std::invalid_argument(
          "unknown argument " + argument +
          "; usage: [--coverage 1|2|3] | [--batch B] [--heads H] "
          "[--queries Q] [--keys K] [--mask none|full|key_row|chunk] "
          "[--mask-dtype bf16|f16|f32] [--out-dtype bf16|f16|f32] "
          "[--smooth 0|1] [--seed S]");
    }
  }
  if (o.custom) {
    // The unbiased entry writes bf16 only.
    if (o.mask == Mask::None && o.out_dtype != HIP_DTYPE_BFLOAT16)
      throw std::invalid_argument("mask none requires out_dtype bf16");
    // The chunk mask holds 2 * queries rows.
    const int64_t elements = static_cast<int64_t>(o.shape.batch) *
                             o.shape.heads * 2 * o.shape.cols *
                             (o.shape.rows + 3);
    if (elements > kMaxScoreElements)
      throw std::invalid_argument(
          "batch * heads * queries * keys is too large for the reference");
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
    const Case c = make_case(options.shape, options.mask, options.mask_dtype,
                             options.out_dtype, options.smooth);
    std::cout << "gqa/softmax custom case: " << case_name(c)
              << " seed=" << options.seed << '\n';
    all_pass = run_case(c, options.seed, stream);
  } else {
    const std::vector<Case> cases = suite_cases(options.coverage);
    std::cout << "gqa/softmax coverage=" << options.coverage << " -> running "
              << cases.size() << " softmax cases + 3 rejected cases\n";
    uint64_t seed = 1;
    for (const Case& c : cases) {
      all_pass = run_case(c, seed++, stream) && all_pass;
    }
    all_pass = run_rejected_case("rejected_u8_output", HIP_DTYPE_BFLOAT16,
                                 HIP_DTYPE_UINT8, false, stream) &&
               all_pass;
    all_pass = run_rejected_case("rejected_u8_bias", HIP_DTYPE_UINT8,
                                 HIP_DTYPE_BFLOAT16, false, stream) &&
               all_pass;
    all_pass = run_rejected_case("rejected_null_bias", HIP_DTYPE_BFLOAT16,
                                 HIP_DTYPE_BFLOAT16, true, stream) &&
               all_pass;
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
