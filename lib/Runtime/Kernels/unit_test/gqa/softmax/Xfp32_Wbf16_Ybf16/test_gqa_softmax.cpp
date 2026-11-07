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
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
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
constexpr int64_t kStridePad = 5;
constexpr uint8_t kFillBytes[2] = {0xa5, 0x5a};

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
// case returning early still reports FAIL.
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
  void finish(bool pass, double rel_l2) {
    row.rel_l2 = rel_l2;
    row.verdict = pass ? "PASS" : "FAIL";
  }
};

// ---- cases -------------------------------------------------------------------

struct Shape {
  int batch, heads, cols, rows;
  int min_coverage;
};

// Bias extents and the block of it each call reads. bias_sq == 1 is the
// one-row (per-key) bias that serves every query row.
struct BiasShape {
  int batch, heads, sq, total, row_offset, col_offset;
};

struct Case {
  Shape shape;
  bool smooth;
  bool biased;
  BiasShape bias;
  int bias_dtype, out_dtype;
};

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

bool run_case(const Case& c, uint64_t seed, hipStream_t stream) {
  const std::string name = case_name(c);
  ResultRow result(name);
  const Shape& s = c.shape;
  const int64_t heads = static_cast<int64_t>(s.batch) * s.heads;
  const int64_t plane = static_cast<int64_t>(s.cols) * s.rows;
  const int64_t out_stride = plane + kStridePad;
  const int out_bytes = dtype_bytes(c.out_dtype);

  uint64_t state = seed | 1u;
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

  DeviceBytes d_scores(scores.size() * sizeof(float));
  upload(d_scores.get(), scores.data(), scores.size() * sizeof(float));
  DeviceBytes d_bias(bias_host.size());
  if (c.biased) {
    upload(d_bias.get(), bias_host.data(), bias_host.size());
  }
  const size_t guard_bytes = static_cast<size_t>(kGuardElements * out_bytes);
  const size_t body_bytes = static_cast<size_t>(heads * out_stride * out_bytes);
  const size_t total_bytes = body_bytes + 2 * guard_bytes;
  DeviceBytes d_out(total_bytes);

  bool pass = true;
  double err2 = 0.0;
  double ref2 = 0.0;
  for (uint8_t fill : kFillBytes) {
    const std::vector<uint8_t> filled(total_bytes, fill);
    upload(d_out.get(), filled.data(), total_bytes);
    void* out_ptr = d_out.get() + guard_bytes;
    const int total_head_queries = static_cast<int>(heads * s.cols);
    int status;
    if (c.biased) {
      status = hip_gqa_softmax_f32_to_out_biased(
          reinterpret_cast<void*>(stream), d_scores.get(), out_ptr,
          total_head_queries, s.rows, s.cols, static_cast<int>(plane),
          static_cast<int>(out_stride), nullptr, s.heads, c.smooth ? 1 : 0,
          d_bias.get(), c.bias.batch, c.bias.heads, c.bias_dtype, c.out_dtype,
          c.bias.sq, c.bias.row_offset, c.bias.total, c.bias.col_offset);
    } else {
      status = hip_gqa_softmax_f32_to_bf16(
          reinterpret_cast<void*>(stream), d_scores.get(), out_ptr,
          total_head_queries, s.rows, s.cols, static_cast<int>(plane),
          static_cast<int>(out_stride), nullptr, s.heads, c.smooth ? 1 : 0);
    }
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
  if (pass) {
    std::printf("PASS %s relL2=%.3e\n", name.c_str(), rel_l2);
  }
  result.finish(pass, rel_l2);
  return pass;
}

// An unsupported dtype or a null bias must return -1 and leave the output
// untouched.
bool run_rejected_case(const char* name, int bias_dtype, int out_dtype,
                       bool null_bias, hipStream_t stream) {
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
  result.finish(pass, pass ? 0.0 : 1.0);
  return pass;
}

// rows (keys) 1 and 7 are under one wave; 33 a partial second wave; 256 a full
// 256-thread block; 1000 makes each thread loop over several keys.
constexpr Shape kShapes[] = {
    {1, 2, 3, 7, 1},   {2, 3, 4, 33, 1},  {1, 4, 2, 1000, 1},
    {1, 1, 1, 1, 2},   {2, 2, 5, 256, 2}, {1, 8, 16, 1000, 3},
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

  std::vector<Case> cases;
  for (const Shape& shape : kShapes) {
    if (shape.min_coverage > coverage) {
      continue;
    }
    // A full [B, H, cols, rows] bias; a broadcast one-row bias
    // [1, H, 1, rows + 3] read at a column offset (the sliding-window case);
    // and a [B, 1, 2*cols, rows] bias read from row offset `cols` (a later
    // prefill chunk).
    const BiasShape full{shape.batch, shape.heads, shape.cols, shape.rows,
                         0, 0};
    const BiasShape key_row{1, shape.heads, 1, shape.rows + 3, 0, 3};
    const BiasShape chunk{shape.batch, 1, 2 * shape.cols, shape.rows,
                          shape.cols, 0};
    for (bool smooth : {false, true}) {
      cases.push_back({shape, smooth, false, {}, 0, HIP_DTYPE_BFLOAT16});
    }
    cases.push_back({shape, false, true, full, HIP_DTYPE_BFLOAT16,
                     HIP_DTYPE_BFLOAT16});
    cases.push_back({shape, true, true, key_row, HIP_DTYPE_BFLOAT16,
                     HIP_DTYPE_BFLOAT16});
    cases.push_back({shape, false, true, chunk, HIP_DTYPE_BFLOAT16,
                     HIP_DTYPE_BFLOAT16});
    cases.push_back({shape, false, true, full, HIP_DTYPE_FLOAT16,
                     HIP_DTYPE_FLOAT16});
    cases.push_back({shape, false, true, chunk, HIP_DTYPE_FLOAT32,
                     HIP_DTYPE_FLOAT32});
  }
  std::cout << "gqa/softmax coverage=" << coverage << " -> running "
            << cases.size() << " softmax cases + 3 rejected cases\n";

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  uint64_t seed = 1;
  for (const Case& c : cases) {
    all_pass = run_case(c, seed++ * 0x9e3779b97f4a7c15ull, stream) && all_pass;
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
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
