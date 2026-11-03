/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for GatherBlockQuantized over 8-bit weights.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
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

constexpr int kBits = 8;
constexpr int kGuardElements = 32;
// A NaN that no case can produce, so an output element the kernel skipped
// fails the comparison just like a guard element it overwrote.
constexpr uint16_t kFillBits = 0x7fff;
constexpr uint8_t kGarbageByte = 0xa5;

struct Shape {
  int64_t rows;
  int64_t cols;
  int block_size;
  int64_t indices;
  int min_coverage;
};

// 160 / 64 leaves a partial last block.
constexpr Shape kShapes[] = {
    {1024, 64, 16, 1, 1},
    {1024, 160, 64, 16, 1},
    {4096, 2048, 32, 16, 2},
    {2048, 960, 32, 2048, 2},
    {8192, 3840, 128, 16, 3},
    {4096, 2048, 64, 2048, 3},
};

struct Category {
  bool compact;
  bool zero_points;
  bool is_signed;
  int indices_rank;
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

void fill_random(std::vector<uint8_t>& bytes, Rng& rng) {
  for (size_t i = 0; i < bytes.size(); i += 8) {
    const uint64_t r = rng.next();
    std::memcpy(bytes.data() + i, &r, std::min<size_t>(8, bytes.size() - i));
  }
}

float half_to_float(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits >> 15) << 31;
  uint32_t exponent = (bits >> 10) & 0x1fu;
  uint32_t mantissa = bits & 0x3ffu;
  uint32_t result_bits = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      result_bits = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x3ffu;
      result_bits = sign | ((127u - 15u + 1u - shift) << 23) |
                    (mantissa << 13);
    }
  } else if (exponent == 31) {
    result_bits = sign | 0x7f800000u | (mantissa << 13);
  } else {
    result_bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
  }
  float result = 0.0f;
  std::memcpy(&result, &result_bits, sizeof(result));
  return result;
}

uint16_t float_to_half_bits(float value) {
  const __half half = __float2half(value);
  uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

// 11 significant bits, so the value is exact in fp16 and its product with a
// 9-bit (q - zp) is exact in fp32.
uint16_t random_scale(Rng& rng) {
  const uint64_t r = rng.next();
  const float magnitude =
      std::ldexp(1.0f + static_cast<float>(r & 1023) / 1024.0f,
                 -4 - static_cast<int>((r >> 10) & 3));
  return float_to_half_bits(((r >> 12) & 1) ? -magnitude : magnitude);
}

// The leading entries reach the last row through a negative index, the first
// and last rows, both out-of-range sides, and values far outside int32.
std::vector<int64_t> make_indices(int64_t rows, int64_t count, Rng& rng) {
  const int64_t special[] = {-1,       0, rows - 1, rows,
                             -rows,    -rows - 1, 1, -2,
                             rows + 5, int64_t(1) << 40,
                             -(int64_t(1) << 40)};
  const int64_t special_count =
      static_cast<int64_t>(sizeof(special) / sizeof(special[0]));
  std::vector<int64_t> indices(count);
  for (int64_t i = 0; i < count; ++i) {
    if (i < special_count) {
      indices[i] = special[i];
    } else {
      indices[i] =
          static_cast<int64_t>(rng.next() % static_cast<uint64_t>(2 * rows)) -
          rows;
    }
  }
  return indices;
}

// The gathered row, or -1 when the index is out of range.
int64_t wrap_index(int64_t index, int64_t rows) {
  const int64_t row = index < 0 ? index + rows : index;
  return (row < 0 || row >= rows) ? -1 : row;
}

std::vector<int64_t> indices_shape(int64_t count, int rank) {
  if (rank == 1) {
    return {count};
  }
  const int64_t batch = count % 2 == 0 ? 2 : 1;
  return {batch, count / batch};
}

int decode_quant(const std::vector<uint8_t>& buffer, int64_t element,
                 bool is_signed) {
  if (is_signed) {
    return static_cast<int8_t>(buffer[element]);
  }
  return buffer[element];
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ != 0) {
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)));
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
  size_t count_ = 0;
};

template <typename T> void upload(T* destination, const std::vector<T>& source) {
  HIP_CHECK(hipMemcpy(destination, source.data(), source.size() * sizeof(T),
                      hipMemcpyHostToDevice));
}

struct Metrics {
  int64_t mismatches = 0;
  int64_t first_mismatch = -1;
  double max_abs = 0.0;
  double rel_l2 = 0.0;
};

Metrics compare(const uint16_t* actual, const std::vector<uint16_t>& expected) {
  Metrics metrics;
  double error_sq = 0.0;
  double reference_sq = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    if (actual[i] != expected[i]) {
      if (metrics.mismatches == 0) {
        metrics.first_mismatch = static_cast<int64_t>(i);
      }
      ++metrics.mismatches;
    }
    const double got = half_to_float(actual[i]);
    const double want = half_to_float(expected[i]);
    const double delta = std::isnan(got)
                             ? std::numeric_limits<double>::infinity()
                             : std::abs(got - want);
    metrics.max_abs = std::max(metrics.max_abs, delta);
    error_sq += delta * delta;
    reference_sq += want * want;
  }
  metrics.rel_l2 = std::sqrt(error_sq / std::max(reference_sq, 1.0e-20));
  return metrics;
}

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
                 env_or("HIPDNN_RESULTS_OP", "gather_block_quantized"),
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
  void finish(double rel_l2, bool pass) {
    row.rel_l2 = rel_l2;
    row.verdict = pass ? "PASS" : "FAIL";
  }
};

std::string case_shape(const Shape& shape, const Category& category) {
  const std::vector<int64_t> dims =
      indices_shape(shape.indices, category.indices_rank);
  std::string indices = std::to_string(dims[0]);
  for (size_t i = 1; i < dims.size(); ++i) {
    indices += "x" + std::to_string(dims[i]);
  }
  char buffer[160];
  std::snprintf(buffer, sizeof(buffer), "R%lld_C%lld_bs%d_I%s%s%s%s",
                static_cast<long long>(shape.rows),
                static_cast<long long>(shape.cols), shape.block_size,
                indices.c_str(), category.compact ? "_compact" : "",
                category.zero_points ? "_zp" : "",
                category.is_signed ? "_signed" : "");
  return buffer;
}

bool run_case(const Shape& shape, uint64_t seed, const Category& category,
              hipStream_t stream) {
  const std::string name = case_shape(shape, category);
  ResultRow result(name, "fixed:elementwise");
  const int64_t blocks =
      (shape.cols + shape.block_size - 1) / shape.block_size;
  const int64_t row_bytes = shape.cols * kBits / 8;

  // Every category of a shape draws the same inputs, so the compact and
  // full-table launches are checked against one reference.
  Rng rng(seed);
  std::vector<uint8_t> table(static_cast<size_t>(shape.rows * row_bytes));
  fill_random(table, rng);
  std::vector<uint16_t> scales(static_cast<size_t>(shape.rows * blocks));
  for (uint16_t& scale : scales) {
    scale = random_scale(rng);
  }
  std::vector<uint8_t> zero_points(
      static_cast<size_t>((shape.rows * blocks * kBits + 7) / 8));
  fill_random(zero_points, rng);
  const std::vector<int64_t> indices =
      make_indices(shape.rows, shape.indices, rng);
  const int default_zp = category.is_signed ? 0 : 1 << (kBits - 1);

  const int64_t output_elements = shape.indices * shape.cols;
  std::vector<uint16_t> expected(static_cast<size_t>(output_elements), 0);
  for (int64_t k = 0; k < shape.indices; ++k) {
    const int64_t row = wrap_index(indices[k], shape.rows);
    if (row < 0) {
      continue;
    }
    for (int64_t col = 0; col < shape.cols; ++col) {
      const int64_t block = row * blocks + col / shape.block_size;
      const int q = decode_quant(table, row * shape.cols + col,
                                 category.is_signed);
      const int zp = category.zero_points
                         ? decode_quant(zero_points, block, category.is_signed)
                         : default_zp;
      expected[k * shape.cols + col] = float_to_half_bits(
          static_cast<float>(q - zp) * half_to_float(scales[block]));
    }
  }

  std::vector<uint8_t> data;
  if (category.compact) {
    data.assign(static_cast<size_t>(shape.indices * row_bytes), kGarbageByte);
    for (int64_t k = 0; k < shape.indices; ++k) {
      const int64_t row = wrap_index(indices[k], shape.rows);
      if (row >= 0) {
        std::memcpy(data.data() + k * row_bytes, table.data() + row * row_bytes,
                    static_cast<size_t>(row_bytes));
      }
    }
  } else {
    data = std::move(table);
  }

  DeviceBuffer<uint8_t> d_data(data.size());
  DeviceBuffer<int64_t> d_indices(indices.size());
  DeviceBuffer<uint16_t> d_scales(scales.size());
  DeviceBuffer<uint8_t> d_zero_points(zero_points.size());
  const std::vector<uint16_t> fill(
      static_cast<size_t>(output_elements + 2 * kGuardElements), kFillBits);
  DeviceBuffer<uint16_t> d_output(fill.size());
  upload(d_data.get(), data);
  upload(d_indices.get(), indices);
  upload(d_scales.get(), scales);
  upload(d_zero_points.get(), zero_points);
  upload(d_output.get(), fill);

  const std::vector<int64_t> index_dims =
      indices_shape(shape.indices, category.indices_rank);
  std::vector<int64_t> output_dims = index_dims;
  output_dims.push_back(shape.cols);
  const int64_t data_dims[2] = {shape.rows, shape.cols};
  const int64_t scale_dims[2] = {shape.rows, blocks};
  const int status = hip_gather_block_quantized(
      reinterpret_cast<void*>(stream), d_data.get(), d_indices.get(),
      d_scales.get(),
      category.zero_points ? static_cast<const void*>(d_zero_points.get())
                           : nullptr,
      d_output.get() + kGuardElements, data_dims, 2, index_dims.data(),
      static_cast<int>(index_dims.size()), scale_dims, 2, output_dims.data(),
      static_cast<int>(output_dims.size()), kBits, shape.block_size,
      /*gather_axis=*/0, /*quantize_axis=*/1, default_zp,
      category.is_signed ? 1 : 0, /*is_float8_data=*/0,
      /*indices_is_int64=*/1, category.compact ? 1 : 0, HIP_DTYPE_FLOAT16);
  if (status != static_cast<int>(hipSuccess)) {
    std::cout << "FAIL launch status=" << status << " shape=" << name << '\n';
    return false;
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  std::vector<uint16_t> actual(fill.size());
  HIP_CHECK(hipMemcpy(actual.data(), d_output.get(),
                      actual.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
  bool guards_ok = true;
  for (int i = 0; i < kGuardElements; ++i) {
    guards_ok = guards_ok && actual[i] == kFillBits &&
                actual[kGuardElements + output_elements + i] == kFillBits;
  }
  const Metrics metrics = compare(actual.data() + kGuardElements, expected);
  const bool pass = guards_ok && metrics.mismatches == 0;

  std::cout << (pass ? "PASS" : "FAIL") << " relL2=" << metrics.rel_l2
            << " n=" << output_elements << " shape=" << name
            << " mismatches=" << metrics.mismatches
            << " maxAbs=" << metrics.max_abs
            << " guards=" << (guards_ok ? "PASS" : "FAIL");
  if (metrics.first_mismatch >= 0) {
    const int64_t i = metrics.first_mismatch;
    char detail[96];
    std::snprintf(detail, sizeof(detail),
                  " first=%lld got=0x%04x want=0x%04x index=%lld",
                  static_cast<long long>(i),
                  actual[kGuardElements + i], expected[i],
                  static_cast<long long>(indices[i / shape.cols]));
    std::cout << detail;
  }
  std::cout << '\n';
  result.finish(metrics.rel_l2, pass);
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

  int shape_count = 0;
  for (const Shape& shape : kShapes) {
    shape_count += shape.min_coverage <= coverage ? 1 : 0;
  }
  constexpr int category_count = 2 * 2 * 2 * 2;
  std::cout << "gather_block_quantized coverage=" << coverage
            << " cases=" << shape_count * category_count << '\n';

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  uint64_t seed = 0;
  for (const Shape& shape : kShapes) {
    ++seed;
    if (shape.min_coverage > coverage) {
      continue;
    }
    for (bool compact : {false, true}) {
      for (bool zero_points : {false, true}) {
        for (bool is_signed : {false, true}) {
          for (int indices_rank : {1, 2}) {
            all_pass = run_case(shape, seed,
                                {compact, zero_points, is_signed, indices_rank},
                                stream) &&
                       all_pass;
          }
        }
      }
    }
  }
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
