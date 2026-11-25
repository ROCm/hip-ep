/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for GatherBlockQuantized over float8 e4m3fn weights.
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

// block_size applies to the blockwise scales only; per-tensor and per-row
// scales use one block per row. 160 / 64 leaves a partial last block.
constexpr Shape kShapes[] = {
    {1024, 160, 32, 1, 1},
    {4096, 160, 64, 16, 1},
    {2048, 2048, 128, 16, 2},
    {4096, 960, 64, 2048, 2},
    {1024, 64, 16, 16, 3},
    {8192, 3840, 128, 2048, 3},
};

// Scales shape: Tensor [1, 1], Row [rows, 1], Block [rows, ceil(cols / bs)].
enum class Grain { Tensor, Row, Block };

struct Category {
  bool compact;
  Grain grain;
  int indices_rank;
};

const char* grain_name(Grain grain) {
  switch (grain) {
  case Grain::Tensor:
    return "tensor";
  case Grain::Row:
    return "row";
  case Grain::Block:
    return "block";
  }
  return "unknown";
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

// 11 significant bits, so the value is exact in fp16 and its product with an
// e4m3fn value (4 significant bits) is exact in fp32.
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

// OCP e4m3fn: exponent bias 7, no infinities. Callers never pass the NaN
// encoding S.1111.111.
float decode_e4m3fn(uint8_t byte) {
  const float sign = (byte & 0x80) ? -1.0f : 1.0f;
  const int exponent = (byte >> 3) & 0x0f;
  const int mantissa = byte & 0x07;
  if (exponent == 0) {
    return sign * std::ldexp(static_cast<float>(mantissa), -9);
  }
  return sign * std::ldexp(static_cast<float>(8 + mantissa), exponent - 10);
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

std::string case_shape(const Shape& shape, int block_size,
                       const Category& category) {
  const std::vector<int64_t> dims =
      indices_shape(shape.indices, category.indices_rank);
  std::string indices = std::to_string(dims[0]);
  for (size_t i = 1; i < dims.size(); ++i) {
    indices += "x" + std::to_string(dims[i]);
  }
  char buffer[160];
  std::snprintf(buffer, sizeof(buffer), "R%lld_C%lld_bs%d_I%s_%s%s",
                static_cast<long long>(shape.rows),
                static_cast<long long>(shape.cols), block_size,
                indices.c_str(), grain_name(category.grain),
                category.compact ? "_compact" : "");
  return buffer;
}

bool run_case(const Shape& shape, uint64_t seed, const Category& category,
              hipStream_t stream) {
  const int block_size = category.grain == Grain::Block
                             ? shape.block_size
                             : static_cast<int>(shape.cols);
  const std::string name = case_shape(shape, block_size, category);
  ResultRow result(name, "fixed:elementwise");
  const int64_t blocks = (shape.cols + block_size - 1) / block_size;
  const int64_t scale_rows = category.grain == Grain::Tensor ? 1 : shape.rows;
  const int64_t row_bytes = shape.cols * kBits / 8;

  // The compact and full-table launches of a grain draw the same inputs, so
  // they are checked against one reference.
  Rng rng(seed);
  std::vector<uint8_t> table(static_cast<size_t>(shape.rows * row_bytes));
  fill_random(table, rng);
  for (uint8_t& byte : table) {
    if ((byte & 0x7f) == 0x7f) {
      byte ^= 0x01;
    }
  }
  std::vector<uint16_t> scales(static_cast<size_t>(scale_rows * blocks));
  for (uint16_t& scale : scales) {
    scale = random_scale(rng);
  }
  const std::vector<int64_t> indices =
      make_indices(shape.rows, shape.indices, rng);

  const int64_t output_elements = shape.indices * shape.cols;
  std::vector<uint16_t> expected(static_cast<size_t>(output_elements), 0);
  for (int64_t k = 0; k < shape.indices; ++k) {
    const int64_t row = wrap_index(indices[k], shape.rows);
    if (row < 0) {
      continue;
    }
    for (int64_t col = 0; col < shape.cols; ++col) {
      const int64_t block =
          (scale_rows == 1 ? 0 : row) * blocks + col / block_size;
      expected[k * shape.cols + col] = float_to_half_bits(
          decode_e4m3fn(table[row * shape.cols + col]) *
          half_to_float(scales[block]));
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
  const std::vector<uint16_t> fill(
      static_cast<size_t>(output_elements + 2 * kGuardElements), kFillBits);
  DeviceBuffer<uint16_t> d_output(fill.size());
  upload(d_data.get(), data);
  upload(d_indices.get(), indices);
  upload(d_scales.get(), scales);
  upload(d_output.get(), fill);

  const std::vector<int64_t> index_dims =
      indices_shape(shape.indices, category.indices_rank);
  std::vector<int64_t> output_dims = index_dims;
  output_dims.push_back(shape.cols);
  const int64_t data_dims[2] = {shape.rows, shape.cols};
  const int64_t scale_dims[2] = {scale_rows, blocks};
  const int status = hip_gather_block_quantized(
      reinterpret_cast<void*>(stream), d_data.get(), d_indices.get(),
      d_scales.get(), /*zero_points=*/nullptr,
      d_output.get() + kGuardElements, data_dims, 2, index_dims.data(),
      static_cast<int>(index_dims.size()), scale_dims, 2, output_dims.data(),
      static_cast<int>(output_dims.size()), kBits, block_size,
      /*gather_axis=*/0, /*quantize_axis=*/1, /*default_zp=*/0,
      /*is_signed_data=*/0, /*is_float8_data=*/1,
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
  constexpr int category_count = 2 * 3 * 2;
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
      for (Grain grain : {Grain::Tensor, Grain::Row, Grain::Block}) {
        for (int indices_rank : {1, 2}) {
          all_pass = run_case(shape, seed, {compact, grain, indices_rank},
                              stream) &&
                     all_pass;
        }
      }
    }
  }
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
