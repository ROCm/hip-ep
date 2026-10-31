/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for hip_cast with bf16 on one side of the cast.
 *
 * Covers bf16 <-> {f32, f16, f64, i64, i32, i16, i8, u8} (16 pairs), plus a
 * few pre-existing non-bf16 pairs as a regression check, the unsupported-pair
 * path and num_elements == 0. Every output is compared bit-exactly with a
 * CPU reference computed in this file with plain integer bit operations; it
 * does not use the HIP conversion intrinsics the kernel uses.
 *
 * Reference semantics (same as the kernel and PyTorch): every bf16 cast goes
 * through fp32. fp32 -> bf16 and fp32 -> fp16 round to nearest even, so f64
 * and i64 sources round twice (first to fp32). float -> integer truncates
 * toward zero. A NaN result matches any NaN (payloads may differ).
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

// ---- dtype helpers ---------------------------------------------------------

constexpr int kF32 = HIP_DTYPE_FLOAT32;
constexpr int kF16 = HIP_DTYPE_FLOAT16;
constexpr int kBF16 = HIP_DTYPE_BFLOAT16;
constexpr int kF64 = HIP_DTYPE_FLOAT64;
constexpr int kI64 = HIP_DTYPE_INT64;
constexpr int kI32 = HIP_DTYPE_INT32;
constexpr int kI16 = HIP_DTYPE_INT16;
constexpr int kI8 = HIP_DTYPE_INT8;
constexpr int kU8 = HIP_DTYPE_UINT8;
constexpr int kU16 = HIP_DTYPE_UINT16;

int dtype_size(int dtype) {
  switch (dtype) {
  case kF64: case kI64: return 8;
  case kF32: case kI32: return 4;
  case kF16: case kBF16: case kI16: case kU16: return 2;
  case kI8: case kU8: return 1;
  default: return 0;
  }
}

const char* dtype_name(int dtype) {
  switch (dtype) {
  case kF32: return "f32";
  case kF16: return "f16";
  case kBF16: return "bf16";
  case kF64: return "f64";
  case kI64: return "i64";
  case kI32: return "i32";
  case kI16: return "i16";
  case kI8: return "i8";
  case kU8: return "u8";
  case kU16: return "u16";
  default: return "?";
  }
}

template <typename T> T get(const std::vector<uint8_t>& bytes, int64_t i) {
  T value;
  std::memcpy(&value, bytes.data() + i * sizeof(T), sizeof(T));
  return value;
}

template <typename T> void put(std::vector<uint8_t>& bytes, int64_t i, T v) {
  std::memcpy(bytes.data() + i * sizeof(T), &v, sizeof(T));
}

uint64_t raw_element(const uint8_t* bytes, int64_t i, int size) {
  uint64_t value = 0;
  std::memcpy(&value, bytes + i * size, static_cast<size_t>(size));
  return value;
}

uint32_t float_bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  return bits;
}

float bits_float(uint32_t bits) {
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// ---- reference conversions (integer bit operations only) -------------------

// fp32 -> bf16, round to nearest even, NaN stays NaN.
uint16_t f32_to_bf16(float f) {
  uint32_t x = float_bits(f);
  if ((x & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<uint16_t>((x >> 16) | 0x40u);
  }
  x += 0x7fffu + ((x >> 16) & 1u);
  return static_cast<uint16_t>(x >> 16);
}

float bf16_to_f32(uint16_t bits) {
  return bits_float(static_cast<uint32_t>(bits) << 16);
}

// fp32 -> fp16, round to nearest even, overflow to inf, subnormals kept.
uint16_t f32_to_f16(float f) {
  const uint32_t x = float_bits(f);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const uint32_t exp = (x >> 23) & 0xffu;
  const uint32_t man = x & 0x7fffffu;
  if (exp == 0xffu) {
    return static_cast<uint16_t>(sign | (man ? 0x7e00u : 0x7c00u));
  }
  const int e = static_cast<int>(exp) - 127 + 15;
  if (e >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }
  if (e >= 1) {
    uint32_t r = (static_cast<uint32_t>(e) << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) {
      ++r;  // may carry into the exponent, up to 0x7c00 = inf
    }
    return static_cast<uint16_t>(sign | r);
  }
  if (exp == 0) {
    return static_cast<uint16_t>(sign);  // fp32 zero or subnormal
  }
  // fp16 subnormal: value / 2^-24 = (man | 1 << 23) >> (126 - exp).
  const uint32_t m = man | 0x800000u;
  const int shift = 126 - static_cast<int>(exp);  // >= 14
  if (shift >= 25) {
    return static_cast<uint16_t>(sign);  // below half the smallest subnormal
  }
  uint32_t q = m >> shift;
  const uint32_t rem = m & ((1u << shift) - 1u);
  const uint32_t half = 1u << (shift - 1);
  if (rem > half || (rem == half && (q & 1u))) {
    ++q;
  }
  return static_cast<uint16_t>(sign | q);
}

float f16_to_f32(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits >> 15) << 31;
  uint32_t exponent = (bits >> 10) & 0x1fu;
  uint32_t mantissa = bits & 0x3ffu;
  uint32_t result = 0;
  if (exponent == 0) {
    if (mantissa == 0) {
      result = sign;
    } else {
      int shift = 0;
      while ((mantissa & 0x400u) == 0) {
        mantissa <<= 1;
        ++shift;
      }
      mantissa &= 0x3ffu;
      result = sign | ((127u - 15u + 1u - shift) << 23) | (mantissa << 13);
    }
  } else if (exponent == 31) {
    result = sign | 0x7f800000u | (mantissa << 13);
  } else {
    result = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
  }
  return bits_float(result);
}

// Source element -> fp32, as the kernel's CastIO<Src>::load does.
float load_as_float(int dtype, const std::vector<uint8_t>& in, int64_t i) {
  switch (dtype) {
  case kF32: return get<float>(in, i);
  case kF16: return f16_to_f32(get<uint16_t>(in, i));
  case kBF16: return bf16_to_f32(get<uint16_t>(in, i));
  case kF64: return static_cast<float>(get<double>(in, i));
  case kI64: return static_cast<float>(get<int64_t>(in, i));
  case kI32: return static_cast<float>(get<int32_t>(in, i));
  case kI16: return static_cast<float>(get<int16_t>(in, i));
  case kI8: return static_cast<float>(get<int8_t>(in, i));
  case kU8: return static_cast<float>(get<uint8_t>(in, i));
  default: std::abort();
  }
}

// fp32 -> destination element, as the kernel's CastIO<Dst>::store does.
void store_from_float(int dtype, std::vector<uint8_t>& out, int64_t i,
                      float f) {
  switch (dtype) {
  case kF32: put<float>(out, i, f); break;
  case kF16: put<uint16_t>(out, i, f32_to_f16(f)); break;
  case kBF16: put<uint16_t>(out, i, f32_to_bf16(f)); break;
  case kF64: put<double>(out, i, static_cast<double>(f)); break;
  case kI64: put<int64_t>(out, i, static_cast<int64_t>(f)); break;
  case kI32: put<int32_t>(out, i, static_cast<int32_t>(f)); break;
  case kI16: put<int16_t>(out, i, static_cast<int16_t>(f)); break;
  case kI8: put<int8_t>(out, i, static_cast<int8_t>(f)); break;
  case kU8: put<uint8_t>(out, i, static_cast<uint8_t>(f)); break;
  default: std::abort();
  }
}

std::vector<uint8_t> reference(int src, int dst, const std::vector<uint8_t>& in,
                               int64_t n) {
  std::vector<uint8_t> out(static_cast<size_t>(n * dtype_size(dst)));
  for (int64_t i = 0; i < n; ++i) {
    if (src == kI64 && dst == kI32) {
      // Pre-existing integer path: plain truncation, not through fp32.
      put<int32_t>(out, i, static_cast<int32_t>(get<int64_t>(in, i)));
    } else {
      store_from_float(dst, out, i, load_as_float(src, in, i));
    }
  }
  return out;
}

bool is_nan_bits(int dtype, uint64_t bits) {
  switch (dtype) {
  case kF32: return (bits & 0x7fffffffu) > 0x7f800000u;
  case kF16: return (bits & 0x7fffu) > 0x7c00u;
  case kBF16: return (bits & 0x7fffu) > 0x7f80u;
  case kF64:
    return (bits & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
  default: return false;
  }
}

// ---- input generation ------------------------------------------------------

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

// Special values first, then generated ones, so every case of at least
// ~100 elements contains all of the specials.
std::vector<uint8_t> make_f32(int64_t n, Rng& rng) {
  const float specials[] = {
      0.0f, -0.0f, INFINITY, -INFINITY, NAN, 1.0f, -1.0f,
      1.00390625f,   // 1 + 2^-8: bf16 tie, rounds down to 1.0 (even)
      -1.00390625f,
      1.01171875f,   // 1 + 3 * 2^-8: bf16 tie, rounds up (even)
      1.0078125f,    // exact in bf16
      3.14159f, -2.5f, 65504.0f, 65520.0f /* fp16 overflow tie -> inf */,
      1e-40f, -1e-40f, 1e-45f,  // fp32 subnormals
      FLT_MIN, -FLT_MIN, FLT_MAX, -FLT_MAX,
      3.3895314e38f,  // largest finite bf16
      3.4e38f,        // rounds to +inf in bf16
      bits_float(0x7f7f8000u),  // tie between max bf16 and inf -> inf
      bits_float(0x00008000u),  // subnormal tie, rounds down
      bits_float(0x00018000u),  // subnormal tie, rounds up
      6.1035156e-5f,  // smallest normal fp16
      5.9604645e-8f,  // smallest subnormal fp16
      2.9802322e-8f,  // half of it: fp16 tie, rounds to 0
  };
  const int64_t ns = static_cast<int64_t>(sizeof(specials) / sizeof(float));
  std::vector<uint8_t> v(static_cast<size_t>(n * 4));
  for (int64_t i = 0; i < n; ++i) {
    if (i < ns) {
      put<float>(v, i, specials[i]);
      continue;
    }
    const uint64_t r = rng.next();
    uint32_t bits = 0;
    switch (r & 3) {
    case 0:  // any bit pattern, including NaN, inf and subnormals
      bits = static_cast<uint32_t>(r >> 32);
      break;
    case 1: {  // around a bf16 rounding tie
      const uint32_t low[3] = {0x7fffu, 0x8000u, 0x8001u};
      bits = (static_cast<uint32_t>(r >> 32) & 0xffff0000u) |
             low[(r >> 2) % 3];
      break;
    }
    default:  // ordinary magnitudes 2^-27 .. 2^27
      bits = (static_cast<uint32_t>(r >> 63) << 31) |
             (static_cast<uint32_t>(100 + (r >> 8) % 55) << 23) |
             (static_cast<uint32_t>(r >> 16) & 0x7fffffu);
      break;
    }
    put<uint32_t>(v, i, bits);
  }
  return v;
}

// 16-bit sources: a few specials, then every bit pattern in order.
std::vector<uint8_t> make_enumerated16(int64_t n,
                                       const std::vector<uint16_t>& specials) {
  std::vector<uint8_t> v(static_cast<size_t>(n * 2));
  const int64_t ns = static_cast<int64_t>(specials.size());
  for (int64_t i = 0; i < n; ++i) {
    put<uint16_t>(v, i,
                  i < ns ? specials[static_cast<size_t>(i)]
                         : static_cast<uint16_t>((i - ns) & 0xffff));
  }
  return v;
}

std::vector<uint8_t> make_enumerated8(int64_t n) {
  std::vector<uint8_t> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    v[static_cast<size_t>(i)] = static_cast<uint8_t>(i & 0xff);
  }
  return v;
}

std::vector<uint8_t> make_f64(int64_t n, Rng& rng) {
  const double specials[] = {
      0.0, -0.0, INFINITY, -INFINITY, NAN, 1.0, -1.0, 1e300, -1e300, 1e-300,
      -1e-300, DBL_MAX, 3.4e38, 1e-40,
      1.0 + 0x1p-8,           // exact in fp32, bf16 tie -> 1.0
      1.0 + 0x1p-8 + 0x1p-40, // rounds to the tie in fp32 first -> 1.0
      65504.0, 3.141592653589793,
  };
  const int64_t ns = static_cast<int64_t>(sizeof(specials) / sizeof(double));
  std::vector<uint8_t> v(static_cast<size_t>(n * 8));
  for (int64_t i = 0; i < n; ++i) {
    if (i < ns) {
      put<double>(v, i, specials[i]);
      continue;
    }
    const uint64_t r = rng.next();
    uint64_t bits = 0;
    if (r & 1) {
      bits = rng.next();  // any bit pattern
    } else {  // ordinary magnitudes 2^-40 .. 2^40
      bits = ((r >> 63) << 63) |
             (static_cast<uint64_t>(1023 - 40 + (r >> 8) % 81) << 52) |
             (rng.next() & 0xfffffffffffffull);
    }
    put<uint64_t>(v, i, bits);
  }
  return v;
}

std::vector<uint8_t> make_i64(int64_t n, Rng& rng) {
  const int64_t specials[] = {
      0, 1, -1, 255, 256, 257, -257, 65535, 16777216, 16777217, -16777217,
      16777219, INT32_MAX, INT32_MIN, (int64_t(1) << 40) + 12345,
      -(int64_t(1) << 53) + 1, int64_t(1) << 62, INT64_MAX, INT64_MIN,
  };
  const int64_t ns = static_cast<int64_t>(sizeof(specials) / sizeof(int64_t));
  std::vector<uint8_t> v(static_cast<size_t>(n * 8));
  for (int64_t i = 0; i < n; ++i) {
    int64_t value = 0;
    if (i < ns) {
      value = specials[i];
    } else {
      const uint64_t r = rng.next();
      value = (r & 1) ? static_cast<int64_t>(rng.next())
                      : static_cast<int64_t>((r >> 8) % (1u << 26)) -
                            (int64_t(1) << 25);
    }
    put<int64_t>(v, i, value);
  }
  return v;
}

std::vector<uint8_t> make_i32(int64_t n, Rng& rng) {
  const int32_t specials[] = {0, 1, -1, 257, 16777216, 16777217, -16777217,
                              16777219, INT32_MAX, INT32_MIN};
  const int64_t ns = static_cast<int64_t>(sizeof(specials) / sizeof(int32_t));
  std::vector<uint8_t> v(static_cast<size_t>(n * 4));
  for (int64_t i = 0; i < n; ++i) {
    put<int32_t>(v, i,
                 i < ns ? specials[i]
                        : static_cast<int32_t>(rng.next() & 0xffffffffu));
  }
  return v;
}

// bf16 values whose truncation fits the integer destination, because a
// float -> integer cast of an out-of-range value or NaN is undefined (the
// same holds for the pre-existing fp32/fp16 -> integer kernels).
std::vector<uint8_t> make_bf16_in_range(int64_t n, int dst, Rng& rng) {
  const bool is_signed = dst != kU8;
  const int bits = dtype_size(dst) * 8;
  // Largest exponent whose bf16 values all fit: signed |x| < 2^(bits-1),
  // u8 x < 256.
  const int max_exp = is_signed ? bits - 2 : 7;
  std::vector<uint16_t> specials = {
      0x0000, 0x8000 /* -0 */, 0x3f80 /* 1 */, 0x3f00 /* 0.5 */,
      0x3fc0 /* 1.5 */, 0x3f7f /* 0.996 */, 0x4000 /* 2 */,
      static_cast<uint16_t>(((max_exp + 127) << 7) | 0x7f),  // largest
  };
  if (is_signed) {
    const uint16_t negatives[] = {
        0xbf80 /* -1 */, 0xbf00 /* -0.5 */, 0xbfc0 /* -1.5 */,
        0xbf7f /* -0.996 */,
        static_cast<uint16_t>(0x8000 | ((max_exp + 127) << 7) | 0x7f),
        // -2^(bits-1): the smallest value of the destination type.
        static_cast<uint16_t>(0x8000 | ((bits - 1 + 127) << 7)),
    };
    specials.insert(specials.end(), std::begin(negatives),
                    std::end(negatives));
  }
  const int64_t ns = static_cast<int64_t>(specials.size());
  std::vector<uint8_t> v(static_cast<size_t>(n * 2));
  for (int64_t i = 0; i < n; ++i) {
    if (i < ns) {
      put<uint16_t>(v, i, specials[static_cast<size_t>(i)]);
      continue;
    }
    const uint64_t r = rng.next();
    const int e = -10 + static_cast<int>((r >> 8) % (max_exp + 11));
    const uint16_t sign = (is_signed && (r & 1)) ? 0x8000 : 0;
    put<uint16_t>(v, i,
                  static_cast<uint16_t>(sign | ((e + 127) << 7) |
                                        ((r >> 32) & 0x7f)));
  }
  return v;
}

std::vector<uint8_t> make_input(int src, int dst, int64_t n, uint64_t seed) {
  Rng rng(seed);
  switch (src) {
  case kF32: return make_f32(n, rng);
  case kF64: return make_f64(n, rng);
  case kI64: return make_i64(n, rng);
  case kI32: return make_i32(n, rng);
  case kF16:
    return make_enumerated16(n, {0x3c00, 0x7c00, 0xfc00, 0x7e00, 0x8000,
                                 0x7bff, 0x0001, 0x0400});
  case kBF16:
    if (dtype_size(dst) <= 8 && dst != kF32 && dst != kF16 && dst != kF64) {
      return make_bf16_in_range(n, dst, rng);
    }
    return make_enumerated16(n, {0x3f80, 0x7f80, 0xff80, 0x7fc0, 0x8000,
                                 0x7f7f, 0x0001, 0x477f, 0x4780, 0x3380});
  case kI16:
    return make_enumerated16(n, {0x0000, 0x0001, 0xffff, 0x0101, 0x7fff,
                                 0x8000});
  case kI8:
  case kU8:
    return make_enumerated8(n);
  default:
    std::abort();
  }
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

void upload(uint8_t* device, const std::vector<uint8_t>& host) {
  HIP_CHECK(hipMemcpy(device, host.data(), host.size(), hipMemcpyHostToDevice));
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

std::string pair_name(int src, int dst) {
  return std::string(dtype_name(src)) + "_to_" + dtype_name(dst);
}

// Supported pair: output must match the reference bit for bit (any NaN for
// a NaN reference), and the guards and every element must be written.
bool run_cast_case(int src, int dst, int64_t n, uint64_t seed,
                   hipStream_t stream) {
  const std::string name = pair_name(src, dst) + "_n" + std::to_string(n);
  ResultRow result(name);
  const int ssize = dtype_size(src);
  const int dsize = dtype_size(dst);
  const std::vector<uint8_t> input = make_input(src, dst, n, seed);
  const std::vector<uint8_t> expected = reference(src, dst, input, n);

  DeviceBytes d_in(input.size());
  upload(d_in.get(), input);
  const size_t out_bytes =
      static_cast<size_t>((n + 2 * kGuardElements) * dsize);
  DeviceBytes d_out(out_bytes);
  const size_t guard_bytes = static_cast<size_t>(kGuardElements * dsize);

  bool pass = true;
  for (uint8_t fill : kFillBytes) {
    upload(d_out.get(), std::vector<uint8_t>(out_bytes, fill));
    const int status = hip_cast(reinterpret_cast<void*>(stream), d_in.get(),
                                d_out.get() + guard_bytes, n, src, dst);
    if (status != 0) {
      std::cout << "FAIL " << name << " hip_cast returned " << status << '\n';
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
    const uint8_t* got = out.data() + guard_bytes;
    for (int64_t i = 0; i < n; ++i) {
      const uint64_t g = raw_element(got, i, dsize);
      const uint64_t w = raw_element(expected.data(), i, dsize);
      if (g == w || (is_nan_bits(dst, w) && is_nan_bits(dst, g))) {
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
        std::printf(" first=%lld in=0x%llx got=0x%llx want=0x%llx",
                    static_cast<long long>(first),
                    static_cast<unsigned long long>(
                        raw_element(input.data(), first, ssize)),
                    static_cast<unsigned long long>(
                        raw_element(got, first, dsize)),
                    static_cast<unsigned long long>(
                        raw_element(expected.data(), first, dsize)));
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

// Unsupported pair: hip_cast must return -1 and leave the output untouched.
bool run_unsupported_case(int src, int dst, hipStream_t stream) {
  const int64_t n = 257;
  const std::string name = "unsupported_" + pair_name(src, dst);
  ResultRow result(name);
  const int ssize = std::max(dtype_size(src), 1);
  const int dsize = std::max(dtype_size(dst), 1);
  DeviceBytes d_in(static_cast<size_t>(n * ssize));
  HIP_CHECK(hipMemset(d_in.get(), 0, static_cast<size_t>(n * ssize)));
  const size_t out_bytes = static_cast<size_t>(n * dsize);
  DeviceBytes d_out(out_bytes);
  upload(d_out.get(), std::vector<uint8_t>(out_bytes, kFillBytes[0]));
  const int status = hip_cast(reinterpret_cast<void*>(stream), d_in.get(),
                              d_out.get(), n, src, dst);
  HIP_CHECK(hipStreamSynchronize(stream));
  const std::vector<uint8_t> out = download(d_out.get(), out_bytes);
  const bool untouched =
      std::all_of(out.begin(), out.end(),
                  [](uint8_t b) { return b == kFillBytes[0]; });
  const bool pass = status == -1 && untouched;
  std::cout << (pass ? "PASS " : "FAIL ") << name << " status=" << status
            << " output_untouched=" << (untouched ? "yes" : "no")
            << " (an 'unsupported conversion' message above is expected)\n";
  result.finish(pass);
  return pass;
}

// num_elements == 0: hip_cast returns 0 and writes nothing.
bool run_empty_case(hipStream_t stream) {
  ResultRow result("bf16_to_f32_n0");
  DeviceBytes d_in(16);
  DeviceBytes d_out(16);
  upload(d_out.get(), std::vector<uint8_t>(16, kFillBytes[0]));
  const int status = hip_cast(reinterpret_cast<void*>(stream), d_in.get(),
                              d_out.get(), 0, kBF16, kF32);
  HIP_CHECK(hipStreamSynchronize(stream));
  const std::vector<uint8_t> out = download(d_out.get(), 16);
  const bool untouched =
      std::all_of(out.begin(), out.end(),
                  [](uint8_t b) { return b == kFillBytes[0]; });
  const bool pass = status == 0 && untouched;
  std::cout << (pass ? "PASS " : "FAIL ") << "bf16_to_f32_n0 status=" << status
            << '\n';
  result.finish(pass);
  return pass;
}

struct Size {
  int64_t n;
  int min_coverage;
};

// 255/256/257 straddle the 256-thread block; 70001 holds every 16-bit
// pattern after the specials; the last one is a large tensor.
constexpr Size kSizes[] = {
    {1, 1}, {257, 1}, {70001, 1}, {255, 2}, {256, 2}, {4099, 2},
    {1048579, 3},
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

  std::vector<std::pair<int, int>> pairs;
  for (int other : {kF32, kF16, kF64, kI64, kI32, kI16, kI8, kU8}) {
    pairs.push_back({kBF16, other});
    pairs.push_back({other, kBF16});
  }
  // Pre-existing non-bf16 paths, to show they are unchanged.
  pairs.push_back({kF32, kF16});
  pairs.push_back({kF16, kF32});
  pairs.push_back({kI64, kI32});
  pairs.push_back({kU8, kF32});

  const std::pair<int, int> unsupported[] = {
      {kBF16, kU16}, {kU16, kBF16}, {kF64, kI8}};

  int size_count = 0;
  for (const Size& size : kSizes) {
    size_count += size.min_coverage <= coverage ? 1 : 0;
  }
  const int cases = static_cast<int>(pairs.size()) * size_count +
                    static_cast<int>(sizeof(unsupported) /
                                     sizeof(unsupported[0])) +
                    1;
  std::cout << "cast coverage=" << coverage << " -> running " << size_count
            << " size(s) x " << pairs.size() << " dtype pair(s) + 4 edge cases = "
            << cases << " cast cases\n";

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  for (size_t p = 0; p < pairs.size(); ++p) {
    for (const Size& size : kSizes) {
      if (size.min_coverage > coverage) {
        continue;
      }
      const uint64_t seed = (static_cast<uint64_t>(p) << 32) ^
                            static_cast<uint64_t>(size.n);
      all_pass = run_cast_case(pairs[p].first, pairs[p].second, size.n, seed,
                               stream) &&
                 all_pass;
    }
  }
  for (const auto& pair : unsupported) {
    all_pass = run_unsupported_case(pair.first, pair.second, stream) && all_pass;
  }
  all_pass = run_empty_case(stream) && all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
