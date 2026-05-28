/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- hip-onnx-runner.cpp - Run an ONNX model via MorphiZen EP ----------===//
//
// Loads an ONNX model, generates random inputs, runs one inference via the
// MorphiZen execution provider, and reports timing.
/*
Usage: hip-onnx-runner.exe [options]

Options:
-m, --model               Path to .onnx model
-L, --l2norm              Compare two dirs: dir1,dir2 (same *.bin set); each
 file must be ..._<type>.bin (fp32,fp16,i64,...); element-wise L2; no -m
-n,--no-ep                CPU only; skip EP registration (flag)
-d, --dump-level 0=off, 1=dump inputs to <stem>_i_dump/,
  2=outputs to <stem>_o_dump/, 3=both (default: 0)
-s, --seed                RNG seed for random inputs (default:42)
-i, --input-dir           Directory with input_<idx>_<name>_<type>.bin
 only; empty = random inputs
-o, --graph-opt-level
 session_options.SetGraphOptimizationLevel(level),
 0 = ORT_DISABLE_ALL,
 1 = ORT_ENABLE_BASIC,
 2 = ORT_ENABLE_EXTENDED,
 99 = ORT_ENABLE_ALL,
 -1 = default, not call this function  (default: -1)
-p, --positive-only       Generate positive-only random inputs [0.1, 256.0]
 for Sqrt/Reciprocal testing (flag)
*/
//===----------------------------------------------------------------------===//

#include <onnxruntime_cxx_api.h>

#include "CrashHandler.h"
#include "hip/timing.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#if _WIN32
#include <codecvt>
#include <locale>
#endif

#include "../common/minioptions.h"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int64_t calculate_product(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  for (auto d : shape)
    n *= d;
  return n;
}

static size_t element_byte_size(ONNXTensorElementDataType t) {
  switch (t) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return 8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return 8;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return 4;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return 2;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return 1;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return 1;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return 2;
  default:
    std::cerr << "Unsupported element type: " << t << "\n";
    std::exit(1);
  }
}

static std::string sanitize_filename_component(const std::string &name) {
  std::string s = name;
  for (auto &c : s) {
    if (c == '/' || c == '\\' || c == ':' || c == '*' || c == '?' || c == '"' ||
        c == '<' || c == '>' || c == '|' || c == '\0')
      c = '_';
  }
  if (s.empty())
    s = "tensor";
  return s;
}

// Short suffix before ".bin" for typed dumps (e.g. name_fp32.bin).
static const char *onnx_elem_type_tag(ONNXTensorElementDataType t) {
  switch (t) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    return "fp32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    return "fp16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    return "fp64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    return "bf16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    return "i64";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    return "i32";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    return "i16";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    return "i8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    return "u8";
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    return "u16";
  default:
    std::cerr << "Unsupported element type for dump tag: " << t << "\n";
    std::exit(1);
  }
}

static bool type_tag_to_onnx(const std::string &tag,
                             ONNXTensorElementDataType &out) {
  if (tag == "fp32")
    out = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  else if (tag == "fp16")
    out = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  else if (tag == "fp64")
    out = ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
  else if (tag == "bf16")
    out = ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16;
  else if (tag == "i64")
    out = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  else if (tag == "i32")
    out = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  else if (tag == "i16")
    out = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
  else if (tag == "i8")
    out = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
  else if (tag == "u8")
    out = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
  else if (tag == "u16")
    out = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
  else
    return false;
  return true;
}

// IEEE binary16 -> float32 (little-endian element order in buffers).
static float fp16_bits_to_float(uint16_t h) {
  const uint32_t s = static_cast<uint32_t>(h & 0x8000u) << 16;
  int32_t e = (h >> 10) & 0x1f;
  int32_t m = h & 0x3ff;
  uint32_t v;
  if (e == 0) {
    if (m == 0) {
      v = s;
    } else {
      while ((m & 0x400) == 0) {
        m <<= 1;
        e -= 1;
      }
      e += 1;
      m &= ~0x400;
      v = s | static_cast<uint32_t>(e + (127 - 15)) << 23 |
          static_cast<uint32_t>(m) << 13;
    }
  } else if (e == 31) {
    v = s | 0x7f800000u | (m != 0 ? 0x00400000u : 0u);
  } else {
    v = s | static_cast<uint32_t>(e + (127 - 15)) << 23 |
        static_cast<uint32_t>(m) << 13;
  }
  float f;
  std::memcpy(&f, &v, sizeof(f));
  return f;
}

static uint32_t float_bits_as_u32(float f) {
  uint32_t u;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

static float u32_as_float_bits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// float32 -> IEEE binary16 bits (same layout as Ort::Float16_t / ONNX FLOAT16).
// Non-intrinsic path from FP16 (Marat Dukhan, MIT license).
static uint16_t float_to_fp16_bits(float f) {
  const uint32_t kScaleToInfBits = 0x77800000u;  // 0x1.0p+112f
  const uint32_t kScaleToZeroBits = 0x08800000u; // 0x1.0p-110f
  const float scale_to_inf = u32_as_float_bits(kScaleToInfBits);
  const float scale_to_zero = u32_as_float_bits(kScaleToZeroBits);

  const float saturated_f = std::fabs(f) * scale_to_inf;
  float base = saturated_f * scale_to_zero;

  const uint32_t w = float_bits_as_u32(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & 0x80000000u;
  uint32_t bias = shl1_w & 0xFF000000u;
  if (bias < 0x71000000u)
    bias = 0x71000000u;

  base = u32_as_float_bits((bias >> 1) + 0x07800000u) + base;
  const uint32_t bits = float_bits_as_u32(base);
  const uint32_t exp_bits = (bits >> 13) & 0x00007C00u;
  const uint32_t mantissa_bits = bits & 0x00000FFFu;
  const uint32_t nonsign = exp_bits + mantissa_bits;

  if (shl1_w > 0xFF000000u)
    return static_cast<uint16_t>((sign >> 16) | 0x7E00u);
  return static_cast<uint16_t>((sign >> 16) | nonsign);
}

// BFloat16 -> float32 (simple: copy sign bit + upper 7 bits of
// exponent/mantissa to f32).
static float bf16_bits_to_float(uint16_t bf16) {
  uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

// float32 -> BFloat16 bits (round-to-nearest-even truncation).
static uint16_t float_to_bf16_bits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Round to nearest even: check if we need to round up
  uint32_t lsb = (bits >> 16) & 1;
  uint32_t rounding_bias = 0x7fff + lsb;
  bits += rounding_bias;
  return static_cast<uint16_t>(bits >> 16);
}

// Random fill matching element type (do not reinterpret float bits as
// int/fp16). Integers: uniform over the full representable range of each type.
// Float32/fp16: keep a modest interval (same order as the old float-only path)
// to avoid surprising huge magnitudes in models that expect bounded inputs.
// positive_only: if true, generate only positive values for float types (for
// Sqrt/Reciprocal testing)
static void fill_random_input_buffer(char *dst, size_t nbytes,
                                     ONNXTensorElementDataType et,
                                     std::mt19937 &rng,
                                     bool positive_only = false) {
  std::uniform_real_distribution<float> fdist(positive_only ? 0.1f : -256.0f,
                                              positive_only ? 256.0f : 255.0f);
  std::uniform_int_distribution<int> idist_i16(-32768, 32767);
  std::uniform_int_distribution<int> idist_i8(-128, 127);
  std::uniform_int_distribution<int> idist_u8(0, 255);
  std::uniform_int_distribution<int> idist_u16(0, 65535);
  std::uniform_int_distribution<uint32_t> idist_u32(0u, UINT32_MAX - 1000);
  std::uniform_int_distribution<uint64_t> idist_u64(0ull, ~0ull);

  switch (et) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
    auto *p = reinterpret_cast<float *>(dst);
    const size_t n = nbytes / sizeof(float);
    for (size_t i = 0; i < n; ++i)
      p[i] = fdist(rng);
    return;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
    auto *p = reinterpret_cast<uint16_t *>(dst);
    const size_t n = nbytes / sizeof(uint16_t);
    for (size_t i = 0; i < n; ++i)
      p[i] = float_to_fp16_bits(fdist(rng));
    return;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
    auto *p = reinterpret_cast<double *>(dst);
    const size_t n = nbytes / sizeof(double);
    for (size_t i = 0; i < n; ++i)
      p[i] = static_cast<double>(fdist(rng));
    return;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: {
    auto *p = reinterpret_cast<uint16_t *>(dst);
    const size_t n = nbytes / sizeof(uint16_t);
    for (size_t i = 0; i < n; ++i)
      p[i] = float_to_bf16_bits(fdist(rng));
    return;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
    auto *p = reinterpret_cast<int64_t *>(dst);
    const size_t n = nbytes / sizeof(int64_t);
    for (size_t i = 0; i < n; ++i) {
      const uint64_t u = idist_u64(rng);
      std::memcpy(&p[i], &u, sizeof(p[i]));
    }
    return;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
    auto *p = reinterpret_cast<int32_t *>(dst);
    const size_t n = nbytes / sizeof(int32_t);
    for (size_t i = 0; i < n; ++i) {
      const uint32_t u = idist_u32(rng);
      std::memcpy(&p[i], &u, sizeof(p[i]));
    }
    return;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
    auto *p = reinterpret_cast<int16_t *>(dst);
    const size_t n = nbytes / sizeof(int16_t);
    for (size_t i = 0; i < n; ++i)
      p[i] = static_cast<int16_t>(idist_i16(rng));
    return;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
    auto *p = reinterpret_cast<int8_t *>(dst);
    const size_t n = nbytes / sizeof(int8_t);
    for (size_t i = 0; i < n; ++i)
      p[i] = static_cast<int8_t>(idist_i8(rng));
    return;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
    auto *p = reinterpret_cast<uint8_t *>(dst);
    const size_t n = nbytes / sizeof(uint8_t);
    for (size_t i = 0; i < n; ++i)
      p[i] = static_cast<uint8_t>(idist_u8(rng));
    return;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
    auto *p = reinterpret_cast<uint16_t *>(dst);
    const size_t n = nbytes / sizeof(uint16_t);
    for (size_t i = 0; i < n; ++i)
      p[i] = static_cast<uint16_t>(idist_u16(rng));
    return;
  }
  default:
    std::cerr << "Unsupported element type for random fill: " << et << "\n";
    std::exit(1);
  }
}

static std::filesystem::path
tensor_dump_bin_path(const std::filesystem::path &dir, const char *io_prefix,
                     size_t index, const std::string &ort_name,
                     ONNXTensorElementDataType et) {
  const std::string safe = sanitize_filename_component(ort_name);
  return dir / (std::string(io_prefix) + "_" + std::to_string(index) + "_" +
                safe + "_" + onnx_elem_type_tag(et) + ".bin");
}

// Raw tensor bytes for an Ort::Value (must be a tensor).
static bool ort_tensor_raw_bytes(const Ort::Value &v, const void *&out_ptr,
                                 size_t &out_size) {
  if (!v.IsTensor())
    return false;
  auto info = v.GetTensorTypeAndShapeInfo();
  const ONNXTensorElementDataType et = info.GetElementType();
  const int64_t ec = info.GetElementCount();
  if (ec < 0)
    return false;
  out_size = static_cast<size_t>(ec) * element_byte_size(et);
  switch (et) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
    out_ptr = v.GetTensorData<float>();
    return true;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
    out_ptr = v.GetTensorData<Ort::Float16_t>();
    return true;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
    out_ptr = v.GetTensorData<double>();
    return true;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
    out_ptr = v.GetTensorData<Ort::BFloat16_t>();
    return true;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
    out_ptr = v.GetTensorData<int64_t>();
    return true;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
    out_ptr = v.GetTensorData<int32_t>();
    return true;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
    out_ptr = v.GetTensorData<int16_t>();
    return true;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
    out_ptr = v.GetTensorData<int8_t>();
    return true;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
    out_ptr = v.GetTensorData<uint8_t>();
    return true;
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
    out_ptr = v.GetTensorData<uint16_t>();
    return true;
  default:
    return false;
  }
}

static void dump_raw_file(const std::filesystem::path &path, const void *data,
                          size_t nbytes) {
  std::ofstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "Failed to open for write: " << path.string() << "\n";
    std::exit(1);
  }
  f.write(static_cast<const char *>(data),
          static_cast<std::streamsize>(nbytes));
  if (!f) {
    std::cerr << "Failed to write: " << path.string() << "\n";
    std::exit(1);
  }
}

static bool load_raw_file(const std::filesystem::path &path, void *dest,
                          size_t expected_nbytes) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "Failed to open for read: " << path.string() << "\n";
    return false;
  }
  f.seekg(0, std::ios::end);
  const auto end = f.tellg();
  if (end < 0) {
    std::cerr << "Failed to size: " << path.string() << "\n";
    return false;
  }
  const auto sz = static_cast<size_t>(end);
  f.seekg(0);
  if (sz != expected_nbytes) {
    std::cerr << "Size mismatch for " << path.string() << ": expected "
              << expected_nbytes << " bytes, file has " << sz << "\n";
    return false;
  }
  f.read(static_cast<char *>(dest),
         static_cast<std::streamsize>(expected_nbytes));
  if (!f) {
    std::cerr << "Failed to read: " << path.string() << "\n";
    return false;
  }
  return true;
}

static std::string trim_string(std::string s) {
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  return s;
}

// *.bin regular files in dir (for -L pairing by matching basenames).
static std::vector<std::string>
list_l2compare_filenames(const std::filesystem::path &dir) {
  std::vector<std::string> names;
  std::error_code ec;
  std::filesystem::directory_iterator it(dir, ec);
  if (ec) {
    std::cerr << "Cannot list directory: " << dir.string() << " ("
              << ec.message() << ")\n";
    return names;
  }
  const std::filesystem::directory_iterator end;
  for (; it != end; ++it) {
    if (!it->is_regular_file())
      continue;
    const std::string fn = it->path().filename().string();
    if (fn.size() < 4 || fn.compare(fn.size() - 4, 4, ".bin") != 0)
      continue;
    names.push_back(fn);
  }
  std::sort(names.begin(), names.end());
  return names;
}

static bool read_entire_file(const std::filesystem::path &path,
                             std::vector<char> &out) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    std::cerr << "Failed to open: " << path.string() << "\n";
    return false;
  }
  f.seekg(0, std::ios::end);
  const auto end = f.tellg();
  if (end < 0) {
    std::cerr << "Failed to size: " << path.string() << "\n";
    return false;
  }
  out.resize(static_cast<size_t>(end));
  f.seekg(0);
  f.read(out.data(), static_cast<std::streamsize>(out.size()));
  if (!f) {
    std::cerr << "Failed to read: " << path.string() << "\n";
    return false;
  }
  return true;
}

// Parse *.bin basename ..._<tag>.bin into element type; *typed_out false if tag
// is missing or unknown.
static bool parse_dump_filename_elem_type(const std::string &filename,
                                          ONNXTensorElementDataType &et,
                                          bool *typed_out) {
  if (filename.size() < 4 ||
      filename.compare(filename.size() - 4, 4, ".bin") != 0) {
    *typed_out = false;
    return true;
  }
  const std::string base = filename.substr(0, filename.size() - 4);
  const size_t pos = base.rfind('_');
  if (pos == std::string::npos || pos + 1 >= base.size()) {
    *typed_out = false;
    return true;
  }
  const std::string tag = base.substr(pos + 1);
  if (type_tag_to_onnx(tag, et)) {
    *typed_out = true;
    return true;
  }
  *typed_out = false;
  return true;
}

static bool squared_l2_diff_elementwise(
    const std::vector<char> &a, const std::vector<char> &b,
    ONNXTensorElementDataType et, double *out_sq,
    size_t *out_used_elems = nullptr, size_t *out_skipped_nonfinite = nullptr) {
  const size_t es = element_byte_size(et);
  if (a.size() != b.size() || a.size() % es != 0) {
    std::cerr << "Size not aligned to element type (" << es << " bytes/elem)\n";
    return false;
  }
  const size_t n = a.size() / es;
  long double s = 0;
  size_t used_elems = 0;
  // Skips NaN/Inf on either side so (Inf-Inf) etc. never poisons the sum.
  size_t skipped_nonfinite = 0;
  switch (et) {
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
    const auto *pa = reinterpret_cast<const float *>(a.data());
    const auto *pb = reinterpret_cast<const float *>(b.data());
    for (size_t i = 0; i < n; ++i) {
      const float fa = pa[i], fb = pb[i];
      if (!std::isfinite(static_cast<double>(fa)) ||
          !std::isfinite(static_cast<double>(fb))) {
        skipped_nonfinite++;
        continue;
      }
      const double d = static_cast<double>(fa) - static_cast<double>(fb);
      s += d * d;
      used_elems++;
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
    const auto *pa = reinterpret_cast<const uint16_t *>(a.data());
    const auto *pb = reinterpret_cast<const uint16_t *>(b.data());
    for (size_t i = 0; i < n; ++i) {
      const float fa = fp16_bits_to_float(pa[i]);
      const float fb = fp16_bits_to_float(pb[i]);
      if (!std::isfinite(static_cast<double>(fa)) ||
          !std::isfinite(static_cast<double>(fb))) {
        skipped_nonfinite++;
        continue;
      }
      const double d = static_cast<double>(fa) - static_cast<double>(fb);
      s += d * d;
      used_elems++;
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
    const auto *pa = reinterpret_cast<const double *>(a.data());
    const auto *pb = reinterpret_cast<const double *>(b.data());
    for (size_t i = 0; i < n; ++i) {
      if (!std::isfinite(pa[i]) || !std::isfinite(pb[i])) {
        skipped_nonfinite++;
        continue;
      }
      const double d = pa[i] - pb[i];
      s += d * d;
      used_elems++;
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16: {
    const auto *pa = reinterpret_cast<const uint16_t *>(a.data());
    const auto *pb = reinterpret_cast<const uint16_t *>(b.data());
    for (size_t i = 0; i < n; ++i) {
      const float fa = bf16_bits_to_float(pa[i]);
      const float fb = bf16_bits_to_float(pb[i]);
      if (!std::isfinite(static_cast<double>(fa)) ||
          !std::isfinite(static_cast<double>(fb))) {
        skipped_nonfinite++;
        continue;
      }
      const double d = static_cast<double>(fa) - static_cast<double>(fb);
      s += d * d;
      used_elems++;
    }
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
    const auto *pa = reinterpret_cast<const int64_t *>(a.data());
    const auto *pb = reinterpret_cast<const int64_t *>(b.data());
    for (size_t i = 0; i < n; ++i) {
      const double d = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
      s += d * d;
    }
    used_elems = n;
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
    const auto *pa = reinterpret_cast<const int32_t *>(a.data());
    const auto *pb = reinterpret_cast<const int32_t *>(b.data());
    for (size_t i = 0; i < n; ++i) {
      const double d = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
      s += d * d;
    }
    used_elems = n;
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
    const auto *pa = reinterpret_cast<const int16_t *>(a.data());
    const auto *pb = reinterpret_cast<const int16_t *>(b.data());
    for (size_t i = 0; i < n; ++i) {
      const double d = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
      s += d * d;
    }
    used_elems = n;
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
    const auto *pa = reinterpret_cast<const int8_t *>(a.data());
    const auto *pb = reinterpret_cast<const int8_t *>(b.data());
    for (size_t i = 0; i < n; ++i) {
      const double d = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
      s += d * d;
    }
    used_elems = n;
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
    const auto *pa = reinterpret_cast<const uint8_t *>(a.data());
    const auto *pb = reinterpret_cast<const uint8_t *>(b.data());
    for (size_t i = 0; i < n; ++i) {
      const double d = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
      s += d * d;
    }
    used_elems = n;
    break;
  }
  case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
    const auto *pa = reinterpret_cast<const uint16_t *>(a.data());
    const auto *pb = reinterpret_cast<const uint16_t *>(b.data());
    for (size_t i = 0; i < n; ++i) {
      const double d = static_cast<double>(pa[i]) - static_cast<double>(pb[i]);
      s += d * d;
    }
    used_elems = n;
    break;
  }
  default:
    std::cerr << "Unsupported element type for L2\n";
    return false;
  }
  if (out_used_elems)
    *out_used_elems = used_elems;
  if (out_skipped_nonfinite)
    *out_skipped_nonfinite = skipped_nonfinite;
  *out_sq = static_cast<double>(s);
  return true;
}

// Compare two directories: same set of *.bin basenames, pairwise L2. Returns
// exit code 0 on success.
static int run_l2norm_output_dumps(const std::string &dir1_str,
                                   const std::string &dir2_str) {
  const std::filesystem::path d1(dir1_str);
  const std::filesystem::path d2(dir2_str);
  std::error_code ec;
  if (!std::filesystem::is_directory(d1, ec)) {
    std::cerr << "Not a directory: " << dir1_str << "\n";
    return 1;
  }
  if (!std::filesystem::is_directory(d2, ec)) {
    std::cerr << "Not a directory: " << dir2_str << "\n";
    return 1;
  }

  std::vector<std::string> names1 = list_l2compare_filenames(d1);
  std::vector<std::string> names2 = list_l2compare_filenames(d2);
  if (names1.empty()) {
    std::cerr << "No *.bin files in: " << dir1_str << "\n";
    return 1;
  }
  if (names1 != names2) {
    std::cerr << "Filename sets differ between directories.\n";
    std::set<std::string> s1(names1.begin(), names1.end());
    std::set<std::string> s2(names2.begin(), names2.end());
    std::cerr << "Only in first dir:\n";
    for (const auto &n : s1)
      if (!s2.count(n))
        std::cerr << "  " << n << "\n";
    std::cerr << "Only in second dir:\n";
    for (const auto &n : s2)
      if (!s1.count(n))
        std::cerr << "  " << n << "\n";
    return 1;
  }

  long double combined_sq = 0;
  size_t total_used_elems = 0;
  size_t total_skipped_nonfinite = 0;
  for (const std::string &fn : names1) {
    std::vector<char> a, b;
    if (!read_entire_file(d1 / fn, a) || !read_entire_file(d2 / fn, b))
      return 1;
    if (a.size() != b.size()) {
      std::cerr << "Size mismatch for " << fn << ": " << a.size() << " vs "
                << b.size() << "\n";
      return 1;
    }
    bool typed = false;
    ONNXTensorElementDataType et = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    if (!parse_dump_filename_elem_type(fn, et, &typed)) {
      std::cerr << "Bad dump filename: " << fn << "\n";
      return 1;
    }
    if (!typed) {
      std::cerr
          << fn
          << ": missing or unknown type tag; expected name ending with _fp32, "
             "_fp16, _fp64, _bf16, _i64, _i32, _i16, _i8, _u8, or _u16 before "
             ".bin\n";
      return 1;
    }
    // Bitwise-equal buffers => L2 is 0 (fast path).
    double sq = 0;
    size_t used_elems = 0;
    size_t skipped_nonfinite = 0;
    if (a != b) {
      if (!squared_l2_diff_elementwise(a, b, et, &sq, &used_elems,
                                       &skipped_nonfinite))
        return 1;
    } else {
      used_elems = a.size() / element_byte_size(et);
    }
    if ((et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
         et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) &&
        used_elems == 0 && a.size() / element_byte_size(et) > 0) {
      const size_t n_skip = a.size() / element_byte_size(et);
      std::cerr << "Warning: " << fn
                << " - no finite elements to compare; L2(diff)=0; " << n_skip
                << " element(s) skipped (non-finite)\n";
    }
    std::cout << fn << ": L2(diff) = " << std::sqrt(sq) << " (" << a.size()
              << " bytes, " << onnx_elem_type_tag(et) << " element-wise";
    if (et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        et == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
      std::cout << "; " << used_elems << " elems compared";
      if (skipped_nonfinite > 0)
        std::cout << ", " << skipped_nonfinite
                  << " element(s) skipped (non-finite)";
    } else if (used_elems > 0) {
      std::cout << "; " << used_elems << " elems";
    }
    std::cout << ")\n";
    combined_sq += static_cast<long double>(sq);
    total_used_elems += used_elems;
    total_skipped_nonfinite += skipped_nonfinite;
  }
  std::cout << "Combined L2 (stacked diffs): "
            << std::sqrt(static_cast<double>(combined_sq))
            << " (total_elems: " << total_used_elems + total_skipped_nonfinite
            << ", total_used_elems: " << total_used_elems
            << ", skipped_nonfinite: " << total_skipped_nonfinite << ")\n";
  return 0;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char *argv[]) {
  // Disabled: cpptrace terminate handler turns uncaught ORT/MorphiZen throws
  // into a stack trace without reaching the catch blocks below.
  // hip::install_crash_handlers("hip-onnx-runner");
  MiniOptions mo;
  mo.add_option("m", "model", "Path to .onnx model", "");
  mo.add_option("L", "l2norm",
                "Compare two dirs: dir1,dir2 (same *.bin set); each file must "
                "be ..._<type>.bin (fp32,fp16,i64,...); element-wise L2; no -m",
                "");
  mo.add_option("n", "no-ep", "CPU only; skip EP registration", "false", true);
  mo.add_option(
      "d", "dump-level",
      "0=off, 1=dump inputs to <stem>_i_dump/, 2=outputs to <stem>_o_dump/, "
      "3=both",
      "0");
  mo.add_option("s", "seed", "RNG seed for random inputs", "42");
  mo.add_option("i", "input-dir",
                "Directory with input_<idx>_<name>_<type>.bin only; empty = "
                "random inputs",
                "");
  mo.add_option("o", "graph-opt-level",
                "session_options.SetGraphOptimizationLevel(level), "
                "  0 = ORT_DISABLE_ALL,  "
                "  1 = ORT_ENABLE_BASIC,  "
                "  2 = ORT_ENABLE_EXTENDED,  "
                " 99 = ORT_ENABLE_ALL,  "
                " -1 = default, not call this function ",
                "-1");
  mo.add_option(
      "p", "positive-only",
      "Generate positive-only random inputs (for Sqrt/Reciprocal testing)",
      "false", true);
  mo.add_option(
      "E", "strict-ep",
      "Set session.disable_cpu_ep_fallback=1 (fail if any node is not on "
      "MorphiZen EP)",
      "false", true);

  try {
    mo.parse(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << e.what() << "\n\n";
    mo.print_help(argv[0]);
    return 1;
  }

  if (argc == 1) {
    mo.print_help(argv[0]);
    return 1;
  }

  std::vector<std::string> l2norm_arg = mo.get_vector<std::string>("l2norm");
  std::string model_path_str = trim_string(mo.get<std::string>("model"));
  const bool no_ep = mo.get<bool>("no-ep");
  const int dump_level = mo.get<int>("dump-level");
  std::mt19937 rng(mo.get<unsigned int>("seed"));
  std::string input_dir_str = trim_string(mo.get<std::string>("input-dir"));
  const bool use_input_files = !input_dir_str.empty();
  const int graph_optimization_level = mo.get<int>("graph-opt-level");
  const bool positive_only = mo.get<bool>("positive-only");
  const bool strict_ep = mo.get<bool>("strict-ep");

  if ((l2norm_arg.size() == 2) && !l2norm_arg[0].empty() &&
      !l2norm_arg[1].empty()) {
    return run_l2norm_output_dumps(l2norm_arg[0], l2norm_arg[1]);
  }

  if (model_path_str.empty()) {
    std::cerr << "Error: --model is required for inference.\n\n";
    mo.print_help(argv[0]);
    return 1;
  }
  if (dump_level < 0 || dump_level > 3) {
    std::cerr << "Error: --dump-level must be 0, 1, 2, or 3.\n\n";
    mo.print_help(argv[0]);
    return 1;
  }

  // ORT environment
  Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "hip-onnx-runner");

  const std::string kEpName = "MorphiZenExecutionProvider";
#ifdef _WIN32
  const std::string ep_lib_name = "onnxruntime_morphizen_ep.dll";
#else
  const std::string ep_lib_name = "libonnxruntime_morphizen_ep.so";
#endif

  if (!no_ep) {
    // Search order: $MORPHIZEN_EP_LIB (full path), then cwd, then sibling
    // dirs of the runner exe (../lib for the install layout, same dir for a
    // single-folder drop). First hit wins.
    std::filesystem::path lib_path;
    if (const char *env_lib = std::getenv("MORPHIZEN_EP_LIB");
        env_lib && *env_lib) {
      lib_path = std::filesystem::u8path(env_lib);
    } else {
      std::vector<std::filesystem::path> candidates{
          std::filesystem::u8path(ep_lib_name),
      };
      std::error_code ec;
      auto exe_dir = std::filesystem::weakly_canonical(
                         std::filesystem::u8path(argv[0]), ec)
                         .parent_path();
      if (!ec && !exe_dir.empty()) {
        candidates.push_back(exe_dir / ep_lib_name);
        candidates.push_back(exe_dir.parent_path() / "lib" / ep_lib_name);
      }
      for (const auto &p : candidates) {
        if (std::filesystem::exists(p)) {
          lib_path = p;
          break;
        }
      }
    }
    if (lib_path.empty() || !std::filesystem::exists(lib_path)) {
      std::cerr << "EP library not found: " << ep_lib_name << "\n"
                << "Set MORPHIZEN_EP_LIB to its full path, or use -n.\n";
      return 1;
    }
    std::cout << "Registering EP: " << lib_path.string() << "\n";
    auto *status = Ort::GetApi().RegisterExecutionProviderLibrary(
        env, kEpName.c_str(), lib_path.c_str());
    if (status) {
      std::cerr << "RegisterExecutionProviderLibrary failed: "
                << Ort::GetApi().GetErrorMessage(status) << "\n";
      return 1;
    }
  }

  // Session options
  Ort::SessionOptions session_opts;
  session_opts.SetLogSeverityLevel(ORT_LOGGING_LEVEL_ERROR);
  if (graph_optimization_level != -1) {
    std::cout << "Setting graph_optimization_level to "
              << graph_optimization_level << "\n";
    session_opts.SetGraphOptimizationLevel(
        static_cast<GraphOptimizationLevel>(graph_optimization_level));
  }
  if (!no_ep) {
    // Collect devices for this EP
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto &dev : env.GetEpDevices()) {
      if (dev.EpName() == kEpName)
        devices.emplace_back(dev);
    }
    if (devices.empty()) {
      std::cerr << "No devices found for EP: " << kEpName << "\n";
      return 1;
    }
    std::cout << "Found " << devices.size() << " EP device(s)\n";

    // ORT >=1.24 added an AppendExecutionProvider_V2 overload that accepts
    // Ort::KeyValuePairs alongside the legacy std::unordered_map<string,string>
    // version. Passing a brace-init `{}` is ambiguous against both candidates
    // on the prebuilt Linux ORT package we ship here. Spell out the map type
    // so we always bind to the original overload and stay compatible with the
    // older ORT version we still link against on Windows.
    session_opts.AppendExecutionProvider_V2(
        env, devices, std::unordered_map<std::string, std::string>{});
    if (strict_ep) {
      session_opts.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    }
  }

  // Create session
  auto model_path = std::filesystem::path(model_path_str);
  std::cout << "Loading model: " << model_path.string() << "\n";

  std::unique_ptr<Ort::Session> session;
  {
    auto t0 = std::chrono::steady_clock::now();
    try {
#if _WIN32
      session = std::make_unique<Ort::Session>(
          env, model_path.wstring().c_str(), session_opts);
#else
      session = std::make_unique<Ort::Session>(
          env, model_path.u8string().c_str(), session_opts);
#endif
    } catch (const Ort::Exception &e) {
      std::cerr << "Session creation failed (Ort::Exception): " << e.what()
                << "\n";
      return 1;
    } catch (const std::exception &e) {
      std::cerr << "Session creation failed (std::exception): " << e.what()
                << "\n";
      return 1;
    } catch (...) {
      std::cerr << "Session creation failed: unknown exception\n";
      return 1;
    }
    std::cout << "Session created in "
              << static_cast<int>(elapsed_since(t0) * 1000) << " ms\n";
    std::cout.flush();
  }

  std::cerr << "[STEP] AllocatorWithDefaultOptions" << std::endl;
  Ort::AllocatorWithDefaultOptions allocator;

  size_t input_count = 0;
  size_t output_count = 0;
  std::vector<std::string> input_names_str;
  std::vector<const char *> input_names;
  std::vector<std::vector<int64_t>> input_shapes;
  std::vector<ONNXTensorElementDataType> input_types;
  std::vector<std::string> output_names_str;
  std::vector<const char *> output_names;

  try {
    std::cerr << "[STEP] GetInputCount" << std::endl;
    input_count = session->GetInputCount();
    std::cerr << "[STEP] input_count=" << input_count << std::endl;

    for (size_t i = 0; i < input_count; ++i) {
      std::cerr << "[STEP] GetInputNameAllocated[" << i << "]" << std::endl;
      auto name_ptr = session->GetInputNameAllocated(i, allocator);
      input_names_str.push_back(name_ptr.get());
      std::cerr << "[STEP]   name=" << input_names_str.back() << std::endl;
      std::cerr << "[STEP] GetInputTypeInfo[" << i << "]" << std::endl;
      auto type_info = session->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      input_shapes.push_back(tensor_info.GetShape());
      input_types.push_back(tensor_info.GetElementType());
    }
    for (auto &s : input_names_str)
      input_names.push_back(s.c_str());

    std::cerr << "[STEP] GetOutputCount" << std::endl;
    output_count = session->GetOutputCount();
    std::cerr << "[STEP] output_count=" << output_count << std::endl;

    for (size_t i = 0; i < output_count; ++i) {
      std::cerr << "[STEP] GetOutputNameAllocated[" << i << "]" << std::endl;
      auto name_ptr = session->GetOutputNameAllocated(i, allocator);
      output_names_str.push_back(name_ptr.get());
      std::cerr << "[STEP]   name=" << output_names_str.back() << std::endl;
    }
    for (auto &s : output_names_str)
      output_names.push_back(s.c_str());
  } catch (const Ort::Exception &e) {
    std::cerr << "I/O metadata collection failed (Ort::Exception): " << e.what()
              << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "I/O metadata collection failed (std::exception): " << e.what()
              << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "I/O metadata collection failed: unknown exception"
              << std::endl;
    return 1;
  }

  std::cout << "Inputs: " << input_count << "  Outputs: " << output_count
            << std::endl;

  std::filesystem::path input_dir_path;
  if (use_input_files) {
    input_dir_path = std::filesystem::path(input_dir_str);
    std::error_code ec;
    if (!std::filesystem::is_directory(input_dir_path, ec)) {
      std::cerr << "Error: -i is not a directory: " << input_dir_str << "\n";
      return 1;
    }
  }

  std::cerr << "[STEP] build input tensors" << std::endl;
  std::vector<std::vector<char>> input_buffers(input_count);
  std::vector<Ort::Value> input_tensors;

  for (size_t i = 0; i < input_count; ++i) {
    auto shape = input_shapes[i];
    if (!shape.empty() && shape[0] == -1)
      shape[0] = 1;
    // Any remaining dynamic dim becomes 1 so we have a concrete shape.
    for (auto &d : shape) {
      if (d < 0)
        d = 1;
    }

    size_t elem_size = element_byte_size(input_types[i]);
    int64_t n_elems = calculate_product(shape);
    std::cerr << "[STEP]   input[" << i << "] elem_size=" << elem_size
              << " n_elems=" << n_elems << " shape=[";
    for (auto d : shape)
      std::cerr << d << ",";
    std::cerr << "]" << std::endl;

    if (use_input_files) {
      const auto bin_path = tensor_dump_bin_path(
          input_dir_path, "input", i, input_names_str[i], input_types[i]);
      std::cerr << "[STEP]   bin_path=" << bin_path.string() << std::endl;
      std::error_code ec;
      if (!std::filesystem::exists(bin_path, ec)) {
        std::cerr << "Missing input file: " << bin_path.string() << std::endl;
        return 1;
      }
      // Sidecar `.shape` file: comma-separated concrete dims. Required when
      // the ONNX has dynamic dims AND the dim split isn't a 1-D vector --
      // file size alone cannot tell `[1,8]` from `[8,1]` for input_ids.
      const auto shape_path = bin_path.string() + ".shape";
      if (std::filesystem::exists(shape_path, ec)) {
        std::ifstream sf(shape_path);
        std::string line;
        std::getline(sf, line);
        std::vector<int64_t> from_shape_file;
        size_t pos = 0;
        while (pos < line.size()) {
          size_t comma = line.find(',', pos);
          std::string tok = line.substr(pos, comma - pos);
          if (!tok.empty()) {
            try {
              from_shape_file.push_back(std::stoll(tok));
            } catch (...) {
              std::cerr << "Bad value in " << shape_path << ": '" << tok << "'"
                        << std::endl;
              return 1;
            }
          }
          if (comma == std::string::npos)
            break;
          pos = comma + 1;
        }
        if (from_shape_file.size() != shape.size()) {
          std::cerr << "Rank mismatch in " << shape_path << ": got "
                    << from_shape_file.size() << " dims, ONNX rank "
                    << shape.size() << std::endl;
          return 1;
        }
        shape = from_shape_file;
        std::cerr << "[STEP]   shape from sidecar: [";
        for (auto d : shape)
          std::cerr << d << ",";
        std::cerr << "]" << std::endl;
      } else {
        // Fallback: replace any remaining -1 dim by file-size split on first
        // dynamic dim found.  Only safe when there's exactly one dyn dim.
        int dyn_idx = -1;
        int dyn_count = 0;
        for (size_t d = 0; d < shape.size(); ++d) {
          if (shape[d] < 0) {
            dyn_idx = static_cast<int>(d);
            ++dyn_count;
          }
        }
        if (dyn_count > 1) {
          std::cerr << "Input '" << input_names_str[i] << "' has " << dyn_count
                    << " dynamic dims; create " << shape_path
                    << " with concrete shape (comma-separated)." << std::endl;
          return 1;
        }
        if (dyn_count == 1) {
          std::ifstream f(bin_path, std::ios::binary | std::ios::ate);
          const std::streamsize file_sz = f.tellg();
          f.close();
          int64_t inner = elem_size;
          for (size_t d = 0; d < shape.size(); ++d) {
            if (static_cast<int>(d) != dyn_idx)
              inner *= shape[d];
          }
          if (inner > 0 && file_sz > 0 && (file_sz % inner) == 0) {
            shape[dyn_idx] = file_sz / inner;
            std::cerr
                << "[STEP]   shape inferred from file (single dyn dim): [";
            for (auto d : shape)
              std::cerr << d << ",";
            std::cerr << "]" << std::endl;
          }
        }
      }

      n_elems = calculate_product(shape);
      const size_t expected_bytes = static_cast<size_t>(n_elems) * elem_size;
      input_buffers[i].resize(expected_bytes);
      if (!load_raw_file(bin_path, input_buffers[i].data(),
                         input_buffers[i].size())) {
        std::cerr << "[STEP]   load_raw_file FAILED" << std::endl;
        return 1;
      }
      std::cout << "Loaded input " << i << " from " << bin_path.string() << " ("
                << expected_bytes << " bytes)" << std::endl;
    } else {
      input_buffers[i].resize(n_elems * elem_size);
      fill_random_input_buffer(input_buffers[i].data(), input_buffers[i].size(),
                               input_types[i], rng, positive_only);
    }

    std::cerr << "[STEP]   CreateTensor[" << i << "]" << std::endl;
    Ort::MemoryInfo mem =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensors.push_back(Ort::Value::CreateTensor(
        mem, input_buffers[i].data(), input_buffers[i].size(), shape.data(),
        shape.size(), input_types[i]));
    std::cerr << "[STEP]   CreateTensor[" << i << "] OK" << std::endl;
  }

  const std::string dump_stem = model_path.stem().string();
  const std::string dump_tag = no_ep ? "_cpu" : "";
  const std::filesystem::path dump_dir_inputs =
      std::filesystem::path(".") / (dump_stem + dump_tag + "_i_dump");
  const std::filesystem::path dump_dir_outputs =
      std::filesystem::path(".") / (dump_stem + dump_tag + "_o_dump");
  if (dump_level == 1 || dump_level == 3) {
    std::filesystem::create_directories(dump_dir_inputs);
    for (size_t i = 0; i < input_count; ++i) {
      const auto path = tensor_dump_bin_path(
          dump_dir_inputs, "input", i, input_names_str[i], input_types[i]);
      dump_raw_file(path, input_buffers[i].data(), input_buffers[i].size());
      std::cout << "Dumped input tensor to " << path.string() << "\n";
    }
  }

  std::cerr << "[STEP] Run" << std::endl;
  std::cout << "Running inference..." << std::endl;
  std::vector<Ort::Value> outputs;
  {
    auto t0 = std::chrono::steady_clock::now();
    try {
      outputs = session->Run(Ort::RunOptions{}, input_names.data(),
                             input_tensors.data(), input_count,
                             output_names.data(), output_count);
      std::cout << "Inference: "
                << static_cast<int64_t>(elapsed_since(t0) * 1e6) << " us\n";
      std::cout << "OK - " << outputs.size() << " output tensor(s)\n";
    } catch (const Ort::Exception &e) {
      std::cerr << "Inference failed (Ort::Exception): " << e.what()
                << std::endl;
      return 1;
    } catch (const std::exception &e) {
      std::cerr << "Inference failed (std::exception): " << e.what()
                << std::endl;
      return 1;
    }
  }

  if (dump_level == 2 || dump_level == 3) {
    std::filesystem::create_directories(dump_dir_outputs);
    for (size_t i = 0; i < outputs.size(); ++i) {
      const void *ptr = nullptr;
      size_t nbytes = 0;
      if (!ort_tensor_raw_bytes(outputs[i], ptr, nbytes)) {
        std::cerr << "Cannot dump output " << i
                  << " (unsupported or not a "
                     "tensor)\n";
        return 1;
      }
      auto out_info = outputs[i].GetTensorTypeAndShapeInfo();
      const ONNXTensorElementDataType oet = out_info.GetElementType();
      const auto path = tensor_dump_bin_path(dump_dir_outputs, "output", i,
                                             output_names_str[i], oet);
      dump_raw_file(path, ptr, nbytes);
      std::cout << "Dumped output tensor to " << path.string() << "\n";
    }
  }

  // Release session and all Ort::Values before unloading the EP DLL. Calling
  // UnregisterExecutionProviderLibrary while the session still uses the EP
  // causes use-after-free / AV (e.g. 0xC0000005) during later teardown.
  outputs.clear();
  input_tensors.clear();
  session.reset();

  if (!no_ep) {
    Ort::GetApi().UnregisterExecutionProviderLibrary(env, kEpName.c_str());
  }

  return 0;
}
