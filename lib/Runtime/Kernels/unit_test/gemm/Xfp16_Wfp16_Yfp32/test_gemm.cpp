/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for the CK GEMM entry points with fp16 operands and
 * fp32 output (hip_ck_gemm_run, hip_ref_gemm_run).
 */

#include "hip_custom_kernels.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#ifndef HIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS
#ifdef HIPDNN_LUT_LINKED_EXTERNALLY
#include "lut_fb_path.h"
static const unsigned char kLutBlob0[] = {
#embed HIPDNN_LUT_FB
};
extern "C" const unsigned char* const kCkGemmLutBlobs[1] = {kLutBlob0};
extern "C" const size_t kCkGemmLutBlobSizes[1] = {sizeof(kLutBlob0)};
extern "C" const size_t kCkGemmLutBlobCount = 1;
#else
extern "C" int hip_ck_gemm_lut_candidates(int64_t, int64_t, int64_t, int64_t,
                                          int, int, int, int, int*, int) {
  return 0;
}
#endif
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <set>
#include <string>
#include <thread>
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

// ---- Leaf configuration ----

using AbT = uint16_t;
using DT = float;
constexpr int kAbDtype = HIP_DTYPE_FLOAT16;
constexpr int kDDtype = HIP_DTYPE_FLOAT32;
constexpr bool kAlphaApplies = true;

struct Combo {
  const char* name;
  int trans_a;
  bool bias;
};
constexpr Combo kCombos[] = {
    {"TN", 1, false},
};
constexpr float kAlphas[] = {1.0f, 0.375f};

// Small enough to cross every layout and alignment at every tier, large enough
// to span several tiles of every instance.
struct Dims {
  int64_t m, n, k;
};
constexpr Dims kDims[] = {
    {200, 136, 168}, {196, 136, 168}, {198, 136, 168}, {199, 136, 168},
    {200, 135, 168}, {200, 136, 164}, {200, 136, 167}, {199, 135, 167},
};

enum class Layout { kSingle, kBatched, kPadded, kBroadcastB };
constexpr Layout kLayouts[] = {Layout::kSingle, Layout::kBatched,
                               Layout::kPadded, Layout::kBroadcastB};
constexpr int64_t kCategoricalBatch = 3;

// Column-major: a score GEMM is m = keys, n = query rows, k = head dim, batch =
// heads; a linear layer is m = out features, n = tokens, k = in features.
struct TypicalShape {
  int64_t m, n, k, batch;
  bool score;
  int min_coverage;
};
constexpr TypicalShape kTypical[] = {
    {1280, 100, 1280, 1, false, 1},  {448, 448, 64, 20, true, 1},
    {64, 448, 448, 20, false, 2},    {5120, 100, 1280, 1, false, 2},
    {1280, 1500, 5120, 1, false, 3}, {1500, 1, 64, 20, true, 3},
    {777, 1, 128, 32, true, 3},      {128, 1, 777, 32, false, 3},
};

constexpr int kTimedIters = 10;
constexpr int64_t kGuard = 64;

// ---- Element helpers ----

float half_to_float(uint16_t bits) {
  __half h;
  std::memcpy(&h, &bits, sizeof(bits));
  return __half2float(h);
}

uint16_t float_to_half_bits(float value) {
  const __half half = __float2half(value);
  uint16_t bits = 0;
  std::memcpy(&bits, &half, sizeof(bits));
  return bits;
}

template <typename T> double to_double(T v);
template <> double to_double<uint16_t>(uint16_t v) { return half_to_float(v); }
template <> double to_double<float>(float v) { return v; }

template <typename T> T from_float(float v);
template <> uint16_t from_float<uint16_t>(float v) {
  return float_to_half_bits(v);
}
template <> float from_float<float>(float v) { return v; }

// All-ones bytes are a NaN in fp16 and fp32 alike, so hipMemset(0xff) fills a
// buffer with it and a skipped element fails the comparison.
template <typename T> T fill_value() {
  T v;
  std::memset(&v, 0xff, sizeof(v));
  return v;
}

template <typename T> bool is_fill(T v) {
  const T f = fill_value<T>();
  return std::memcmp(&v, &f, sizeof(T)) == 0;
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
  float uniform() {
    return static_cast<float>(static_cast<double>(next() >> 40) /
                                  static_cast<double>(1ull << 24) * 2.0 -
                              1.0);
  }

private:
  uint64_t state_;
};

// ---- Problem description (hip_ck_gemm_run's column-major convention) ----

struct Problem {
  int64_t m, n, k, batch;
  int trans_a;
  bool bias;
  float alpha;
  int64_t lda, ldb, ldd;
  int64_t stride_a, stride_b, stride_d;
  int64_t a_rows() const { return trans_a ? k : m; }
  int64_t a_cols() const { return trans_a ? m : k; }
  int64_t a_elems() const { return (batch - 1) * stride_a + lda * a_cols(); }
  int64_t b_elems() const { return (batch - 1) * stride_b + ldb * n; }
  int64_t d_elems() const { return (batch - 1) * stride_d + ldd * n; }
};

// The WMMA instances' fp32 accumulation error grows linearly with k (relL2
// about 3e-9 * k measured), where FMA accumulation grows with sqrt(k).
double tolerance(const Problem& p) { return 1.0e-6 + 1.0e-8 * p.k; }

int64_t round_up(int64_t v, int64_t to) { return (v + to - 1) / to * to; }

// Padded leading dimensions stay multiples of 8, the alignment the WMMA
// instances require of a non-packed ld.
Problem make_problem(int64_t m, int64_t n, int64_t k, int64_t batch,
                     int trans_a, bool bias, float alpha, Layout layout) {
  Problem p{m, n, k, batch, trans_a, bias, alpha, trans_a ? k : m,
            k, m, 0, 0,     0};
  int64_t gap = 0;
  if (layout == Layout::kPadded) {
    p.lda = round_up(p.lda, 8) + 8;
    p.ldb = round_up(p.ldb, 8) + 8;
    p.ldd = round_up(p.ldd, 8) + 8;
    gap = 64;
  }
  p.stride_a = p.lda * p.a_cols() + gap;
  p.stride_b = layout == Layout::kBroadcastB ? 0 : p.ldb * n + gap;
  p.stride_d = p.ldd * n + gap;
  return p;
}

Problem packed(const Problem& p) {
  const Layout layout =
      p.stride_b == 0 ? Layout::kBroadcastB : Layout::kBatched;
  return make_problem(p.m, p.n, p.k, p.batch, p.trans_a, p.bias, p.alpha,
                      layout);
}

std::string format_alpha(float alpha) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%g", alpha);
  return buffer;
}

std::string problem_shape(const char* combo, const Problem& p, Layout layout) {
  std::string s = std::string(combo) + "_M" + std::to_string(p.m) + "_N" +
                  std::to_string(p.n) + "_K" + std::to_string(p.k) + "_B" +
                  std::to_string(p.batch);
  if (layout == Layout::kPadded) {
    s += "_pad";
  } else if (layout == Layout::kBroadcastB) {
    s += "_bcastB";
  }
  if (p.alpha != 1.0f) {
    s += "_a" + format_alpha(p.alpha);
  }
  return s;
}

// Values inside the matrices are random; leading-dimension and batch gaps keep
// the NaN fill, so a kernel that reads them corrupts its output.
template <typename T>
std::vector<T> make_operand(int64_t rows, int64_t cols, int64_t ld,
                            int64_t stride, int64_t batch, Rng& rng) {
  const int64_t copies = stride == 0 ? 1 : batch;
  std::vector<T> v(static_cast<size_t>((copies - 1) * stride + ld * cols),
                   fill_value<T>());
  for (int64_t b = 0; b < copies; ++b) {
    for (int64_t c = 0; c < cols; ++c) {
      for (int64_t r = 0; r < rows; ++r) {
        v[b * stride + c * ld + r] = from_float<T>(rng.uniform());
      }
    }
  }
  return v;
}

template <typename T>
std::vector<T> repack(const std::vector<T>& src, int64_t rows, int64_t cols,
                      int64_t ld, int64_t stride, int64_t batch, int64_t dst_ld,
                      int64_t dst_stride) {
  const int64_t copies = stride == 0 ? 1 : batch;
  std::vector<T> dst(
      static_cast<size_t>((copies - 1) * dst_stride + dst_ld * cols),
      fill_value<T>());
  for (int64_t b = 0; b < copies; ++b) {
    for (int64_t c = 0; c < cols; ++c) {
      std::memcpy(dst.data() + b * dst_stride + c * dst_ld,
                  src.data() + b * stride + c * ld,
                  static_cast<size_t>(rows) * sizeof(T));
    }
  }
  return dst;
}

template <typename T> std::vector<double> to_doubles(const std::vector<T>& v) {
  std::vector<double> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    out[i] = to_double(v[i]);
  }
  return out;
}

void parallel_for(int64_t count, double work,
                  const std::function<void(int64_t, int64_t)>& body) {
  const unsigned hw = std::max(1u, std::thread::hardware_concurrency());
  const int64_t threads =
      work < 1.0e6 ? 1 : std::min<int64_t>(static_cast<int64_t>(hw), count);
  if (threads <= 1) {
    body(0, count);
    return;
  }
  std::vector<std::thread> pool;
  const int64_t chunk = (count + threads - 1) / threads;
  for (int64_t begin = 0; begin < count; begin += chunk) {
    pool.emplace_back(body, begin, std::min(count, begin + chunk));
  }
  for (std::thread& t : pool) {
    t.join();
  }
}

// alpha * op(A) B per batch, without bias, dense [batch][n][m].
std::vector<double> reference(const Problem& p, const std::vector<double>& a,
                              const std::vector<double>& b) {
  std::vector<double> out(static_cast<size_t>(p.batch * p.n * p.m));
  const double alpha = p.alpha;
  const double work = static_cast<double>(p.batch) * p.n * p.m * p.k;
  parallel_for(p.batch * p.n, work, [&](int64_t begin, int64_t end) {
    std::vector<double> acc(static_cast<size_t>(p.m));
    for (int64_t col = begin; col < end; ++col) {
      const int64_t bi = col / p.n;
      const int64_t j = col % p.n;
      const double* bcol = b.data() + bi * p.stride_b + j * p.ldb;
      const double* abase = a.data() + bi * p.stride_a;
      double* dst = out.data() + col * p.m;
      if (p.trans_a) {
        for (int64_t i = 0; i < p.m; ++i) {
          const double* arow = abase + i * p.lda;
          double s = 0.0;
          for (int64_t q = 0; q < p.k; ++q) {
            s += arow[q] * bcol[q];
          }
          dst[i] = alpha * s;
        }
      } else {
        std::fill(acc.begin(), acc.end(), 0.0);
        for (int64_t q = 0; q < p.k; ++q) {
          const double bv = bcol[q];
          const double* acol = abase + q * p.lda;
          for (int64_t i = 0; i < p.m; ++i) {
            acc[i] += acol[i] * bv;
          }
        }
        for (int64_t i = 0; i < p.m; ++i) {
          dst[i] = alpha * acc[i];
        }
      }
    }
  });
  return out;
}

struct Check {
  double rel_l2 = std::numeric_limits<double>::infinity();
  double max_abs = 0.0;
  int64_t overwritten = 0;
  int64_t first_overwritten = -1;
};

// `got` is the whole output allocation; the matrix starts at `offset`. Every
// element outside the matrices must still hold the fill.
template <typename T>
Check check_output(const std::vector<T>& got, int64_t offset, int64_t m,
                   int64_t n, int64_t batch, int64_t ldd, int64_t stride_d,
                   const std::vector<double>& want,
                   const std::vector<double>* bias) {
  Check c;
  std::vector<uint8_t> inside(got.size(), 0);
  double err_sq = 0.0;
  double ref_sq = 0.0;
  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t j = 0; j < n; ++j) {
      for (int64_t i = 0; i < m; ++i) {
        const int64_t idx = offset + b * stride_d + j * ldd + i;
        inside[idx] = 1;
        const double w = want[(b * n + j) * m + i] + (bias ? (*bias)[i] : 0.0);
        const double g = to_double(got[idx]);
        const double delta = std::isfinite(g)
                                 ? std::abs(g - w)
                                 : std::numeric_limits<double>::infinity();
        c.max_abs = std::max(c.max_abs, delta);
        err_sq += delta * delta;
        ref_sq += w * w;
      }
    }
  }
  for (size_t i = 0; i < got.size(); ++i) {
    if (!inside[i] && !is_fill(got[i])) {
      if (c.overwritten == 0) {
        c.first_overwritten = static_cast<int64_t>(i);
      }
      ++c.overwritten;
    }
  }
  c.rel_l2 = std::sqrt(err_sq / std::max(ref_sq, 1.0e-30));
  return c;
}

template <typename T> class DeviceBuffer {
public:
  explicit DeviceBuffer(size_t count) : count_(count) {
    if (count_ != 0) {
      HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&data_), bytes()));
    }
  }
  ~DeviceBuffer() {
    if (data_) {
      (void)hipFree(data_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  T* get() const { return data_; }
  size_t count() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }
  void upload(const std::vector<T>& src) {
    HIP_CHECK(hipMemcpy(data_, src.data(), src.size() * sizeof(T),
                        hipMemcpyHostToDevice));
  }
  void fill() {
    if (data_) {
      HIP_CHECK(hipMemset(data_, 0xff, bytes()));
    }
  }
  std::vector<T> download() const {
    std::vector<T> out(count_);
    HIP_CHECK(hipMemcpy(out.data(), data_, bytes(), hipMemcpyDeviceToHost));
    return out;
  }

private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

// ---- Cold-cache timing, as in the matmul_nbits leaves ----
// The timed launches rotate over copies of the operands a launch reads, so
// each copy is read again only after kColdRotationBytes of other copies and
// comes from DRAM, as consecutive layers' weights do. HIPDNN_KERNEL_UT_WARM=1
// times one copy back to back instead.

constexpr size_t kColdRotationBytes = size_t(256) << 20;
constexpr size_t kColdSettleBytes = size_t(128) << 20;
constexpr int kColdMaxCopies = 4096;

bool warm_timing() {
  const char* v = std::getenv("HIPDNN_KERNEL_UT_WARM");
  return v && v[0] == '1';
}

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
    if (warm_timing() || bytes_ == 0) {
      return;
    }
    count_ = 1 + static_cast<int>(std::min<size_t>(
                     kColdMaxCopies - 1,
                     (kColdRotationBytes + bytes_ - 1) / bytes_));
    stride_ = stride;
    if (hipMalloc(&slab_, stride_ * (count_ - 1)) != hipSuccess) {
      (void)hipGetLastError();
      std::fprintf(stderr,
                   "cold-cache copies do not fit; timing on warm caches\n");
      slab_ = nullptr;
      count_ = 1;
      return;
    }
    for (size_t i = 0; i < parts_.size(); ++i) {
      if (parts_[i].bytes) {
        HIP_CHECK(hipMemcpy(part(1, i), parts_[i].src, parts_[i].bytes,
                            hipMemcpyDeviceToDevice));
      }
    }
    for (int filled = 1; filled < count_ - 1;) {
      const int more = std::min(filled, count_ - 1 - filled);
      HIP_CHECK(hipMemcpy(static_cast<char*>(slab_) + stride_ * filled, slab_,
                          stride_ * more, hipMemcpyDeviceToDevice));
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

// Holds the stream until the host releases it, so every timed launch is queued
// before the first runs and the reading is GPU time, not submission rate.
__global__ void hold_stream_kernel(const volatile int* release) {
  for (int spin = 0; spin < (1 << 16) && *release == 0; ++spin) {
    __builtin_amdgcn_s_sleep(127);
  }
}

int* hold_release_flag() {
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

double rotated_mean_ms(hipStream_t stream, const ColdCopies& copies,
                       const std::function<void(int)>& launch) {
  const int n = copies.count();
  int turn = 0;
  if (n > 1) {
    const size_t settle = std::min<size_t>(
        n, (kColdSettleBytes + copies.bytes() - 1) / copies.bytes());
    for (size_t i = 0; i < settle; ++i) {
      launch(turn++ % n);
    }
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  volatile int* release = hold_release_flag();
  void* release_dev = nullptr;
  if (release &&
      hipHostGetDevicePointer(&release_dev, const_cast<int*>(release), 0) ==
          hipSuccess) {
    *release = 0;
    hold_stream_kernel<<<1, 1, 0, stream>>>(
        static_cast<const int*>(release_dev));
  }
  hipEvent_t start;
  hipEvent_t stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start, stream));
  for (int i = 0; i < kTimedIters; ++i) {
    launch(turn++ % n);
  }
  HIP_CHECK(hipEventRecord(stop, stream));
  if (release) {
    *release = 1;
  }
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  return ms / kTimedIters;
}

// ---- out/results.csv ----

class CsvWriter {
public:
  CsvWriter() {
    const char* path = std::getenv("HIPDNN_RESULTS_CSV");
    if (!path || !path[0]) {
      return;
    }
    bool need_header = true;
    if (FILE* probe = std::fopen(path, "rb")) {
      std::fseek(probe, 0, SEEK_END);
      need_header = std::ftell(probe) == 0;
      std::fclose(probe);
    }
    f_ = std::fopen(path, "a");
    if (f_ && need_header) {
      std::fprintf(f_,
                   "op,leaf,arch,mode,shape,config,time_ms,relL2,verdict\n");
      std::fflush(f_);
    }
  }
  ~CsvWriter() {
    if (f_) {
      std::fclose(f_);
    }
  }
  CsvWriter(const CsvWriter&) = delete;
  CsvWriter& operator=(const CsvWriter&) = delete;
  void write(const std::string& shape, const std::string& config,
             double time_ms, double rel_l2, const char* verdict) {
    if (!f_) {
      return;
    }
    std::fprintf(f_, "%s,%s,%s,%s,%s,%s,%.6f,%.6e,%s\n",
                 env_or("HIPDNN_RESULTS_OP", "gemm"),
                 env_or("HIPDNN_RESULTS_LEAF", "Xfp16_Wfp16_Yfp32"),
                 env_or("HIPDNN_RESULTS_ARCH", "unknown"),
                 env_or("HIPDNN_RESULTS_MODE", "unknown"), shape.c_str(),
                 config.c_str(), time_ms, rel_l2, verdict);
    std::fflush(f_);
  }

private:
  static const char* env_or(const char* name, const char* dflt) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? v : dflt;
  }
  FILE* f_ = nullptr;
};

struct Tally {
  int rows = 0;
  int lookup = 0;
  int autotune = 0;
  int ref = 0;
  int declined = 0;
  int fail = 0;
  double max_rel_l2 = 0.0;
};

CsvWriter* g_csv = nullptr;
Tally g_tally;

bool emit(const std::string& shape, const std::string& config, double time_ms,
          double rel_l2, bool pass, const std::string& detail) {
  const bool declined = config == "declined";
  const char* verdict = pass ? "PASS" : "FAIL";
  g_csv->write(shape, config, time_ms, rel_l2, verdict);
  ++g_tally.rows;
  if (!pass) {
    ++g_tally.fail;
  } else if (declined) {
    ++g_tally.declined;
  }
  if (config.rfind("lookup:", 0) == 0) {
    ++g_tally.lookup;
  } else if (config.rfind("autotune:", 0) == 0) {
    ++g_tally.autotune;
  } else if (config == "fixed:ref") {
    ++g_tally.ref;
  }
  if (!declined && std::isfinite(rel_l2)) {
    g_tally.max_rel_l2 = std::max(g_tally.max_rel_l2, rel_l2);
  }
  std::printf("%-8s %-44s %-28s relL2=%.3e time=%.4f ms%s%s\n", verdict,
              shape.c_str(), config.c_str(), rel_l2, time_ms,
              detail.empty() ? "" : " ", detail.c_str());
  std::fflush(stdout);
  return pass;
}

std::string verdict_detail(const Check& c) {
  if (c.overwritten == 0) {
    return "maxAbs=" + std::to_string(c.max_abs);
  }
  return "maxAbs=" + std::to_string(c.max_abs) +
         " overwritten=" + std::to_string(c.overwritten) +
         " first=" + std::to_string(c.first_overwritten);
}

// ---- Entry points ----

std::string instance_name(int instance) {
  const char* name = hip_ck_gemm_instance_name(instance);
  return name ? name : "?";
}

struct Operands {
  const void* a;
  const void* b;
  const void* bias;
  void* d;
};

int ck_launch(hipStream_t stream, int instance, const Problem& p,
              const Operands& x) {
  return hip_ck_gemm_run(stream, instance, x.a, x.b, p.bias ? x.bias : nullptr,
                         x.d, p.m, p.n, p.k, p.batch, p.trans_a, 0, kAbDtype,
                         kDDtype, p.alpha, p.lda, p.ldb, p.ldd, p.stride_a,
                         p.stride_b, p.stride_d);
}

int ref_launch(hipStream_t stream, const Problem& p, const Operands& x) {
  return hip_ref_gemm_run(stream, x.a, x.b, x.d, p.m, p.n, p.k, p.batch,
                          p.trans_a, 0, kAbDtype, kDDtype, p.alpha, p.lda,
                          p.ldb, p.ldd, p.stride_a, p.stride_b, p.stride_d);
}

struct Route {
  int instance = -1;
  std::string config = "fixed:ref";
  std::string launch_error;
};

// Production's order (real/ck_gemm_select.h): the table's proposals first, then
// every instance; the first one hip_ck_gemm_run accepts is used here.
Route select_route(hipStream_t stream, const Problem& p, const Operands& x,
                   bool lookup) {
  Route r;
  auto accepted = [&](int instance, const char* prefix) {
    if (ck_launch(stream, instance, p, x) != 0) {
      return false;
    }
    const hipError_t e = hipStreamSynchronize(stream);
    if (e != hipSuccess) {
      (void)hipGetLastError();
      r.instance = instance;
      r.config = prefix + instance_name(instance);
      r.launch_error = hipGetErrorString(e);
      return true;
    }
    r.instance = instance;
    r.config = prefix + instance_name(instance);
    return true;
  };
  if (lookup) {
    int proposed[3];
    const int count =
        hip_ck_gemm_lut_candidates(p.m, p.n, p.k, p.batch, p.trans_a, kAbDtype,
                                   kDDtype, p.bias ? 1 : 0, proposed, 3);
    for (int c = 0; c < count; ++c) {
      if (accepted(proposed[c], "lookup:")) {
        return r;
      }
    }
  }
  const int count = hip_ck_gemm_num_instances();
  for (int i = 0; i < count; ++i) {
    if (accepted(i, "autotune:")) {
      return r;
    }
  }
  return r;
}

// One GEMM problem: the route production would take, then the reference GEMM
// on packed copies of the same operands. hip_ref_gemm_run has no bias, so its
// row is checked against the product alone.
bool run_gemm_case(const std::string& shape, const Problem& p, uint64_t seed,
                   hipStream_t stream, bool lookup) {
  Rng rng(seed);
  const std::vector<AbT> a = make_operand<AbT>(p.a_rows(), p.a_cols(), p.lda,
                                               p.stride_a, p.batch, rng);
  const std::vector<AbT> b =
      make_operand<AbT>(p.k, p.n, p.ldb, p.stride_b, p.batch, rng);
  std::vector<DT> bias;
  if (p.bias) {
    bias = make_operand<DT>(p.m, 1, p.m, 0, 1, rng);
  }
  const Problem q = packed(p);
  const std::vector<AbT> aq = repack(a, p.a_rows(), p.a_cols(), p.lda,
                                     p.stride_a, p.batch, q.lda, q.stride_a);
  const std::vector<AbT> bq =
      repack(b, p.k, p.n, p.ldb, p.stride_b, p.batch, q.ldb, q.stride_b);
  const std::vector<double> want = reference(q, to_doubles(aq), to_doubles(bq));
  const std::vector<double> bias_d = to_doubles(bias);

  DeviceBuffer<AbT> d_a(a.size());
  DeviceBuffer<AbT> d_b(b.size());
  DeviceBuffer<DT> d_bias(bias.size());
  DeviceBuffer<DT> d_d(static_cast<size_t>(p.d_elems() + kGuard));
  DeviceBuffer<AbT> d_aq(aq.size());
  DeviceBuffer<AbT> d_bq(bq.size());
  DeviceBuffer<DT> d_dq(static_cast<size_t>(q.d_elems() + kGuard));
  d_a.upload(a);
  d_b.upload(b);
  if (p.bias) {
    d_bias.upload(bias);
  }
  d_aq.upload(aq);
  d_bq.upload(bq);

  bool pass = true;
  const Operands x{d_a.get(), d_b.get(), d_bias.get(), d_d.get()};
  d_d.fill();
  const Route route = select_route(stream, p, x, lookup);
  if (!route.launch_error.empty()) {
    pass =
        emit(shape, route.config, 0.0, std::numeric_limits<double>::infinity(),
             false, "launch error after accept: " + route.launch_error) &&
        pass;
  } else if (route.instance >= 0) {
    d_d.fill();
    const int rc = ck_launch(stream, route.instance, p, x);
    HIP_CHECK(hipStreamSynchronize(stream));
    if (rc != 0) {
      pass = emit(shape, route.config, 0.0,
                  std::numeric_limits<double>::infinity(), false,
                  "refused on rerun rc=" + std::to_string(rc)) &&
             pass;
    } else {
      const Check c =
          check_output(d_d.download(), 0, p.m, p.n, p.batch, p.ldd, p.stride_d,
                       want, p.bias ? &bias_d : nullptr);
      const ColdCopies copies({{d_a.get(), d_a.bytes()},
                               {d_b.get(), d_b.bytes()},
                               {d_bias.get(), d_bias.bytes()}});
      const double ms = rotated_mean_ms(stream, copies, [&](int i) {
        (void)ck_launch(stream, route.instance, p,
                        {copies.part(i, 0), copies.part(i, 1),
                         copies.part(i, 2), d_d.get()});
      });
      pass = emit(shape, route.config, ms, c.rel_l2,
                  c.rel_l2 <= tolerance(p) && c.overwritten == 0,
                  verdict_detail(c)) &&
             pass;
    }
  }

  const std::string ref_shape = route.instance >= 0 ? shape + "_ref" : shape;
  const Operands xq{d_aq.get(), d_bq.get(), nullptr, d_dq.get()};
  d_dq.fill();
  const int rc = ref_launch(stream, q, xq);
  if (rc != 0) {
    return emit(ref_shape, "fixed:ref", 0.0,
                std::numeric_limits<double>::infinity(), false,
                "hip_ref_gemm_run refused rc=" + std::to_string(rc)) &&
           pass;
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  const Check c = check_output(d_dq.download(), 0, q.m, q.n, q.batch, q.ldd,
                               q.stride_d, want, nullptr);
  const ColdCopies copies(
      {{d_aq.get(), d_aq.bytes()}, {d_bq.get(), d_bq.bytes()}});
  const double ms = rotated_mean_ms(stream, copies, [&](int i) {
    (void)ref_launch(
        stream, q, {copies.part(i, 0), copies.part(i, 1), nullptr, d_dq.get()});
  });
  return emit(ref_shape, "fixed:ref", ms, c.rel_l2,
              c.rel_l2 <= tolerance(q) && c.overwritten == 0,
              verdict_detail(c)) &&
         pass;
}

// ---- Rejections ----

// `call` must return non-zero and leave the fill in `out` untouched.
bool expect_decline(const std::string& shape, DeviceBuffer<uint8_t>& out,
                    hipStream_t stream, const std::function<int()>& call,
                    const std::string& precondition_failure) {
  if (!precondition_failure.empty()) {
    return emit(shape, "declined", 0.0, 0.0, false, precondition_failure);
  }
  out.fill();
  const int rc = call();
  HIP_CHECK(hipStreamSynchronize(stream));
  const std::vector<uint8_t> after = out.download();
  const bool untouched = std::all_of(after.begin(), after.end(),
                                     [](uint8_t v) { return v == 0xff; });
  std::string detail;
  if (rc == 0) {
    detail = "accepted";
  } else if (!untouched) {
    detail = "declined but wrote the output";
  }
  return emit(shape, "declined", 0.0, 0.0, rc != 0 && untouched, detail);
}

// Number of instances that accept `p`; every accepted launch is synchronized.
int count_accepting(hipStream_t stream, const Problem& p, const Operands& x) {
  int accepted = 0;
  const int count = hip_ck_gemm_num_instances();
  for (int i = 0; i < count; ++i) {
    if (ck_launch(stream, i, p, x) == 0) {
      ++accepted;
      HIP_CHECK(hipStreamSynchronize(stream));
    }
  }
  return accepted;
}

bool run_registry_case(hipStream_t stream) {
  const int count = hip_ck_gemm_num_instances();
  std::set<std::string> names;
  bool ok = count > 0;
  for (int i = 0; i < count; ++i) {
    const char* name = hip_ck_gemm_instance_name(i);
    ok = ok && name && names.insert(name).second;
  }
  ok = ok && hip_ck_gemm_instance_name(-1) == nullptr &&
       hip_ck_gemm_instance_name(count) == nullptr;
  const Problem p =
      make_problem(64, 64, 64, 1, 0, false, 1.0f, Layout::kSingle);
  const Operands none{nullptr, nullptr, nullptr, nullptr};
  ok = ok && ck_launch(stream, -1, p, none) != 0 &&
       ck_launch(stream, count, p, none) != 0;
  return emit("registry_N" + std::to_string(count), "fixed:registry", 0.0, 0.0,
              ok, ok ? "" : "names not unique or out-of-range index accepted");
}

// hip_ref_gemm_run checks packing and its int32-index and grid-y limits before
// touching memory, so the oversized cases pass sizes with a small buffer.
bool run_ref_rejections(hipStream_t stream) {
  bool pass = true;
  {
    const Problem p =
        make_problem(64, 64, 64, 1, 0, false, 1.0f, Layout::kPadded);
    DeviceBuffer<AbT> d_a(static_cast<size_t>(p.a_elems()));
    DeviceBuffer<AbT> d_b(static_cast<size_t>(p.b_elems()));
    DeviceBuffer<uint8_t> d_d(static_cast<size_t>(p.d_elems()) * sizeof(DT));
    d_a.fill();
    d_b.fill();
    pass = expect_decline(
               "ref_M64_N64_K64_B1_pad", d_d, stream,
               [&] {
                 return ref_launch(stream, p,
                                   {d_a.get(), d_b.get(), nullptr, d_d.get()});
               },
               "") &&
           pass;
  }
  DeviceBuffer<uint8_t> dummy(4096);
  const Problem int32 =
      make_problem(65536, 8, 65536, 1, 0, false, 1.0f, Layout::kSingle);
  pass =
      expect_decline(
          "ref_M65536_N8_K65536_B1_int32", dummy, stream,
          [&] {
            return ref_launch(stream, int32,
                              {dummy.get(), dummy.get(), nullptr, dummy.get()});
          },
          "") &&
      pass;
  const Problem grid_y =
      make_problem(1048576, 1, 1, 1, 0, false, 1.0f, Layout::kSingle);
  pass =
      expect_decline(
          "ref_M1048576_N1_K1_B1_gridy", dummy, stream,
          [&] {
            return ref_launch(stream, grid_y,
                              {dummy.get(), dummy.get(), nullptr, dummy.get()});
          },
          "") &&
      pass;
  return pass;
}

// fp16 -> fp32 has only the TN no-bias combo, so every instance must refuse NN
// and bias for a problem some instance accepts as TN without bias.
bool run_leaf_rejections(hipStream_t stream) {
  const Problem tn =
      make_problem(64, 64, 64, 1, 1, false, 1.0f, Layout::kSingle);
  DeviceBuffer<AbT> d_a(static_cast<size_t>(tn.a_elems()));
  DeviceBuffer<AbT> d_b(static_cast<size_t>(tn.b_elems()));
  DeviceBuffer<DT> d_bias(static_cast<size_t>(tn.m));
  DeviceBuffer<uint8_t> d_d(static_cast<size_t>(tn.d_elems()) * sizeof(DT));
  HIP_CHECK(hipMemset(d_a.get(), 0, d_a.bytes()));
  HIP_CHECK(hipMemset(d_b.get(), 0, d_b.bytes()));
  HIP_CHECK(hipMemset(d_bias.get(), 0, d_bias.bytes()));
  const Operands x{d_a.get(), d_b.get(), d_bias.get(), d_d.get()};
  const bool control = count_accepting(stream, tn, x) > 0;
  const char* precondition = control ? "" : "no instance accepts TN";

  bool pass = true;
  const Problem nn =
      make_problem(64, 64, 64, 1, 0, false, 1.0f, Layout::kSingle);
  pass = expect_decline(
             problem_shape("NN", nn, Layout::kSingle), d_d, stream,
             [&] { return count_accepting(stream, nn, x) == 0 ? 1 : 0; },
             precondition) &&
         pass;
  const Problem bias =
      make_problem(64, 64, 64, 1, 1, true, 1.0f, Layout::kSingle);
  pass = expect_decline(
             problem_shape("TN_bias", bias, Layout::kSingle), d_d, stream,
             [&] { return count_accepting(stream, bias, x) == 0 ? 1 : 0; },
             precondition) &&
         pass;
  return pass;
}

constexpr int kLeafRejectionCases = 2;

int resolve_coverage(int argc, char** argv) {
  int tier = 3;
  if (const char* env = std::getenv("HIPDNN_UT_COVERAGE")) {
    tier = std::atoi(env);
  }
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--coverage") == 0 && i + 1 < argc) {
      tier = std::atoi(argv[++i]);
    } else {
      return 0;
    }
  }
  return tier >= 1 && tier <= 3 ? tier : 0;
}

bool lookup_mode() {
  const char* mode = std::getenv("HIPDNN_KERNEL_UT_MODE");
  if (!mode || !mode[0]) {
    mode = std::getenv("HIPDNN_RESULTS_MODE");
  }
  return !(mode && std::strcmp(mode, "autotune") == 0);
}

} // namespace

int main(int argc, char** argv) {
  const int coverage = resolve_coverage(argc, argv);
  if (coverage == 0) {
    std::cerr << "usage: " << argv[0] << " [--coverage 1|2|3]\n";
    return 2;
  }
  const bool lookup = lookup_mode();
  if (!lookup) {
#ifdef _WIN32
    _putenv_s("HIPDNN_CK_GEMM_AUTOTUNE_MODE", "online");
#else
    setenv("HIPDNN_CK_GEMM_AUTOTUNE_MODE", "online", 1);
#endif
  }

  int typical = 0;
  for (const TypicalShape& s : kTypical) {
    typical += s.min_coverage <= coverage ? 1 : 0;
  }
  constexpr int combos = static_cast<int>(sizeof(kCombos) / sizeof(kCombos[0]));
  constexpr int categorical =
      combos * static_cast<int>(sizeof(kDims) / sizeof(kDims[0])) *
      static_cast<int>(sizeof(kLayouts) / sizeof(kLayouts[0])) *
      static_cast<int>(sizeof(kAlphas) / sizeof(kAlphas[0]));
  constexpr int rejections = 1 + 3 + kLeafRejectionCases;
  std::printf("gemm coverage=%d mode=%s -> running %d typical shape(s) x %d "
              "combo(s) + %d categorical + %d registry/rejection = %d cases\n",
              coverage, lookup ? "lookup" : "autotune", typical, combos,
              categorical, rejections,
              typical * combos + categorical + rejections);
  std::fflush(stdout);

  CsvWriter csv;
  g_csv = &csv;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = run_registry_case(stream);
  uint64_t seed = 0;

  for (const Combo& combo : kCombos) {
    for (const Dims& d : kDims) {
      for (Layout layout : kLayouts) {
        for (float alpha : kAlphas) {
          const int64_t batch =
              layout == Layout::kSingle ? 1 : kCategoricalBatch;
          const Problem p = make_problem(d.m, d.n, d.k, batch, combo.trans_a,
                                         combo.bias, alpha, layout);
          all_pass = run_gemm_case(problem_shape(combo.name, p, layout), p,
                                   ++seed, stream, lookup) &&
                     all_pass;
        }
      }
    }
  }

  for (const TypicalShape& s : kTypical) {
    if (s.min_coverage > coverage) {
      continue;
    }
    const float alpha =
        kAlphaApplies && s.score
            ? static_cast<float>(1.0 / std::sqrt(static_cast<double>(s.k)))
            : 1.0f;
    for (const Combo& combo : kCombos) {
      const Layout layout = s.batch > 1 ? Layout::kBatched : Layout::kSingle;
      const Problem p = make_problem(s.m, s.n, s.k, s.batch, combo.trans_a,
                                     combo.bias, alpha, layout);
      all_pass = run_gemm_case(problem_shape(combo.name, p, layout), p, ++seed,
                               stream, lookup) &&
                 all_pass;
    }
  }

  all_pass = run_ref_rejections(stream) && all_pass;
  all_pass = run_leaf_rejections(stream) && all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::printf("summary rows=%d lookup=%d autotune=%d ref=%d declined=%d "
              "fail=%d maxRelL2=%.3e tolerance=1e-6+1e-8*k\n",
              g_tally.rows, g_tally.lookup, g_tally.autotune, g_tally.ref,
              g_tally.declined, g_tally.fail, g_tally.max_rel_l2);
  std::printf("%s\n", all_pass ? "ALL PASS" : "SOME FAIL");
  return all_pass ? 0 : 1;
}
