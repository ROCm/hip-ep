/* Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License. */
#include <hip/hip_runtime.h>
#include "hip_custom_kernels.h"

#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <set>
#include <cerrno>
#include <climits>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(_e),   \
                   __FILE__, __LINE__);                                        \
      std::exit(3);                                                            \
    }                                                                          \
  } while (0)

// ---- Tiny inline coverage-tier resolver + CSV writer ----
namespace hipdnn_ep_test {
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
    if (!path || !path[0]) return;
    bool need_header = true;
    if (FILE* probe = std::fopen(path, "rb")) {
      std::fseek(probe, 0, SEEK_END);
      need_header = std::ftell(probe) == 0;
      std::fclose(probe);
    }
    f_ = std::fopen(path, "a");
    if (!f_) throw std::runtime_error("cannot open requested results CSV");
    if (f_ && need_header) {
      std::fprintf(f_, "op,leaf,arch,mode,shape,config,time_ms,relL2,verdict\n");
      flush();
    }
  }
  ~CsvWriter() { if (f_) std::fclose(f_); }
  void write(const CsvRow& r) {
    if (!f_) return;
    std::fprintf(f_, "%s,%s,%s,%s,%s,%s,%.6f,%.6e,%s\n",
                 env_or("HIPDNN_RESULTS_OP", "unknown"),
                 env_or("HIPDNN_RESULTS_LEAF", "unknown"),
                 env_or("HIPDNN_RESULTS_ARCH", "unknown"),
                 env_or("HIPDNN_RESULTS_MODE", "unknown"), r.shape.c_str(),
                 r.config.c_str(), r.time_ms, r.rel_l2, r.verdict.c_str());
    flush();
  }

 private:
  void flush() {
    if (std::fflush(f_) != 0 || std::ferror(f_)) {
      std::fclose(f_);
      f_ = nullptr;
      throw std::runtime_error("cannot write requested results CSV");
    }
  }
  static const char* env_or(const char* name, const char* dflt) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? v : dflt;
  }
  FILE* f_ = nullptr;
};
}  // namespace hipdnn_ep_test

// ---- host fp16 conversions (round to nearest even), no compiler-rt needed ----
static uint16_t float_to_half_bits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF) return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const int shift = 14 - exp;
    uint32_t half = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u), mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1u))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) ++half;
  return static_cast<uint16_t>(sign | half);
}

static float half_bits_to_float(uint16_t h) {
  const uint32_t sign = (h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1Fu, mant = h & 0x3FFu, x;
  if (exp == 0) {
    if (mant == 0) {
      x = sign;
    } else {
      exp = 1;
      while (!(mant & 0x400u)) { mant <<= 1; --exp; }
      mant &= 0x3FFu;
      x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    x = sign | 0x7F800000u | (mant << 13);
  } else {
    x = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

// ---- small helpers ----
template <class Fn>
static void parallel_for(int64_t n, Fn&& fn) {
  const int64_t nt = std::max<int64_t>(1, std::min<int64_t>(16, std::thread::hardware_concurrency()));
  if (n < 4096 || nt == 1) {
    for (int64_t i = 0; i < n; ++i) fn(i);
    return;
  }
  std::vector<std::thread> pool;
  const int64_t chunk = (n + nt - 1) / nt;
  for (int64_t t = 0; t < nt; ++t) {
    const int64_t b = t * chunk, e = std::min<int64_t>(n, b + chunk);
    if (b >= e) break;
    pool.emplace_back([&fn, b, e] { for (int64_t i = b; i < e; ++i) fn(i); });
  }
  for (auto& th : pool) th.join();
}

template <class T>
static T* to_device(const std::vector<T>& h) {
  T* d = nullptr;
  HIP_CHECK(hipMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
  if (!h.empty()) HIP_CHECK(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}

template <class T>
static std::vector<T> from_device(const T* d, int64_t n) {
  std::vector<T> h(static_cast<size_t>(n));
  if (n > 0) HIP_CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
  return h;
}

// ---- Cold-cache timing ----
// In a model, a kernel's weights are read once per step and the rest of the
// model streams through the GPU caches before they are read again, so every
// launch reads them from DRAM. Back-to-back launches on one buffer find them in
// L2 / the Infinity Cache instead, and flushing the caches between launches
// does not restore DRAM reads either: a read-only flush leaves the reused lines
// resident, and a write flush leaves dirty lines for the timed launch to write
// back. So the timed launches rotate over copies of the weights, back to back,
// the way consecutive layers read their own: each copy is read again only after
// kColdRotationBytes of other copies. HIPDNN_KERNEL_UT_WARM=1 times one copy
// back to back instead.
namespace hipdnn_ep_test {
// Several times the largest last-level cache among the targets (96 MB on
// gfx1100).
constexpr size_t kColdRotationBytes = size_t(256) << 20;
// Untimed reads before the timed launches, so that the copies' initial fill
// has left the caches.
constexpr size_t kColdSettleBytes = size_t(128) << 20;
// Bounds the copies of operands of a few KB, whose launches are bound by
// latency rather than DRAM bandwidth; that many copies still keep each one out
// of L2 between its reads.
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
    if (warmTiming() || bytes_ == 0) return;
    count_ = 1 + static_cast<int>(std::min<size_t>(
                     kColdMaxCopies - 1,
                     (kColdRotationBytes + bytes_ - 1) / bytes_));
    stride_ = stride;
    if (hipMalloc(&slab_, stride_ * (count_ - 1)) != hipSuccess) {
      std::fprintf(stderr, "cold-cache copies do not fit; timing on warm caches\n");
      slab_ = nullptr;
      count_ = 1;
      return;
    }
    // Copy 1 from the buffers, then double the run of filled copies.
    for (size_t i = 0; i < parts_.size(); ++i)
      if (parts_[i].bytes)
        (void)hipMemcpy(part(1, i), parts_[i].src, parts_[i].bytes,
                        hipMemcpyDeviceToDevice);
    for (int filled = 1; filled < count_ - 1;) {
      const int more = std::min(filled, count_ - 1 - filled);
      (void)hipMemcpy(static_cast<char*>(slab_) + stride_ * filled, slab_,
                      stride_ * more, hipMemcpyDeviceToDevice);
      filled += more;
    }
  }
  ~ColdCopies() {
    if (slab_) (void)hipFree(slab_);
  }
  ColdCopies(const ColdCopies&) = delete;
  ColdCopies& operator=(const ColdCopies&) = delete;

  int count() const { return count_; }
  size_t bytes() const { return bytes_; }
  // Part `i` of copy `c`.
  void* part(int c, size_t i) const {
    if (c == 0 || parts_[i].bytes == 0) return const_cast<void*>(parts_[i].src);
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
// queued before the first one runs and the reading is the GPU's time rather
// than the host's submission rate. Gives up after about 0.25 s in case the
// queue cannot take all the launches.
__global__ void holdStreamKernel(const volatile int* release) {
  for (int spin = 0; spin < (1 << 16) && *release == 0; ++spin)
    __builtin_amdgcn_s_sleep(127);
}

// Host-mapped release flag for holdStreamKernel, or null if it is unavailable.
// Coherent, so the spinning kernel sees the host's write without a flush.
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

// Mean milliseconds per launch over `iters` back-to-back launches on `stream`,
// launch(c) reading copy c. Untimed launches over the rotation go first, until
// kColdSettleBytes have been read.
template <class Launch>
double rotatedMeanMs(hipStream_t stream, int iters, const ColdCopies& copies,
                     Launch&& launch) {
  const int n = copies.count();
  int turn = 0;
  if (n > 1) {
    const size_t settle = std::min<size_t>(
        n, (kColdSettleBytes + copies.bytes() - 1) / copies.bytes());
    for (size_t i = 0; i < settle; ++i) launch(turn++ % n);
  }
  // Drained, so the hold below holds the timed launches rather than being
  // released while earlier launches are still queued ahead of it.
  (void)hipStreamSynchronize(stream);
  volatile int* release = holdReleaseFlag();
  void* release_dev = nullptr;
  if (release &&
      hipHostGetDevicePointer(&release_dev, const_cast<int*>(release), 0) ==
          hipSuccess) {
    *release = 0;
    holdStreamKernel<<<1, 1, 0, stream>>>(static_cast<const int*>(release_dev));
  }
  hipEvent_t start, stop;
  (void)hipEventCreate(&start);
  (void)hipEventCreate(&stop);
  std::fprintf(stderr, "HIPDNN_UT_TIMING begin calls=%d role=reported\n", iters);
  std::fflush(stderr);
  (void)hipEventRecord(start, stream);
  for (int i = 0; i < iters; ++i) launch(turn++ % n);
  (void)hipEventRecord(stop, stream);
  if (release) *release = 1;
  (void)hipEventSynchronize(stop);
  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, stop);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  std::fprintf(stderr, "HIPDNN_UT_TIMING end calls=%d role=reported\n", iters);
  std::fflush(stderr);
  return iters > 0 ? ms / iters : 0.0;
}
}  // namespace hipdnn_ep_test


static void report(const char* label, const std::string& shape, const std::string& config, int status,
                   double ms, double rel_l2, bool pass) {
  hipdnn_ep_test::CsvWriter csv;
  hipdnn_ep_test::CsvRow row;
  row.shape = shape;
  row.config = config;
  row.time_ms = ms;
  row.rel_l2 = rel_l2;
  row.verdict = pass ? "PASS" : "FAIL";
  csv.write(row);
  std::printf("  %-10s %-36s status=%d  %.4f ms  relL2=%.3e  %s\n", label, shape.c_str(), status, ms, rel_l2,
              pass ? "PASS" : "*** FAIL ***");
}

static const char* arg_str(int argc, char** argv, const char* name) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
  return nullptr;
}


static double arg_f64(int argc, char** argv, const char* name, double dflt) {
  const char* v = arg_str(argc, argv, name);
  return v ? std::atof(v) : dflt;
}

// "1x1024x8192" -> {1, 1024, 8192}; "" or "scalar" -> {}
static std::vector<int64_t> parse_shape(const char* s) {
  std::vector<int64_t> d;
  if (!s || !*s || std::strcmp(s, "scalar") == 0) return d;
  const char* p = s;
  while (*p) {
    d.push_back(std::atoll(p));
    while (*p && *p != 'x') ++p;
    if (*p == 'x') ++p;
  }
  return d;
}

static int64_t checked_product(std::initializer_list<int64_t> values) {
  int64_t product = 1;
  for (int64_t v : values) {
    if (v <= 0 || product > std::numeric_limits<int64_t>::max() / v)
      throw std::invalid_argument("nonpositive or overflowing geometry");
    product *= v;
  }
  if (static_cast<uint64_t>(product) > std::numeric_limits<size_t>::max() / 8)
    throw std::overflow_error("allocation size overflow");
  return product;
}

static int64_t integer(const char* text) {
  char* end = nullptr;
  errno = 0;
  const long long value = std::strtoll(text, &end, 10);
  if (errno || !text[0] || !end || *end) throw std::invalid_argument("invalid integer argument");
  return static_cast<int64_t>(value);
}

static bool options(int argc, char** argv, std::initializer_list<const char*> allowed) {
  std::set<std::string> seen;
  for (int i = 1; i < argc; i += 2) {
    bool known = false;
    for (const char* name : allowed) if (std::strcmp(argv[i], name) == 0) known = true;
    if (!known || i + 1 >= argc || !seen.insert(argv[i]).second) {
      std::fprintf(stderr, "unknown, duplicate or incomplete argument: %s\n", argv[i]);
      return false;
    }
  }
  return true;
}

static int64_t opt(int argc, char** argv, const char* key, int64_t fallback) {
  const char* value = arg_str(argc, argv, key);
  return value ? integer(value) : fallback;
}

template <class T> static T number(double x) { return static_cast<T>(x); }
template <> uint16_t number<uint16_t>(double x) { return float_to_half_bits(static_cast<float>(x)); }
template <class T> static double real(T x) { return static_cast<double>(x); }
template <> double real<uint16_t>(uint16_t x) { return half_bits_to_float(x); }

template <class T> static std::vector<T> values(int64_t n, unsigned seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> distribution(lo, hi);
  std::vector<T> result(static_cast<size_t>(n));
  for (T& value : result) value = number<T>(distribution(rng));
  return result;
}

using Value = float;
struct Case { int64_t rows = 1, norm = 1152; bool bias = false, mean = false, inv_std = false, stash_half = false;
              float eps = 1e-5f; };

static std::vector<Case> build_cases(int tier) {
  const std::vector<std::pair<int64_t, int64_t>> lengths = {{1, 7}, {3, 1152}, {17, 3073}};
  std::vector<Case> result;
  for (int i = 0; i < tier; ++i)
    for (bool bias : {false, true}) for (bool mean : {false, true})
      for (bool inv : {false, true}) for (bool half : {false, true}) {
        if (sizeof(Value) == 4 && half) continue;
        for (float eps : {0.0f, 1e-5f}) result.push_back({lengths[i].first, lengths[i].second, bias, mean, inv, half, eps});
      }
  return result;
}

static bool run_case(const Case& c, int iters, unsigned seed) {
  const int64_t count = checked_product({c.rows, c.norm});
  auto x = values<Value>(count, seed, -1.0f, 1.0f);
  auto scale = values<Value>(c.norm, seed + 1, 0.5f, 1.5f);
  auto bias = values<Value>(c.norm, seed + 2, -0.2f, 0.2f);
  std::vector<double> ref(static_cast<size_t>(count)), means(static_cast<size_t>(c.rows)), invs(static_cast<size_t>(c.rows));
  parallel_for(c.rows, [&](int64_t row) {
    double sum = 0.0, sq = 0.0;
    for (int64_t i = 0; i < c.norm; ++i) { const double v = real(x[row * c.norm + i]); sum += v; sq += v * v; }
    const double mean = sum / c.norm;
    const double inverse = 1.0 / std::sqrt(std::max(0.0, sq / c.norm - mean * mean) + c.eps);
    means[row] = c.stash_half ? real(number<uint16_t>(mean)) : static_cast<float>(mean);
    invs[row] = c.stash_half ? real(number<uint16_t>(inverse)) : static_cast<float>(inverse);
    for (int64_t i = 0; i < c.norm; ++i)
      ref[row * c.norm + i] = (real(x[row * c.norm + i]) - mean) * inverse * real(scale[i]) + (c.bias ? real(bias[i]) : 0.0);
  });
  Value* dx = to_device(x); Value* ds = to_device(scale); Value* db = c.bias ? to_device(bias) : nullptr;
  Value* dy = nullptr; HIP_CHECK(hipMalloc(&dy, static_cast<size_t>(count) * sizeof(Value)));
  void* dm = nullptr; void* dv = nullptr;
  const size_t stat_bytes = static_cast<size_t>(c.rows) * (c.stash_half ? 2 : 4);
  if (c.mean) HIP_CHECK(hipMalloc(&dm, stat_bytes));
  if (c.inv_std) HIP_CHECK(hipMalloc(&dv, stat_bytes));
  hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream));
  const int dtype = sizeof(Value) == 2 ? HIP_DTYPE_FLOAT16 : HIP_DTYPE_FLOAT32;
  const int stash = c.stash_half ? HIP_DTYPE_FLOAT16 : HIP_DTYPE_FLOAT32;
  int status = 0;
  auto launch = [&](const void* input, const void* gamma, const void* beta) {
    const int rc = hip_layer_norm(stream, input, gamma, beta, dy, dm, dv, c.rows, c.norm, c.eps, dtype, stash);
    if (rc && !status) status = rc;
  };
  for (int w = 0; w < 3; ++w) launch(dx, ds, db);
  HIP_CHECK(hipStreamSynchronize(stream));
  const hipdnn_ep_test::ColdCopies copies({{dx, x.size() * sizeof(Value)}, {ds, scale.size() * sizeof(Value)},
                                         {db, c.bias ? bias.size() * sizeof(Value) : 0}});
  double ms = 0.0;
  if (!status) ms = hipdnn_ep_test::rotatedMeanMs(stream, iters, copies, [&](int i) {
    launch(copies.part(i, 0), copies.part(i, 1), c.bias ? copies.part(i, 2) : nullptr);
  });
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto got = from_device(dy, count);
  double ss_error = 0.0, ss_ref = 0.0; bool correct = true;
  const double atol = sizeof(Value) == 2 ? 0.004 : 2e-5;
  for (size_t i = 0; i < got.size(); ++i) {
    const double error = real(got[i]) - ref[i]; ss_error += error * error; ss_ref += ref[i] * ref[i];
    if (!std::isfinite(real(got[i])) || std::abs(error) > atol + atol * std::abs(ref[i])) correct = false;
  }
  auto check_stats = [&](const void* ptr, const std::vector<double>& expected) {
    if (!ptr) return;
    const auto half = c.stash_half ? from_device(static_cast<const uint16_t*>(ptr), c.rows) : std::vector<uint16_t>();
    const auto full = !c.stash_half ? from_device(static_cast<const float*>(ptr), c.rows) : std::vector<float>();
    const double tol = c.stash_half ? 0.002 : 2e-5;
    for (int64_t i = 0; i < c.rows; ++i) {
      const double v = c.stash_half ? real(half[i]) : full[i];
      if (!std::isfinite(v) || std::abs(v - expected[i]) > tol + tol * std::abs(expected[i])) correct = false;
    }
  };
  check_stats(dm, means); check_stats(dv, invs);
  const double l2 = std::sqrt(ss_error / std::max(ss_ref, 1e-30));
  const bool pass = !status && correct && std::isfinite(ms) && ms > 0;
  char epsilon[32];
  std::snprintf(epsilon, sizeof(epsilon), "%.9g", static_cast<double>(c.eps));
  const std::string shape = std::to_string(c.rows) + "x" + std::to_string(c.norm) + "_bias" + std::to_string(c.bias) +
    "_mean" + std::to_string(c.mean) + "_inv" + std::to_string(c.inv_std) + "_stash" + (c.stash_half ? "fp16" : "fp32") + "_eps" + epsilon;
  report("layer_norm", shape, "fixed:hip_layer_norm", status, ms, l2, pass);
  HIP_CHECK(hipStreamDestroy(stream)); HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(ds)); HIP_CHECK(hipFree(dy));
  if (db) HIP_CHECK(hipFree(db)); if (dm) HIP_CHECK(hipFree(dm)); if (dv) HIP_CHECK(hipFree(dv));
  return pass;
}

int main(int argc, char** argv) {
  if (!options(argc, argv, {"--rows", "--norm", "--bias", "--mean", "--inv-std", "--stash", "--eps", "--iters", "--coverage"})) return 2;
  try {
    Case c;
    c.rows = opt(argc, argv, "--rows", 1); c.norm = opt(argc, argv, "--norm", 1152);
    const auto flag = [&](const char* name) { const int64_t v = opt(argc, argv, name, 0);
      if (v != 0 && v != 1) throw std::invalid_argument("boolean flag must be 0 or 1"); return v == 1; };
    c.bias = flag("--bias"); c.mean = flag("--mean"); c.inv_std = flag("--inv-std");
    const char* stash = arg_str(argc, argv, "--stash");
    if (stash && std::strcmp(stash, "fp16") && std::strcmp(stash, "fp32")) throw std::invalid_argument("stash must be fp16 or fp32");
    c.stash_half = stash && std::strcmp(stash, "fp16") == 0;
    if (sizeof(Value) == 4 && c.stash_half) throw std::invalid_argument("FP32 I/O supports FP32 statistics only");
    if (const char* eps = arg_str(argc, argv, "--eps")) {
      char* end = nullptr; errno = 0; c.eps = std::strtof(eps, &end);
      if (errno || !eps[0] || !end || *end || !std::isfinite(c.eps) || c.eps < 0) throw std::invalid_argument("invalid epsilon");
    }
    const int64_t count = opt(argc, argv, "--iters", 20);
    const char* coverage = arg_str(argc, argv, "--coverage");
    if (!coverage) coverage = std::getenv("HIPDNN_UT_COVERAGE");
    const int64_t tier = coverage ? integer(coverage) : 3;
    checked_product({c.rows, c.norm});
    if (count < 1 || count > INT_MAX || tier < 1 || tier > 3) throw std::invalid_argument("invalid iters/coverage");
    const bool single = arg_str(argc, argv, "--rows") || arg_str(argc, argv, "--norm") ||
      arg_str(argc, argv, "--bias") || arg_str(argc, argv, "--mean") || arg_str(argc, argv, "--inv-std") ||
      arg_str(argc, argv, "--stash") || arg_str(argc, argv, "--eps");
    if (single && (!arg_str(argc, argv, "--rows") || !arg_str(argc, argv, "--norm")))
      throw std::invalid_argument("a single LayerNormalization case requires --rows and --norm");
    hipDeviceProp_t device{}; HIP_CHECK(hipGetDeviceProperties(&device, 0));
    std::printf("LayerNorm Xfp32_Wfp32_Yfp32; GPU=%s arch=%s; owning-main cold rotation; iterations=%lld\n",
                device.name, device.gcnArchName, static_cast<long long>(count));
    const auto cases = single ? std::vector<Case>{c} : build_cases(static_cast<int>(tier));
    std::printf("coverage=%lld -> running %zu LayerNormalization cases\n", static_cast<long long>(tier), cases.size());
    int failures = 0; unsigned seed = 1234;
    for (const auto& item : cases) if (!run_case(item, static_cast<int>(count), seed++)) ++failures;
    std::printf("%s\n", failures ? "SOME FAILED" : "ALL PASS");
    return failures ? 1 : 0;
  } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 2; }
}
