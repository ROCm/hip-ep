/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for com.microsoft GroupNorm (hip_group_norm) with
 * fp16 input, scale, bias and output, both layouts, with and without SiLU,
 * against a CPU fp64 reference computed from the rounded inputs.
 *
 * Every shape also runs without a workspace (the fused path) and with a
 * workspace too small for the planned split (a shrunk split), with misaligned
 * tensors (scalar access), and twice with the same workspace to check that
 * the output is bitwise deterministic.
 */

#include "hip_custom_kernels.h"

#include <hip/hip_bf16.h>
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

// The one dtype of this leaf: input, scale, bias and output.
using Elem = __half;

constexpr int kGuardElements = 32;
constexpr unsigned char kFillByte = 0x7f;

struct Shape {
  int64_t n;
  int64_t c;
  int64_t spatial;
  int64_t groups;
  int min_coverage;
};

constexpr Shape kShapes[] = {
    {1, 64, 16, 32, 1},        // tiny: fused path
    {2, 40, 1000, 8, 1},       // odd channels per group (5)
    {1, 320, 4096, 32, 1},     // channels per group 10
    {2, 128, 997, 1, 1},       // G = 1, prime spatial
    {1, 256, 257, 256, 1},     // G = C
    {1, 64, 1, 32, 1},         // spatial = 1
    {3, 96, 4096, 32, 2},      // N = 3
    {1, 4096, 64, 1, 2},       // a group wider than one block
    {2, 3000, 40, 1, 2},       // uneven slices of one group
    {1, 512, 9216, 32, 2},     // VAE 96x96
    {1, 512, 36864, 32, 3},    // VAE 192x192
    {1, 256, 147456, 32, 3},   // VAE 384x384
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
  // Uniform in [-1, 1).
  double uniform() {
    return static_cast<double>(next() >> 11) * 0x1.0p-52 - 1.0;
  }

private:
  uint64_t state_;
};

template <typename T> struct TypeInfo;

template <> struct TypeInfo<__half> {
  static constexpr int kDtype = HIP_DTYPE_FLOAT16;
  static constexpr const char *kName = "fp16";
  static constexpr double kAtol = 4e-3, kRtol = 4e-3;
  static constexpr double kLargeOffset = 256.0;
  static constexpr double kWideRange = 3e4;
  static constexpr double kComputeEps = 0x1.0p-24;
  static __half from(double v) { return __float2half(static_cast<float>(v)); }
  static double to(__half v) { return __half2float(v); }
};

template <> struct TypeInfo<__hip_bfloat16> {
  static constexpr int kDtype = HIP_DTYPE_BFLOAT16;
  static constexpr const char *kName = "bf16";
  static constexpr double kAtol = 3e-2, kRtol = 2e-2;
  static constexpr double kLargeOffset = 32.0;
  static constexpr double kWideRange = 1e15;
  static constexpr double kComputeEps = 0x1.0p-24;
  static __hip_bfloat16 from(double v) {
    return __float2bfloat16(static_cast<float>(v));
  }
  static double to(__hip_bfloat16 v) { return __bfloat162float(v); }
};

template <> struct TypeInfo<float> {
  static constexpr int kDtype = HIP_DTYPE_FLOAT32;
  static constexpr const char *kName = "fp32";
  static constexpr double kAtol = 1e-4, kRtol = 1e-4;
  static constexpr double kLargeOffset = 1e4;
  static constexpr double kWideRange = 1e15;
  static constexpr double kComputeEps = 0x1.0p-24;
  static float from(double v) { return static_cast<float>(v); }
  static double to(float v) { return v; }
};

template <> struct TypeInfo<double> {
  static constexpr int kDtype = HIP_DTYPE_FLOAT64;
  static constexpr const char *kName = "fp64";
  static constexpr double kAtol = 1e-9, kRtol = 1e-9;
  static constexpr double kLargeOffset = 1e8;
  static constexpr double kWideRange = 1e100;
  static constexpr double kComputeEps = 0x1.0p-53;
  static double from(double v) { return v; }
  static double to(double v) { return v; }
};

class DeviceBytes {
public:
  explicit DeviceBytes(size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0)
      HIP_CHECK(hipMalloc(&data_, bytes_));
  }
  ~DeviceBytes() {
    if (data_)
      (void)hipFree(data_);
  }
  DeviceBytes(const DeviceBytes &) = delete;
  DeviceBytes &operator=(const DeviceBytes &) = delete;
  char *get() { return static_cast<char *>(data_); }
  size_t size() const { return bytes_; }

private:
  void *data_ = nullptr;
  size_t bytes_ = 0;
};

struct CsvRow {
  std::string shape, config;
  double time_ms = 0.0;
  double rel_l2 = 0.0;
  std::string verdict;
};

class CsvWriter {
public:
  CsvWriter() {
    const char *path = std::getenv("HIPDNN_RESULTS_CSV");
    if (!path || !path[0])
      return;
    bool need_header = true;
    if (FILE *probe = std::fopen(path, "rb")) {
      std::fseek(probe, 0, SEEK_END);
      need_header = std::ftell(probe) == 0;
      std::fclose(probe);
    }
    f_ = std::fopen(path, "a");
    if (f_ && need_header) {
      std::fprintf(f_, "op,leaf,arch,mode,shape,config,time_ms,relL2,verdict\n");
      std::fflush(f_);
    }
  }
  ~CsvWriter() {
    if (f_)
      std::fclose(f_);
  }
  void write(const CsvRow &r) {
    if (!f_)
      return;
    std::fprintf(f_, "%s,%s,%s,%s,%s,%s,%.6f,%.6e,%s\n",
                 env_or("HIPDNN_RESULTS_OP", "group_norm"),
                 env_or("HIPDNN_RESULTS_LEAF", "unknown"),
                 env_or("HIPDNN_RESULTS_ARCH", "unknown"),
                 env_or("HIPDNN_RESULTS_MODE", "unknown"), r.shape.c_str(),
                 r.config.c_str(), r.time_ms, r.rel_l2, r.verdict.c_str());
    std::fflush(f_);
  }

private:
  static const char *env_or(const char *name, const char *dflt) {
    const char *v = std::getenv(name);
    return (v && v[0]) ? v : dflt;
  }
  FILE *f_ = nullptr;
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
  ResultRow(const ResultRow &) = delete;
  ResultRow &operator=(const ResultRow &) = delete;
  ~ResultRow() { CsvWriter().write(row); }
  void finish(double rel_l2, bool pass) {
    row.rel_l2 = rel_l2;
    row.verdict = pass ? "PASS" : "FAIL";
  }
};

// kLargeOffset puts |mean| / stddev near the dtype's precision limit (each
// value keeps a few bits below the unit spread); E[x^2] - mean^2 fails there.
// kOutliers adds sparse +-32 spikes on top, so some threads start their
// accumulation far from the group mean.
// kWideRange spans most of the dtype's range (squared deviations still fit
// the fp32 accumulator for fp16), to catch intermediate overflow.
// kConstantGroup makes group 0 one repeated large value: the output must be
// exactly beta, which folding the mean into the bias does not give.
// kNonFinite puts a NaN inside group 0 and +Inf at the first element of
// group 1: those groups must be NaN and every other group unaffected.
enum class Inputs {
  kSmallOffset,
  kLargeOffset,
  kOutliers,
  kWideRange,
  kConstantGroup,
  kNonFinite
};

const char *inputs_name(Inputs inputs) {
  switch (inputs) {
  case Inputs::kSmallOffset:
    return "offset";
  case Inputs::kLargeOffset:
    return "large";
  case Inputs::kOutliers:
    return "outliers";
  case Inputs::kWideRange:
    return "wide";
  case Inputs::kConstantGroup:
    return "constant";
  case Inputs::kNonFinite:
    return "nonfinite";
  }
  return "?";
}

enum class WorkspaceMode { kFull, kNone, kSmall, kMisaligned };

const char *mode_name(WorkspaceMode mode) {
  switch (mode) {
  case WorkspaceMode::kFull:
    return "split";
  case WorkspaceMode::kNone:
    return "no-workspace";
  case WorkspaceMode::kSmall:
    return "small-workspace";
  default:
    return "misaligned";
  }
}

int64_t offset_of(const Shape &s, int channels_last, int64_t n, int64_t c,
                  int64_t p) {
  return channels_last ? (n * s.spatial + p) * s.c + c
                       : (n * s.c + c) * s.spatial + p;
}

// Reference from the values the kernel actually reads, accumulated in long
// double relative to the group's first value so it stays exact for fp64
// inputs with a large offset. *conditioning receives |mean| * |gamma| / std
// per element: rounding the mean costs about eps times that in the output. A
// group without spread has an exactly representable mean, so its
// conditioning is 0. NaN and Inf propagate as in the exact formula.
std::vector<double> reference(const Shape &s, int channels_last, int activation,
                              double epsilon, const std::vector<double> &x,
                              const std::vector<double> &gamma,
                              const std::vector<double> &beta,
                              std::vector<double> *conditioning) {
  using LD = long double;
  std::vector<double> y(x.size());
  conditioning->assign(x.size(), 0.0);
  const int64_t cpg = s.c / s.groups;
  const LD count = static_cast<LD>(cpg * s.spatial);
  for (int64_t n = 0; n < s.n; ++n) {
    for (int64_t g = 0; g < s.groups; ++g) {
      const LD shift = x[offset_of(s, channels_last, n, g * cpg, 0)];
      LD sum = 0.0L;
      for (int64_t c = g * cpg; c < (g + 1) * cpg; ++c)
        for (int64_t p = 0; p < s.spatial; ++p)
          sum += x[offset_of(s, channels_last, n, c, p)] - shift;
      const LD mean = shift + sum / count;
      LD sq = 0.0L;
      for (int64_t c = g * cpg; c < (g + 1) * cpg; ++c)
        for (int64_t p = 0; p < s.spatial; ++p) {
          const LD d = x[offset_of(s, channels_last, n, c, p)] - mean;
          sq += d * d;
        }
      const LD inv = 1.0L / std::sqrt(sq / count + epsilon);
      const LD cond = sq == 0.0L ? 0.0L : std::abs(mean) * inv;
      for (int64_t c = g * cpg; c < (g + 1) * cpg; ++c)
        for (int64_t p = 0; p < s.spatial; ++p) {
          const int64_t o = offset_of(s, channels_last, n, c, p);
          double v = static_cast<double>((x[o] - mean) * inv * gamma[c]) +
                     beta[c];
          if (activation)
            v = v / (1.0 + std::exp(-v));
          y[o] = v;
          (*conditioning)[o] = static_cast<double>(cond * std::abs(gamma[c]));
        }
    }
  }
  return y;
}

std::string case_shape(const Shape &s) {
  char buffer[96];
  std::snprintf(buffer, sizeof(buffer), "N%lld_C%lld_HW%lld_G%lld",
                static_cast<long long>(s.n), static_cast<long long>(s.c),
                static_cast<long long>(s.spatial),
                static_cast<long long>(s.groups));
  return buffer;
}

template <typename T>
bool run_case(const Shape &s, uint64_t seed, int channels_last,
              WorkspaceMode mode, Inputs inputs, hipStream_t stream) {
  using Info = TypeInfo<T>;
  const int activation = static_cast<int>(seed & 1);
  const float epsilon = 1e-5f;
  char config[128];
  std::snprintf(config, sizeof(config), "%s:%s:%s:%s:silu%d", Info::kName,
                channels_last ? "nhwc" : "nchw", mode_name(mode),
                inputs_name(inputs), activation);
  const std::string name = case_shape(s);
  ResultRow result(name, config);

  const int64_t elements = s.n * s.c * s.spatial;
  Rng rng(seed);
  std::vector<T> hx(elements), hgamma(s.c), hbeta(s.c);
  std::vector<double> x(elements), gamma(s.c), beta(s.c);
  const int64_t cpg = s.c / s.groups;
  for (int64_t n = 0; n < s.n; ++n)
    for (int64_t c = 0; c < s.c; ++c) {
      const double group_shift = ((c / cpg) % 3) + 0.5 * n;
      const bool large = inputs == Inputs::kLargeOffset ||
                         inputs == Inputs::kOutliers;
      const double shift = large
                               ? Info::kLargeOffset * (1.0 + 0.1 * group_shift)
                               : 4.0 * group_shift - 2.0;
      for (int64_t p = 0; p < s.spatial; ++p) {
        const int64_t o = offset_of(s, channels_last, n, c, p);
        double v = shift + rng.uniform();
        if (inputs == Inputs::kOutliers && (c * s.spatial + p) % 61 == 0)
          v += (p & 1) ? 32.0 : -32.0;
        if (inputs == Inputs::kWideRange)
          v = Info::kWideRange * rng.uniform();
        if (inputs == Inputs::kConstantGroup && c < cpg)
          v = Info::kLargeOffset * (1.0 + 0.5 * n);
        hx[o] = Info::from(v);
        x[o] = Info::to(hx[o]);
      }
    }
  if (inputs == Inputs::kNonFinite) {
    for (int64_t n = 0; n < s.n; ++n) {
      const int64_t nan_at =
          s.spatial > 1 ? offset_of(s, channels_last, n, 0, s.spatial / 2)
                        : offset_of(s, channels_last, n, cpg - 1, 0);
      hx[nan_at] = Info::from(std::numeric_limits<double>::quiet_NaN());
      x[nan_at] = Info::to(hx[nan_at]);
      if (s.groups > 1) {
        const int64_t inf_at = offset_of(s, channels_last, n, cpg, 0);
        hx[inf_at] = Info::from(std::numeric_limits<double>::infinity());
        x[inf_at] = Info::to(hx[inf_at]);
      }
    }
  }
  for (int64_t c = 0; c < s.c; ++c) {
    hgamma[c] = Info::from(1.0 + 0.5 * rng.uniform());
    hbeta[c] = Info::from(0.25 * rng.uniform());
    gamma[c] = Info::to(hgamma[c]);
    beta[c] = Info::to(hbeta[c]);
  }
  std::vector<double> conditioning;
  const std::vector<double> want = reference(
      s, channels_last, activation, epsilon, x, gamma, beta, &conditioning);

  // Misaligned tensors start one element past a 256-byte boundary.
  const size_t skew = mode == WorkspaceMode::kMisaligned ? sizeof(T) : 0;
  const size_t data_bytes = elements * sizeof(T);
  const size_t guard_bytes = kGuardElements * sizeof(T);
  DeviceBytes d_x(data_bytes + skew);
  DeviceBytes d_gamma(s.c * sizeof(T));
  DeviceBytes d_beta(s.c * sizeof(T));
  DeviceBytes d_y(data_bytes + 2 * guard_bytes + skew);
  T *x_ptr = reinterpret_cast<T *>(d_x.get() + skew);
  T *y_ptr = reinterpret_cast<T *>(d_y.get() + guard_bytes + skew);
  HIP_CHECK(hipMemcpy(x_ptr, hx.data(), data_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gamma.get(), hgamma.data(), s.c * sizeof(T),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_beta.get(), hbeta.data(), s.c * sizeof(T),
                      hipMemcpyHostToDevice));

  size_t ws_bytes = hip_group_norm_workspace_bytes(
      s.n, s.c, s.spatial, s.groups, channels_last, Info::kDtype);
  if (mode == WorkspaceMode::kNone)
    ws_bytes = 0;
  else if (mode == WorkspaceMode::kSmall)
    ws_bytes /= 7;
  DeviceBytes d_ws(ws_bytes);

  auto launch = [&]() {
    HIP_CHECK(hipMemset(d_y.get(), kFillByte, d_y.size()));
    return hip_group_norm(stream, x_ptr, d_gamma.get(), d_beta.get(), y_ptr,
                          s.n, s.c, s.spatial, s.groups, channels_last,
                          activation, epsilon,
                          ws_bytes ? d_ws.get() : nullptr, ws_bytes,
                          Info::kDtype);
  };
  auto download = [&](std::vector<unsigned char> &out) {
    HIP_CHECK(hipStreamSynchronize(stream));
    out.resize(d_y.size());
    HIP_CHECK(hipMemcpy(out.data(), d_y.get(), out.size(),
                        hipMemcpyDeviceToHost));
  };

  int status = launch();
  if (status != static_cast<int>(hipSuccess)) {
    std::cout << "FAIL launch status=" << status << " shape=" << name << ' '
              << config << '\n';
    return false;
  }
  std::vector<unsigned char> first;
  download(first);

  bool deterministic = true;
  if (mode == WorkspaceMode::kFull) {
    std::vector<unsigned char> second;
    if (launch() != static_cast<int>(hipSuccess))
      return false;
    download(second);
    deterministic = first == second;
  }

  bool guards_ok = true;
  const size_t tail = skew + guard_bytes + data_bytes;
  for (size_t i = 0; i < guard_bytes; ++i)
    guards_ok = guards_ok && first[skew + i] == kFillByte &&
                first[tail + i] == kFillByte;

  std::vector<T> got(elements);
  std::memcpy(got.data(), first.data() + skew + guard_bytes, data_bytes);
  int64_t bad = 0, first_bad = -1;
  double max_abs = 0.0, err_sq = 0.0, ref_sq = 0.0;
  for (int64_t i = 0; i < elements; ++i) {
    const double g = Info::to(got[i]);
    if (std::isnan(want[i])) {
      if (!std::isnan(g) && bad++ == 0)
        first_bad = i;
      continue;
    }
    const double d = std::isfinite(g) ? std::abs(g - want[i])
                                      : std::numeric_limits<double>::infinity();
    // Even an exact algorithm rounds the mean to the compute type; allow a
    // few ulps of it on top of the dtype tolerance.
    const double tol = Info::kAtol + Info::kRtol * std::abs(want[i]) +
                       16.0 * Info::kComputeEps * conditioning[i];
    if (!(d <= tol)) {
      if (bad++ == 0)
        first_bad = i;
    }
    max_abs = std::max(max_abs, d);
    err_sq += d * d;
    ref_sq += want[i] * want[i];
  }
  const double rel_l2 = std::sqrt(err_sq / std::max(ref_sq, 1e-30));
  const bool pass = status == 0 && bad == 0 && guards_ok && deterministic;

  std::cout << (pass ? "PASS" : "FAIL") << " relL2=" << rel_l2
            << " maxAbs=" << max_abs << " shape=" << name << ' ' << config
            << " bad=" << bad << " guards=" << (guards_ok ? "ok" : "BAD")
            << " deterministic=" << (deterministic ? "yes" : "NO");
  if (first_bad >= 0)
    std::cout << " first=" << first_bad << " got=" << Info::to(got[first_bad])
              << " want=" << want[first_bad];
  std::cout << '\n';
  result.finish(rel_l2, pass);
  return pass;
}

template <typename T>
bool run_dtype(const Shape &s, uint64_t seed, int coverage,
               hipStream_t stream) {
  bool ok = true;
  // The fallbacks walk the whole tensor per group; keep them to small shapes.
  const bool all_modes = s.min_coverage < 3 || coverage < 3;
  for (int channels_last : {1, 0}) {
    for (Inputs inputs :
         {Inputs::kSmallOffset, Inputs::kLargeOffset, Inputs::kOutliers,
          Inputs::kWideRange, Inputs::kConstantGroup, Inputs::kNonFinite})
      ok = run_case<T>(s, seed, channels_last, WorkspaceMode::kFull, inputs,
                       stream) &&
           ok;
    if (!all_modes)
      continue;
    for (WorkspaceMode mode : {WorkspaceMode::kNone, WorkspaceMode::kSmall,
                               WorkspaceMode::kMisaligned})
      ok = run_case<T>(s, seed, channels_last, mode, Inputs::kSmallOffset,
                       stream) &&
           ok;
    // The fused fallback has its own variance and apply paths.
    for (Inputs inputs :
         {Inputs::kOutliers, Inputs::kConstantGroup, Inputs::kNonFinite})
      ok = run_case<T>(s, seed, channels_last, WorkspaceMode::kNone, inputs,
                       stream) &&
           ok;
  }
  return ok;
}

// A zero-sized N, C or spatial extent must succeed without touching memory.
template <typename T> bool run_empty(hipStream_t stream) {
  using Info = TypeInfo<T>;
  constexpr Shape kEmpty[] = {
      {0, 32, 16, 8, 1}, {2, 0, 16, 1, 1}, {2, 32, 0, 8, 1}};
  DeviceBytes d_y(kGuardElements * sizeof(T));
  bool ok = true;
  for (const Shape &s : kEmpty)
    for (int channels_last : {1, 0}) {
      HIP_CHECK(hipMemset(d_y.get(), kFillByte, d_y.size()));
      const size_t ws = hip_group_norm_workspace_bytes(
          s.n, s.c, s.spatial, s.groups, channels_last, Info::kDtype);
      const int status = hip_group_norm(
          stream, d_y.get(), d_y.get(), d_y.get(), d_y.get(), s.n, s.c,
          s.spatial, s.groups, channels_last, 0, 1e-5f, nullptr, 0,
          Info::kDtype);
      HIP_CHECK(hipStreamSynchronize(stream));
      std::vector<unsigned char> out(d_y.size());
      HIP_CHECK(hipMemcpy(out.data(), d_y.get(), out.size(),
                          hipMemcpyDeviceToHost));
      bool untouched = true;
      for (unsigned char b : out)
        untouched = untouched && b == kFillByte;
      const bool pass = status == 0 && untouched;
      std::cout << (pass ? "PASS" : "FAIL") << " empty shape=" << case_shape(s)
                << ' ' << Info::kName << ':'
                << (channels_last ? "nhwc" : "nchw") << " status=" << status
                << " workspace=" << ws << '\n';
      ok = ok && pass;
    }
  return ok;
}

// More than 2^31 elements, so any 32-bit offset arithmetic would wrap. The
// reference is streamed per group to keep host memory at two copies of the
// tensor in T.
template <typename T> bool run_large(int channels_last, hipStream_t stream) {
  using Info = TypeInfo<T>;
  using LD = long double;
  const Shape s = {1, 32, (int64_t(1) << 26) + 3, 8, 3};
  const int64_t elements = s.n * s.c * s.spatial;
  const int64_t cpg = s.c / s.groups;
  const float epsilon = 1e-5f;
  Rng rng(77);
  std::vector<T> hx(elements), hgamma(s.c), hbeta(s.c);
  for (int64_t c = 0; c < s.c; ++c) {
    const double shift = 4.0 * ((c / cpg) % 3) - 2.0;
    for (int64_t p = 0; p < s.spatial; ++p)
      hx[offset_of(s, channels_last, 0, c, p)] =
          Info::from(shift + rng.uniform());
    hgamma[c] = Info::from(1.0 + 0.5 * rng.uniform());
    hbeta[c] = Info::from(0.25 * rng.uniform());
  }
  DeviceBytes d_x(elements * sizeof(T)), d_y(elements * sizeof(T));
  DeviceBytes d_gamma(s.c * sizeof(T)), d_beta(s.c * sizeof(T));
  HIP_CHECK(hipMemcpy(d_x.get(), hx.data(), d_x.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_gamma.get(), hgamma.data(), d_gamma.size(),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_beta.get(), hbeta.data(), d_beta.size(),
                      hipMemcpyHostToDevice));
  const size_t ws_bytes = hip_group_norm_workspace_bytes(
      s.n, s.c, s.spatial, s.groups, channels_last, Info::kDtype);
  DeviceBytes d_ws(ws_bytes);
  const int status = hip_group_norm(
      stream, d_x.get(), d_gamma.get(), d_beta.get(), d_y.get(), s.n, s.c,
      s.spatial, s.groups, channels_last, 0, epsilon,
      ws_bytes ? d_ws.get() : nullptr, ws_bytes, Info::kDtype);
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<T> got(elements);
  HIP_CHECK(hipMemcpy(got.data(), d_y.get(), d_y.size(), hipMemcpyDeviceToHost));

  int64_t bad = 0;
  double max_abs = 0.0;
  for (int64_t g = 0; g < s.groups; ++g) {
    const LD count = static_cast<LD>(cpg * s.spatial);
    const LD shift = Info::to(hx[offset_of(s, channels_last, 0, g * cpg, 0)]);
    LD sum = 0.0L;
    for (int64_t c = g * cpg; c < (g + 1) * cpg; ++c)
      for (int64_t p = 0; p < s.spatial; ++p)
        sum += Info::to(hx[offset_of(s, channels_last, 0, c, p)]) - shift;
    const LD mean = shift + sum / count;
    LD sq = 0.0L;
    for (int64_t c = g * cpg; c < (g + 1) * cpg; ++c)
      for (int64_t p = 0; p < s.spatial; ++p) {
        const LD d = Info::to(hx[offset_of(s, channels_last, 0, c, p)]) - mean;
        sq += d * d;
      }
    const LD inv = 1.0L / std::sqrt(sq / count + epsilon);
    for (int64_t c = g * cpg; c < (g + 1) * cpg; ++c) {
      const double gamma = Info::to(hgamma[c]), beta = Info::to(hbeta[c]);
      const double cond =
          static_cast<double>(std::abs(mean) * inv) * std::abs(gamma);
      for (int64_t p = 0; p < s.spatial; ++p) {
        const int64_t o = offset_of(s, channels_last, 0, c, p);
        const double want =
            static_cast<double>((Info::to(hx[o]) - mean) * inv * gamma) + beta;
        const double v = Info::to(got[o]);
        const double d = std::isfinite(v)
                             ? std::abs(v - want)
                             : std::numeric_limits<double>::infinity();
        const double tol = Info::kAtol + Info::kRtol * std::abs(want) +
                           16.0 * Info::kComputeEps * cond;
        bad += !(d <= tol);
        max_abs = std::max(max_abs, d);
      }
    }
  }
  const bool pass = status == 0 && bad == 0;
  std::cout << (pass ? "PASS" : "FAIL") << " large shape=" << case_shape(s)
            << ' ' << Info::kName << ':' << (channels_last ? "nhwc" : "nchw")
            << " elements=" << elements << " maxAbs=" << max_abs
            << " bad=" << bad << '\n';
  return pass;
}

bool parse_options(int argc, char **argv, int *coverage, bool *large) {
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--coverage" && i + 1 < argc) {
      *coverage = std::atoi(argv[++i]);
    } else if (argument == "--large") {
      *large = true;
    } else {
      std::cerr << "usage: " << argv[0] << " [--coverage 1|2|3] [--large]\n";
      return false;
    }
  }
  return *coverage >= 1 && *coverage <= 3;
}

} // namespace

int main(int argc, char **argv) {
  int coverage = 3;
  bool large = false;
  if (!parse_options(argc, argv, &coverage, &large))
    return 2;
  std::cout << "group_norm " << TypeInfo<Elem>::kName
            << " coverage=" << coverage << '\n';

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  bool all_pass = true;
  uint64_t seed = 0;
  for (const Shape &shape : kShapes) {
    ++seed;
    if (shape.min_coverage > coverage)
      continue;
    all_pass = run_dtype<Elem>(shape, seed, coverage, stream) && all_pass;
  }
  all_pass = run_empty<Elem>(stream) && all_pass;
  if (large)
    for (int channels_last : {1, 0})
      all_pass = run_large<Elem>(channels_last, stream) && all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
}
