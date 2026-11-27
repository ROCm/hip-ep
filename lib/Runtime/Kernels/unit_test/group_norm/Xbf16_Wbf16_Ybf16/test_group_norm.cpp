/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for com.microsoft GroupNorm (hip_group_norm) with
 * bf16 input, scale, bias and output, both layouts, with and without SiLU,
 * against a CPU fp64 reference computed from the rounded inputs.
 *
 * Every shape also runs without a workspace (the fused path) and with a
 * workspace too small for the planned split (a shrunk split), with misaligned
 * tensors (scalar access), and twice with the same workspace to check that
 * the output is bitwise deterministic.
 *
 * Every case that passes is timed with the shared UT timer on rotated copies
 * of its inputs. --n --c --spatial --groups (with optional --layout, --silu
 * and --workspace) run one given case through the same check and timer.
 */

#include "hip_custom_kernels.h"

#include "../../common/hip_adaptive_timing.h"

#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
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
using Elem = __hip_bfloat16;
constexpr const char *kLeaf = "group_norm/Xbf16_Wbf16_Ybf16";

constexpr int kGuardElements = 32;
// A shape this small takes a few microseconds per call. On some HIP runtimes
// the event timer then returns non-positive elapsed times, and the shared
// timer rejects the case. Smaller shapes are checked for correctness only.
constexpr int64_t kMinTimedElements = int64_t(1) << 20;
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

// Shared UT timer. The timed launches read rotated copies of the inputs, so
// that each copy has left the caches by its next read; HIPDNN_KERNEL_UT_WARM=1
// times one copy back to back instead.
constexpr size_t kColdRotationBytes = size_t(256) << 20;
// Untimed reads before the timed launches, so that the copies' initial fill
// has left the caches.
constexpr size_t kColdSettleBytes = size_t(128) << 20;
constexpr int kColdMaxCopies = 4096;

bool warm_timing() {
  const char *v = std::getenv("HIPDNN_KERNEL_UT_WARM");
  return v && v[0] == '1';
}

// Copies of the buffers one launch reads. Copy 0 is the buffers themselves;
// the others share one allocation.
class ColdCopies {
public:
  struct Part {
    const void *src;
    size_t bytes;
  };

  explicit ColdCopies(std::initializer_list<Part> parts) : parts_(parts) {
    size_t stride = 0;
    for (const Part &p : parts_) {
      offsets_.push_back(stride);
      stride += (p.bytes + 255) & ~size_t(255);
      bytes_ += p.bytes;
    }
    if (warm_timing() || bytes_ == 0)
      return;
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
    for (size_t i = 0; i < parts_.size(); ++i)
      if (parts_[i].bytes)
        HIP_CHECK(hipMemcpy(part(1, i), parts_[i].src, parts_[i].bytes,
                            hipMemcpyDeviceToDevice));
    for (int filled = 1; filled < count_ - 1;) {
      const int more = std::min(filled, count_ - 1 - filled);
      HIP_CHECK(hipMemcpy(static_cast<char *>(slab_) + stride_ * filled, slab_,
                          stride_ * more, hipMemcpyDeviceToDevice));
      filled += more;
    }
  }
  ~ColdCopies() {
    if (slab_)
      (void)hipFree(slab_);
  }
  ColdCopies(const ColdCopies &) = delete;
  ColdCopies &operator=(const ColdCopies &) = delete;

  int count() const { return count_; }
  size_t bytes() const { return bytes_; }
  // Part `i` of copy `c`.
  char *part(int c, size_t i) const {
    if (c == 0 || parts_[i].bytes == 0)
      return static_cast<char *>(const_cast<void *>(parts_[i].src));
    return static_cast<char *>(slab_) + stride_ * (c - 1) + offsets_[i];
  }

private:
  std::vector<Part> parts_;
  std::vector<size_t> offsets_;
  size_t bytes_ = 0;
  size_t stride_ = 0;
  int count_ = 1;
  void *slab_ = nullptr;
};

// Holds the stream until the host sets *release, so that every timed launch is
// queued before the first one runs and the reading is the GPU's time rather
// than the host's submission rate. Gives up after about 0.25 s in case the
// queue cannot take all the launches.
__global__ void hold_stream_kernel(const volatile int *release) {
  for (int spin = 0; spin < (1 << 16) && *release == 0; ++spin)
    __builtin_amdgcn_s_sleep(127);
}

// Host-mapped release flag for hold_stream_kernel, or null if unavailable.
// Coherent, so the spinning kernel sees the host's write without a flush.
int *hold_release_flag() {
  static int *flag = [] {
    void *p = nullptr;
    return hipHostMalloc(&p, sizeof(int),
                         hipHostMallocMapped | hipHostMallocCoherent) ==
                   hipSuccess
               ? static_cast<int *>(p)
               : nullptr;
  }();
  return flag;
}

// Mean milliseconds of the complete op; launch(c) reads operand copy c.
template <class Launch>
double rotated_mean_ms(hipStream_t stream, const ColdCopies &copies,
                       Launch &&launch) {
  return hipdnn_ep_test::timing::adaptiveRotatedMeanMs(
      stream, copies, launch, hold_release_flag(),
      [&](const int *flag) { hold_stream_kernel<<<1, 1, 0, stream>>>(flag); },
      warm_timing(), kColdSettleBytes, "group_norm", "hip_group_norm");
}

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
    hipdnn_ep_test::timing::emitCsvBinding(r.shape, r.config, r.verdict,
                                           r.rel_l2, r.time_ms);
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
              WorkspaceMode mode, Inputs inputs, int activation,
              hipStream_t stream) {
  using Info = TypeInfo<T>;
  const float epsilon = 1e-5f;
  // The op has no tuned configurations: every row names the fixed path.
  char config[128];
  std::snprintf(config, sizeof(config), "fixed:%s:%s:%s:%s:silu%d",
                Info::kName, channels_last ? "nhwc" : "nchw", mode_name(mode),
                inputs_name(inputs), activation);
  const std::string name = case_shape(s);
  hipdnn_ep_test::timing::CaseScope timing_case(kLeaf, name, "hip_group_norm");
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
  // Timed after the check, so the timed launches never feed the comparison.
  // The copies keep the skew of a misaligned case.
  if (pass && s.n * s.c * s.spatial >= kMinTimedElements) {
    ColdCopies copies({{d_x.get(), data_bytes + skew},
                       {d_gamma.get(), s.c * sizeof(T)},
                       {d_beta.get(), s.c * sizeof(T)}});
    result.row.time_ms = rotated_mean_ms(stream, copies, [&](int copy) {
      return hip_group_norm(stream, copies.part(copy, 0) + skew,
                            copies.part(copy, 1), copies.part(copy, 2), y_ptr,
                            s.n, s.c, s.spatial, s.groups, channels_last,
                            activation, epsilon,
                            ws_bytes ? d_ws.get() : nullptr, ws_bytes,
                            Info::kDtype);
    });
  }
  return pass;
}

template <typename T>
bool run_dtype(const Shape &s, uint64_t seed, int coverage,
               hipStream_t stream) {
  bool ok = true;
  const int activation = static_cast<int>(seed & 1);
  // The fallbacks walk the whole tensor per group; keep them to small shapes.
  const bool all_modes = s.min_coverage < 3 || coverage < 3;
  for (int channels_last : {1, 0}) {
    for (Inputs inputs :
         {Inputs::kSmallOffset, Inputs::kLargeOffset, Inputs::kOutliers,
          Inputs::kWideRange, Inputs::kConstantGroup, Inputs::kNonFinite})
      ok = run_case<T>(s, seed, channels_last, WorkspaceMode::kFull, inputs,
                       activation, stream) &&
           ok;
    if (!all_modes)
      continue;
    for (WorkspaceMode mode : {WorkspaceMode::kNone, WorkspaceMode::kSmall,
                               WorkspaceMode::kMisaligned})
      ok = run_case<T>(s, seed, channels_last, mode, Inputs::kSmallOffset,
                       activation, stream) &&
           ok;
    // The fused fallback has its own variance and apply paths.
    for (Inputs inputs :
         {Inputs::kOutliers, Inputs::kConstantGroup, Inputs::kNonFinite})
      ok = run_case<T>(s, seed, channels_last, WorkspaceMode::kNone, inputs,
                       activation, stream) &&
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

// More than 2^31 elements, so any 32-bit offset arithmetic would wrap.
constexpr Shape kLargeShape = {1, 32, (int64_t(1) << 26) + 3, 8, 3};

// The suite caps the channels-first apply grid at kApplyMaxBlocks blocks,
// above every kShapes entry. This shape's odd spatial forces one value per
// vector, so its apply needs kApplyMaxBlocks + 1 blocks and runs the
// grid-stride kernel that serves tensors past the 32-bit launch limit.
constexpr int64_t kApplyMaxBlocks = int64_t(1) << 17;
constexpr Shape kGridStrideShape = {1, 32, (int64_t(1) << 20) + 3, 8, 1};

// The reference is streamed per group to keep host memory at two copies of
// the tensor in T.
template <typename T>
bool run_large(const Shape &s, const char *label, int channels_last,
               hipStream_t stream) {
  using Info = TypeInfo<T>;
  using LD = long double;
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
  std::cout << (pass ? "PASS" : "FAIL") << ' ' << label
            << " shape=" << case_shape(s)
            << ' ' << Info::kName << ':' << (channels_last ? "nhwc" : "nchw")
            << " elements=" << elements << " maxAbs=" << max_abs
            << " bad=" << bad << '\n';
  return pass;
}

struct Options {
  int coverage = 3;
  bool large = false;
  // One explicit case (make test_custom): --n --c --spatial --groups.
  bool custom = false;
  Shape shape = {0, 0, 0, 0, 1};
  int channels_last = 0;
  int activation = 0;
  WorkspaceMode mode = WorkspaceMode::kFull;
};

void usage(const char *exe) {
  std::cerr << "usage: " << exe << " [--coverage 1|2|3] [--large]\n"
            << "       " << exe
            << " --n N --c C --spatial HW --groups G [--layout nchw|nhwc]"
               " [--silu 0|1] [--workspace split|none|small|misaligned]\n";
}

// Returns false on an unknown, repeated, incomplete or unsupported argument.
bool parse_options(int argc, char **argv, Options *o) {
  std::vector<std::string> seen;
  int64_t *dims[] = {&o->shape.n, &o->shape.c, &o->shape.spatial,
                     &o->shape.groups};
  const char *dim_flags[] = {"--n", "--c", "--spatial", "--groups"};
  bool have_dim[4] = {false, false, false, false};
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (std::find(seen.begin(), seen.end(), argument) != seen.end())
      return false;
    seen.push_back(argument);
    if (argument == "--large") {
      o->large = true;
      continue;
    }
    if (i + 1 >= argc)
      return false;
    const std::string value = argv[++i];
    bool matched = false;
    for (int d = 0; d < 4; ++d)
      if (argument == dim_flags[d]) {
        char *end = nullptr;
        *dims[d] = std::strtoll(value.c_str(), &end, 10);
        if (end == value.c_str() || *end != '\0')
          return false;
        have_dim[d] = matched = o->custom = true;
      }
    if (matched)
      continue;
    if (argument == "--coverage") {
      o->coverage = std::atoi(value.c_str());
    } else if (argument == "--layout") {
      if (value != "nchw" && value != "nhwc")
        return false;
      o->channels_last = value == "nhwc";
      o->custom = true;
    } else if (argument == "--silu") {
      if (value != "0" && value != "1")
        return false;
      o->activation = value == "1";
      o->custom = true;
    } else if (argument == "--workspace") {
      if (value == "split")
        o->mode = WorkspaceMode::kFull;
      else if (value == "none")
        o->mode = WorkspaceMode::kNone;
      else if (value == "small")
        o->mode = WorkspaceMode::kSmall;
      else if (value == "misaligned")
        o->mode = WorkspaceMode::kMisaligned;
      else
        return false;
      o->custom = true;
    } else {
      return false;
    }
  }
  if (o->coverage < 1 || o->coverage > 3)
    return false;
  if (!o->custom)
    return true;
  const Shape &s = o->shape;
  return have_dim[0] && have_dim[1] && have_dim[2] && have_dim[3] &&
         s.n >= 1 && s.c >= 1 && s.spatial >= 1 && s.groups >= 1 &&
         s.c % s.groups == 0;
}

} // namespace

int main(int argc, char **argv) try {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    usage(argv[0]);
    return 2;
  }
  const int coverage = options.coverage;
  std::cout << "group_norm " << TypeInfo<Elem>::kName
            << " coverage=" << coverage << '\n';
  hipdnn_ep_test::timing::printTimingMode();

  // The kernel reads the cap once, so it is set before the first launch. The
  // --large and custom runs keep the default to time the production path.
  const bool grid_stride = !options.custom && !options.large;
  if (grid_stride) {
    const std::string cap = std::to_string(kApplyMaxBlocks);
#ifdef _WIN32
    _putenv_s("HIPDNN_EP_GROUPNORM_APPLY_MAX_BLOCKS", cap.c_str());
#else
    setenv("HIPDNN_EP_GROUPNORM_APPLY_MAX_BLOCKS", cap.c_str(), 1);
#endif
  }

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  if (options.custom) {
    const bool pass = run_case<Elem>(options.shape, 1, options.channels_last,
                                     options.mode, Inputs::kSmallOffset,
                                     options.activation, stream);
    HIP_CHECK(hipStreamDestroy(stream));
    std::cout << (pass ? "ALL PASS" : "SOME FAIL") << '\n';
    return pass ? 0 : 1;
  }
  bool all_pass = true;
  uint64_t seed = 0;
  for (const Shape &shape : kShapes) {
    ++seed;
    if (shape.min_coverage > coverage)
      continue;
    all_pass = run_dtype<Elem>(shape, seed, coverage, stream) && all_pass;
  }
  all_pass = run_empty<Elem>(stream) && all_pass;
  if (grid_stride)
    all_pass =
        run_large<Elem>(kGridStrideShape, "grid-stride", 0, stream) && all_pass;
  if (options.large)
    for (int channels_last : {1, 0})
      all_pass =
          run_large<Elem>(kLargeShape, "large", channels_last, stream) &&
          all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
} catch (const std::exception &error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 2;
}
