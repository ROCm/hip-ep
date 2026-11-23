/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Standalone unit test for hip_elementwise_binary_bcast (Add, Mul, Min, Max on
 * 4-D operands with per-axis broadcasting) in fp32, fp16 and bf16, against a
 * CPU reference that applies the same operation in fp32 and rounds to the
 * output type. Every comparison is bit-exact (a NaN matches any NaN).
 *
 * The kernel has several paths, chosen from the operand shapes, the
 * alignment of the buffers and the output volume: both operands the output's
 * shape (flat), one operand the output's shape and the other varying along a
 * single axis with a power-of-two extent below it, the same with any extent
 * below it that is a whole number of 16-byte vectors (the per-channel path),
 * and the general kernel. The shapes cover each of them for each dtype. Every
 * shape runs in both operand orders, with aligned buffers, with the dense
 * operand and with the output one element off a 16-byte boundary, and with
 * infinities and extremes in the inputs. The output has guard regions on both
 * sides that must stay untouched, and every element must be written.
 *
 * Every case that passes and is large enough is timed with the shared UT
 * timer on rotated copies of its inputs. --n --c --h --w (with optional --lhs,
 * --rhs, --op, --dtype and --align) run one given case through the same check
 * and timer.
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

constexpr const char *kLeaf = "elementwise_bcast/Xany_Wany_Yany";

constexpr int kGuardElements = 32;
// A shape this small takes a few microseconds per call. On some HIP runtimes
// the event timer then returns non-positive elapsed times, and the shared
// timer rejects the case. Smaller shapes are checked for correctness only.
constexpr int64_t kMinTimedElements = int64_t(1) << 20;
constexpr unsigned char kFillByte = 0x7f;

// op codes of hip_elementwise_binary_bcast.
constexpr int kAdd = 0, kMul = 1, kMin = 2, kMax = 3;
constexpr int kOps[] = {kAdd, kMul, kMin, kMax};

const char *op_name(int op) {
  switch (op) {
  case kAdd: return "add";
  case kMul: return "mul";
  case kMin: return "min";
  default:   return "max";
  }
}

// The operation in fp32, as the kernel evaluates it. Min and Max are the
// kernel's own (a < b ? a : b, a > b ? a : b), not a NaN-propagating minimum.
float apply(int op, float a, float b) {
  switch (op) {
  case kAdd: return a + b;
  case kMul: return a * b;
  case kMin: return a < b ? a : b;
  default:   return a > b ? a : b;
  }
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
  // Uniform in [-1, 1).
  double uniform() {
    return static_cast<double>(next() >> 11) * 0x1.0p-52 - 1.0;
  }

private:
  uint64_t state_;
};

template <typename T> struct TypeInfo;

template <> struct TypeInfo<float> {
  static constexpr int kDtype = HIP_DTYPE_FLOAT32;
  static constexpr const char *kName = "fp32";
  static constexpr float kMax = 3.4028234663852886e38f;
  static constexpr float kTiny = 1.17549435e-38f; // smallest normal
  static float from(float v) { return v; }
  static float to(float v) { return v; }
};

template <> struct TypeInfo<__half> {
  static constexpr int kDtype = HIP_DTYPE_FLOAT16;
  static constexpr const char *kName = "fp16";
  static constexpr float kMax = 65504.0f;
  static constexpr float kTiny = 6.103515625e-05f; // smallest normal
  static __half from(float v) { return __float2half(v); }
  static float to(__half v) { return __half2float(v); }
};

template <> struct TypeInfo<__hip_bfloat16> {
  static constexpr int kDtype = HIP_DTYPE_BFLOAT16;
  static constexpr const char *kName = "bf16";
  static constexpr float kMax = 3.3895313892515355e38f;
  static constexpr float kTiny = 1.17549435e-38f; // smallest normal
  static __hip_bfloat16 from(float v) { return __float2bfloat16(v); }
  static float to(__hip_bfloat16 v) { return __bfloat162float(v); }
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
      warm_timing(), kColdSettleBytes, "elementwise_bcast", "hip_elementwise_binary_bcast");
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
                 env_or("HIPDNN_RESULTS_OP", "elementwise_bcast"),
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

// A shape is two operand shapes; the output shape is their per-axis maximum.
// An axis of extent 1 broadcasts. "Dense" below means an operand with the
// output's shape.
struct CaseShape {
  const char *name;
  int64_t lhs[4];
  int64_t rhs[4];
  int min_coverage;
};

constexpr CaseShape kShapes[] = {
    // One operand dense, the other varying along one axis (a per-channel
    // bias or scale). The extent below the channel axis is noted.
    {"nchw_28x28", {2, 24, 28, 28}, {1, 24, 1, 1}, 1},     // 784: whole vectors
    {"nchw_14x14", {2, 40, 14, 14}, {1, 40, 1, 1}, 1},     // 196: fp32 only
    {"nchw_7x7", {1, 1000, 7, 7}, {1, 1000, 1, 1}, 1},     // 49: no vectors
    {"rank3_3136", {1, 3, 40, 3136}, {1, 1, 40, 1}, 1},    // axis 2, 3136
    {"nchw_56x56", {1, 64, 56, 56}, {1, 64, 1, 1}, 2},     // 3136
    {"axis2_16", {2, 8, 24, 16}, {1, 1, 24, 1}, 2},        // axis 2, 16
    {"axis0_48", {7, 2, 3, 8}, {7, 1, 1, 1}, 2},           // axis 0, 48
    {"vec_8", {3, 5, 1, 8}, {1, 5, 1, 1}, 2},              // exactly 8
    {"vec_4", {3, 5, 1, 4}, {1, 5, 1, 1}, 2},              // exactly 4
    {"pow2_64", {2, 64, 8, 8}, {1, 64, 1, 1}, 3},          // 64: power of two
    {"last_axis", {2, 3, 4, 9}, {1, 1, 1, 9}, 3},          // nothing below it
    // Other forms of the same entry point.
    {"scalar", {4, 1, 32, 32}, {1, 1, 1, 1}, 2},           // a one-element operand
    {"flat", {2, 3, 8, 8}, {2, 3, 8, 8}, 1},               // neither broadcasts
    {"both_broadcast", {2, 1, 4, 8}, {1, 5, 1, 8}, 2},     // each broadcasts
    // Large enough to be timed.
    {"nchw_28x28_big", {16, 128, 28, 28}, {1, 128, 1, 1}, 2},
    {"nchw_56x56_big", {8, 256, 56, 56}, {1, 256, 1, 1}, 3},
};

enum class Align { kAligned, kDense, kOutput };
enum class Inputs { kRandom, kExtremes };

const char *align_name(Align a) {
  switch (a) {
  case Align::kAligned: return "aligned";
  case Align::kDense:   return "dense_off";
  default:              return "output_off";
  }
}

struct Variant {
  Align align;
  Inputs inputs;
};
// The categorical situations, run on every shape at every coverage tier.
constexpr Variant kVariants[] = {
    {Align::kAligned, Inputs::kRandom},
    {Align::kAligned, Inputs::kExtremes},
    {Align::kDense, Inputs::kRandom},
    {Align::kOutput, Inputs::kRandom},
};

int64_t volume(const int64_t *d) { return d[0] * d[1] * d[2] * d[3]; }

std::string shape_string(const int64_t *d) {
  char buf[96];
  std::snprintf(buf, sizeof(buf), "%lldx%lldx%lldx%lld", (long long)d[0],
                (long long)d[1], (long long)d[2], (long long)d[3]);
  return buf;
}

// Row-major strides of an operand shape, 0 on an axis it broadcasts.
void strides_of(const int64_t *shape, int64_t *stride) {
  int64_t run = 1;
  for (int d = 3; d >= 0; --d) {
    stride[d] = shape[d] == 1 ? 0 : run;
    run *= shape[d];
  }
}

// Fills one operand: uniform values, and in the extremes variant every 7th
// element replaced by a value from a list of extremes (a NaN only where the
// op propagates one the way the reference does: Add and Mul).
template <typename T>
void fill_operand(std::vector<T> *host, std::vector<float> *value, int op,
                  Inputs inputs, Rng *rng) {
  using Info = TypeInfo<T>;
  const float inf = std::numeric_limits<float>::infinity();
  const float extremes[] = {0.0f,  -0.0f,        1.0f,           -1.0f,
                            Info::kTiny, -Info::kTiny, Info::kMax, -Info::kMax,
                            inf,   -inf,         std::nanf("")};
  const size_t usable =
      op == kAdd || op == kMul ? sizeof(extremes) / sizeof(float)
                               : sizeof(extremes) / sizeof(float) - 1;
  for (size_t i = 0; i < host->size(); ++i) {
    float v = static_cast<float>(4.0 * rng->uniform());
    if (inputs == Inputs::kExtremes && i % 7 == 3)
      v = extremes[(i / 7) % usable];
    (*host)[i] = Info::from(v);
    (*value)[i] = Info::to((*host)[i]);
  }
}

template <typename T> bool same_bits(T a, T b) {
  using Info = TypeInfo<T>;
  const float fa = Info::to(a), fb = Info::to(b);
  if (std::isnan(fa) || std::isnan(fb))
    return std::isnan(fa) && std::isnan(fb);
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// One case: shapes, op, order, alignment and inputs. `swap` exchanges the
// operands, so that the dense one is the right-hand side.
template <typename T>
bool run_case(const CaseShape &s, bool swap, int op, Align align,
              Inputs inputs, uint64_t seed, hipStream_t stream,
              bool allow_timing) {
  using Info = TypeInfo<T>;
  int64_t lhs4[4], rhs4[4], out4[4];
  for (int d = 0; d < 4; ++d) {
    lhs4[d] = swap ? s.rhs[d] : s.lhs[d];
    rhs4[d] = swap ? s.lhs[d] : s.rhs[d];
    out4[d] = std::max(lhs4[d], rhs4[d]);
  }
  const int64_t ln = volume(lhs4), rn = volume(rhs4), on = volume(out4);
  const bool lhs_dense = volume(lhs4) == on;

  char config[160];
  std::snprintf(config, sizeof(config), "fixed:%s:%s:%s:%s:%s", Info::kName,
                op_name(op), swap ? "dense_rhs" : "dense_lhs",
                align_name(align),
                inputs == Inputs::kRandom ? "random" : "extremes");
  const std::string name = shape_string(lhs4) + "+" + shape_string(rhs4);
  hipdnn_ep_test::timing::CaseScope timing_case(kLeaf, name,
                                                "hip_elementwise_binary_bcast");
  ResultRow result(name, config);

  Rng rng(seed);
  std::vector<T> hl(ln), hr(rn);
  std::vector<float> fl(ln), fr(rn);
  fill_operand(&hl, &fl, op, inputs, &rng);
  fill_operand(&hr, &fr, op, inputs, &rng);

  // The reference: the op in fp32, rounded to the output type.
  int64_t lstride[4], rstride[4];
  strides_of(lhs4, lstride);
  strides_of(rhs4, rstride);
  std::vector<T> want(on);
  {
    int64_t i = 0;
    for (int64_t a = 0; a < out4[0]; ++a)
      for (int64_t b = 0; b < out4[1]; ++b)
        for (int64_t c = 0; c < out4[2]; ++c)
          for (int64_t d = 0; d < out4[3]; ++d, ++i) {
            const int64_t li = a * lstride[0] + b * lstride[1] +
                               c * lstride[2] + d * lstride[3];
            const int64_t ri = a * rstride[0] + b * rstride[1] +
                               c * rstride[2] + d * rstride[3];
            want[i] = Info::from(apply(op, fl[li], fr[ri]));
          }
  }

  // A skew of one element puts a buffer off a 16-byte boundary. The dense
  // operand is the one the per-channel path reads in whole vectors.
  const size_t skew = sizeof(T);
  const size_t lskew = (align == Align::kDense && lhs_dense) ? skew : 0;
  const size_t rskew = (align == Align::kDense && !lhs_dense) ? skew : 0;
  const size_t oskew = align == Align::kOutput ? skew : 0;
  const size_t guard_bytes = kGuardElements * sizeof(T);
  const size_t data_bytes = on * sizeof(T);

  DeviceBytes d_l(ln * sizeof(T) + lskew), d_r(rn * sizeof(T) + rskew);
  DeviceBytes d_o(data_bytes + 2 * guard_bytes + oskew);
  HIP_CHECK(hipMemcpy(d_l.get() + lskew, hl.data(), ln * sizeof(T),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_r.get() + rskew, hr.data(), rn * sizeof(T),
                      hipMemcpyHostToDevice));
  T *l_ptr = reinterpret_cast<T *>(d_l.get() + lskew);
  T *r_ptr = reinterpret_cast<T *>(d_r.get() + rskew);
  T *o_ptr = reinterpret_cast<T *>(d_o.get() + guard_bytes + oskew);

  auto launch = [&]() {
    HIP_CHECK(hipMemset(d_o.get(), kFillByte, d_o.size()));
    return hip_elementwise_binary_bcast(stream, l_ptr, r_ptr, o_ptr, lhs4, rhs4,
                                        out4, op, Info::kDtype);
  };
  const int status = launch();
  if (status != 0) {
    std::cout << "FAIL launch status=" << status << " shape=" << name << ' '
              << config << '\n';
    return false;
  }
  HIP_CHECK(hipStreamSynchronize(stream));
  std::vector<unsigned char> out(d_o.size());
  HIP_CHECK(hipMemcpy(out.data(), d_o.get(), out.size(),
                      hipMemcpyDeviceToHost));

  bool guards_ok = true;
  const size_t tail = oskew + guard_bytes + data_bytes;
  for (size_t i = 0; i < guard_bytes; ++i)
    guards_ok = guards_ok && out[oskew + i] == kFillByte &&
                out[tail + i] == kFillByte;
  for (size_t i = 0; i < oskew; ++i)
    guards_ok = guards_ok && out[i] == kFillByte;

  std::vector<T> got(on);
  std::memcpy(got.data(), out.data() + oskew + guard_bytes, data_bytes);
  int64_t bad = 0, first_bad = -1;
  double err_sq = 0.0, ref_sq = 0.0;
  for (int64_t i = 0; i < on; ++i) {
    if (!same_bits(got[i], want[i])) {
      if (bad++ == 0)
        first_bad = i;
    }
    const double g = Info::to(got[i]), w = Info::to(want[i]);
    if (std::isfinite(g) && std::isfinite(w)) {
      err_sq += (g - w) * (g - w);
      ref_sq += w * w;
    }
  }
  const double rel_l2 = std::sqrt(err_sq / std::max(ref_sq, 1e-30));
  const bool pass = bad == 0 && guards_ok;

  std::cout << (pass ? "PASS" : "FAIL") << " relL2=" << rel_l2
            << " shape=" << name << ' ' << config << " bad=" << bad
            << " guards=" << (guards_ok ? "ok" : "BAD");
  if (first_bad >= 0)
    std::cout << " first=" << first_bad << " got=" << Info::to(got[first_bad])
              << " want=" << Info::to(want[first_bad]);
  std::cout << '\n';
  result.finish(rel_l2, pass);

  // Timed after the check, so the timed launches never feed the comparison.
  // Aligned buffers only: the timed path is the common one.
  if (pass && allow_timing && align == Align::kAligned &&
      on >= kMinTimedElements) {
    ColdCopies copies({{d_l.get(), ln * sizeof(T)}, {d_r.get(), rn * sizeof(T)}});
    result.row.time_ms = rotated_mean_ms(stream, copies, [&](int copy) {
      return hip_elementwise_binary_bcast(stream, copies.part(copy, 0),
                                          copies.part(copy, 1), o_ptr, lhs4,
                                          rhs4, out4, op, Info::kDtype);
    });
  }
  return pass;
}

// The contract at the edges: an empty output launches nothing and succeeds,
// an output beyond 32-bit indexing is declined (-2) before any access, and an
// unsupported dtype is refused (-1).
template <typename T> bool run_contract(hipStream_t stream) {
  using Info = TypeInfo<T>;
  bool ok = true;
  {
    const int64_t l[4] = {0, 3, 4, 4}, r[4] = {1, 3, 1, 1}, o[4] = {0, 3, 4, 4};
    const int status =
        hip_elementwise_binary_bcast(stream, nullptr, nullptr, nullptr, l, r, o,
                                     kAdd, Info::kDtype);
    ok = ok && status == 0;
    std::cout << (status == 0 ? "PASS" : "FAIL") << " contract empty "
              << Info::kName << " status=" << status << '\n';
  }
  {
    const int64_t l[4] = {65536, 65536, 2, 1}, r[4] = {1, 65536, 1, 1},
                  o[4] = {65536, 65536, 2, 1};
    const int status =
        hip_elementwise_binary_bcast(stream, nullptr, nullptr, nullptr, l, r, o,
                                     kAdd, Info::kDtype);
    ok = ok && status == -2;
    std::cout << (status == -2 ? "PASS" : "FAIL") << " contract oversize "
              << Info::kName << " status=" << status << '\n';
  }
  {
    const int64_t l[4] = {1, 2, 2, 2}, r[4] = {1, 2, 1, 1}, o[4] = {1, 2, 2, 2};
    DeviceBytes d(64 * sizeof(float));
    const int status = hip_elementwise_binary_bcast(
        stream, d.get(), d.get(), d.get(), l, r, o, kAdd, HIP_DTYPE_INT32);
    ok = ok && status == -1;
    std::cout << (status == -1 ? "PASS" : "FAIL")
              << " contract unsupported dtype status=" << status << '\n';
  }
  return ok;
}

template <typename T>
bool run_dtype(const CaseShape &s, uint64_t seed, hipStream_t stream) {
  const bool timed_shape = volume(s.lhs) >= kMinTimedElements ||
                           volume(s.rhs) >= kMinTimedElements;
  bool ok = true;
  for (bool swap : {false, true})
    for (int op : kOps)
      for (const Variant &v : kVariants) {
        // A shape large enough to be timed runs the aligned, random variant
        // only: the others would repeat a long CPU reference for no new path.
        if (timed_shape &&
            !(v.align == Align::kAligned && v.inputs == Inputs::kRandom))
          continue;
        ok = run_case<T>(s, swap, op, v.align, v.inputs, ++seed, stream,
                         true) &&
             ok;
      }
  return ok;
}

// One axis of a custom operand: full is the output's extent on every axis.
struct Form {
  const char *name;
  bool keep[4]; // axis kept (not broadcast)
};
constexpr Form kForms[] = {
    {"full", {true, true, true, true}},
    {"n", {true, false, false, false}},
    {"c", {false, true, false, false}},
    {"h", {false, false, true, false}},
    {"w", {false, false, false, true}},
    {"scalar", {false, false, false, false}},
};

const Form *find_form(const std::string &name) {
  for (const Form &f : kForms)
    if (name == f.name)
      return &f;
  return nullptr;
}

enum class DType { kFp32, kFp16, kBf16 };

struct Options {
  int coverage = 3;
  // One explicit case (make test_custom): --n --c --h --w.
  bool custom = false;
  int64_t out[4] = {0, 0, 0, 0};
  const Form *lhs = &kForms[0];
  const Form *rhs = &kForms[2];
  int op = kAdd;
  DType dtype = DType::kFp16;
  Align align = Align::kAligned;
};

void usage(const char *exe) {
  std::cerr << "usage: " << exe << " [--coverage 1|2|3]\n"
            << "       " << exe
            << " --n N --c C --h H --w W [--lhs full|n|c|h|w|scalar]"
               " [--rhs full|n|c|h|w|scalar] [--op add|mul|min|max]"
               " [--dtype fp32|fp16|bf16] [--align aligned|dense|output]\n";
}

// Returns false on an unknown, repeated, incomplete or unsupported argument.
bool parse_options(int argc, char **argv, Options *o) {
  std::vector<std::string> seen;
  const char *dim_flags[] = {"--n", "--c", "--h", "--w"};
  bool have_dim[4] = {false, false, false, false};
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (std::find(seen.begin(), seen.end(), argument) != seen.end())
      return false;
    seen.push_back(argument);
    if (i + 1 >= argc)
      return false;
    const std::string value = argv[++i];
    bool matched = false;
    for (int d = 0; d < 4; ++d)
      if (argument == dim_flags[d]) {
        char *end = nullptr;
        o->out[d] = std::strtoll(value.c_str(), &end, 10);
        if (end == value.c_str() || *end != '\0')
          return false;
        have_dim[d] = matched = o->custom = true;
      }
    if (matched)
      continue;
    if (argument == "--coverage") {
      o->coverage = std::atoi(value.c_str());
    } else if (argument == "--lhs" || argument == "--rhs") {
      const Form *f = find_form(value);
      if (!f)
        return false;
      (argument == "--lhs" ? o->lhs : o->rhs) = f;
      o->custom = true;
    } else if (argument == "--op") {
      if (value == "add")
        o->op = kAdd;
      else if (value == "mul")
        o->op = kMul;
      else if (value == "min")
        o->op = kMin;
      else if (value == "max")
        o->op = kMax;
      else
        return false;
      o->custom = true;
    } else if (argument == "--dtype") {
      if (value == "fp32")
        o->dtype = DType::kFp32;
      else if (value == "fp16")
        o->dtype = DType::kFp16;
      else if (value == "bf16")
        o->dtype = DType::kBf16;
      else
        return false;
      o->custom = true;
    } else if (argument == "--align") {
      if (value == "aligned")
        o->align = Align::kAligned;
      else if (value == "dense")
        o->align = Align::kDense;
      else if (value == "output")
        o->align = Align::kOutput;
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
  for (int d = 0; d < 4; ++d) {
    if (!have_dim[d] || o->out[d] < 1)
      return false;
    // An axis that both operands broadcast has the extent 1.
    if (!o->lhs->keep[d] && !o->rhs->keep[d] && o->out[d] != 1)
      return false;
  }
  // The output must be addressable by the kernel's 32-bit index.
  return volume(o->out) < (int64_t(1) << 32);
}

template <typename T>
bool run_custom(const Options &o, hipStream_t stream) {
  CaseShape s{"custom", {0, 0, 0, 0}, {0, 0, 0, 0}, 1};
  for (int d = 0; d < 4; ++d) {
    s.lhs[d] = o.lhs->keep[d] ? o.out[d] : 1;
    s.rhs[d] = o.rhs->keep[d] ? o.out[d] : 1;
  }
  return run_case<T>(s, false, o.op, o.align, Inputs::kRandom, 1, stream, true);
}

} // namespace

int main(int argc, char **argv) try {
  Options options;
  if (!parse_options(argc, argv, &options)) {
    usage(argv[0]);
    return 2;
  }
  const int coverage = options.coverage;
  hipdnn_ep_test::timing::printTimingMode();

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  if (options.custom) {
    bool pass = false;
    switch (options.dtype) {
    case DType::kFp32: pass = run_custom<float>(options, stream); break;
    case DType::kFp16: pass = run_custom<__half>(options, stream); break;
    case DType::kBf16: pass = run_custom<__hip_bfloat16>(options, stream); break;
    }
    HIP_CHECK(hipStreamDestroy(stream));
    std::cout << (pass ? "ALL PASS" : "SOME FAIL") << '\n';
    return pass ? 0 : 1;
  }

  int shapes = 0;
  for (const CaseShape &s : kShapes)
    shapes += s.min_coverage <= coverage;
  std::cout << "elementwise_bcast coverage=" << coverage << " -> running "
            << shapes << " typical shape(s) x 3 dtype(s) x 2 operand order(s) x "
            << "4 op(s) x " << sizeof(kVariants) / sizeof(kVariants[0])
            << " alignment/input variant(s)\n";

  bool all_pass = true;
  uint64_t seed = 0;
  for (const CaseShape &s : kShapes) {
    seed += 1000;
    if (s.min_coverage > coverage)
      continue;
    all_pass = run_dtype<float>(s, seed, stream) && all_pass;
    all_pass = run_dtype<__half>(s, seed + 1, stream) && all_pass;
    all_pass = run_dtype<__hip_bfloat16>(s, seed + 2, stream) && all_pass;
  }
  all_pass = run_contract<float>(stream) && all_pass;
  all_pass = run_contract<__half>(stream) && all_pass;
  all_pass = run_contract<__hip_bfloat16>(stream) && all_pass;
  HIP_CHECK(hipStreamDestroy(stream));

  std::cout << (all_pass ? "ALL PASS" : "SOME FAIL") << '\n';
  return all_pass ? 0 : 1;
} catch (const std::exception &error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 2;
}
