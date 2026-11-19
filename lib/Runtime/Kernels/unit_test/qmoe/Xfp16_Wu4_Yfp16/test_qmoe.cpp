/* Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License. */
#include "hip_custom_kernels.h"
#include <hip/hip_runtime.h>
#include "../../common/hip_adaptive_timing.h"
#include "../../common/ut_gpu_execution_scope.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
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
    if (!f_)
      throw std::runtime_error("cannot open requested results CSV");
    if (f_ && need_header) {
      std::fprintf(f_,
                   "op,leaf,arch,mode,shape,config,time_ms,relL2,verdict\n");
      flush();
    }
  }
  ~CsvWriter() {
    if (f_)
      std::fclose(f_);
  }
  void write(const CsvRow &r) {
    timing::emitCsvBinding(r.shape, r.config, r.verdict, r.rel_l2, r.time_ms);
    if (!f_)
      return;
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
  static const char *env_or(const char *name, const char *dflt) {
    const char *v = std::getenv(name);
    return (v && v[0]) ? v : dflt;
  }
  FILE *f_ = nullptr;
};
} // namespace hipdnn_ep_test

// ---- host fp16 conversions (round to nearest even), no compiler-rt needed
// ----
static uint16_t float_to_half_bits(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000u;
  const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF)
    return static_cast<uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0u));
  if (exp >= 31)
    return static_cast<uint16_t>(sign | 0x7C00u);
  if (exp <= 0) {
    if (exp < -10)
      return static_cast<uint16_t>(sign);
    mant |= 0x800000u;
    const int shift = 14 - exp;
    uint32_t half = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u), mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1u)))
      ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
  const uint32_t rem = mant & 0x1FFFu;
  if (rem > 0x1000u || (rem == 0x1000u && (half & 1u)))
    ++half;
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
      while (!(mant & 0x400u)) {
        mant <<= 1;
        --exp;
      }
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
template <class Fn> static void parallel_for(int64_t n, Fn &&fn) {
  const int64_t nt = std::max<int64_t>(
      1, (std::thread::hardware_concurrency() < 16 ? std::thread::hardware_concurrency() : 16));
  if (n < 4096 || nt == 1) {
    for (int64_t i = 0; i < n; ++i)
      fn(i);
    return;
  }
  std::vector<std::thread> pool;
  const int64_t chunk = (n + nt - 1) / nt;
  for (int64_t t = 0; t < nt; ++t) {
    const int64_t b = t * chunk, e = n < b + chunk ? n : b + chunk;
    if (b >= e)
      break;
    pool.emplace_back([&fn, b, e] {
      for (int64_t i = b; i < e; ++i)
        fn(i);
    });
  }
  for (auto &th : pool)
    th.join();
}

template <class T> static T *to_device(const std::vector<T> &h) {
  T *d = nullptr;
  HIP_CHECK(hipMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
  if (!h.empty())
    HIP_CHECK(
        hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  return d;
}

template <class T> static std::vector<T> from_device(const T *d, int64_t n) {
  std::vector<T> h(static_cast<size_t>(n));
  if (n > 0)
    HIP_CHECK(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost));
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
    if (warmTiming() || bytes_ == 0)
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
    // Copy 1 from the buffers, then double the run of filled copies.
    for (size_t i = 0; i < parts_.size(); ++i)
      if (parts_[i].bytes)
        (void)hipMemcpy(part(1, i), parts_[i].src, parts_[i].bytes,
                        hipMemcpyDeviceToDevice);
    for (int filled = 1; filled < count_ - 1;) {
      const int more = std::min(filled, count_ - 1 - filled);
      (void)hipMemcpy(static_cast<char *>(slab_) + stride_ * filled, slab_,
                      stride_ * more, hipMemcpyDeviceToDevice);
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
  void *part(int c, size_t i) const {
    if (c == 0 || parts_[i].bytes == 0)
      return const_cast<void *>(parts_[i].src);
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
extern "C" void qmoeUtHoldStream(hipStream_t stream, const int *release);

// Host-mapped release flag for holdStreamKernel, or null if it is unavailable.
// Coherent, so the spinning kernel sees the host's write without a flush.
inline int *holdReleaseFlag() {
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

// Mean milliseconds per launch over `iters` back-to-back launches on `stream`,
// launch(c) reading copy c. Untimed launches over the rotation go first, until
// kColdSettleBytes have been read.
template <class Launch>
double rotatedMeanMs(hipStream_t stream, int /*legacy_iters*/, const ColdCopies &copies,
                     Launch &&launch) {
  return timing::adaptiveRotatedMeanMs(
      stream, copies, launch, holdReleaseFlag(),
      [&](const int *flag) { qmoeUtHoldStream(stream, flag); },
      warmTiming(), kColdSettleBytes, "qmoe_fp16_wu4", "wrap_qmoe");
}

} // namespace hipdnn_ep_test

static void report(const char *label, const std::string &shape,
                   const std::string &config, int status, double ms,
                   double rel_l2, bool pass) {
  hipdnn_ep_test::CsvWriter csv;
  hipdnn_ep_test::CsvRow row;
  row.shape = shape;
  row.config = config;
  row.time_ms = ms;
  row.rel_l2 = rel_l2;
  row.verdict = pass ? "PASS" : "FAIL";
  csv.write(row);
  std::printf("  %-10s %-36s status=%d  %.4f ms  relL2=%.3e  %s\n", label,
              shape.c_str(), status, ms, rel_l2,
              pass ? "PASS" : "*** FAIL ***");
}

static const char *arg_str(int argc, char **argv, const char *name) {
  for (int i = 1; i + 1 < argc; ++i)
    if (std::strcmp(argv[i], name) == 0)
      return argv[i + 1];
  return nullptr;
}

static double arg_f64(int argc, char **argv, const char *name, double dflt) {
  const char *v = arg_str(argc, argv, name);
  return v ? std::atof(v) : dflt;
}

// "1x1024x8192" -> {1, 1024, 8192}; "" or "scalar" -> {}
static std::vector<int64_t> parse_shape(const char *s) {
  std::vector<int64_t> d;
  if (!s || !*s || std::strcmp(s, "scalar") == 0)
    return d;
  const char *p = s;
  while (*p) {
    d.push_back(std::atoll(p));
    while (*p && *p != 'x')
      ++p;
    if (*p == 'x')
      ++p;
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

static int64_t integer(const char *text) {
  char *end = nullptr;
  errno = 0;
  const long long value = std::strtoll(text, &end, 10);
  if (errno || !text[0] || !end || *end)
    throw std::invalid_argument("invalid integer argument");
  return static_cast<int64_t>(value);
}

static bool options(int argc, char **argv,
                    std::initializer_list<const char *> allowed) {
  std::set<std::string> seen;
  for (int i = 1; i < argc; i += 2) {
    bool known = false;
    for (const char *name : allowed)
      if (std::strcmp(argv[i], name) == 0)
        known = true;
    if (!known || i + 1 >= argc || !seen.insert(argv[i]).second) {
      std::fprintf(stderr, "unknown, duplicate or incomplete argument: %s\n",
                   argv[i]);
      return false;
    }
  }
  return true;
}

static int64_t opt(int argc, char **argv, const char *key, int64_t fallback) {
  const char *value = arg_str(argc, argv, key);
  return value ? integer(value) : fallback;
}

template <class T> static T number(double x) { return static_cast<T>(x); }
template <> uint16_t number<uint16_t>(double x) {
  return float_to_half_bits(static_cast<float>(x));
}
template <class T> static double real(T x) { return static_cast<double>(x); }
template <> double real<uint16_t>(uint16_t x) { return half_bits_to_float(x); }

template <class T>
static std::vector<T> values(int64_t n, unsigned seed, float lo, float hi) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> distribution(lo, hi);
  std::vector<T> result(static_cast<size_t>(n));
  for (T &value : result)
    value = number<T>(distribution(rng));
  return result;
}

static bool captureOnly() {
  const char *value = std::getenv("HIPDNN_UT_CAPTURE_ONLY");
  return value && std::strcmp(value, "1") == 0;
}

template <class Launch>
static double
completeCalls(hipStream_t stream, int iters,
              std::initializer_list<hipdnn_ep_test::ColdCopies::Part> parts,
              Launch &&launch, int &status) {
  if (captureOnly()) {
    std::fprintf(stderr, "HIPDNN_UT_TIMING begin calls=1 role=reported\n");
    std::fflush(stderr);
    launch(nullptr, 0);
    HIP_CHECK(hipStreamSynchronize(stream));
    std::fprintf(stderr, "HIPDNN_UT_TIMING end calls=1 role=reported\n");
    std::fflush(stderr);
    if (!status)
      std::puts("UT_CAPTURE_ONLY complete_calls=1 timing=disabled "
                "stream_gate=disabled");
    return 0.0;
  }
  for (int i = 0; i < 3; ++i)
    launch(nullptr, 0);
  HIP_CHECK(hipStreamSynchronize(stream));
  if (status)
    return 0.0;
  const hipdnn_ep_test::ColdCopies copies(parts);
  const double ms = hipdnn_ep_test::rotatedMeanMs(
      stream, iters, copies, [&](int i) { launch(&copies, i); return status; });
  HIP_CHECK(hipStreamSynchronize(stream));
  return ms;
}

static int64_t volume(const std::vector<int64_t> &shape) {
  int64_t n = 1;
  if (shape.size() > 8)
    throw std::invalid_argument("rank exceeds 8");
  for (int64_t d : shape) {
    if (d <= 0 || n > INT_MAX / d)
      throw std::invalid_argument("nonpositive or oversized shape");
    n *= d;
  }
  return n;
}

static std::vector<int64_t> shapeArg(const char *value) {
  if (!value || !*value)
    throw std::invalid_argument("missing shape");
  if (std::strcmp(value, "scalar") == 0)
    return {};
  std::vector<int64_t> result;
  const char *p = value;
  while (*p) {
    char *end = nullptr;
    errno = 0;
    const long long d = std::strtoll(p, &end, 10);
    if (errno || end == p || d <= 0 || (*end && *end != 'x'))
      throw std::invalid_argument("malformed shape");
    result.push_back(d);
    if (!*end)
      break;
    p = end + 1;
    if (!*p)
      throw std::invalid_argument("trailing shape separator");
  }
  volume(result);
  return result;
}

static std::string shapeText(const std::vector<int64_t> &shape) {
  std::string text;
  for (int64_t d : shape)
    text += (text.empty() ? "" : "x") + std::to_string(d);
  return text.empty() ? "scalar" : text;
}

static std::vector<int64_t> broadcast(const std::vector<int64_t> &a,
                                      const std::vector<int64_t> &b) {
  const size_t rank = std::max(a.size(), b.size());
  std::vector<int64_t> out(rank, 1);
  for (size_t i = 0; i < rank; ++i) {
    const int64_t x = i < a.size() ? a[a.size() - 1 - i] : 1;
    const int64_t y = i < b.size() ? b[b.size() - 1 - i] : 1;
    if (x != y && x != 1 && y != 1)
      throw std::invalid_argument("incompatible broadcast");
    out[rank - 1 - i] = std::max(x, y);
  }
  volume(out);
  return out;
}

static int64_t broadcastIndex(int64_t index, const std::vector<int64_t> &out,
                              const std::vector<int64_t> &in) {
  int64_t result = 0, stride = 1;
  for (size_t i = 0; i < out.size(); ++i) {
    const int64_t coord = index % out[out.size() - 1 - i];
    index /= out[out.size() - 1 - i];
    if (i < in.size()) {
      const int64_t d = in[in.size() - 1 - i];
      if (d != 1)
        result += coord * stride;
      stride *= d;
    }
  }
  return result;
}

static int coverage(int argc, char **argv) {
  const char *value = arg_str(argc, argv, "--coverage");
  if (!value)
    value = std::getenv("HIPDNN_UT_COVERAGE");
  const int64_t tier = value ? integer(value) : 3;
  if (tier < 1 || tier > 3)
    throw std::invalid_argument("coverage must be 1, 2 or 3");
  return static_cast<int>(tier);
}

static int iterations(int argc, char **argv) {
  const int64_t n = opt(argc, argv, "--iters", 20);
  if (n < 1 || n > 100000)
    throw std::invalid_argument("invalid iteration count");
  return static_cast<int>(n);
}

static bool passedTime(int status, bool correct, double ms) {
  return !status && correct && std::isfinite(ms) && (captureOnly() || ms > 0);
}

static void deviceInfo(const char *label) {
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("%s; GPU=%s arch=%s; owning inline cold rotation; default "
              "iterations=20\n",
              label, prop.name, prop.gcnArchName);
}

#ifndef HIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS
#include "matmul_nbits_autotune.h"

#ifndef HIPDNN_LUT_LINKED_EXTERNALLY
// MODE=autotune, and MODE=lookup on a chip with no table: no FlatBuffers LUT is
// linked. Empty resolve() lets the kernel link and fall back to its runtime
// sweep, and extendedConfigsMeasured() is false, as on a device whose table
// names no extended entry, so that sweep times the original entries.
// MODE=lookup links the real matmul_nbits_autotune.cpp resolver instead (see
// Makefile), which defines these symbols, so this stub must not also define
// them.
namespace hipdnn_ep {
namespace matmul_nbits_autotune {
Result resolve(const Request &, WmmaValidator, GemvValidator, void *) {
  return {};
}
Stats stats() { return {}; }
bool extendedConfigsMeasured() { return false; }
} // namespace matmul_nbits_autotune
} // namespace hipdnn_ep
#else
// One-line C23 #embed of the real LUT .fb -- HIPDNN_LUT_FB is defined by a
// tiny Makefile-generated header (a plain #define, not a data file) so the
// path never has to survive hipcc's Windows -D quoting (which mangles
// embedded quote characters).
#include "lut_fb_path.h"
static const unsigned char kLutBlob0[] = {
#embed HIPDNN_LUT_FB
};
extern "C" const unsigned char *const kMatmulNbitsLutBlobs[1] = {kLutBlob0};
extern "C" const size_t kMatmulNbitsLutBlobSizes[1] = {sizeof(kLutBlob0)};
extern "C" const size_t kMatmulNbitsLutBlobCount = 1;
#endif // HIPDNN_LUT_LINKED_EXTERNALLY
#endif // HIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS

#include "../../../../hipdnn_ep_runtime.h"
#include "../../../../op_profile.h"
#include "../../../../real/zp_unpack_cache.h"
#include <array>
#include <numeric>
#undef OP_PROFILE
#define OP_PROFILE(...) ((void)0)
#undef OP_ROCTX_MARK
#define OP_ROCTX_MARK(...) ((void)0)

struct RuntimeState {
  hipStream_t stream = nullptr;
  void *scratch = nullptr;
  size_t bytes = 0;
  hipdnn_ep_real::ZpUnpackCache cache;
};
extern "C" void *hipdnn_ep_state_get_stream(RuntimeState *state) {
  return state->stream;
}
extern "C" void *hipdnn_ep_state_get_qmoe_scratch(RuntimeState *state) {
  return state->scratch;
}
extern "C" int hipdnn_ep_state_ensure_qmoe_scratch(RuntimeState *state,
                                                   size_t bytes) {
  // A conservative arena is allocated before all complete calls. No timed
  // allocation or replacement is allowed through this narrow state service.
  return bytes <= state->bytes ? 0 : -1;
}
namespace hipdnn_ep_real {
ZpUnpackCache::~ZpUnpackCache() {
  for (auto &kv : u8)
    HIP_CHECK(hipFree(kv.second.first));
  for (auto &kv : fp16)
    HIP_CHECK(hipFree(kv.second.first));
}
ZpUnpackCache *get_or_create_zp_cache(RuntimeState *state) {
  return &state->cache;
}
const void *unpackForTest(ZpUnpackCache &cache, void *stream,
                          const void *packed, int n, int groups, bool two_bit) {
  std::lock_guard<std::mutex> lock(cache.mu);
  const size_t bytes = static_cast<size_t>(n) * groups;
  auto found = cache.u8.find(packed);
  if (found != cache.u8.end() && found->second.second >= bytes)
    return found->second.first;
  void *dst = nullptr;
  if (hipMalloc(&dst, bytes) != hipSuccess)
    return nullptr;
  if (two_bit)
    hip_matmul_nbits_unpack_zp_u8_2bit(stream, packed, dst, n, groups);
  else
    hip_matmul_nbits_unpack_zp_u8(stream, packed, dst, n, groups);
  if (hipGetLastError() != hipSuccess) {
    HIP_CHECK(hipFree(dst));
    return nullptr;
  }
  if (found != cache.u8.end()) {
    HIP_CHECK(hipFree(found->second.first));
    found->second = {dst, bytes};
  } else
    cache.u8.emplace(packed, std::make_pair(dst, bytes));
  return dst;
}
const void *lookup_or_unpack_zp_u8(ZpUnpackCache &c, void *s, const void *p,
                                   int n, int g) {
  return unpackForTest(c, s, p, n, g, false);
}
const void *lookup_or_unpack_zp_u8_2bit(ZpUnpackCache &c, void *s,
                                        const void *p, int n, int g) {
  return unpackForTest(c, s, p, n, g, true);
}
} // namespace hipdnn_ep_real
int wrap_qmoe(RuntimeState *, const void *, const void *, const void *,
              const void *, const void *, const void *, const void *,
              const void *, const void *, const void *, const void *,
              const void *, const void *, const void *, const void *, void *,
              int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t,
              int64_t, int64_t, float, float, float, int64_t, int64_t,
              const void *, int64_t, float);

struct Case {
  int64_t m = 1, h = 128, inter = 128, experts = 8, topk = 2, gs = 32;
  int64_t fc1_bias = 1, fc2_bias = 1, zp = 0, normalize = 1, router_weights = 0;
  std::string routing = "balanced";
  float alpha = 1.0f, beta = 0.0f,
        limit = std::numeric_limits<float>::infinity();
};
static void validate(const Case &c) {
  volume({c.m, c.h});
  volume({c.m, c.topk, c.inter});
  volume({c.m, c.topk, c.h});
  if (c.h % 32 || c.inter % 32 || c.h > 16384 || c.inter > 16384 ||
      c.experts < 1 || c.experts > 1024 || c.topk < 1 ||
      c.topk > (c.experts < 32 ? c.experts : 32) || c.gs < 32 || c.gs > 1024 ||
      (c.gs & (c.gs - 1)) || c.h % c.gs || c.inter % c.gs || c.fc1_bias < 0 ||
      c.fc1_bias > 1 || c.fc2_bias < 0 || c.fc2_bias > 1 || c.zp < 0 ||
      c.zp > 1 || c.normalize < 0 || c.normalize > 1 || c.router_weights < 0 ||
      c.router_weights > 1 || !std::isfinite(c.alpha) ||
      !std::isfinite(c.beta) || std::isnan(c.limit) || c.limit <= 0 ||
      (c.routing != "balanced" && c.routing != "skewed" && c.routing != "ties"))
    throw std::invalid_argument(
        "unsupported QMoE shape/quantization/routing/activation");
  const uint64_t weight_bytes =
      static_cast<uint64_t>(c.experts) * c.h * c.inter * 3 / 2;
  if (weight_bytes > (uint64_t(8) << 30))
    throw std::invalid_argument("QMoE UT weight allocation exceeds 8 GiB");
}

struct Weight {
  int64_t e, n, k, gs, groups;
  std::vector<uint8_t> packed, zeros;
  std::vector<uint16_t> scales, bias;
  Weight(int64_t experts, int64_t rows, int64_t inner, int64_t block, bool zp,
         bool has_bias, unsigned seed)
      : e(experts), n(rows), k(inner), gs(block), groups(inner / block),
        packed(static_cast<size_t>(experts) * rows * inner / 2),
        zeros(zp ? static_cast<size_t>(experts) * rows * ((groups + 1) / 2)
                 : 0),
        scales(static_cast<size_t>(experts) * rows * groups),
        bias(has_bias ? static_cast<size_t>(experts) * rows : 0) {
    std::mt19937 rng(seed);
    for (auto &p : packed)
      p = static_cast<uint8_t>(rng());
    for (auto &z : zeros)
      z = static_cast<uint8_t>(rng());
    std::uniform_real_distribution<float> scale(0.004f, 0.012f),
        b(-0.02f, 0.02f);
    for (auto &s : scales)
      s = number<uint16_t>(scale(rng));
    for (auto &v : bias)
      v = number<uint16_t>(b(rng));
  }
  int code(int64_t expert, int64_t row, int64_t col) const {
    const uint8_t byte = packed[(expert * n + row) * (k / 2) + col / 2];
    return col % 2 ? byte >> 4 : byte & 15;
  }
  int zero(int64_t expert, int64_t row, int64_t group) const {
    if (zeros.empty())
      return 8;
    const uint8_t byte =
        zeros[(expert * n + row) * ((groups + 1) / 2) + group / 2];
    return group % 2 ? byte >> 4 : byte & 15;
  }
  float scale(int64_t expert, int64_t row, int64_t group) const {
    return real(scales[(expert * n + row) * groups + group]);
  }
  float b(int64_t expert, int64_t row) const {
    return bias.empty() ? 0 : real(bias[expert * n + row]);
  }
  std::vector<float> dequant(int64_t expert) const {
    std::vector<float> w(n * k);
    for (int64_t row = 0; row < n; ++row)
      for (int64_t col = 0; col < k; ++col) {
        const int64_t group = col / gs;
        // WMMA consumes dequantized FP16 weights, not unrounded FP32 products.
        w[row * k + col] = real(number<uint16_t>(
            (code(expert, row, col) - zero(expert, row, group)) *
            scale(expert, row, group)));
      }
    return w;
  }
};

struct Quantized {
  std::vector<int8_t> data;
  std::vector<float> scales;
  Quantized(const std::vector<float> &x, int64_t gs)
      : data(x.size()), scales(x.size() / gs) {
    for (size_t g = 0; g < scales.size(); ++g) {
      float max = 0;
      for (int64_t i = 0; i < gs; ++i)
        max = std::max(max, std::abs(x[g * gs + i]));
      scales[g] = max > 0 ? max * (1.0f / 127.0f) : 1.0f;
      const float inverse = max > 0 ? 127.0f / max : 0.0f;
      for (int64_t i = 0; i < gs; ++i)
        data[g * gs + i] = static_cast<int8_t>(
            std::max(-127, std::min(127, static_cast<int>(std::nearbyint(
                                             x[g * gs + i] * inverse)))));
    }
  }
};
static double quantDot(const Quantized &x, const Weight &w, int64_t expert,
                       int64_t row) {
  double sum = 0;
  for (int64_t g = 0; g < w.groups; ++g) {
    int dot = 0;
    const int zp = w.zero(expert, row, g);
    for (int64_t i = 0; i < w.gs; ++i)
      dot += static_cast<int>(x.data[g * w.gs + i]) *
             (w.code(expert, row, g * w.gs + i) - zp);
    sum += static_cast<double>(x.scales[g] * w.scale(expert, row, g)) * dot;
  }
  return sum;
}
static double floatDot(const std::vector<float> &x, const float *w) {
  // Eight independent double accumulators keep the full reference vectorizable
  // without relying on GPU tiling, sampled outputs or external BLAS/data files.
  double s[8] = {};
  for (size_t i = 0; i < x.size(); i += 8)
    for (int j = 0; j < 8; ++j)
      s[j] += static_cast<double>(x[i + j]) * w[i + j];
  return std::accumulate(std::begin(s), std::end(s), 0.0);
}
static float activate(float gate, float up, const Case &c) {
  const double g = std::min(gate, c.limit),
               l = std::max(-c.limit, std::min(up, c.limit));
  return static_cast<float>(g / (1.0 + std::exp(-c.alpha * g)) * (l + c.beta));
}

template <class Fn> static void parallelExperts(int64_t count, Fn &&fn) {
  const int64_t available = std::thread::hardware_concurrency();
  const int64_t capped = count < 16 ? count : 16;
  const int64_t ready = available < capped ? available : capped;
  const int64_t threads = ready > 0 ? ready : 1;
  std::vector<std::thread> pool;
  for (int64_t t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      for (int64_t e = t; e < count; e += threads)
        fn(e);
    });
  for (auto &t : pool)
    t.join();
}

static std::vector<double>
reference(const Case &c, const std::vector<uint16_t> &x,
          const std::vector<uint16_t> &logits, const std::vector<uint16_t> &rw,
          const Weight &w1, const Weight &w2, bool fused, bool dp4a,
          std::vector<int32_t> &selected,
          std::vector<uint16_t> &selected_weights,
          std::vector<double> &contribution_magnitude) {
  selected.resize(c.m * c.topk);
  selected_weights.resize(selected.size());
  std::vector<std::vector<int64_t>> expert_slots(c.experts);
  for (int64_t token = 0; token < c.m; ++token) {
    std::vector<int64_t> order(c.experts);
    std::iota(order.begin(), order.end(), 0);
    // Only top-k is needed. An independent selection reference also avoids
    // MSVC sort instantiating a second std::min<int64_t> in HIP host code.
    for (int64_t rank = 0; rank < c.topk; ++rank) {
      int64_t best = rank;
      for (int64_t j = rank + 1; j < c.experts; ++j) {
        const float p = real(logits[token * c.experts + order[j]]);
        const float q = real(logits[token * c.experts + order[best]]);
        if (p > q || (p == q && order[j] < order[best])) best = j;
      }
      std::swap(order[rank], order[best]);
    }
    double maximum = real(logits[token * c.experts + order[0]]), all_sum = 0,
           top_sum = 0;
    std::vector<double> weights(c.topk);
    for (int64_t e = 0; e < c.experts; ++e)
      all_sum += std::exp(real(logits[token * c.experts + e]) - maximum);
    for (int64_t r = 0; r < c.topk; ++r) {
      const int64_t e = order[r], slot = token * c.topk + r;
      selected[slot] = e;
      expert_slots[e].push_back(slot);
      weights[r] =
          rw.empty() ? std::exp(real(logits[token * c.experts + e]) - maximum) /
                           all_sum
                     : real(rw[token * c.experts + e]);
      top_sum += weights[r];
    }
    for (int64_t r = 0; r < c.topk; ++r)
      selected_weights[token * c.topk + r] = number<uint16_t>(
          c.normalize ? (top_sum ? weights[r] / top_sum : 0.0) : weights[r]);
  }
  std::vector<float> per_slot(static_cast<size_t>(c.m) * c.topk * c.h);
  parallelExperts(c.experts, [&](int64_t e) {
    if (expert_slots[e].empty())
      return;
    const auto dw1 = fused ? std::vector<float>{} : w1.dequant(e);
    const auto dw2 = fused ? std::vector<float>{} : w2.dequant(e);
    for (int64_t slot : expert_slots[e]) {
      const int64_t token = slot / c.topk;
      std::vector<float> input(c.h), mid(c.inter);
      for (int64_t i = 0; i < c.h; ++i)
        input[i] = real(x[token * c.h + i]);
      const Quantized qin(input, c.gs);
      auto first = [&](int64_t row) {
        double value = 0;
        if (dp4a)
          value = quantDot(qin, w1, e, row);
        else if (!fused)
          value = floatDot(input, dw1.data() + row * c.h);
        else
          for (int64_t i = 0; i < c.h; ++i)
            value += input[i] *
                     (w1.code(e, row, i) - w1.zero(e, row, i / c.gs)) *
                     static_cast<double>(w1.scale(e, row, i / c.gs));
        const float biased = static_cast<float>(value + w1.b(e, row));
        return fused ? biased
                     : static_cast<float>(real(number<uint16_t>(biased)));
      };
      for (int64_t i = 0; i < c.inter; ++i)
        mid[i] = activate(first(2 * i), first(2 * i + 1), c);
      float row_scale = 1.0f;
      if (!fused) {
        float max = 0;
        for (float v : mid)
          max = std::max(max, std::abs(v));
        if (max > 1024.0f)
          row_scale = std::exp2(std::ceil(std::log2(max / 1024.0f)));
        for (float &v : mid)
          v = real(number<uint16_t>(v / row_scale));
      }
      const Quantized qmid(mid, c.gs);
      for (int64_t row = 0; row < c.h; ++row) {
        double value = 0;
        if (dp4a)
          value = quantDot(qmid, w2, e, row);
        else if (!fused)
          value = floatDot(mid, dw2.data() + row * c.inter);
        else
          for (int64_t i = 0; i < c.inter; ++i)
            value += mid[i] * (w2.code(e, row, i) - w2.zero(e, row, i / c.gs)) *
                     static_cast<double>(w2.scale(e, row, i / c.gs));
        if (!fused)
          value = row_scale * real(number<uint16_t>(value));
        value += w2.b(e, row);
        per_slot[slot * c.h + row] =
            static_cast<float>(value * real(selected_weights[slot]));
      }
    }
  });
  std::vector<double> out(c.m * c.h);
  contribution_magnitude.assign(out.size(), 0.0);
  for (int64_t token = 0; token < c.m; ++token) {
    std::vector<int64_t> order(c.topk);
    std::iota(order.begin(), order.end(), 0);
    if (!fused) {
      // Prefill preserves the original host loop: ascending expert IDs,
      // stable by routing slot, with an FP16 store after every contribution.
      // Decode instead accumulates all contributions before the final store.
      for (int64_t r = 1; r < c.topk; ++r) {
        const int64_t slot = order[r];
        int64_t j = r;
        while (j > 0 && selected[token * c.topk + order[j - 1]] >
                            selected[token * c.topk + slot]) {
          order[j] = order[j - 1];
          --j;
        }
        order[j] = slot;
      }
    }
    for (int64_t row = 0; row < c.h; ++row) {
      double sum = 0;
      for (int64_t r : order) {
        const float term = per_slot[(token * c.topk + r) * c.h + row];
        contribution_magnitude[token * c.h + row] += std::abs(term);
        sum += term;
        if (!fused)
          sum = real(number<uint16_t>(sum));
      }
      out[token * c.h + row] = real(number<uint16_t>(sum));
    }
  }
  return out;
}

static bool runCase(const Case &c, int iters, unsigned seed) {
  const std::string shape =
      "M" + std::to_string(c.m) + "_H" + std::to_string(c.h) + "_I" +
      std::to_string(c.inter) + "_E" + std::to_string(c.experts) + "_k" +
      std::to_string(c.topk) + "_gs" + std::to_string(c.gs) + "_b1" +
      std::to_string(c.fc1_bias) + "_b2" + std::to_string(c.fc2_bias) + "_zp" +
      std::to_string(c.zp) + "_norm" + std::to_string(c.normalize) + "_rw" +
      std::to_string(c.router_weights) + "_" + c.routing + "_alpha" +
      std::to_string(c.alpha) + "_beta" + std::to_string(c.beta) + "_limit" +
      std::to_string(c.limit);
  hipdnn_ep_test::timing::CaseScope timing_case("qmoe/Xfp16_Wu4_Yfp16", shape, "wrap_qmoe");
  auto x = values<uint16_t>(c.m * c.h, seed, -0.5f, 0.5f);
  auto logits = values<uint16_t>(c.m * c.experts, seed + 1, -2.0f, 2.0f);
  auto rw = c.router_weights
                ? values<uint16_t>(c.m * c.experts, seed + 2, 0.1f, 0.9f)
                : std::vector<uint16_t>{};
  for (int64_t t = 0; t < c.m; ++t) {
    if (c.routing == "ties") {
      for (int64_t e = 0; e < c.experts; ++e) {
        logits[t * c.experts + e] = 0;
        if (!rw.empty())
          rw[t * c.experts + e] = 0;
      }
    } else
      for (int64_t r = 0; r < c.topk; ++r) {
        const int64_t e =
            c.routing == "skewed" ? r : (t * c.topk + r) % c.experts;
        logits[t * c.experts + e] = number<uint16_t>(4.0f + 0.1f * r);
      }
  }
  Weight w1(c.experts, 2 * c.inter, c.h, c.gs, c.zp, c.fc1_bias, seed + 3);
  Weight w2(c.experts, c.h, c.inter, c.gs, c.zp, c.fc2_bias, seed + 4);
  const bool fused = c.m <= hipdnn_ep_qmoe_fused_max_tokens();
  const bool dp4a = fused && hipdnn_ep_matmul_dp4a_enabled();
  if (!fused && (!hip_matmul_nbits_wmma_smallm_grouped_supported(
                     -1, 16, 2 * c.inter, c.h, 4, c.gs, 2) ||
                 !hip_matmul_nbits_wmma_smallm_grouped_supported(
                     -1, 32, c.h, c.inter, 4, c.gs, 2)))
    throw std::invalid_argument(
        "QMoE UT prefill requires the production grouped WMMA surface");
  std::vector<int32_t> selected;
  std::vector<uint16_t> selected_weights;
  std::vector<double> contribution_magnitude;
  const auto ref = reference(c, x, logits, rw, w1, w2, fused, dp4a, selected,
                             selected_weights, contribution_magnitude);
  hipdnn_ep_test::GpuExecutionScope execution([] { HIP_CHECK(hipDeviceSynchronize()); });
  auto *dx = to_device(x);
  auto *dl = to_device(logits);
  auto *dr = to_device(rw);
  auto *dw1 = to_device(w1.packed);
  auto *ds1 = to_device(w1.scales);
  auto *dz1 = to_device(w1.zeros);
  auto *db1 = to_device(w1.bias);
  auto *dw2 = to_device(w2.packed);
  auto *ds2 = to_device(w2.scales);
  auto *dz2 = to_device(w2.zeros);
  auto *db2 = to_device(w2.bias);
  uint16_t *dy = nullptr;
  HIP_CHECK(hipMalloc(&dy, ref.size() * 2));
  RuntimeState state;
  HIP_CHECK(hipStreamCreate(&state.stream));
  const uint64_t rows = static_cast<uint64_t>(c.m) * c.topk;
  const uint64_t arena =
      rows * (16 * static_cast<uint64_t>(c.h + c.inter) + 512) +
      static_cast<uint64_t>(c.experts + 1) * 256;
  if (arena > std::numeric_limits<size_t>::max())
    throw std::invalid_argument("QMoE scratch overflow");
  state.bytes = static_cast<size_t>(arena);
  HIP_CHECK(hipMalloc(&state.scratch, state.bytes));
  int status = 0;
  auto launch = [&](const hipdnn_ep_test::ColdCopies *copies, int i) {
    auto part = [&](size_t n, const void *original) {
      return copies ? copies->part(i, n) : original;
    };
    const int rc = wrap_qmoe(
        &state, dx, dl, rw.empty() ? nullptr : dr, part(0, dw1), part(1, ds1),
        w1.bias.empty() ? nullptr : part(3, db1), part(4, dw2), part(5, ds2),
        w2.bias.empty() ? nullptr : part(7, db2), nullptr, nullptr, nullptr,
        w1.zeros.empty() ? nullptr : part(2, dz1),
        w2.zeros.empty() ? nullptr : part(6, dz2), nullptr, dy, c.m, c.h,
        c.inter, c.experts, c.topk, 4, c.gs, 1, 0, c.alpha, c.beta, c.limit,
        c.normalize, 2, nullptr, HIPDNN_EP_QMOE_ROUTING_SOFTMAX, 1.0f);
    if (rc && !status)
      status = rc;
  };
  const double ms = completeCalls(state.stream, iters,
                                  {{dw1, w1.packed.size()},
                                   {ds1, w1.scales.size() * 2},
                                   {dz1, w1.zeros.size()},
                                   {db1, w1.bias.size() * 2},
                                   {dw2, w2.packed.size()},
                                   {ds2, w2.scales.size() * 2},
                                   {dz2, w2.zeros.size()},
                                   {db2, w2.bias.size() * 2}},
                                  launch, status);
  const auto got = from_device(dy, ref.size());
  double error = 0, square = 0;
  bool correct = true;
  size_t output_failures = 0, index_failures = 0, weight_failures = 0;
  double max_output_error = 0, max_route_error = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double value = real(got[i]), delta = value - ref[i];
    error += delta * delta;
    square += ref[i] * ref[i];
    max_output_error = std::max(max_output_error, std::abs(delta));
    // Cancellation can make the final sum tiny although its FP16 expert
    // terms are not. Retain the existing accuracy limits, with a prefill-only
    // floor of one relative FP16 ULP of the absolute contribution sum.
    const double cancellation_floor =
        fused ? 0.0 : std::ldexp(contribution_magnitude[i], -10);
    const double tolerance =
        std::max(0.0005 + 0.005 * std::abs(ref[i]), cancellation_floor);
    if (!std::isfinite(value) || std::abs(delta) > tolerance) {
      if (output_failures++ < 8)
        std::fprintf(stderr,
                     "QMOE_CHECK output token=%zu channel=%zu actual=%.9g "
                     "reference=%.9g abs_error=%.9g tolerance=%.9g "
                     "cancellation_floor=%.9g\n",
                     i / c.h, i % c.h, value, ref[i], std::abs(delta), tolerance,
                     cancellation_floor);
      correct = false;
    }
  }
  // The production wrapper's first scratch views are its routing indices and
  // weights. Check all of them independently, in addition to the full output.
  const auto indices =
      from_device(static_cast<const int32_t *>(state.scratch), selected.size());
  const size_t weights_offset =
      (selected.size() * sizeof(int32_t) + 63) & ~size_t(63);
  const auto route_weights = from_device(
      reinterpret_cast<const uint16_t *>(
          static_cast<const char *>(state.scratch) + weights_offset),
      selected_weights.size());
  for (size_t i = 0; i < selected.size(); ++i)
    if (indices[i] != selected[i]) {
      if (index_failures++ < 8)
        std::fprintf(stderr,
                     "QMOE_CHECK expert token=%zu slot=%zu actual=%d reference=%d\n",
                     i / c.topk, i % c.topk, indices[i], selected[i]);
      correct = false;
    }
  for (size_t i = 0; i < selected_weights.size(); ++i) {
    const double delta = std::abs(real(route_weights[i]) - real(selected_weights[i]));
    max_route_error = std::max(max_route_error, delta);
    if (delta > 0.001) {
      if (weight_failures++ < 8)
        std::fprintf(stderr,
                     "QMOE_CHECK route_weight token=%zu slot=%zu actual=%.9g "
                     "reference=%.9g abs_error=%.9g tolerance=0.001\n",
                     i / c.topk, i % c.topk, real(route_weights[i]),
                     real(selected_weights[i]), delta);
      correct = false;
    }
  }
  const double l2 = std::sqrt(error / std::max(square, 1e-30));
  correct &= l2 <= 0.005;
  if (!correct)
    std::fprintf(stderr,
                 "QMOE_CHECK summary output_failures=%zu expert_failures=%zu "
                 "route_weight_failures=%zu max_output_error=%.9g "
                 "max_route_weight_error=%.9g rel_l2=%.9g rel_l2_limit=0.005\n",
                 output_failures, index_failures, weight_failures,
                 max_output_error, max_route_error, l2);
  const bool pass = passedTime(status, correct, ms);
  report("qmoe", shape,
         dp4a    ? "fixed:wrap_qmoe:dp4a"
         : fused ? "fixed:wrap_qmoe:fp_fused"
                 : "fixed:wrap_qmoe:grouped",
         status, ms, l2, pass);
  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dl));
  HIP_CHECK(hipFree(dr));
  HIP_CHECK(hipFree(dw1));
  HIP_CHECK(hipFree(ds1));
  HIP_CHECK(hipFree(dz1));
  HIP_CHECK(hipFree(db1));
  HIP_CHECK(hipFree(dw2));
  HIP_CHECK(hipFree(ds2));
  HIP_CHECK(hipFree(dz2));
  HIP_CHECK(hipFree(db2));
  HIP_CHECK(hipFree(dy));
  HIP_CHECK(hipFree(state.scratch));
  HIP_CHECK(hipStreamDestroy(state.stream));
  return pass;
}

int main(int argc, char **argv) {
  hipdnn_ep_test::timing::printTimingMode();
  if (!options(argc, argv,
               {"--tokens", "--hidden", "--inter", "--experts", "--topk",
                "--block-size", "--fc1-bias", "--fc2-bias", "--zero-points",
                "--normalize", "--router-weights", "--routing-pattern",
                "--alpha", "--beta", "--limit", "--iters", "--coverage"}))
    return 2;
  try {
    const int tier = coverage(argc, argv), iters = iterations(argc, argv);
    std::vector<Case> cases;
    bool custom = false;
    for (int i = 1; i < argc; i += 2)
      if (std::strcmp(argv[i], "--iters") && std::strcmp(argv[i], "--coverage"))
        custom = true;
    if (custom) {
      for (const char *name : {"--tokens", "--hidden", "--inter", "--experts",
                               "--topk", "--block-size"})
        if (!arg_str(argc, argv, name))
          throw std::invalid_argument("incomplete explicit QMoE geometry");
      Case c;
      c.m = opt(argc, argv, "--tokens", 1);
      c.h = opt(argc, argv, "--hidden", 128);
      c.inter = opt(argc, argv, "--inter", 128);
      c.experts = opt(argc, argv, "--experts", 8);
      c.topk = opt(argc, argv, "--topk", 2);
      c.gs = opt(argc, argv, "--block-size", 32);
      c.fc1_bias = opt(argc, argv, "--fc1-bias", 1);
      c.fc2_bias = opt(argc, argv, "--fc2-bias", 1);
      c.zp = opt(argc, argv, "--zero-points", 0);
      c.normalize = opt(argc, argv, "--normalize", 1);
      c.router_weights = opt(argc, argv, "--router-weights", 0);
      if (const char *text = arg_str(argc, argv, "--routing-pattern"))
        c.routing = text;
      for (const auto &pair : {std::make_pair("--alpha", &c.alpha),
                               std::make_pair("--beta", &c.beta),
                               std::make_pair("--limit", &c.limit)})
        if (const char *text = arg_str(argc, argv, pair.first)) {
          char *end = nullptr;
          errno = 0;
          *pair.second = std::strtof(text, &end);
          if (errno || end == text || *end)
            throw std::invalid_argument("invalid activation float");
        }
      cases.push_back(c);
    } else {
      if (captureOnly())
        throw std::invalid_argument("capture-only requires one explicit shape");
      size_t index = 0;
      for (int gs : {32, 64, 128})
        for (int b1 : {0, 1})
          for (int b2 : {0, 1})
            for (int zp : {0, 1})
              for (int norm : {0, 1})
                for (int rw : {0, 1})
                  for (const char *pattern : {"balanced", "skewed", "ties"}) {
                    Case c;
                    c.gs = gs;
                    c.fc1_bias = b1;
                    c.fc2_bias = b2;
                    c.zp = zp;
                    c.normalize = norm;
                    c.router_weights = rw;
                    c.routing = pattern;
                    const int64_t shapes[] = {1, 2, 8, 9, 17, 33};
                    c.m = tier == 1   ? (index % 2 ? 9 : 1)
                          : tier == 2 ? shapes[index % 4]
                                      : shapes[index % 6];
                    const int64_t geometry[][4] = {{128, 128, 8, 1},
                                                   {256, 128, 8, 2},
                                                   {128, 256, 16, 4},
                                                   {256, 256, 16, 8}};
                    const auto &g = geometry[index % (tier == 1 ? 2 : 4)];
                    c.h = g[0];
                    c.inter = g[1];
                    c.experts = g[2];
                    c.topk = g[3];
                    c.limit = index % 2
                                  ? 0.25f
                                  : std::numeric_limits<float>::infinity();
                    c.alpha = index % 2 ? 1.702f : 1.0f;
                    c.beta = index % 2 ? 1.0f : 0.0f;
                    cases.push_back(c);
                    ++index;
                  }
    }
    for (const Case &c : cases)
      validate(c);
    deviceInfo("QMoE Xfp16_Wu4_Yfp16");
    std::printf(
        "coverage=%d -> %zu cases; packed weights/ZP, bias pairs, "
        "normalized/external routing and decode/grouped prefill; full output\n",
        tier, cases.size());
    int failures = 0;
    for (size_t i = 0; i < cases.size(); ++i)
      if (!runCase(cases[i], iters, 6000 + i))
        ++failures;
    return failures ? 1 : 0;
  } catch (const std::exception &error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 2;
  }
}
#undef HIP_CHECK
#include "../../../../real/qmoe.cpp"
