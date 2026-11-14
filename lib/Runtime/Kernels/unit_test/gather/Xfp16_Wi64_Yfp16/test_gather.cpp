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

using Value = uint16_t;
struct Case { int64_t outer = 1, axis_size = 257, inner = 8, nidx = 17; };

static std::vector<Case> build_cases(int tier) {
  const std::vector<std::pair<int64_t, int64_t>> lengths = {{17, 7}, {257, 33}, {4096, 128}};
  std::vector<Case> result;
  for (int i = 0; i < tier; ++i)
    for (int64_t outer : {int64_t(1), int64_t(3)})
      for (int64_t inner : {int64_t(1), int64_t(7), int64_t(8), int64_t(17), int64_t(128)})
        result.push_back({outer, lengths[i].first, inner, lengths[i].second});
  return result;
}

static bool run_case(const Case& c, int iters, unsigned seed) {
  const int64_t data_n = checked_product({c.outer, c.axis_size, c.inner});
  const int64_t out_n = checked_product({c.outer, c.nidx, c.inner});
  // Reproducible raw storage values; Gather must copy every output bit exactly.
  std::mt19937_64 rng(seed);
  std::vector<Value> input(static_cast<size_t>(data_n));
  for (Value& value : input) value = static_cast<Value>(rng());
  std::vector<int64_t> indices(static_cast<size_t>(c.nidx));
  for (int64_t i = 0; i < c.nidx; ++i) indices[i] = rng() % c.axis_size;
  const int64_t special[] = {0, c.axis_size - 1, -1, -c.axis_size, c.axis_size, -c.axis_size - 1};
  for (int64_t i = 0; i < std::min<int64_t>(c.nidx, 6); ++i) indices[i] = special[i];
  std::vector<Value> reference(static_cast<size_t>(out_n));
  parallel_for(out_n, [&](int64_t i) {
    int64_t index = indices[(i / c.inner) % c.nidx];
    if (index < 0) index += c.axis_size;
    reference[i] = index < 0 || index >= c.axis_size ? Value(0) :
      input[((i / (c.inner * c.nidx)) * c.axis_size + index) * c.inner + i % c.inner];
  });
  Value* dx = to_device(input);
  int64_t* di = to_device(indices);
  Value* dy = nullptr;
  HIP_CHECK(hipMalloc(&dy, static_cast<size_t>(out_n) * sizeof(Value)));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  int status = 0;
  auto launch = [&](const void* data, const void* idx) {
    const int rc = hip_gather(stream, data, idx, dy, c.outer == 1 ? 0 : 1, data_n,
                             c.nidx, out_n, c.axis_size, c.inner, sizeof(Value), sizeof(int64_t));
    if (rc && !status) status = rc;
  };
  for (int w = 0; w < 3; ++w) launch(dx, di);
  HIP_CHECK(hipStreamSynchronize(stream));
  const hipdnn_ep_test::ColdCopies copies({{dx, input.size() * sizeof(Value)},
                                         {di, indices.size() * sizeof(int64_t)}});
  // Sparse access may reuse only a small fraction of a large copied table.
  // Disclose the touched-byte upper bound; full allocation size is not proof
  // that reads missed the cache. Do not turn this model into a DRAM counter.
  const uint64_t touched = static_cast<uint64_t>(c.outer) *
      std::min(c.nidx, c.axis_size) * c.inner * sizeof(Value);
  std::printf("Gather inputs: seeded synthetic indices; touched-byte upper bound=%llu, copies=%d; "
              "cache residency is not measured\n", static_cast<unsigned long long>(touched), copies.count());
  double ms = 0.0;
  if (!status) ms = hipdnn_ep_test::rotatedMeanMs(stream, iters, copies, [&](int i) {
    launch(copies.part(i, 0), copies.part(i, 1));
  });
  HIP_CHECK(hipStreamSynchronize(stream));
  const auto got = from_device(dy, out_n);
  const bool correct = std::memcmp(got.data(), reference.data(), got.size() * sizeof(Value)) == 0;
  const bool pass = !status && correct && std::isfinite(ms) && ms > 0;
  const std::string shape = std::to_string(c.outer) + "x" + std::to_string(c.axis_size) + "x" +
                            std::to_string(c.inner) + "_idx" + std::to_string(c.nidx);
  report("gather", shape, "fixed:hip_gather", status, ms, correct ? 0.0 : 1.0, pass);
  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(dx)); HIP_CHECK(hipFree(di)); HIP_CHECK(hipFree(dy));
  return pass;
}

int main(int argc, char** argv) {
  if (!options(argc, argv, {"--outer", "--axis-size", "--inner", "--nidx", "--iters", "--coverage"})) return 2;
  try {
    Case c;
    c.outer = opt(argc, argv, "--outer", 1); c.axis_size = opt(argc, argv, "--axis-size", 257);
    c.inner = opt(argc, argv, "--inner", 8); c.nidx = opt(argc, argv, "--nidx", 17);
    const int64_t count = opt(argc, argv, "--iters", 20);
    const char* coverage = arg_str(argc, argv, "--coverage");
    if (!coverage) coverage = std::getenv("HIPDNN_UT_COVERAGE");
    const int64_t tier = coverage ? integer(coverage) : 3;
    checked_product({c.outer, c.axis_size, c.inner}); checked_product({c.outer, c.nidx, c.inner});
    if (count < 1 || count > INT_MAX || tier < 1 || tier > 3) throw std::invalid_argument("invalid iters/coverage");
    const bool single = arg_str(argc, argv, "--outer") || arg_str(argc, argv, "--axis-size") ||
                        arg_str(argc, argv, "--inner") || arg_str(argc, argv, "--nidx");
    if (single && !arg_str(argc, argv, "--axis-size"))
      throw std::invalid_argument("a single Gather case requires --axis-size");
    hipDeviceProp_t device{}; HIP_CHECK(hipGetDeviceProperties(&device, 0));
    std::printf("Gather Xfp16_Wi64_Yfp16; GPU=%s arch=%s; owning-main cold rotation; iterations=%lld\n",
                device.name, device.gcnArchName, static_cast<long long>(count));
    const auto cases = single ? std::vector<Case>{c} : build_cases(static_cast<int>(tier));
    std::printf("coverage=%lld -> running %zu Gather cases\n", static_cast<long long>(tier), cases.size());
    int failures = 0; unsigned seed = 1234;
    for (const auto& item : cases) if (!run_case(item, static_cast<int>(count), seed++)) ++failures;
    std::printf("%s\n", failures ? "SOME FAILED" : "ALL PASS");
    return failures ? 1 : 0;
  } catch (const std::exception& error) { std::fprintf(stderr, "%s\n", error.what()); return 2; }
}
