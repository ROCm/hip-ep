/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Standalone QGEMM numeric and latency UT. Inputs and CPU references are
// generated in-process. Links production custom_kernels in packaging builds.
// Cold weight rotation and CSV schema follow the MatMulNBits leaves.
#include <hip/hip_runtime.h>
#include "hip_custom_kernels.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace hipdnn_ep_test {
inline int resolveCoverageTier(int argc, char** argv, int default_tier = 3) {
  int tier = default_tier;
  if (const char* env = std::getenv("HIPDNN_UT_COVERAGE")) tier = std::atoi(env);
  for (int i = 1; i < argc; ++i)
    if (std::strcmp(argv[i], "--coverage") == 0 && i + 1 < argc)
      tier = std::atoi(argv[++i]);
  return tier < 1 ? 1 : (tier > 3 ? 3 : tier);
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
    const char* path = std::getenv("HIPDNN_RESULTS_CSV");
    if (!path || !path[0]) return;
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
  ~CsvWriter() { if (f_) std::fclose(f_); }
  void write(const CsvRow& r) {
    if (!f_) return;
    std::fprintf(f_, "%s,%s,%s,%s,%s,%s,%.6f,%.6e,%s\n",
                 env_or("HIPDNN_RESULTS_OP", "unknown"),
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
}  // namespace hipdnn_ep_test

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
  (void)hipEventRecord(start, stream);
  for (int i = 0; i < iters; ++i) launch(turn++ % n);
  (void)hipEventRecord(stop, stream);
  if (release) *release = 1;
  (void)hipEventSynchronize(stop);
  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, stop);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(stop);
  return iters > 0 ? ms / iters : 0.0;
}
}  // namespace hipdnn_ep_test


#define HIP_CHECK(call) do { \
  const hipError_t error = (call); \
  if (error != hipSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(error)); \
    std::exit(2); \
  } \
} while (0)

struct Buffer {
  void* p = nullptr;
  explicit Buffer(size_t bytes) { if (bytes) HIP_CHECK(hipMalloc(&p, bytes)); }
  ~Buffer() { if (p) (void)hipFree(p); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

template<class T> static void upload(Buffer& b, const std::vector<T>& v) {
  HIP_CHECK(hipMemcpy(b.p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
}

static std::vector<uint8_t> pack(const std::vector<int>& values, int bits) {
  std::vector<uint8_t> bytes((values.size() * bits + 7) / 8, 0);
  for (size_t i = 0; i < values.size(); ++i) {
    if (bits == 8) bytes[i] = static_cast<uint8_t>(values[i]);
    else bytes[i / 2] |= static_cast<uint8_t>(values[i] & 15) << ((i % 2) * 4);
  }
  return bytes;
}

struct Case {
  int M, N, K, bits;
  bool trans_a, trans_b, channel, signed_b;
  int bias; // 0 absent, 1 column, 2 row, 3 full matrix
  bool wide = false;
};

// A count of zero selects the original hip_qgemm ABI (all rows); otherwise
// use hip_qgemm_live_rows with device seqlens=count-1. Out-of-window counts
// follow the launcher's full-window fallback. Padding output is not checked.
template<class AType, class YType>
static bool runCase(const Case& c, int id, int count) {
  const int checked_rows = count < 1 || count > c.M ? c.M : count;
  std::vector<AType> a(size_t(c.M) * c.K);
  std::vector<int> b(size_t(c.K) * c.N), zero(c.N);
  std::vector<float> scale(c.N);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = static_cast<AType>(std::is_unsigned_v<AType>
        ? int((i * 7 + id * 3) % 17) : int((i * 7 + id * 3) % 17) - 8);
  for (size_t i = 0; i < b.size(); ++i)
    b[i] = c.signed_b ? int((i * 11 + id) % 15) - 7 : int((i * 11 + id) % 16);
  for (int n = 0; n < c.N; ++n) {
    zero[n] = c.signed_b ? n % 3 - 1 : n % 4;
    scale[n] = float(1 + n % 3) / 64;
  }
  const auto packed_b = pack(b, c.bits), packed_zero = pack(zero, c.bits);
  const int d0 = c.bias == 3 ? c.M : 1, d1 = c.bias == 2 ? 1 : c.N;
  std::vector<int32_t> bias(size_t(d0) * d1);
  for (size_t i = 0; i < bias.size(); ++i) bias[i] = int(i % 13) - 6;
  constexpr int az = 2, cz = -2, yz = 7;
  const int bz = c.signed_b ? -1 : 3;
  constexpr float mab = 1.0f / 64, mc = 0.25f;
  std::vector<YType> reference(size_t(checked_rows) * c.N), output(size_t(c.M) * c.N);
  for (int m = 0; m < checked_rows; ++m) for (int n = 0; n < c.N; ++n) {
    int64_t dot = 0;
    for (int k = 0; k < c.K; ++k) {
      const int av = a[c.trans_a ? k * c.M + m : m * c.K + k];
      const int bv = b[c.trans_b ? n * c.K + k : k * c.N + n];
      dot += int64_t(av - az) * (bv - (c.channel ? zero[n] : bz));
    }
    float value = (c.channel ? mab * scale[n] : mab) * float(dot);
    if (c.bias) value += mc * float(bias[(c.bias == 3 ? m : 0) * d1 + (d1 == 1 ? 0 : n)] - cz);
    value = std::nearbyint(value) + yz;
    reference[size_t(m) * c.N + n] = static_cast<YType>(std::clamp(value,
        float(std::numeric_limits<YType>::lowest()), float(std::numeric_limits<YType>::max())));
  }
  Buffer da(a.size() * sizeof(AType)), db(packed_b.size());
  Buffer dc(c.bias ? bias.size() * sizeof(int32_t) : 0);
  Buffer ds(c.channel ? scale.size() * sizeof(float) : 0), dz(c.channel ? packed_zero.size() : 0);
  Buffer dy(output.size() * sizeof(YType)), live(count ? sizeof(int32_t) : 0);
  const size_t ws = hip_qgemm_workspace_bytes(c.M, c.N, c.K);
  Buffer scratch(ws);
  upload(da, a); upload(db, packed_b);
  if (c.bias) upload(dc, bias);
  if (c.channel) { upload(ds, scale); upload(dz, packed_zero); }
  const int32_t seq = count - 1;
  if (count) HIP_CHECK(hipMemcpy(live.p, &seq, sizeof(seq), hipMemcpyHostToDevice));
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  const hipdnn_ep_test::ColdCopies weights({{db.p, packed_b.size()},
      {ds.p, c.channel ? scale.size() * sizeof(float) : 0},
      {dz.p, c.channel ? packed_zero.size() : 0}});
  constexpr int a_dtype = std::is_unsigned_v<AType> ? HIP_DTYPE_UINT16 : HIP_DTYPE_INT8;
  constexpr int y_dtype = std::is_unsigned_v<YType> ? HIP_DTYPE_UINT16 : HIP_DTYPE_INT16;
  auto launch = [&](int copy) {
    int status;
    if (count == 0) {
      status = hip_qgemm(stream, da.p, weights.part(copy, 0), dc.p,
          weights.part(copy, 1), weights.part(copy, 2), dy.p, c.M, c.N, c.K,
          c.trans_a, c.trans_b, a_dtype, c.signed_b ? HIP_DTYPE_INT8 : HIP_DTYPE_UINT8,
          HIP_DTYPE_INT32, y_dtype, c.bits, c.bias ? d0 : 0, c.bias ? d1 : 0,
          mab, mc, az, bz, cz, yz, scratch.p, ws);
    } else {
      status = hip_qgemm_live_rows(stream, da.p, weights.part(copy, 0), dc.p,
          weights.part(copy, 1), weights.part(copy, 2), dy.p, c.M, c.N, c.K,
          c.trans_a, c.trans_b, a_dtype, c.signed_b ? HIP_DTYPE_INT8 : HIP_DTYPE_UINT8,
          HIP_DTYPE_INT32, y_dtype, c.bits, c.bias ? d0 : 0, c.bias ? d1 : 0,
          mab, mc, az, bz, cz, yz, scratch.p, ws, static_cast<const int32_t*>(live.p));
    }
    if (status != 0) { std::fprintf(stderr, "QGEMM returned %d\n", status); std::exit(2); }
  };
  for (int i = 0; i < 3; ++i) launch(0);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(output.data(), dy.p, output.size() * sizeof(YType), hipMemcpyDeviceToHost));
  size_t errors = 0;
  double ssd = 0, ssr = 0;
  for (size_t i = 0; i < reference.size(); ++i) {
    const double diff = double(output[i]) - reference[i];
    errors += diff != 0;
    ssd += diff * diff; ssr += double(reference[i]) * reference[i];
  }
  const double ms = hipdnn_ep_test::rotatedMeanMs(stream, 100, weights, launch);
  // Validate again after repeated launches, including workspace reuse.
  HIP_CHECK(hipMemcpy(output.data(), dy.p, output.size() * sizeof(YType), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < reference.size(); ++i) errors += output[i] != reference[i];
  HIP_CHECK(hipStreamDestroy(stream));
  const double rel_l2 = std::sqrt(ssr > 0 ? ssd / ssr : ssd);
  char shape[96], config[160];
  std::snprintf(shape, sizeof(shape), "M%d_N%d_K%d", c.M, c.N, c.K);
  std::snprintf(config, sizeof(config), "W%d_%s_ta%d_tb%d_pc%d_bias%d_live%d_%s",
      c.bits, c.signed_b ? "signed" : "unsigned", c.trans_a, c.trans_b,
      c.channel, c.bias, count, weights.count() > 1 ? "cold" : "warm");
  std::printf("%s %s A=%s Y=%s %.6f ms errors=%zu %s\n", shape, config,
      c.wide ? "u16" : "i8", c.wide ? "u16" : "i16", ms, errors, errors ? "FAIL" : "PASS");
  hipdnn_ep_test::CsvWriter csv;
  csv.write({shape, config, ms, rel_l2, errors ? "FAIL" : "PASS"});
  return errors == 0;
}

static void defaultEnvironment(const char* key, const char* value) {
  if (std::getenv(key)) return;
#ifdef _WIN32
  if (_putenv_s(key, value) != 0) std::exit(2);
#else
  if (setenv(key, value, 0) != 0) std::exit(2);
#endif
}

int main(int argc, char** argv) {
  const int tier = hipdnn_ep_test::resolveCoverageTier(argc, argv);
  hipDeviceProp_t props{};
  HIP_CHECK(hipGetDeviceProperties(&props, 0));
  defaultEnvironment("HIPDNN_RESULTS_OP", "qgemm");
  defaultEnvironment("HIPDNN_RESULTS_LEAF", "Xint_Wint_Yint");
  defaultEnvironment("HIPDNN_RESULTS_ARCH", props.gcnArchName);
  defaultEnvironment("HIPDNN_RESULTS_MODE", "lookup");
  std::printf("QGEMM: %s (%s), coverage=%d; exact CPU reference\n", props.name, props.gcnArchName, tier);
  std::vector<Case> cases;
  // Every tier covers the original 34 categorical numeric cases.
  for (int bits : {4, 8}) for (bool sign : {false, true}) for (bool channel : {false, true}) {
    cases.push_back({1,37,257,bits,false,true,channel,sign,1});
    cases.push_back({17,35,65,bits,false,true,channel,sign,0});
    cases.push_back({16,32,64,bits,true,false,channel,sign,2});
    cases.push_back({64,64,128,bits,false,false,channel,sign,3});
  }
  cases.push_back({1,32,8192,4,false,true,false,true,0});
  cases.push_back({2,32,8192,8,false,true,true,true,1});
  int failed = 0, total = 0;
  auto check = [&](const Case& c, int count) {
    ++total;
    const bool pass = c.wide ? runCase<uint16_t,uint16_t>(c,c.M+c.N+c.K+c.bits,count)
                            : runCase<int8_t,int16_t>(c,c.M+c.N+c.K+c.bits,count);
    failed += !pass;
  };
  for (const auto& c : cases) check(c,0);
  cases = {{256,256,512,4,false,true,false,true,0},
           {128,128,4096,8,false,true,true,true,1},
           {129,35,65,4,true,false,true,false,2},
           {256,1024,512,4,false,true,true,true,1,true}};
  if (tier >= 2) cases.push_back({512,512,1024,4,false,true,true,true,3});
  if (tier >= 3) cases.push_back({512,1024,1024,4,false,true,false,true,0,true});
  for (const auto& c : cases) {
    // All tiers cover null, short, both tile boundaries and full/fallback.
    for (int count : {0,1,5,63,64,65,127,128,129,c.M,c.M+1,-1}) check(c,count);
  }
  std::printf("QGEMM: %d cases, %d failed\n", total, failed);
  return failed ? 1 : 0;
}
