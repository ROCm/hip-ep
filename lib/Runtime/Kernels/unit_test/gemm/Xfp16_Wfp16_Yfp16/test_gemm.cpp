/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Self-contained fp16 GEMM kernel-unit-test leaf under unit_test/.
 * Timing uses unit_test/common adaptive complete-call policy
 * (--timing screen|standard|confirm via HIPDNN_UT_TIMING).
 */
#include <hip/hip_runtime.h>
#include "../../common/hip_adaptive_timing.h"
#include <hip/hip_fp16.h>
#include "hip_custom_kernels.h"
#include "gemm_autotune.h"

#ifndef HIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS
#if !defined(HIPDNN_LUT_LINKED_EXTERNALLY)
namespace hipdnn_ep {
namespace gemm_autotune {
Result resolve(const Request&, WmmaValidator, GemvValidator, TiledFmaValidator,
               void*) {
  return {};
}
Stats stats() { return {}; }
}  // namespace gemm_autotune
}  // namespace hipdnn_ep
#else
#include "lut_fb_path.h"
extern "C" const unsigned char kGemmLutData[] = {
#embed HIPDNN_LUT_FB
};
extern "C" const size_t kGemmLutData_size = sizeof(kGemmLutData);
#endif
#endif  // HIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <thread>
#include <vector>

#define HIP_CHECK(x)                                                           \
  do {                                                                         \
    hipError_t e = (x);                                                        \
    if (e != hipSuccess) {                                                     \
      std::fprintf(stderr, "HIP %s:%d %s\n", __FILE__, __LINE__,               \
                   hipGetErrorString(e));                                      \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static constexpr int kGemmDtype = 0;
static constexpr const char* kDtypeName = "fp16";
static constexpr const char* kLeafName = "Xfp16_Wfp16_Yfp16";
static constexpr double kTolerance = 6e-2;
using Elem = __half;

static float toFloat(Elem value) { return __half2float(value); }
static Elem fromFloat(float value) { return __float2half(value); }

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
  ~CsvWriter() {
    if (f_) std::fclose(f_);
  }
  void write(const CsvRow& r) {
    timing::emitCsvBinding(r.shape, r.config, r.verdict, r.rel_l2, r.time_ms);
    if (!f_) return;
    std::fprintf(f_, "%s,%s,%s,%s,%s,%s,%.6f,%.6e,%s\n",
                 env_or("HIPDNN_RESULTS_OP", "gemm"),
                 env_or("HIPDNN_RESULTS_LEAF", kLeafName),
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

constexpr size_t kColdRotationBytes = size_t(256) << 20;
constexpr size_t kColdSettleBytes = size_t(128) << 20;
constexpr int kColdMaxCopies = 4096;

inline bool warmTiming() {
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
    if (warmTiming() || bytes_ == 0) return;
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

__global__ void holdStreamKernel(const volatile int* release) {
  for (int spin = 0; spin < (1 << 16) && *release == 0; ++spin)
    __builtin_amdgcn_s_sleep(127);
}

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

template <class Launch>
double rotatedMeanMs(hipStream_t stream, int /*legacy_iters*/,
                     const ColdCopies& copies, Launch&& launch) {
  return timing::adaptiveRotatedMeanMs(
      stream, copies, launch, holdReleaseFlag(),
      [&](const int* flag) { holdStreamKernel<<<1, 1, 0, stream>>>(flag); },
      warmTiming(), kColdSettleBytes, "gemm_fp16");
}
}  // namespace hipdnn_ep_test

static std::string beside_exe(const char* name) {
  char* exe = nullptr;
  if (_get_pgmptr(&exe) != 0 || !exe) return name;
  const std::string path(exe);
  const size_t slash = path.find_last_of("\\/");
  return slash == std::string::npos ? name : path.substr(0, slash + 1) + name;
}

template <typename Fn>
static std::vector<std::string> capture_stderr(Fn&& fn) {
  static const std::string capture = beside_exe("_config_capture.tmp");
  const char* const kCapture = capture.c_str();
  const char* leaf_stderr = std::getenv("HIPDNN_KERNEL_UT_STDERR");
  const bool redirect = !leaf_stderr || !leaf_stderr[0];
  const char* path = redirect ? kCapture : leaf_stderr;
  std::streamoff start = 0;
  std::fflush(stderr);
  if (redirect) {
    if (!std::freopen(kCapture, "w", stderr)) {
      std::freopen("CON", "w", stderr);
      fn();
      return {};
    }
  } else {
    std::ifstream log(path, std::ios::binary | std::ios::ate);
    if (log) start = log.tellg();
  }
  fn();
  std::fflush(stderr);
  if (redirect) std::freopen("CON", "w", stderr);

  std::vector<std::string> lines;
  {
    std::ifstream log(path, std::ios::binary);
    log.seekg(start);
    for (std::string line; std::getline(log, line);) {
      if (!line.empty() && line.back() == '\r') line.pop_back();
      lines.push_back(line);
    }
  }
  if (redirect) std::remove(kCapture);
  return lines;
}

template <typename Fn>
static std::string capture_selected_config(const char* tune_key, Fn&& launch) {
  static std::map<std::string, std::string> selected_by_key;
  _putenv_s("HIPDNN_EP_DEBUG", "1");
  _putenv_s("HIPDNN_GEMM_LUT_LOG", "1");

  std::string lut, tuned, final_cfg;
  for (const std::string& line : capture_stderr(launch)) {
    if (line.find("LUT hit") != std::string::npos) {
      const size_t arrow = line.find("->");
      lut = "lookup:" + (arrow == std::string::npos ? line : line.substr(arrow));
    } else if (line.find("best config") != std::string::npos ||
               line.find("best cfg") != std::string::npos) {
      tuned = "autotune:" + line;
    } else if (line.find(" final ") != std::string::npos &&
               line.find("[gemm]") != std::string::npos) {
      const size_t arrow = line.find("->");
      final_cfg =
          "final:" + (arrow == std::string::npos ? line : line.substr(arrow));
    }
  }

  std::string selected =
      !lut.empty() ? lut : (!tuned.empty() ? tuned : final_cfg);
  if (selected.empty()) {
    auto it = selected_by_key.find(tune_key);
    if (it != selected_by_key.end()) selected = it->second + " (cached)";
    else selected = "unknown";
  } else {
    selected_by_key[tune_key] = selected;
  }
  // Strip commas so CSV rows stay one field.
  for (char& c : selected)
    if (c == ',') c = ';';
  return selected;
}

static void cpuReference(const std::vector<float>& a, const std::vector<float>& b,
                         const std::vector<float>& c, std::vector<float>& y,
                         int m, int n, int k, int ta, int tb, float alpha,
                         float beta, int c0, int c1) {
  auto rows = [&](int first, int last) {
    for (int i = first; i < last; ++i)
      for (int j = 0; j < n; ++j) {
        float acc = 0.0f;
        for (int kk = 0; kk < k; ++kk) {
          const float av = ta ? a[kk * m + i] : a[i * k + kk];
          const float bv = tb ? b[j * k + kk] : b[kk * n + j];
          acc += av * bv;
        }
        float value = alpha * acc;
        if (beta != 0.0f)
          value += beta * c[(c0 == 1 ? 0 : i) * c1 + (c1 == 1 ? 0 : j)];
        y[i * n + j] = value;
      }
  };
  const unsigned hw = std::thread::hardware_concurrency();
  const unsigned threads = std::min<unsigned>(hw ? hw : 4, std::max(1, m));
  if (threads == 1 || m < 8) {
    rows(0, m);
    return;
  }
  std::vector<std::thread> pool;
  const int chunk = (m + static_cast<int>(threads) - 1) / static_cast<int>(threads);
  for (unsigned t = 0; t < threads; ++t) {
    const int first = static_cast<int>(t) * chunk,
              last = std::min(m, first + chunk);
    if (first < last) pool.emplace_back(rows, first, last);
  }
  for (auto& thread : pool) thread.join();
}

struct Case {
  int m, n, k, ta, tb, c0, c1;
  float alpha, beta;
  const char* phase;
};

static int runCase(const Case& test) {
  const size_t aCount = test.ta ? static_cast<size_t>(test.k) * test.m
                                : static_cast<size_t>(test.m) * test.k;
  const size_t bCount = test.tb ? static_cast<size_t>(test.n) * test.k
                                : static_cast<size_t>(test.k) * test.n;
  const size_t yCount = static_cast<size_t>(test.m) * test.n;
  const size_t cCount =
      test.beta == 0.0f ? 0 : static_cast<size_t>(test.c0) * test.c1;
  std::mt19937 rng(1234 + test.m + test.n + test.k + test.ta + test.tb);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
  std::vector<float> af(aCount), bf(bCount), cf(cCount), reference(yCount);
  for (float& value : af) value = dist(rng);
  for (float& value : bf) value = dist(rng);
  for (float& value : cf) value = dist(rng);
  cpuReference(af, bf, cf, reference, test.m, test.n, test.k, test.ta, test.tb,
               test.alpha, test.beta, test.c0, test.c1);

  std::vector<Elem> a(aCount), b(bCount), c(cCount), y(yCount);
  for (size_t i = 0; i < aCount; ++i) a[i] = fromFloat(af[i]);
  for (size_t i = 0; i < bCount; ++i) b[i] = fromFloat(bf[i]);
  for (size_t i = 0; i < cCount; ++i) c[i] = fromFloat(cf[i]);

  void *da = nullptr, *db = nullptr, *dc = nullptr, *dy = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc(&da, aCount * sizeof(Elem)));
  HIP_CHECK(hipMalloc(&db, bCount * sizeof(Elem)));
  HIP_CHECK(hipMalloc(&dy, yCount * sizeof(Elem)));
  if (cCount) HIP_CHECK(hipMalloc(&dc, cCount * sizeof(Elem)));
  HIP_CHECK(hipMemcpy(da, a.data(), aCount * sizeof(Elem), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(db, b.data(), bCount * sizeof(Elem), hipMemcpyHostToDevice));
  if (cCount)
    HIP_CHECK(hipMemcpy(dc, c.data(), cCount * sizeof(Elem), hipMemcpyHostToDevice));

  char shape_buf[96];
  std::snprintf(shape_buf, sizeof(shape_buf), "M%d_N%d_K%d_ta%d_tb%d%s", test.m,
                test.n, test.k, test.ta, test.tb,
                test.beta != 0.0f ? "_bias" : "");
  hipdnn_ep_test::timing::CaseScope timing_case(
      (std::string("gemm/") + kLeafName).c_str(), shape_buf, "hip_gemm");

  char tune_key[64];
  std::snprintf(tune_key, sizeof(tune_key), "%dx%dx%d_ta%d_tb%d_d%d", test.m,
                test.n, test.k, test.ta, test.tb, kGemmDtype);

  int status = 0;
  const std::string selected_config = capture_selected_config(tune_key, [&] {
    for (int w = 0; w < 3; ++w) {
      status = hip_gemm(stream, da, db, dc, dy, test.m, test.n, test.k,
                        test.alpha, test.beta, test.ta, test.tb, kGemmDtype,
                        test.c0, test.c1);
      if (status != 0) return;
    }
    if (hipStreamSynchronize(stream) != hipSuccess) status = 1;
  });

  double avg_ms = 0.0;
  double worst = 0.0;
  int bad = 0;
  if (status == 0) {
    HIP_CHECK(hipMemcpy(y.data(), dy, yCount * sizeof(Elem),
                        hipMemcpyDeviceToHost));
    double ssd = 0.0, ssr = 0.0;
    for (size_t i = 0; i < yCount; ++i) {
      const double got = static_cast<double>(toFloat(y[i]));
      const double ref = static_cast<double>(reference[i]);
      const double error = std::abs(got - ref);
      const double relative = error / std::max(1.0, std::abs(ref));
      worst = std::max(worst, relative);
      if (relative > kTolerance) ++bad;
      ssd += error * error;
      ssr += ref * ref;
    }
    const double rel_l2 = ssr > 0.0 ? std::sqrt(ssd / ssr) : std::sqrt(ssd);

    // Weight side is B; rotate it the way consecutive layers read DRAM.
    const hipdnn_ep_test::ColdCopies weights({{db, bCount * sizeof(Elem)}});
    avg_ms = hipdnn_ep_test::rotatedMeanMs(
        stream, /*legacy*/ 20, weights, [&](int i) {
          return hip_gemm(stream, da, weights.part(i, 0), dc, dy, test.m, test.n,
                          test.k, test.alpha, test.beta, test.ta, test.tb,
                          kGemmDtype, test.c0, test.c1);
        });

    const bool pass = bad == 0;
    std::printf(
        "%s %s M=%d N=%d K=%d ta=%d tb=%d bias=%d dtype=%d max_rel=%.4g "
        "time_ms=%.4f fail=%d/%zu config=%s\n",
        kDtypeName, test.phase, test.m, test.n, test.k, test.ta, test.tb,
        test.beta != 0.0f, kGemmDtype, worst, avg_ms, bad, yCount,
        selected_config.c_str());

    hipdnn_ep_test::CsvWriter csv;
    hipdnn_ep_test::CsvRow row;
    row.shape = shape_buf;
    row.config = selected_config;
    row.time_ms = avg_ms;
    row.rel_l2 = rel_l2;
    row.verdict = pass ? "PASS" : "FAIL";
    csv.write(row);

    hipFree(da);
    hipFree(db);
    hipFree(dy);
    if (dc) hipFree(dc);
    hipStreamDestroy(stream);
    return pass ? 0 : 1;
  }

  std::fprintf(stderr, "%s %s hip_gemm rc=%d\n", kDtypeName, test.phase, status);
  hipFree(da);
  hipFree(db);
  hipFree(dy);
  if (dc) hipFree(dc);
  hipStreamDestroy(stream);
  return 1;
}

int main(int argc, char** argv) {
  hipdnn_ep_test::timing::printTimingMode();
  static const Case cases[] = {
      {1, 2048, 2048, 0, 1, 1, 2048, 1.0f, 0.0f, "GemvNt"},
      {1, 4096, 4096, 0, 0, 1, 4096, 1.0f, 0.25f, "GemvNn-bias"},
      {16, 4096, 11008, 0, 1, 1, 4096, 1.0f, 0.0f, "Wmma"},
      {64, 4096, 14336, 1, 1, 64, 1, 1.0f, 0.25f, "fixed-layout-bias"},
      {128, 1024, 4096, 1, 0, 128, 1, 1.0f, 0.0f, "fixed-layout"},
      {512, 2048, 2048, 0, 0, 1, 2048, 1.0f, 0.25f, "fixed-layout-bias"},
      {1, 11008, 4096, 0, 1, 1, 11008, 1.0f, 0.25f, "GemvNt-bias"},
      {1, 4096, 14336, 0, 0, 1, 4096, 1.0f, 0.0f, "GemvNn"},
      {16, 1024, 4096, 0, 1, 1, 1024, 0.75f, 0.0f, "Wmma"},
      {16, 2048, 2048, 1, 0, 16, 1, 1.0f, 0.25f, "fixed-layout-bias"},
      {64, 4096, 4096, 0, 1, 1, 4096, 1.0f, 0.25f, "Wmma-bias"},
      {64, 11008, 4096, 1, 1, 64, 1, 1.0f, 0.0f, "fixed-layout"},
      {128, 4096, 14336, 0, 0, 1, 4096, 1.0f, 0.25f, "fixed-layout-bias"},
      {128, 1024, 4096, 0, 1, 1, 1024, 1.0f, 0.0f, "Wmma"},
      {512, 4096, 4096, 1, 1, 512, 1, 1.0f, 0.25f, "fixed-layout-bias"},
      {512, 11008, 4096, 0, 0, 1, 11008, 1.0f, 0.0f, "fixed-layout"},
      {1024, 2048, 2048, 0, 1, 1, 2048, 1.0f, 0.25f, "Wmma-bias"},
      {1024, 4096, 4096, 1, 0, 1024, 1, 1.0f, 0.0f, "fixed-layout"},
      {16, 4096, 14336, 1, 1, 16, 1, 1.0f, 0.0f, "fixed-layout"},
      {64, 1024, 4096, 0, 0, 1, 1024, 1.0f, 0.25f, "fixed-layout-bias"},
      {128, 2048, 2048, 1, 0, 128, 1, 1.0f, 0.0f, "fixed-layout"},
      {512, 4096, 14336, 0, 1, 1, 4096, 1.0f, 0.25f, "Wmma-bias"},
      {1024, 1024, 4096, 0, 0, 1, 1024, 1.0f, 0.0f, "fixed-layout"},
      {16, 11008, 4096, 0, 1, 1, 11008, 1.0f, 0.25f, "Wmma-bias"},
      {64, 2048, 2048, 1, 1, 64, 1, 1.0f, 0.0f, "fixed-layout"},
      {128, 4096, 4096, 0, 0, 1, 4096, 1.0f, 0.25f, "fixed-layout-bias"},
      {512, 1024, 4096, 1, 0, 512, 1, 1.0f, 0.0f, "fixed-layout"},
      {1024, 11008, 4096, 0, 1, 1, 11008, 1.0f, 0.25f, "Wmma-bias"},
      {128, 11008, 4096, 1, 1, 128, 1, 1.0f, 0.0f, "fixed-layout"},
      {64, 4096, 14336, 0, 0, 1, 4096, 1.0f, 0.25f, "fixed-layout-bias"},
  };

  std::vector<int> positional;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--coverage") == 0) {
      ++i;
      continue;
    }
    if (std::strcmp(argv[i], "--timing") == 0) {
      ++i;
      continue;
    }
    positional.push_back(std::atoi(argv[i]));
  }
  if (positional.size() >= 5) {
    const int ta = positional[3], tb = positional[4];
    const Case custom{positional[0], positional[1], positional[2], ta, tb,
                       ta ? positional[0] : 1, ta ? 1 : positional[1], 1.0f, 0.0f,
                       "custom"};
    return runCase(custom);
  }

  const int tier = hipdnn_ep_test::resolveCoverageTier(argc, argv);
  static const size_t kTier1[] = {0, 1, 2, 3, 4, 5};
  static const size_t kTier2[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,
                                   10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
  const size_t total = sizeof(cases) / sizeof(cases[0]);
  const size_t* indexes = tier == 1 ? kTier1 : (tier == 2 ? kTier2 : nullptr);
  const size_t count = tier == 1   ? sizeof(kTier1) / sizeof(kTier1[0])
                       : tier == 2 ? sizeof(kTier2) / sizeof(kTier2[0])
                                   : total;
  std::printf("%s coverage=%d -> running %zu/%zu comprehensive cases (dtype=%d)\n",
              kDtypeName, tier, count, total, kGemmDtype);
  int failures = 0;
  for (size_t i = 0; i < count; ++i)
    failures |= runCase(cases[indexes ? indexes[i] : i]);
  std::printf("%s %s (%d/%zu case(s) failing)\n", kDtypeName,
              failures ? "SOME FAILED" : "ALL PASS", failures, count);
  return failures ? 1 : 0;
}
