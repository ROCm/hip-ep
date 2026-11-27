/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Self-contained fp16 GEMM kernel-unit-test leaf under unit_test/.
 * Timing uses unit_test/common adaptive complete-call policy
 * (--timing screen|standard|confirm via HIPDNN_UT_TIMING).
 */
#include "../../common/ut_gpu_execution_scope.h"
#include <hip/hip_runtime.h>
#include "../../common/hip_adaptive_timing.h"
#include "../../common/ut_gemm_rank_input.h"
#ifdef HIPDNN_UT_GPU_PHASE_HEADER
#include HIPDNN_UT_GPU_PHASE_HEADER
#elif !defined(HIPDNN_UT_GPU_PHASE_LINKED)
namespace hipdnn_ep_test {
struct GpuPhase { void begin() {} void finish(bool = false) {} void timingBegin() {} void timingEnd() {} };
}
#endif
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
#include <iomanip>
#include <sstream>
#include <climits>
#include <cstdint>
#include <map>
#include <memory>
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
static constexpr int kOutputDtype = 0;
using Output = Elem;
static float outputFloat(Output v) { return toFloat(v); }
static Output outputFromFloat(float v) { return fromFloat(v); }

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
                     const ColdCopies& copies, Launch&& launch,
                     const timing::RankOptions* ranking = nullptr) {
  return timing::adaptiveRotatedMeanMs(
      stream, copies, launch, holdReleaseFlag(),
      [&](const int* flag) { holdStreamKernel<<<1, 1, 0, stream>>>(flag); },
      warmTiming(), kColdSettleBytes, "gemm_fp16", "hip_gemm_strided_batched", timing::NoPreparation{}, ranking);
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
      !final_cfg.empty() ? final_cfg : (!lut.empty() ? lut : tuned);
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

struct Case {
  int m, n, k, ta, tb, c0, c1;
  float alpha, beta;
  const char* phase;
  int batches = 1, a_group = 1, b_group = 1;
  int64_t sa = -1, sb = -1, sc = -1, sy = -1;
};

static Case normalizedCase(Case t) {
  if (t.m <= 0 || t.n <= 0 || t.k < 0 || t.ta < 0 || t.ta > 1 ||
      t.tb < 0 || t.tb > 1 || t.batches < 1 || t.batches > 65535 ||
      t.a_group < 1 || t.b_group < 1 || t.batches % t.a_group ||
      t.batches % t.b_group || t.sa < -1 || t.sb < -1 || t.sc < -1 ||
      t.sy < -1 || !std::isfinite(t.alpha) || !std::isfinite(t.beta))
    throw std::invalid_argument("GEMM: invalid dimensions, groups, strides or scales");
  if (t.c1 == -1) t.c1 = t.n;
  if ((t.c0 != 1 && t.c0 != t.m) || (t.c1 != 1 && t.c1 != t.n))
    throw std::invalid_argument("GEMM: invalid C broadcast");
  if (t.sa < 0) t.sa = int64_t(t.m) * t.k;
  if (t.sb < 0) t.sb = int64_t(t.n) * t.k;
  if (t.sc < 0) t.sc = int64_t(t.c0) * t.c1;
  if (t.sy < 0) t.sy = int64_t(t.m) * t.n;
  if (t.batches > 1 && t.sy < int64_t(t.m) * t.n)
    throw std::invalid_argument("GEMM: overlapping output batches");
  return t;
}

static size_t operandCount(int batches, int group, int64_t stride,
                           uint64_t matrix, size_t element) {
  const uint64_t last = uint64_t(batches / group - 1);
  const uint64_t limit = uint64_t(PTRDIFF_MAX) / element;
  if (matrix > limit || (last && uint64_t(stride) > (limit - matrix) / last))
    throw std::invalid_argument("GEMM: operand extent overflow");
  return size_t(last * uint64_t(stride) + matrix);
}

static void cpuReference(const std::vector<float>& a, const std::vector<float>& b,
                         const std::vector<float>& c, std::vector<float>& y,
                         const Case& t) {
  // Independent FP32 accumulators across output columns make NN references
  // vectorizable without reassociating the reduction over K.
  auto rows = [&](int64_t first, int64_t last) {
    for (int64_t index = first; index < last; ++index) {
      const int batch = int(index / t.m), row = int(index % t.m);
      const size_t ao = size_t(batch / t.a_group) * t.sa;
      const size_t bo = size_t(batch / t.b_group) * t.sb;
      const size_t co = size_t(batch) * t.sc, yo = size_t(batch) * t.sy;
      for (int j = 0; j < t.n; j += 16) {
        float acc[16] = {};
        const int width = std::min(16, t.n - j);
        for (int kk = 0; kk < t.k; ++kk) {
          const float av = a[ao + (t.ta ? size_t(kk)*t.m+row : size_t(row)*t.k+kk)];
          for (int v = 0; v < width; ++v) {
            const float bv = b[bo + (t.tb ? size_t(j+v)*t.k+kk : size_t(kk)*t.n+j+v)];
            acc[v] += av * bv;
          }
        }
        for (int v = 0; v < width; ++v) {
          float value = t.alpha * acc[v];
          if (t.beta != 0.0f)
            value += t.beta * c[co + size_t(t.c0 == 1 ? 0 : row)*t.c1 + (t.c1 == 1 ? 0 : j+v)];
          y[yo + size_t(row)*t.n+j+v] = value;
        }
      }
    }
  };
  const int64_t total = int64_t(t.batches) * t.m;
  const unsigned hw = std::thread::hardware_concurrency();
  const unsigned threads = unsigned(std::min<int64_t>(hw ? hw : 4, total));
  const int64_t chunk = (total + threads - 1) / threads;
  std::vector<std::thread> pool;
  for (unsigned i = 0; i < threads; ++i) {
    const int64_t first = i * chunk, last = std::min(total, first + chunk);
    if (first < last) pool.emplace_back(rows, first, last);
  }
  for (auto& thread : pool) thread.join();
}

static int runCase(const Case& raw, const hipdnn_ep_test::timing::RankRow* rank_row = nullptr) {
  const Case t = normalizedCase(raw);
  std::ostringstream description;
  description << "M" << t.m << "_N" << t.n << "_K" << t.k << "_ta" << t.ta << "_tb" << t.tb;
  if (t.beta != 0) {
    description << "_bias";
    if (t.c0 != 1 || t.c1 != t.n) description << "_crows" << t.c0 << "_ccols" << t.c1;
    if (t.beta != 1) description << "_beta" << std::setprecision(9) << t.beta;
  }
  if (t.alpha != 1) description << "_alpha" << std::setprecision(9) << t.alpha;
  if (t.batches != 1 || t.a_group != 1 || t.b_group != 1)
    description << "_B" << t.batches << "_ag" << t.a_group << "_bg" << t.b_group;
  if (t.sa != int64_t(t.m)*t.k || t.sb != int64_t(t.n)*t.k ||
      t.sc != int64_t(t.c0)*t.c1 || t.sy != int64_t(t.m)*t.n)
    description << "_sa" << t.sa << "_sb" << t.sb << "_sc" << t.sc << "_sy" << t.sy;
  const std::string shape = rank_row ? rank_row->shape : description.str();
  hipdnn_ep_test::timing::CaseScope timing_case(
      (std::string("gemm/") + kLeafName).c_str(), shape.c_str(), "hip_gemm_strided_batched");
  const UtGemmShape rank_shape{t.m,t.n,t.k,t.ta,t.tb,kGemmDtype,t.c0,t.c1,t.alpha,t.beta,
      kOutputDtype,t.batches,t.a_group,t.b_group,t.sa,t.sb,t.sc,t.sy};
  if (rank_row && std::getenv("HIPDNN_UT_RANK_CATALOG")) {
#ifdef HIPDNN_UT_RANK_GEMM_ADAPTER
    hipdnn_ep_test::timing::printGemmCatalog(rank_shape);
    return 0;
#else
    throw std::runtime_error("UT rank: GEMM catalog requires standalone RANK=1 build");
#endif
  }
#ifdef HIPDNN_UT_RANK_GEMM_ADAPTER
  std::unique_ptr<hipdnn_ep_test::timing::RankOptions> ranking;
  if (rank_row)
    ranking = std::make_unique<hipdnn_ep_test::timing::RankOptions>(
        hipdnn_ep_test::timing::gemmRankOptions(rank_shape,*rank_row,[] { return 0.0; }));
#else
  if (rank_row) throw std::runtime_error("UT rank: GEMM requires standalone RANK=1 build");
#endif
  const size_t ac = std::max<size_t>(1,operandCount(t.batches,t.a_group,t.sa,uint64_t(t.m)*t.k,sizeof(Elem)));
  const size_t bc = std::max<size_t>(1,operandCount(t.batches,t.b_group,t.sb,uint64_t(t.n)*t.k,sizeof(Elem)));
  const size_t cc = t.beta == 0 ? 0 : operandCount(t.batches,1,t.sc,uint64_t(t.c0)*t.c1,sizeof(Output));
  const size_t yc = operandCount(t.batches,1,t.sy,uint64_t(t.m)*t.n,sizeof(Output));
  const float sentinel = -123.5f;
  constexpr size_t guardElements = 32, guardBytes = 64;
  std::vector<float> af(ac),bf(bc),cf(cc),reference(yc,sentinel);
  std::vector<Elem> a(ac),b(bc);
  std::vector<Output> c(cc),y(yc+2*guardElements,outputFromFloat(sentinel));
  std::mt19937 rng(1234+t.m+t.n+t.k+t.ta+t.tb+t.batches);
  std::uniform_real_distribution<float> dist(-0.5f,0.5f);
  // The reference consumes exactly the input/C bit patterns sent to the GPU.
  for (size_t i=0;i<ac;++i) {a[i]=fromFloat(dist(rng));af[i]=toFloat(a[i]);}
  for (size_t i=0;i<bc;++i) {b[i]=fromFloat(dist(rng));bf[i]=toFloat(b[i]);}
  for (size_t i=0;i<cc;++i) {c[i]=outputFromFloat(dist(rng));cf[i]=outputFloat(c[i]);}
  cpuReference(af,bf,cf,reference,t);
  std::printf("CASE_CPU_READY leaf=%s shape=%s entries=%zu reference=full_represented_inputs\n",kLeafName,shape.c_str(),yc);
  std::fflush(stdout);

  hipdnn_ep_test::GpuExecutionScope execution([] {
    if (hipDeviceSynchronize()!=hipSuccess) throw std::runtime_error("GEMM device synchronization failed");
  });
  hipdnn_ep_test::GpuPhase phase;
  phase.begin();
  void *da=nullptr,*db=nullptr,*dc=nullptr,*dy=nullptr,*dw=nullptr;
  hipStream_t stream=nullptr;
  auto cleanup=[&](void*) {
    if (stream) (void)hipStreamSynchronize(stream);
    for (void* p : {da,db,dc,dy,dw}) if(p)(void)hipFree(p);
    if (stream) (void)hipStreamDestroy(stream);
    phase.finish(true);
  };
  const std::unique_ptr<void,decltype(cleanup)> owner(reinterpret_cast<void*>(1),cleanup);
  size_t workspaceBytes=0;
  if (hip_gemm_workspace_size(t.m,t.n,t.k,t.ta,t.tb,kGemmDtype,kOutputDtype,
      t.batches,t.a_group,t.b_group,&workspaceBytes)) return 1;
  HIP_CHECK(hipStreamCreate(&stream));
  HIP_CHECK(hipMalloc(&da,ac*sizeof(Elem)));
  HIP_CHECK(hipMalloc(&db,bc*sizeof(Elem)));
  HIP_CHECK(hipMalloc(&dy,y.size()*sizeof(Output)));
  if(cc)HIP_CHECK(hipMalloc(&dc,cc*sizeof(Output)));
  if(workspaceBytes) {
    HIP_CHECK(hipMalloc(&dw,workspaceBytes+2*guardBytes));
    HIP_CHECK(hipMemset(dw,0xa5,workspaceBytes+2*guardBytes));
  }
  HIP_CHECK(hipMemcpy(da,a.data(),ac*sizeof(Elem),hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(db,b.data(),bc*sizeof(Elem),hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dy,y.data(),y.size()*sizeof(Output),hipMemcpyHostToDevice));
  if(cc)HIP_CHECK(hipMemcpy(dc,c.data(),cc*sizeof(Output),hipMemcpyHostToDevice));
  void* output=static_cast<Output*>(dy)+guardElements;
  void* workspace=dw?static_cast<char*>(dw)+guardBytes:nullptr;
  auto call=[&](const void* weights,void* scratch,size_t bytes) {
    return hip_gemm_strided_batched(stream,da,weights,dc,output,t.m,t.n,t.k,t.alpha,t.beta,
        t.ta,t.tb,kGemmDtype,kOutputDtype,t.c0,t.c1,t.batches,t.sa,t.sb,t.sc,t.sy,
        t.a_group,t.b_group,scratch,bytes);
  };
  // A too-small or aliased caller workspace must fail before any kernel runs.
  if (workspaceBytes && (!call(db,workspace,workspaceBytes-1) || !call(db,output,workspaceBytes)))
    throw std::runtime_error("GEMM workspace guard failed");
  auto validate=[&]() {
    if(call(db,workspace,workspaceBytes)||hipStreamSynchronize(stream)!=hipSuccess||
       hipMemcpy(y.data(),dy,y.size()*sizeof(Output),hipMemcpyDeviceToHost)!=hipSuccess)
      throw std::runtime_error("GEMM launch/readback failed");
    double ssd=0,ssr=0;
    for(size_t i=0;i<guardElements;++i)
      if(outputFloat(y[i])!=sentinel || outputFloat(y[guardElements+yc+i])!=sentinel)
        throw std::runtime_error("GEMM output guard overwritten");
    for(size_t i=0;i<yc;++i) {
      const double got=outputFloat(y[guardElements+i]),ref=reference[i],delta=got-ref;
      const int64_t batch=t.batches==1?0:int64_t(i)/t.sy,
                    local=t.batches==1?int64_t(i):int64_t(i)%t.sy;
      const bool active=batch<t.batches&&local<int64_t(t.m)*t.n;
      if(!std::isfinite(got)||!std::isfinite(ref)||
         (active?std::abs(delta)/std::max(1.,std::abs(ref))>kTolerance:got!=sentinel))
        throw std::runtime_error("GEMM CPU reference or output padding FAIL");
      if(active){ssd+=delta*delta;ssr+=ref*ref;}
    }
    if(dw) {
      unsigned char guards[2*guardBytes];
      if(hipMemcpy(guards,dw,guardBytes,hipMemcpyDeviceToHost)!=hipSuccess||
         hipMemcpy(guards+guardBytes,static_cast<char*>(dw)+guardBytes+workspaceBytes,guardBytes,hipMemcpyDeviceToHost)!=hipSuccess)
        throw std::runtime_error("GEMM workspace guard readback failed");
      for(auto x:guards)if(x!=0xa5)throw std::runtime_error("GEMM workspace guard overwritten");
    }
    return std::sqrt(ssd/std::max(ssr,1e-30));
  };
  const hipdnn_ep_test::ColdCopies weights({{db,bc*sizeof(Elem)}});
  auto launch=[&](int i){return call(weights.part(i,0),workspace,workspaceBytes);};
  if(rank_row) {
#ifdef HIPDNN_UT_RANK_GEMM_ADAPTER
    ranking->validate=[select=ranking->select,validate](size_t candidate){select(candidate);return validate();};
    phase.timingBegin();
    hipdnn_ep_test::rotatedMeanMs(stream,0,weights,launch,ranking.get());
    phase.timingEnd();
    return 0;
#endif
  }
  std::string selected;
  double relL2=0;
  selected=capture_selected_config(shape.c_str(),[&]{relL2=validate();});
  phase.timingBegin();
  const double ms=hipdnn_ep_test::rotatedMeanMs(stream,0,weights,launch);
  phase.timingEnd();
  // The post-timing check also covers reused caller workspace and rotations.
  relL2=validate();
  std::printf("%s %s shape=%s time_ms=%.9f relL2=%.9g config=%s PASS\n",kDtypeName,t.phase,
      shape.c_str(),ms,relL2,selected.c_str());
  hipdnn_ep_test::CsvWriter csv;
  csv.write({shape,selected,ms,relL2,"PASS"});
  return 0;
}

int main(int argc, char** argv) {
  hipdnn_ep_test::timing::printTimingMode();
  if (hipdnn_ep_test::timing::rankRequested()) {
    try {
      for (const auto& row : hipdnn_ep_test::timing::readRankRows(std::string("gemm/") + kLeafName)) {
        const auto s = hipdnn_ep_test::timing::gemmRankShape(row.shape, kGemmDtype, kOutputDtype);
        const Case test{s.m, s.n, s.k, s.ta, s.tb, s.crows, s.ccols, s.alpha, s.beta, "rank",
            s.batches,s.a_group,s.b_group,s.stride_a,s.stride_b,s.stride_c,s.stride_y};
        if (runCase(test, &row)) return 1;
      }
      return 0;
    } catch (const std::exception& error) {
      std::fprintf(stderr, "GEMM UT rank: %s\n", error.what());
      return 1;
    }
  }
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

  Case custom{128,512,256,0,1,1,-1,1.0f,0.0f,"custom"};
  bool customRequested=false;
  std::vector<int> positional;
  try {
    for(int i=1;i<argc;++i) {
      const std::string flag=argv[i];
      if(flag=="--coverage"||flag=="--timing") {if(++i==argc)return 2;continue;}
      int* integer=nullptr;
      int64_t* stride=nullptr;
      float* scale=nullptr;
      if(flag=="--m")integer=&custom.m;
      else if(flag=="--n")integer=&custom.n;
      else if(flag=="--k")integer=&custom.k;
      else if(flag=="--ta")integer=&custom.ta;
      else if(flag=="--tb")integer=&custom.tb;
      else if(flag=="--batch")integer=&custom.batches;
      else if(flag=="--a-group")integer=&custom.a_group;
      else if(flag=="--b-group")integer=&custom.b_group;
      else if(flag=="--crows")integer=&custom.c0;
      else if(flag=="--ccols")integer=&custom.c1;
      else if(flag=="--stride-a")stride=&custom.sa;
      else if(flag=="--stride-b")stride=&custom.sb;
      else if(flag=="--stride-c")stride=&custom.sc;
      else if(flag=="--stride-y")stride=&custom.sy;
      else if(flag=="--alpha")scale=&custom.alpha;
      else if(flag=="--beta")scale=&custom.beta;
      if(integer||stride||scale) {
        if(++i==argc)return 2;
        const std::string value=argv[i];size_t used=0;
        if(integer)*integer=std::stoi(value,&used);
        else if(stride)*stride=std::stoll(value,&used);
        else *scale=std::stof(value,&used);
        if(used!=value.size())return 2;
        customRequested=true;
      } else {
        if(flag.empty()||flag[0]=='-')return 2;
        size_t used=0;const int value=std::stoi(flag,&used);
        if(used!=flag.size())return 2;
        positional.push_back(value);
      }
    }
    if(!positional.empty()) {
      if(customRequested||positional.size()!=5)return 2;
      custom.m=positional[0];custom.n=positional[1];custom.k=positional[2];
      custom.ta=positional[3];custom.tb=positional[4];customRequested=true;
    }
    if(customRequested)return runCase(custom);
  } catch(const std::exception& error) {
    std::fprintf(stderr,"GEMM custom: %s\n",error.what());return 2;
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
    try { failures |= runCase(cases[indexes ? indexes[i] : i]); }
    catch(const std::exception& error){std::fprintf(stderr,"GEMM: %s\n",error.what());return 1;}
  // Every tier crosses transpose, C broadcast, tails, periodic sharing and
  // padded/zero batch strides; tiers thin continuous dimensions only.
  try {
    const int tail = tier == 1 ? 17 : tier == 2 ? 33 : 65;
    for(int ta=0;ta<2;++ta)for(int tb=0;tb<2;++tb) {
      for(int broadcast=0;broadcast<4;++broadcast) {
        Case t{tail,67,129,ta,tb,broadcast&1?tail:1,broadcast&2?67:1,
               0.75f,0.25f,"batched-tail",4,2,1};
        t.sa=int64_t(t.m)*t.k+13;t.sb=0;
        t.sc=int64_t(t.c0)*t.c1+7;t.sy=int64_t(t.m)*t.n+11;
        failures|=runCase(t);
      }
    }
    failures|=runCase({2,257,0,0,0,1,257,1.0f,0.5f,"K-zero",2});
    failures|=runCase({2,256,4096,0,0,1,256,1.0f,0.0f,"parallel-NN",4,1,2});
  } catch(const std::exception& error) {
    std::fprintf(stderr,"GEMM coverage: %s\n",error.what());return 1;
  }
  std::printf("%s %s (%d/%zu case(s) failing)\n", kDtypeName,
              failures ? "SOME FAILED" : "ALL PASS", failures, count);
  return failures ? 1 : 0;
}
