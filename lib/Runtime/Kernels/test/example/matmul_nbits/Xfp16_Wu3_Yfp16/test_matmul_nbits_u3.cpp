/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * custom_kernels MatMulNBits bits=3 (uint3, continuous bitstream) Verification.
 *
 * Unit-test model: this file is entirely self-contained (no example/common/,
 * no python, no on-disk data) -- inputs, the uint3 packing, and the
 * dequant+matmul CPU reference are all generated in-process. The case set is
 * every categorical situation this op supports (group_size in {32,64,128},
 * zero-points on/off, dtype fp16/fp32) crossed with a comprehensive list of
 * typical (M,K,N) shapes -- M in {1,16,64,128,512} (decode/GEMV through
 * several prefill points) round-robined across 4 representative real-model
 * (K,N) layer families (FFN gate/up-proj, attn-proj, FFN down-proj, o-proj) --
 * NOT the full M/K/N shape space, and not a blind M x KN cross (which would
 * multiply by the 12-way categorical cross below and blow the tier3 time
 * budget; see kShapes4's comment). COVERAGE=1|2|3 (default 3) only thins how
 * many typical shapes run; every tier still touches every
 * group_size/zero-points/dtype value. The CPU reference is multithreaded
 * (see genCase() below) so the wider shape set stays inside budget.
 *
 * uint3 packing: value k occupies bits [3k, 3k+3) of row n's bitstream,
 * LSB-first -- a continuous per-row 3-bit bitstream (custom format, NOT an
 * ONNX MatMulNBits convention; see matmul_nbits_kernel.hip), row byte stride
 * = ceil(K*3/8).
 *
 * A:      FP16 (or FP32) row-major [M, K]
 * B:      uint8 packed [N, ceil(K*3/8)], continuous 3-bit bitstream
 * scales: FP16 [N, num_groups_k], per-column per-group
 * zeros:  uint8 [N, num_groups_k], per-column per-group zero point in [0,7]
 *         (optional; default zero point is 4 when absent), passed to the
 *         kernel raw (zp_elem_size=1, plain per-element convention -- the
 *         3-bit-packed zero_points path is exercised by the kernel's own
 *         tests, not this small correctness UT).
 * C:      FP16 (or FP32) row-major [M, N]
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip_custom_kernels.h"

#ifndef HIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS
#include "matmul_nbits_autotune.h"

#ifndef HIPDNN_LUT_LINKED_EXTERNALLY
namespace hipdnn_ep {
namespace matmul_nbits_autotune {
Result resolve(const Request&, WmmaValidator, GemvValidator, void*) { return {}; }
Stats stats() { return {}; }
}  // namespace matmul_nbits_autotune
}  // namespace hipdnn_ep
#else
#include "lut_fb_path.h"
static const unsigned char kLutBlob0[] = {
#embed HIPDNN_LUT_FB
};
extern "C" const unsigned char* const kMatmulNbitsLutBlobs[1]   = { kLutBlob0 };
extern "C" const size_t               kMatmulNbitsLutBlobSizes[1] = { sizeof(kLutBlob0) };
extern "C" const size_t               kMatmulNbitsLutBlobCount    = 1;
#endif  // HIPDNN_LUT_LINKED_EXTERNALLY
#endif  // HIPDNN_KERNEL_UT_LINKS_SHARED_KERNELS

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <string>
#include <thread>
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

// Reports which config the op actually ran, read back from its own
// HIPDNN_MATMUL_LUT_LOG / HIPDNN_MATMUL_AUTOTUNE_LOG output rather than guessed
// from the mode. The op logs a selection only on the first encounter with a
// tune key (which does not include the activation dtype), so a later case
// sharing that key reuses the earlier selection silently -- hence the memo,
// which keeps every row attributable to a real config.
template <typename Fn>
static std::string capture_selected_config(const char* tune_key, Fn&& launch) {
  static std::map<std::string, std::string> selected_by_key;
  static const char* const kCapture = "out/_config_capture.tmp";
  _putenv_s("HIPDNN_MATMUL_AUTOTUNE_LOG", "1");
  _putenv_s("HIPDNN_MATMUL_LUT_LOG", "1");
  std::fflush(stderr);
  FILE* redirected = std::freopen(kCapture, "w", stderr);
  launch();
  std::fflush(stderr);
  std::freopen("CON", "w", stderr);
  if (!redirected) return "log-unavailable";

  std::string lut, tuned;
  {
    // Scoped so the stream is closed before the remove below; Windows refuses
    // to delete a file that is still open.
    std::ifstream log(kCapture);
    std::string line;
    while (std::getline(log, line)) {
      if (line.find("LUT hit") != std::string::npos) {
        lut = "lookup:" + line.substr(line.find("config["));
        const size_t src = line.find('(');
        if (src != std::string::npos)
          lut += " " + line.substr(src, line.find(')', src) + 1 - src);
      } else if (line.find("best config[") != std::string::npos) {
        tuned = "autotune:" + line.substr(line.find("best config["));
        // Drop the tuner's own "(x ms/iter)": it is a peak sample taken under
        // the clock state of the sweep, not this row's time_ms, and showing
        // both invites reading them as the same measurement.
        const size_t timing = tuned.rfind(" (");
        if (timing != std::string::npos &&
            tuned.find("ms/iter", timing) != std::string::npos)
          tuned.erase(timing);
      }
    }
  }
  std::remove(kCapture);

  std::string selected = !lut.empty() ? lut : tuned;
  if (selected.empty()) {
    auto it = selected_by_key.find(tune_key);
    // No selection logged and none remembered: this shape reached a path with
    // no tunable config at all.
    if (it == selected_by_key.end()) return "naive";
    selected = it->second;
  } else {
    selected_by_key[tune_key] = selected;
  }
  std::replace(selected.begin(), selected.end(), ',', ';');
  return selected;
}

static float half_to_float(__half h) {
  uint16_t bits;
  std::memcpy(&bits, &h, sizeof(bits));
  uint32_t sign = (bits >> 15) & 1;
  uint32_t exp = (bits >> 10) & 0x1F;
  uint32_t mant = bits & 0x3FF;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) f = sign << 31;
    else {
      exp = 1;
      while (!(mant & 0x400)) { mant <<= 1; exp--; }
      mant &= 0x3FF;
      f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
  } else if (exp == 31) f = (sign << 31) | (0xFFu << 23) | (mant << 13);
  else f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
  float result;
  std::memcpy(&result, &f, sizeof(result));
  return result;
}

static __half float_to_half(float x) {
  uint32_t f;
  std::memcpy(&f, &x, 4);
  uint32_t s = (f >> 31) & 1;
  int e = (int)((f >> 23) & 0xff) - 127 + 15;
  uint32_t m = f & 0x7fffff;
  uint16_t bits;
  if (e <= 0) bits = (uint16_t)(s << 15);
  else if (e >= 31) bits = (uint16_t)((s << 15) | 0x7c00);
  else bits = (uint16_t)((s << 15) | (e << 10) | (m >> 13));
  __half h;
  std::memcpy(&h, &bits, 2);
  return h;
}

#define HIP_CHECK(call)                                                     \
    do {                                                                    \
        hipError_t err = (call);                                            \
        if (err != hipSuccess) {                                            \
            std::fprintf(stderr, "HIP error at %s:%d code=%d \"%s\"\n",     \
                         __FILE__, __LINE__, err, hipGetErrorString(err));   \
            std::exit(1);                                                   \
        }                                                                   \
    } while (0)

// Packs `vals` (K values, 0..7) into a continuous per-row 3-bit LSB-first
// bitstream: value k occupies bits [3k, 3k+3). row_bytes = ceil(K*3/8).
static void pack3bit(const uint8_t* vals, int K, std::vector<uint8_t>& out) {
  const int row_bytes = (K * 3 + 7) / 8;
  std::vector<uint8_t> buf(row_bytes + 1, 0);
  for (int k = 0; k < K; ++k) {
    const int bitpos = k * 3;
    const int byte0 = bitpos / 8;
    const int shift = bitpos % 8;
    const uint32_t contrib = static_cast<uint32_t>(vals[k]) << shift;
    buf[byte0] = static_cast<uint8_t>(buf[byte0] | (contrib & 0xFF));
    buf[byte0 + 1] = static_cast<uint8_t>(buf[byte0 + 1] | ((contrib >> 8) & 0xFF));
  }
  out.assign(buf.begin(), buf.begin() + row_bytes);
}

// ============================================================
// In-process data generation + CPU fp32 reference (no python, no disk I/O).
// B is randint(0,8) per element, packed via pack3bit(); scales are
// uniform(0.01,0.05); zeros (when present) are integers in [3,5] (uint8,
// plain per-element convention); dequant is (B - zp) * scale, default zp = 4
// when zeros are absent.
// ============================================================
struct CaseData {
  std::vector<__half> A16;
  std::vector<float> A32;
  std::vector<uint8_t> Bpacked;
  std::vector<__half> scales;
  std::vector<uint8_t> zeros;
  std::vector<__half> Cref16;
  std::vector<float> Cref32;
};

static void genCase(int M, int K, int N, int gs, bool zero, bool fp32,
                    unsigned seed, CaseData& out) {
  const int num_groups_k = (K + gs - 1) / gs;
  const int row_bytes = (K * 3 + 7) / 8;

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> adist(-0.5f, 0.5f);
  std::uniform_int_distribution<int> bdist(0, 7);
  std::uniform_real_distribution<float> sdist(0.01f, 0.05f);
  std::uniform_int_distribution<int> zdist(3, 5);

  std::vector<float> A(static_cast<size_t>(M) * K);
  for (auto& v : A) v = adist(rng);

  std::vector<uint8_t> Bq(static_cast<size_t>(N) * K);
  for (auto& v : Bq) v = static_cast<uint8_t>(bdist(rng));

  std::vector<float> scalesF(static_cast<size_t>(N) * num_groups_k);
  for (auto& v : scalesF) v = sdist(rng);

  std::vector<float> zerosF;
  if (zero) {
    zerosF.resize(static_cast<size_t>(N) * num_groups_k);
    for (auto& v : zerosF) v = static_cast<float>(zdist(rng));
  }

  out.Bpacked.assign(static_cast<size_t>(N) * row_bytes, 0);
  std::vector<uint8_t> packed_row;
  for (int n = 0; n < N; ++n) {
    pack3bit(&Bq[static_cast<size_t>(n) * K], K, packed_row);
    std::memcpy(&out.Bpacked[static_cast<size_t>(n) * row_bytes], packed_row.data(), row_bytes);
  }

  out.scales.resize(scalesF.size());
  for (size_t i = 0; i < scalesF.size(); ++i) out.scales[i] = float_to_half(scalesF[i]);
  if (zero) {
    out.zeros.resize(zerosF.size());
    for (size_t i = 0; i < zerosF.size(); ++i) out.zeros[i] = static_cast<uint8_t>(zerosF[i]);
  }

  // Threaded over N (each output column is independent) so the wider M/K/N
  // shapes in the coverage sweep (buildCases() below) stay inside the shared
  // ~30 min tier3 budget across all 9 UT leaves -- see example/README.md
  // "Shape coverage".
  std::vector<float> Cref(static_cast<size_t>(M) * N, 0.0f);
  auto refColumns = [&](int n0, int n1) {
    for (int n = n0; n < n1; ++n) {
      const uint8_t* row = &Bq[static_cast<size_t>(n) * K];
      const float* srow = &scalesF[static_cast<size_t>(n) * num_groups_k];
      const float* zrow = zero ? &zerosF[static_cast<size_t>(n) * num_groups_k] : nullptr;
      for (int m = 0; m < M; ++m) {
        const float* arow = &A[static_cast<size_t>(m) * K];
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
          const int g = k / gs;
          const float zp = zero ? zrow[g] : 4.0f;
          acc += arow[k] * (static_cast<float>(row[k]) - zp) * srow[g];
        }
        Cref[static_cast<size_t>(m) * N + n] = acc;
      }
    }
  };
  {
    const unsigned hw = std::thread::hardware_concurrency();
    const unsigned nthreads = std::min<unsigned>(hw ? hw : 4u,
                                                  N > 0 ? static_cast<unsigned>(N) : 1u);
    if (nthreads <= 1 || N < 64) {
      refColumns(0, N);
    } else {
      std::vector<std::thread> pool;
      const int chunk = (N + static_cast<int>(nthreads) - 1) / static_cast<int>(nthreads);
      for (unsigned t = 0; t < nthreads; ++t) {
        const int n0 = static_cast<int>(t) * chunk;
        const int n1 = std::min(N, n0 + chunk);
        if (n0 >= n1) break;
        pool.emplace_back(refColumns, n0, n1);
      }
      for (auto& th : pool) th.join();
    }
  }

  if (fp32) {
    out.A32.resize(A.size());
    for (size_t i = 0; i < A.size(); ++i) out.A32[i] = A[i];
    out.Cref32 = std::move(Cref);
  } else {
    out.A16.resize(A.size());
    for (size_t i = 0; i < A.size(); ++i) out.A16[i] = float_to_half(A[i]);
    out.Cref16.resize(Cref.size());
    for (size_t i = 0; i < Cref.size(); ++i) out.Cref16[i] = float_to_half(Cref[i]);
  }
}

namespace sweep {

// tier3: M in {1,16,64,128,512} (decode through several prefill points)
// round-robined -- not a full M x KN cross -- across 4 representative
// real-model (K,N) layer families: FFN gate/up-proj, attn-proj (K=2880
// exercises the group_size zero-padding path: not a multiple of 64/128), FFN
// down-proj, and a square o-proj/attn-combine shape. M=1 (decode, cheap)
// touches all 4 families; M=16/128 touch gate/up-proj + attn-proj; M=64/512
// touch down-proj + o-proj -- every M value and every (K,N) family appears at
// least once without paying for the full M x KN cross (which would multiply
// the already-12x categorical cross in buildCases() to 240 cases/leaf).
struct Shape { int M, K, N; };
static const Shape kShapes4[] = {
    {1,   4096,  11008},  // decode, FFN gate/up-proj (K=hidden, N=intermediate)
    {1,   2880,  5120},   // decode, attn-proj (group_size zero-padding path)
    {1,   11008, 4096},   // decode, FFN down-proj (K=intermediate, N=hidden)
    {1,   4096,  4096},   // decode, o-proj / square attn-proj
    {16,  4096,  11008},  // prefill, gate/up-proj
    {16,  2880,  5120},   // prefill, attn-proj
    {64,  11008, 4096},   // prefill, down-proj
    {64,  4096,  4096},   // prefill, o-proj
    {128, 4096,  11008},  // prefill, gate/up-proj
    {128, 2880,  5120},   // prefill, attn-proj
    {512, 11008, 4096},   // prefill, down-proj
    {512, 4096,  4096},   // prefill, o-proj
};
// tier2 (~65%): all 4 M=1 (decode) families + one mid-M point for each of
// M=16/64/128/512, still touching every (K,N) family at least once.
static const Shape kShapes3[] = {
    kShapes4[0], kShapes4[1], kShapes4[2], kShapes4[3],
    kShapes4[5], kShapes4[6], kShapes4[9], kShapes4[11],
};
// tier1: M=1 (smallest family) + one M>1 point -- "1 decode + 1 prefill".
static const Shape kShapes2[] = {kShapes4[1], kShapes4[6]};

static const int kGsArray[] = {32, 64, 128};
static const bool kZeroArray[] = {true, false};
static const bool kDtypeArray[] = {false, true};  // fp16, fp32

struct Case {
  int M, K, N, gs;
  bool zero, fp32;
};

static std::vector<Case> buildCases(int tier) {
  const Shape* shapes = tier == 1 ? kShapes2 : tier == 2 ? kShapes3 : kShapes4;
  const size_t n_shapes = tier == 1 ? sizeof(kShapes2) / sizeof(kShapes2[0])
                        : tier == 2 ? sizeof(kShapes3) / sizeof(kShapes3[0])
                                    : sizeof(kShapes4) / sizeof(kShapes4[0]);
  std::vector<Case> out;
  for (int gs : kGsArray)
    for (bool zero : kZeroArray)
      for (bool fp32 : kDtypeArray)
        for (size_t si = 0; si < n_shapes; ++si)
          out.push_back({shapes[si].M, shapes[si].K, shapes[si].N, gs, zero, fp32});
  return out;
}

static bool runOne(const Case& c, int idx) {
  CaseData d;
  genCase(c.M, c.K, c.N, c.gs, c.zero, c.fp32, /*seed=*/1000u + idx, d);

  const size_t countA = static_cast<size_t>(c.M) * c.K;
  const size_t countC = static_cast<size_t>(c.M) * c.N;
  const size_t elem = c.fp32 ? 4 : 2;

  void *dA = nullptr, *dC = nullptr;
  __half* dS = nullptr;
  uint8_t* dZ = nullptr;
  uint8_t* dB = nullptr;
  HIP_CHECK(hipMalloc(&dA, countA * elem));
  HIP_CHECK(hipMalloc(&dB, d.Bpacked.size()));
  HIP_CHECK(hipMalloc(&dS, d.scales.size() * sizeof(__half)));
  if (c.zero) HIP_CHECK(hipMalloc(&dZ, d.zeros.size()));
  HIP_CHECK(hipMalloc(&dC, countC * elem));
  HIP_CHECK(hipMemcpy(dA, c.fp32 ? (void*)d.A32.data() : (void*)d.A16.data(),
                      countA * elem, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, d.Bpacked.data(), d.Bpacked.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dS, d.scales.data(), d.scales.size() * sizeof(__half), hipMemcpyHostToDevice));
  if (c.zero) HIP_CHECK(hipMemcpy(dZ, d.zeros.data(), d.zeros.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dC, 0, countC * elem));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  auto launch = [&]() {
    // bits=3, zp_elem_size=1 (raw per-group uint8 zero_points, plain
    // per-element convention -- no pre_unpacked_zp_u8/fp16 buffer).
    return hip_matmul_nbits(stream, dA, dB, dS, c.zero ? dZ : nullptr,
                            nullptr, dC, c.M, c.N, c.K, /*batch_count=*/1,
                            /*bits=*/3, c.gs, static_cast<int>(elem),
                            /*zp_elem_size=*/1, nullptr, nullptr);
  };

  int status = 0;
  char tune_key[64];
  std::snprintf(tune_key, sizeof(tune_key), "%dx%dx%d_gs%d_%s", c.M, c.N, c.K,
                c.gs, c.zero ? "z" : "noz");
  const std::string selected_config = capture_selected_config(tune_key, [&] {
    for (int w = 0; w < 3; ++w) status = launch();
    HIP_CHECK(hipStreamSynchronize(stream));
  });

  double avg_ms = 0.0;
  if (status == 0) {
    constexpr int kIters = 20;
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    hipEventRecord(e0, stream);
    for (int i = 0; i < kIters; ++i) launch();
    hipEventRecord(e1, stream);
    HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    hipEventElapsedTime(&ms, e0, e1);
    avg_ms = ms / kIters;
    hipEventDestroy(e0);
    hipEventDestroy(e1);
  }

  bool pass = (status == 0);
  double rel_l2 = 0.0;
  if (status == 0) {
    std::vector<uint8_t> raw(countC * elem);
    HIP_CHECK(hipMemcpy(raw.data(), dC, countC * elem, hipMemcpyDeviceToHost));
    double ssd = 0.0, ssr = 0.0;
    int errors = 0;
    for (size_t i = 0; i < countC; ++i) {
      const float gpu_val = c.fp32 ? reinterpret_cast<float*>(raw.data())[i]
                                    : half_to_float(reinterpret_cast<__half*>(raw.data())[i]);
      const float ref_val = c.fp32 ? d.Cref32[i] : half_to_float(d.Cref16[i]);
      const float diff = std::fabs(gpu_val - ref_val);
      const float tol = std::fabs(ref_val) * 0.05f + 0.1f;
      if (diff > tol) ++errors;
      ssd += (double)diff * diff;
      ssr += (double)ref_val * ref_val;
    }
    rel_l2 = ssr > 0.0 ? std::sqrt(ssd / ssr) : std::sqrt(ssd);
    pass = (errors == 0);
  }

  std::printf("  M=%d N=%d K=%d gs=%d zero=%d dtype=%s  status=%d  %.4f ms  relL2=%.3e  %s\n",
              c.M, c.N, c.K, c.gs, c.zero, c.fp32 ? "fp32" : "fp16", status,
              avg_ms, rel_l2, pass ? "PASS" : "*** FAIL ***");

  {
    using namespace hipdnn_ep_test;
    CsvWriter csv;
    char shape_buf[96];
    std::snprintf(shape_buf, sizeof(shape_buf), "%dx%dx%d_gs%d_%s%s", c.M, c.N,
                 c.K, c.gs, c.fp32 ? "fp32" : "fp16", c.zero ? "" : "_noz");
    CsvRow row;
    row.shape = shape_buf;
    row.config = selected_config;
    row.time_ms = avg_ms;
    row.rel_l2 = rel_l2;
    row.verdict = pass ? "PASS" : "FAIL";
    csv.write(row);
  }

  hipFree(dA);
  hipFree(dB);
  hipFree(dS);
  if (dZ) hipFree(dZ);
  hipFree(dC);
  hipStreamDestroy(stream);
  return pass;
}

static int run(int coverage_tier) {
  std::vector<Case> cases = buildCases(coverage_tier);
  std::printf("coverage=%d -> running %zu matmul_nbits_u3 cases (3 group_size "
             "x 2 zero-points x 2 dtype x typical shapes)\n",
             coverage_tier, cases.size());
  int fail = 0;
  for (size_t i = 0; i < cases.size(); ++i)
    if (!runOne(cases[i], static_cast<int>(i))) ++fail;
  std::printf("\nCoverage sweep done: %zu cases, %d failed\n", cases.size(), fail);
  std::printf("%s\n", fail == 0 ? "ALL PASS" : "SOME FAILED");
  return fail == 0 ? 0 : 1;
}

}  // namespace sweep

static int runCustom(int M, int N, int K, int gs, bool zero, bool fp32) {
  return sweep::runOne({M, K, N, gs, zero, fp32}, /*idx=*/0) ? 0 : 1;
}

int main(int argc, char* argv[]) {
  std::printf("custom_kernels MatMulNBits bits=3 (uint3, continuous bitstream) Verification\n");
  std::printf("=============================================================================\n");

  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("GPU: %s (arch: %s)\n", prop.name, prop.gcnArchName);

  int M = 0, N = 0, K = 0;
  int gs = 128;
  bool no_zeros = false, fp32_mode = false;
  bool have_custom = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--no-zeros") no_zeros = true;
    else if (a == "--fp32") fp32_mode = true;
    else if (a == "--group-size" && i + 1 < argc) gs = std::atoi(argv[++i]);
    else if (sscanf(argv[i], "%dx%dx%d", &M, &K, &N) == 3) have_custom = true;
  }

  if (have_custom)
    return runCustom(M, N, K, gs, !no_zeros, fp32_mode);

  const int tier = hipdnn_ep_test::resolveCoverageTier(argc, argv);
  return sweep::run(tier);
}
