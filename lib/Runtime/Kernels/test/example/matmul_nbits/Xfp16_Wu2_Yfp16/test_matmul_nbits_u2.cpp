/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * custom_kernels MatMulNBits bits=2 (uint2, 4-per-byte packed) Verification.
 *
 * Unit-test model: this file is entirely self-contained (no example/common/,
 * no python, no on-disk data) -- inputs, the uint2 packing, and the
 * dequant+matmul CPU reference are all generated in-process. The case set is
 * every categorical situation this op supports (group_size in {32,64,128},
 * zero-points on/off, dtype fp16/fp32) crossed with a SMALL list of typical
 * (M,K,N) shapes (decode M=1 at real model FFN/attn-proj K,N, plus two
 * smaller prefill-representative shapes) -- NOT the full M/K/N shape space.
 * COVERAGE=1|2|3 (default 3) only thins how many typical shapes run; every
 * tier still touches every group_size/zero-points/dtype value.
 *
 * uint2 packing: value k occupies bits [2k, 2k+2) of row n's bitstream,
 * LSB-first (byte = v0 | v1<<2 | v2<<4 | v3<<6) -- a plain continuous 2-bit
 * stream, K/4 bytes/row, no group_size padding (2 divides 8 evenly and every
 * group_size here is a multiple of 4).
 *
 * A:      FP16 (or FP32) row-major [M, K]
 * B:      uint8 packed [N, K/4], 4 values/byte LSB-first
 * scales: FP16 [N, num_groups_k], per-column per-group
 * zeros:  uint8 [N, num_groups_k], per-column per-group zero point in [0,3]
 *         (optional; default zero point is 2 when absent), passed to the
 *         kernel raw (zp_elem_size=1, direct convention -- no packed/
 *         pre-unpack step; that path is exercised by the kernel's own tests,
 *         not this small correctness UT).
 * C:      FP16 (or FP32) row-major [M, N]
 */
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "hip_custom_kernels.h"
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
extern "C" const unsigned char kMatmulNbitsLutData[] = {
#embed HIPDNN_LUT_FB
};
extern "C" const size_t kMatmulNbitsLutData_size = sizeof(kMatmulNbitsLutData);
#endif  // HIPDNN_LUT_LINKED_EXTERNALLY

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
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

// ============================================================
// In-process data generation + CPU fp32 reference (no python, no disk I/O).
// B is randint(0,4) per element, packed 4-per-byte LSB-first (byte =
// v0|v1<<2|v2<<4|v3<<6); scales are uniform(0.01,0.05); zeros (when present)
// are integers in [1,2] (uint8, direct/unpacked convention); dequant is
// (B - zp) * scale, default zp = 2 when zeros are absent.
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

  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> adist(-0.5f, 0.5f);
  std::uniform_int_distribution<int> bdist(0, 3);
  std::uniform_real_distribution<float> sdist(0.01f, 0.05f);
  std::uniform_int_distribution<int> zdist(1, 2);

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

  // Pack 4 values/byte, LSB-first: byte = v0 | v1<<2 | v2<<4 | v3<<6.
  const int row_bytes = K / 4;
  out.Bpacked.assign(static_cast<size_t>(N) * row_bytes, 0);
  for (int n = 0; n < N; ++n) {
    const uint8_t* row = &Bq[static_cast<size_t>(n) * K];
    uint8_t* prow = &out.Bpacked[static_cast<size_t>(n) * row_bytes];
    for (int j = 0; j < row_bytes; ++j) {
      const uint8_t v0 = row[4 * j], v1 = row[4 * j + 1];
      const uint8_t v2 = row[4 * j + 2], v3 = row[4 * j + 3];
      prow[j] = static_cast<uint8_t>(v0 | (v1 << 2) | (v2 << 4) | (v3 << 6));
    }
  }

  out.scales.resize(scalesF.size());
  for (size_t i = 0; i < scalesF.size(); ++i) out.scales[i] = float_to_half(scalesF[i]);
  if (zero) {
    out.zeros.resize(zerosF.size());
    for (size_t i = 0; i < zerosF.size(); ++i) out.zeros[i] = static_cast<uint8_t>(zerosF[i]);
  }

  std::vector<float> Cref(static_cast<size_t>(M) * N, 0.0f);
  for (int n = 0; n < N; ++n) {
    const uint8_t* row = &Bq[static_cast<size_t>(n) * K];
    const float* srow = &scalesF[static_cast<size_t>(n) * num_groups_k];
    const float* zrow = zero ? &zerosF[static_cast<size_t>(n) * num_groups_k] : nullptr;
    for (int m = 0; m < M; ++m) {
      const float* arow = &A[static_cast<size_t>(m) * K];
      float acc = 0.0f;
      for (int k = 0; k < K; ++k) {
        const int g = k / gs;
        const float zp = zero ? zrow[g] : 2.0f;
        acc += arow[k] * (static_cast<float>(row[k]) - zp) * srow[g];
      }
      Cref[static_cast<size_t>(m) * N + n] = acc;
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

struct Shape { int M, K, N; };
static const Shape kShapes4[] = {
    {1, 4096, 11008},
    {1, 2880, 5120},
    {128, 512, 1024},
    {512, 512, 1024},
};
static const Shape kShapes3[] = {kShapes4[0], kShapes4[2], kShapes4[3]};
static const Shape kShapes2[] = {kShapes4[0], kShapes4[3]};

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
    // bits=2, zp_elem_size=1 (raw per-group uint8 zero_points, direct
    // convention -- no pre_unpacked_zp_u8/fp16 buffer).
    return hip_matmul_nbits(stream, dA, dB, dS, c.zero ? dZ : nullptr,
                            nullptr, dC, c.M, c.N, c.K, /*batch_count=*/1,
                            /*bits=*/2, c.gs, static_cast<int>(elem),
                            /*zp_elem_size=*/1, nullptr, nullptr);
  };

  int status = 0;
  for (int w = 0; w < 3; ++w) status = launch();
  HIP_CHECK(hipStreamSynchronize(stream));

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
    row.config = "final";
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
  std::printf("coverage=%d -> running %zu matmul_nbits_u2 cases (3 group_size "
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
  std::printf("custom_kernels MatMulNBits bits=2 (uint2, 4-per-byte packed) Verification\n");
  std::printf("==========================================================================\n");

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
