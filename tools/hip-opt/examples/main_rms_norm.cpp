//===- main_rms_norm.cpp - Main driver for two-rms_norm DPS test ----------===//
//
// Two chained RMS normalizations:
//   tmp = RMSNorm(A, W0)
//   B   = RMSNorm(tmp, W1)
//
// RMSNorm(x, w)[n,d] = x[n,d] / rms(x[n,:]) * w[d]
// where rms(x) = sqrt(mean(x^2) + epsilon)
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// MLIR-compiled function.
// A, B are memref<?x?xf32, 1> -> 7 args each.
// W0, W1 are memref<?xf32, 1> -> 5 args each (1D: alloc, align, off, size0, stride0).
extern "C" void two_rms_norms(
    float *A_alloc, float *A_align, int64_t A_off,
    int64_t A_s0, int64_t A_s1, int64_t A_st0, int64_t A_st1,
    float *W0_alloc, float *W0_align, int64_t W0_off,
    int64_t W0_s0, int64_t W0_st0,
    float *W1_alloc, float *W1_align, int64_t W1_off,
    int64_t W1_s0, int64_t W1_st0,
    float *B_alloc, float *B_align, int64_t B_off,
    int64_t B_s0, int64_t B_s1, int64_t B_st0, int64_t B_st1);

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err = (call);                                              \
    if (err != hipSuccess) {                                              \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,    \
              hipGetErrorString(err));                                     \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

static void cpu_rms_norm(const float *input, const float *weight, float *output,
                          int64_t N, int64_t D, float epsilon) {
  for (int64_t n = 0; n < N; ++n) {
    float sum_sq = 0.0f;
    for (int64_t d = 0; d < D; ++d) {
      float v = input[n * D + d];
      sum_sq += v * v;
    }
    float rms = std::sqrt(sum_sq / float(D) + epsilon);
    for (int64_t d = 0; d < D; ++d)
      output[n * D + d] = input[n * D + d] / rms * weight[d];
  }
}

int main() {
  const int64_t N = 4, D = 8;
  const float epsilon = 1e-5f;

  printf("Two-rms_norm DPS test:  B = RMSNorm(RMSNorm(A, W0), W1)  [%lld x %lld]\n", N, D);

  std::vector<float> h_A(N * D), h_W0(D), h_W1(D);
  std::vector<float> h_B(N * D, 0.0f), h_ref(N * D);

  for (int64_t i = 0; i < N * D; ++i) h_A[i] = float(i + 1) * 0.1f;
  for (int64_t d = 0; d < D; ++d) h_W0[d] = 1.0f;
  for (int64_t d = 0; d < D; ++d) h_W1[d] = 0.5f;

  // CPU reference: two chained RMSNorms
  std::vector<float> h_tmp(N * D);
  cpu_rms_norm(h_A.data(), h_W0.data(), h_tmp.data(), N, D, epsilon);
  cpu_rms_norm(h_tmp.data(), h_W1.data(), h_ref.data(), N, D, epsilon);

  float *d_A, *d_W0, *d_W1, *d_B;
  HIP_CHECK(hipMalloc(&d_A,  N * D * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_W0, D * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_W1, D * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_B,  N * D * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_A,  h_A.data(),  N * D * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_W0, h_W0.data(), D * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_W1, h_W1.data(), D * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_B, 0, N * D * sizeof(float)));

  printf("Running two_rms_norms on GPU...\n");
  two_rms_norms(
      d_A,  d_A,  0, N, D, D, 1,
      d_W0, d_W0, 0, D, 1,
      d_W1, d_W1, 0, D, 1,
      d_B,  d_B,  0, N, D, D, 1);

  HIP_CHECK(hipMemcpy(h_B.data(), d_B, N * D * sizeof(float), hipMemcpyDeviceToHost));

  printf("\nGPU Result B (%lld x %lld):\n", N, D);
  for (int64_t i = 0; i < N; ++i) {
    printf("  [");
    for (int64_t j = 0; j < D; ++j) printf(" %7.4f", h_B[i * D + j]);
    printf(" ]\n");
  }

  printf("\nCPU Reference:\n");
  for (int64_t i = 0; i < N; ++i) {
    printf("  [");
    for (int64_t j = 0; j < D; ++j) printf(" %7.4f", h_ref[i * D + j]);
    printf(" ]\n");
  }

  float max_diff = 0.0f;
  for (int64_t i = 0; i < N * D; ++i) {
    float diff = std::fabs(h_B[i] - h_ref[i]);
    if (diff > max_diff) max_diff = diff;
  }

  printf("\nMax absolute difference: %e\n", max_diff);
  printf("%s\n", max_diff < 1e-3f ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_W0));
  HIP_CHECK(hipFree(d_W1));
  HIP_CHECK(hipFree(d_B));

  return (max_diff < 1e-3f) ? 0 : 1;
}
