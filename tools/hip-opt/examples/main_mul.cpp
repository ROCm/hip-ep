//===- main_mul.cpp - Main driver for two-mul DPS test --------------------===//
//
// Two chained element-wise multiplies:
//   tmp = A * B       (intermediate)
//   D   = tmp * C     (final output, caller-provided)
//
// Expected result: D = A * B * C
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern "C" void two_muls(
    float *A_alloc, float *A_align, int64_t A_off,
    int64_t A_s0, int64_t A_s1, int64_t A_st0, int64_t A_st1,
    float *B_alloc, float *B_align, int64_t B_off,
    int64_t B_s0, int64_t B_s1, int64_t B_st0, int64_t B_st1,
    float *C_alloc, float *C_align, int64_t C_off,
    int64_t C_s0, int64_t C_s1, int64_t C_st0, int64_t C_st1,
    float *D_alloc, float *D_align, int64_t D_off,
    int64_t D_s0, int64_t D_s1, int64_t D_st0, int64_t D_st1);

#define HIP_CHECK(call)                                                   \
  do {                                                                    \
    hipError_t err = (call);                                              \
    if (err != hipSuccess) {                                              \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__,    \
              hipGetErrorString(err));                                     \
      exit(1);                                                            \
    }                                                                     \
  } while (0)

int main() {
  const int64_t M = 4, N = 4;
  const int64_t numElements = M * N;

  printf("Two-mul DPS test:  D = A * B * C  (%lld x %lld)\n", M, N);

  std::vector<float> h_A(numElements), h_B(numElements), h_C(numElements);
  std::vector<float> h_D(numElements, 0.0f), h_ref(numElements);

  for (int64_t i = 0; i < numElements; ++i) {
    h_A[i] = float(i + 1) * 0.1f;
    h_B[i] = float(i + 1) * 0.2f;
    h_C[i] = 0.5f;
    h_ref[i] = h_A[i] * h_B[i] * h_C[i];
  }

  float *d_A, *d_B, *d_C, *d_D;
  HIP_CHECK(hipMalloc(&d_A, numElements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_B, numElements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_C, numElements * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_D, numElements * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_A, h_A.data(), numElements * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), numElements * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_C, h_C.data(), numElements * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_D, 0, numElements * sizeof(float)));

  printf("Running two_muls on GPU...\n");
  two_muls(
      d_A, d_A, 0, M, N, N, 1,
      d_B, d_B, 0, M, N, N, 1,
      d_C, d_C, 0, M, N, N, 1,
      d_D, d_D, 0, M, N, N, 1);

  HIP_CHECK(hipMemcpy(h_D.data(), d_D, numElements * sizeof(float), hipMemcpyDeviceToHost));

  printf("\nGPU Result D (%lld x %lld):\n", M, N);
  for (int64_t i = 0; i < M; ++i) {
    printf("  [");
    for (int64_t j = 0; j < N; ++j) printf(" %7.4f", h_D[i * N + j]);
    printf(" ]\n");
  }

  printf("\nCPU Reference (A * B * C):\n");
  for (int64_t i = 0; i < M; ++i) {
    printf("  [");
    for (int64_t j = 0; j < N; ++j) printf(" %7.4f", h_ref[i * N + j]);
    printf(" ]\n");
  }

  float max_diff = 0.0f;
  for (int64_t i = 0; i < numElements; ++i) {
    float diff = std::fabs(h_D[i] - h_ref[i]);
    if (diff > max_diff) max_diff = diff;
  }

  printf("\nMax absolute difference: %e\n", max_diff);
  printf("%s\n", max_diff < 1e-4f ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));
  HIP_CHECK(hipFree(d_D));

  return (max_diff < 1e-4f) ? 0 : 1;
}
