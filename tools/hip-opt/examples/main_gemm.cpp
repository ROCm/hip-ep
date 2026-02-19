//===- main_gemm.cpp - Main driver for two-matmul DPS test ----------------===//
//
// This program:
//   1. Allocates device memory for A, B0, B1, C
//   2. Initializes A, B0, B1 with test data and copies to device
//   3. Calls two_matmuls() -- the MLIR-compiled function (test_gemm.mlir)
//      which computes: tmp = A @ B0, then C = tmp @ B1
//   4. Copies result C back to host and compares with CPU reference
//
// After MLIR lowering, each memref<?x?xf32, 1> argument becomes 7 scalars:
//   (allocatedPtr, alignedPtr, offset, size0, size1, stride0, stride1)
// So 4 memrefs = 28 parameters.
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The MLIR-compiled function (test_gemm.mlir -> hip-opt -> llc -> gemm.obj).
// Each memref<?x?xf32, 1> lowers to:
//   ptr, ptr, i64, i64, i64, i64, i64
//   (alloc, align, offset, size[0], size[1], stride[0], stride[1])
extern "C" void two_matmuls(
    float *A_alloc, float *A_align, int64_t A_off,
    int64_t A_s0, int64_t A_s1, int64_t A_st0, int64_t A_st1,
    float *B0_alloc, float *B0_align, int64_t B0_off,
    int64_t B0_s0, int64_t B0_s1, int64_t B0_st0, int64_t B0_st1,
    float *B1_alloc, float *B1_align, int64_t B1_off,
    int64_t B1_s0, int64_t B1_s1, int64_t B1_st0, int64_t B1_st1,
    float *C_alloc, float *C_align, int64_t C_off,
    int64_t C_s0, int64_t C_s1, int64_t C_st0, int64_t C_st1);

static void reference_matmul(const float *A, const float *B, float *C,
                             int64_t M, int64_t K, int64_t N) {
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k)
        sum += A[i * K + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
}

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
  // Two chained matmuls:
  //   tmp[M,N] = A[M,K] @ B0[K,N]
  //   C[M,P]   = tmp[M,N] @ B1[N,P]
  const int64_t M = 4, K = 8, N = 4, P = 4;

  printf("Two-matmul DPS test:\n");
  printf("  tmp[%lld,%lld] = A[%lld,%lld] @ B0[%lld,%lld]\n", M, N, M, K, K, N);
  printf("  C[%lld,%lld]   = tmp[%lld,%lld] @ B1[%lld,%lld]\n", M, P, M, N, N, P);

  // Host data
  std::vector<float> h_A(M * K), h_B0(K * N), h_B1(N * P);
  std::vector<float> h_C(M * P, 0.0f), h_ref(M * P, 0.0f);

  for (int64_t i = 0; i < M * K; ++i) h_A[i] = float(i % 10) / 10.0f;
  for (int64_t i = 0; i < K * N; ++i) h_B0[i] = float((i + 3) % 10) / 10.0f;
  for (int64_t i = 0; i < N * P; ++i) h_B1[i] = float((i + 7) % 10) / 10.0f;

  // Device memory
  float *d_A, *d_B0, *d_B1, *d_C;
  HIP_CHECK(hipMalloc(&d_A,  M * K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_B0, K * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_B1, N * P * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_C,  M * P * sizeof(float)));

  HIP_CHECK(hipMemcpy(d_A,  h_A.data(),  M * K * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B0, h_B0.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B1, h_B1.data(), N * P * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_C, 0, M * P * sizeof(float)));

  printf("\nRunning two_matmuls on GPU...\n");
  two_matmuls(
      d_A,  d_A,  0, M, K, K, 1,
      d_B0, d_B0, 0, K, N, N, 1,
      d_B1, d_B1, 0, N, P, P, 1,
      d_C,  d_C,  0, M, P, P, 1);

  HIP_CHECK(hipMemcpy(h_C.data(), d_C, M * P * sizeof(float), hipMemcpyDeviceToHost));

  // CPU reference: tmp = A @ B0, then C = tmp @ B1
  std::vector<float> h_tmp(M * N, 0.0f);
  reference_matmul(h_A.data(), h_B0.data(), h_tmp.data(), M, K, N);
  reference_matmul(h_tmp.data(), h_B1.data(), h_ref.data(), M, N, P);

  printf("\nGPU Result C (%lld x %lld):\n", M, P);
  for (int64_t i = 0; i < M; ++i) {
    printf("  [");
    for (int64_t j = 0; j < P; ++j) printf(" %8.4f", h_C[i * P + j]);
    printf(" ]\n");
  }

  printf("\nCPU Reference C (%lld x %lld):\n", M, P);
  for (int64_t i = 0; i < M; ++i) {
    printf("  [");
    for (int64_t j = 0; j < P; ++j) printf(" %8.4f", h_ref[i * P + j]);
    printf(" ]\n");
  }

  float max_diff = 0.0f;
  for (int64_t i = 0; i < M * P; ++i) {
    float diff = std::fabs(h_C[i] - h_ref[i]);
    if (diff > max_diff) max_diff = diff;
  }

  printf("\nMax absolute difference: %e\n", max_diff);
  printf("%s\n", max_diff < 1e-4f ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B0));
  HIP_CHECK(hipFree(d_B1));
  HIP_CHECK(hipFree(d_C));

  return (max_diff < 1e-4f) ? 0 : 1;
}
