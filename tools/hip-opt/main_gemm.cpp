//===- main_gemm.cpp - Main driver for GEMM test -------------------------===//
//
// This program:
//   1. Allocates host memory and initializes test matrices A and B
//   2. Allocates GPU device memory via hipMalloc
//   3. Copies A, B to device
//   4. Calls run_gemm() -- the MLIR-compiled function (from test_gemm.mlir)
//   5. Copies result C back to host
//   6. Prints results and compares with CPU reference
//
// Compile with:
//   cl.exe /c /EHsc /I%THEROCK_DIST%/include main_gemm.cpp
//
//===----------------------------------------------------------------------===//

#include <hip/hip_runtime_api.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The MLIR-compiled function (defined in test_gemm.mlir, compiled to gemm.obj)
// After lowering through hip-opt + mlir-translate + llc, the signature is:
//   void run_gemm(float* A, float* B, float* C, int64_t M, int64_t K, int64_t N)
extern "C" void run_gemm(float* A, float* B, float* C,
                          int64_t M, int64_t K, int64_t N);

// CPU reference GEMM: C = A @ B (row-major)
static void reference_gemm(const float* A, const float* B, float* C,
                           int64_t M, int64_t K, int64_t N) {
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      float sum = 0.0f;
      for (int64_t k = 0; k < K; ++k) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
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
  // Matrix dimensions: C(MxN) = A(MxK) @ B(KxN)
  const int64_t M = 4;
  const int64_t K = 8;
  const int64_t N = 4;

  printf("GEMM Test: C(%lld x %lld) = A(%lld x %lld) @ B(%lld x %lld)\n",
         M, N, M, K, K, N);

  // 1. Allocate and initialize host data
  std::vector<float> h_A(M * K);
  std::vector<float> h_B(K * N);
  std::vector<float> h_C(M * N, 0.0f);
  std::vector<float> h_C_ref(M * N, 0.0f);

  // Initialize A and B with simple deterministic values
  for (int64_t i = 0; i < M * K; ++i) {
    h_A[i] = static_cast<float>(i % 10) / 10.0f;
  }
  for (int64_t i = 0; i < K * N; ++i) {
    h_B[i] = static_cast<float>((i + 3) % 10) / 10.0f;
  }

  // Print input matrices
  printf("\nMatrix A (%lld x %lld):\n", M, K);
  for (int64_t i = 0; i < M; ++i) {
    printf("  [");
    for (int64_t j = 0; j < K; ++j) {
      printf(" %5.2f", h_A[i * K + j]);
    }
    printf(" ]\n");
  }

  printf("\nMatrix B (%lld x %lld):\n", K, N);
  for (int64_t i = 0; i < K; ++i) {
    printf("  [");
    for (int64_t j = 0; j < N; ++j) {
      printf(" %5.2f", h_B[i * N + j]);
    }
    printf(" ]\n");
  }

  // 2. Allocate device memory
  float* d_A = nullptr;
  float* d_B = nullptr;
  float* d_C = nullptr;
  HIP_CHECK(hipMalloc(&d_A, M * K * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_B, K * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_C, M * N * sizeof(float)));

  // 3. Copy A, B to device
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), M * K * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), K * N * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_C, 0, M * N * sizeof(float)));

  // 4. Call the MLIR-compiled function
  printf("\nRunning GEMM on GPU via MLIR-compiled function...\n");
  run_gemm(d_A, d_B, d_C, M, K, N);

  // 5. Copy result back to host
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, M * N * sizeof(float), hipMemcpyDeviceToHost));

  // 6. Print GPU result
  printf("\nGPU Result C (%lld x %lld):\n", M, N);
  for (int64_t i = 0; i < M; ++i) {
    printf("  [");
    for (int64_t j = 0; j < N; ++j) {
      printf(" %8.4f", h_C[i * N + j]);
    }
    printf(" ]\n");
  }

  // 7. CPU reference and comparison
  reference_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, K, N);

  printf("\nCPU Reference C (%lld x %lld):\n", M, N);
  for (int64_t i = 0; i < M; ++i) {
    printf("  [");
    for (int64_t j = 0; j < N; ++j) {
      printf(" %8.4f", h_C_ref[i * N + j]);
    }
    printf(" ]\n");
  }

  // Compare
  float max_diff = 0.0f;
  for (int64_t i = 0; i < M * N; ++i) {
    float diff = std::fabs(h_C[i] - h_C_ref[i]);
    if (diff > max_diff) max_diff = diff;
  }

  printf("\nMax absolute difference: %e\n", max_diff);
  if (max_diff < 1e-4f) {
    printf("PASS: GPU result matches CPU reference!\n");
  } else {
    printf("FAIL: Results differ beyond tolerance.\n");
  }

  // 8. Cleanup
  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  return (max_diff < 1e-4f) ? 0 : 1;
}
