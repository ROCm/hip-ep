// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

/**
 * Test for ROCm Gemm custom op using hipBLASLt
 */

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <vector>
#include <iostream>

// Test hipBLASLt GEMM directly (without ORT)
TEST(RocmGemmTest, HipBlasLtDirectGemm) {
  // Check GPU availability
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess || device_count == 0) {
    GTEST_SKIP() << "No AMD GPU available, skipping test";
  }

  // Matrix dimensions: C = A * B
  // A: [M, K], B: [K, N], C: [M, N]
  const int64_t M = 64, N = 32, K = 48;

  // Create hipBLASLt handle
  hipblasLtHandle_t handle;
  ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

  hipStream_t stream;
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);

  // Create matrix layouts
  hipblasLtMatrixLayout_t layout_A, layout_B, layout_C, layout_D;
  ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout_A, HIP_R_32F, M, K, K), HIPBLAS_STATUS_SUCCESS);
  ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout_B, HIP_R_32F, K, N, N), HIPBLAS_STATUS_SUCCESS);
  ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout_C, HIP_R_32F, M, N, N), HIPBLAS_STATUS_SUCCESS);
  ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout_D, HIP_R_32F, M, N, N), HIPBLAS_STATUS_SUCCESS);

  // Create matmul descriptor
  hipblasLtMatmulDesc_t matmul_desc;
  ASSERT_EQ(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
            HIPBLAS_STATUS_SUCCESS);

  // Set transpose operations (no transpose)
  hipblasOperation_t trans = HIPBLAS_OP_N;
  ASSERT_EQ(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                             &trans, sizeof(trans)),
            HIPBLAS_STATUS_SUCCESS);
  ASSERT_EQ(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                             &trans, sizeof(trans)),
            HIPBLAS_STATUS_SUCCESS);

  // Get heuristics
  hipblasLtMatmulPreference_t pref;
  ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);

  size_t max_workspace = 32 * 1024 * 1024;  // 32 MB
  ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &max_workspace, sizeof(max_workspace)),
            HIPBLAS_STATUS_SUCCESS);

  hipblasLtMatmulHeuristicResult_t results[4];
  int returned;
  ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(handle, matmul_desc,
                                             layout_A, layout_B, layout_C, layout_D,
                                             pref, 4, results, &returned),
            HIPBLAS_STATUS_SUCCESS);

  EXPECT_GT(returned, 0) << "Should find at least one algorithm";
  std::cout << "Found " << returned << " algorithms" << std::endl;
  std::cout << "Best algo workspace: " << results[0].workspaceSize << " bytes" << std::endl;

  // Allocate device memory
  float *d_A, *d_B, *d_C, *d_D;
  size_t size_A = M * K * sizeof(float);
  size_t size_B = K * N * sizeof(float);
  size_t size_C = M * N * sizeof(float);
  size_t size_D = M * N * sizeof(float);

  ASSERT_EQ(hipMalloc(&d_A, size_A), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_B, size_B), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_C, size_C), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_D, size_D), hipSuccess);

  // Allocate workspace
  void* workspace = nullptr;
  if (results[0].workspaceSize > 0) {
    ASSERT_EQ(hipMalloc(&workspace, results[0].workspaceSize), hipSuccess);
  }

  // Initialize host data
  std::vector<float> h_A(M * K, 1.0f);
  std::vector<float> h_B(K * N, 1.0f);
  std::vector<float> h_C(M * N, 0.0f);

  ASSERT_EQ(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice), hipSuccess);

  // Execute GEMM: D = alpha * A * B + beta * C
  float alpha = 1.0f, beta = 0.0f;
  ASSERT_EQ(hipblasLtMatmul(handle, matmul_desc, &alpha,
                             d_A, layout_A, d_B, layout_B,
                             &beta, d_C, layout_C, d_D, layout_D,
                             &results[0].algo, workspace, results[0].workspaceSize,
                             stream),
            HIPBLAS_STATUS_SUCCESS);

  // Sync and copy result
  ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

  std::vector<float> h_D(M * N);
  ASSERT_EQ(hipMemcpy(h_D.data(), d_D, size_D, hipMemcpyDeviceToHost), hipSuccess);

  // Verify: D[i,j] = sum_{k} A[i,k] * B[k,j] = K * 1.0 * 1.0 = K
  float expected = static_cast<float>(K);
  EXPECT_NEAR(h_D[0], expected, 1e-3) << "D[0] should be " << K << " (sum of K ones)";

  std::cout << "D[0] = " << h_D[0] << " (expected: " << expected << ")" << std::endl;

  // Cleanup
  if (workspace) hipFree(workspace);
  hipFree(d_D);
  hipFree(d_C);
  hipFree(d_B);
  hipFree(d_A);
  hipblasLtMatmulPreferenceDestroy(pref);
  hipblasLtMatmulDescDestroy(matmul_desc);
  hipblasLtMatrixLayoutDestroy(layout_D);
  hipblasLtMatrixLayoutDestroy(layout_C);
  hipblasLtMatrixLayoutDestroy(layout_B);
  hipblasLtMatrixLayoutDestroy(layout_A);
  hipStreamDestroy(stream);
  hipblasLtDestroy(handle);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
