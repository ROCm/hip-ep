// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

/**
 * Combined test for ROCm Conv and Gemm operations
 * Tests both MIOpen convolution and hipBLASLt GEMM in sequence
 */

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <hipblaslt/hipblaslt.h>
#include <vector>
#include <iostream>

class RocmConvAndGemmTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Check GPU availability
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
      GTEST_SKIP() << "No AMD GPU available, skipping test";
    }
    
    // Create HIP stream
    ASSERT_EQ(hipStreamCreate(&stream_), hipSuccess);
  }
  
  void TearDown() override {
    if (stream_) {
      hipStreamDestroy(stream_);
    }
  }
  
  hipStream_t stream_ = nullptr;
};

TEST_F(RocmConvAndGemmTest, ConvThenGemm) {
  std::cout << "\n=== Testing Conv followed by Gemm ===" << std::endl;
  
  // ========== Part 1: Convolution ==========
  std::cout << "\n[Part 1] Running MIOpen Convolution..." << std::endl;
  
  miopenHandle_t miopen_handle;
  ASSERT_EQ(miopenCreate(&miopen_handle), miopenStatusSuccess);
  ASSERT_EQ(miopenSetStream(miopen_handle, stream_), miopenStatusSuccess);
  
  // Conv dimensions
  const int conv_n = 1, conv_c = 3, conv_h = 8, conv_w = 8;
  const int conv_k = 16, conv_r = 3, conv_s = 3;
  const int conv_pad = 1;
  
  miopenTensorDescriptor_t input_desc, weight_desc, output_desc;
  ASSERT_EQ(miopenCreateTensorDescriptor(&input_desc), miopenStatusSuccess);
  ASSERT_EQ(miopenCreateTensorDescriptor(&weight_desc), miopenStatusSuccess);
  ASSERT_EQ(miopenCreateTensorDescriptor(&output_desc), miopenStatusSuccess);
  
  ASSERT_EQ(miopenSet4dTensorDescriptor(input_desc, miopenFloat, conv_n, conv_c, conv_h, conv_w), miopenStatusSuccess);
  ASSERT_EQ(miopenSet4dTensorDescriptor(weight_desc, miopenFloat, conv_k, conv_c, conv_r, conv_s), miopenStatusSuccess);
  
  miopenConvolutionDescriptor_t conv_desc;
  ASSERT_EQ(miopenCreateConvolutionDescriptor(&conv_desc), miopenStatusSuccess);
  ASSERT_EQ(miopenInitConvolutionDescriptor(conv_desc, miopenConvolution,
                                             conv_pad, conv_pad, 1, 1, 1, 1),
            miopenStatusSuccess);
  
  int out_n, out_c, out_h, out_w;
  ASSERT_EQ(miopenGetConvolutionForwardOutputDim(conv_desc, input_desc, weight_desc,
                                                  &out_n, &out_c, &out_h, &out_w),
            miopenStatusSuccess);
  
  ASSERT_EQ(miopenSet4dTensorDescriptor(output_desc, miopenFloat, out_n, out_c, out_h, out_w),
            miopenStatusSuccess);
  
  std::cout << "Conv output shape: [" << out_n << ", " << out_c << ", " << out_h << ", " << out_w << "]" << std::endl;
  
  float *d_input, *d_weight, *d_output;
  size_t input_size = conv_n * conv_c * conv_h * conv_w * sizeof(float);
  size_t weight_size = conv_k * conv_c * conv_r * conv_s * sizeof(float);
  size_t output_size = out_n * out_c * out_h * out_w * sizeof(float);
  
  ASSERT_EQ(hipMalloc(&d_input, input_size), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_weight, weight_size), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_output, output_size), hipSuccess);
  
  std::vector<float> h_input(conv_n * conv_c * conv_h * conv_w, 1.0f);
  std::vector<float> h_weight(conv_k * conv_c * conv_r * conv_s, 0.1f);
  
  ASSERT_EQ(hipMemcpy(d_input, h_input.data(), input_size, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_weight, h_weight.data(), weight_size, hipMemcpyHostToDevice), hipSuccess);
  
  size_t workspace_size;
  ASSERT_EQ(miopenConvolutionForwardGetWorkSpaceSize(miopen_handle, weight_desc, input_desc,
                                                      conv_desc, output_desc, &workspace_size),
            miopenStatusSuccess);
  
  void* workspace = nullptr;
  if (workspace_size > 0) {
    ASSERT_EQ(hipMalloc(&workspace, workspace_size), hipSuccess);
  }
  
  miopenConvAlgoPerf_t perf_results[4];
  int algo_count;
  ASSERT_EQ(miopenFindConvolutionForwardAlgorithm(miopen_handle, input_desc, d_input,
                                                   weight_desc, d_weight,
                                                   conv_desc, output_desc, d_output,
                                                   4, &algo_count, perf_results,
                                                   workspace, workspace_size, false),
            miopenStatusSuccess);
  
  std::cout << "Found " << algo_count << " conv algorithms" << std::endl;
  
  float alpha = 1.0f, beta = 0.0f;
  ASSERT_EQ(miopenConvolutionForward(miopen_handle, &alpha,
                                      input_desc, d_input,
                                      weight_desc, d_weight,
                                      conv_desc, perf_results[0].fwd_algo, &beta,
                                      output_desc, d_output,
                                      workspace, workspace_size),
            miopenStatusSuccess);
  
  ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
  
  std::vector<float> h_output(out_n * out_c * out_h * out_w);
  ASSERT_EQ(hipMemcpy(h_output.data(), d_output, output_size, hipMemcpyDeviceToHost), hipSuccess);
  
  std::cout << "Conv output[0]: " << h_output[0] << std::endl;
  EXPECT_NE(h_output[0], 0.0f) << "Conv output should be non-zero";
  
  // Cleanup conv
  if (workspace) hipFree(workspace);
  hipFree(d_output);
  hipFree(d_weight);
  hipFree(d_input);
  miopenDestroyConvolutionDescriptor(conv_desc);
  miopenDestroyTensorDescriptor(output_desc);
  miopenDestroyTensorDescriptor(weight_desc);
  miopenDestroyTensorDescriptor(input_desc);
  miopenDestroy(miopen_handle);
  
  // ========== Part 2: GEMM ==========
  std::cout << "\n[Part 2] Running hipBLASLt GEMM..." << std::endl;
  
  hipblasLtHandle_t hipblaslt_handle;
  ASSERT_EQ(hipblasLtCreate(&hipblaslt_handle), HIPBLAS_STATUS_SUCCESS);
  
  const int64_t gemm_m = 64, gemm_n = 32, gemm_k = 48;
  
  hipblasLtMatrixLayout_t layout_A, layout_B, layout_C, layout_D;
  ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout_A, HIP_R_32F, gemm_m, gemm_k, gemm_k), HIPBLAS_STATUS_SUCCESS);
  ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout_B, HIP_R_32F, gemm_k, gemm_n, gemm_n), HIPBLAS_STATUS_SUCCESS);
  ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout_C, HIP_R_32F, gemm_m, gemm_n, gemm_n), HIPBLAS_STATUS_SUCCESS);
  ASSERT_EQ(hipblasLtMatrixLayoutCreate(&layout_D, HIP_R_32F, gemm_m, gemm_n, gemm_n), HIPBLAS_STATUS_SUCCESS);
  
  hipblasLtMatmulDesc_t matmul_desc;
  ASSERT_EQ(hipblasLtMatmulDescCreate(&matmul_desc, HIPBLAS_COMPUTE_32F, HIP_R_32F),
            HIPBLAS_STATUS_SUCCESS);
  
  hipblasOperation_t trans = HIPBLAS_OP_N;
  ASSERT_EQ(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                                             &trans, sizeof(trans)),
            HIPBLAS_STATUS_SUCCESS);
  ASSERT_EQ(hipblasLtMatmulDescSetAttribute(matmul_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                                             &trans, sizeof(trans)),
            HIPBLAS_STATUS_SUCCESS);
  
  hipblasLtMatmulPreference_t pref;
  ASSERT_EQ(hipblasLtMatmulPreferenceCreate(&pref), HIPBLAS_STATUS_SUCCESS);
  
  size_t max_workspace = 32 * 1024 * 1024;
  ASSERT_EQ(hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                   &max_workspace, sizeof(max_workspace)),
            HIPBLAS_STATUS_SUCCESS);
  
  hipblasLtMatmulHeuristicResult_t results[4];
  int returned;
  ASSERT_EQ(hipblasLtMatmulAlgoGetHeuristic(hipblaslt_handle, matmul_desc,
                                             layout_A, layout_B, layout_C, layout_D,
                                             pref, 4, results, &returned),
            HIPBLAS_STATUS_SUCCESS);
  
  std::cout << "Found " << returned << " gemm algorithms" << std::endl;
  
  float *d_A, *d_B, *d_C, *d_D;
  size_t size_A = gemm_m * gemm_k * sizeof(float);
  size_t size_B = gemm_k * gemm_n * sizeof(float);
  size_t size_C = gemm_m * gemm_n * sizeof(float);
  size_t size_D = gemm_m * gemm_n * sizeof(float);
  
  ASSERT_EQ(hipMalloc(&d_A, size_A), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_B, size_B), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_C, size_C), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_D, size_D), hipSuccess);
  
  void* gemm_workspace = nullptr;
  if (results[0].workspaceSize > 0) {
    ASSERT_EQ(hipMalloc(&gemm_workspace, results[0].workspaceSize), hipSuccess);
  }
  
  std::vector<float> h_A(gemm_m * gemm_k, 1.0f);
  std::vector<float> h_B(gemm_k * gemm_n, 1.0f);
  std::vector<float> h_C(gemm_m * gemm_n, 0.0f);
  
  ASSERT_EQ(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_C, h_C.data(), size_C, hipMemcpyHostToDevice), hipSuccess);
  
  float gemm_alpha = 1.0f, gemm_beta = 0.0f;
  ASSERT_EQ(hipblasLtMatmul(hipblaslt_handle, matmul_desc, &gemm_alpha,
                             d_A, layout_A, d_B, layout_B,
                             &gemm_beta, d_C, layout_C, d_D, layout_D,
                             &results[0].algo, gemm_workspace, results[0].workspaceSize,
                             stream_),
            HIPBLAS_STATUS_SUCCESS);
  
  ASSERT_EQ(hipStreamSynchronize(stream_), hipSuccess);
  
  std::vector<float> h_D(gemm_m * gemm_n);
  ASSERT_EQ(hipMemcpy(h_D.data(), d_D, size_D, hipMemcpyDeviceToHost), hipSuccess);
  
  float expected = static_cast<float>(gemm_k);
  std::cout << "Gemm output D[0] = " << h_D[0] << " (expected: " << expected << ")" << std::endl;
  EXPECT_NEAR(h_D[0], expected, 1e-3);
  
  std::cout << "\n=== Both Conv and Gemm operations completed successfully! ===" << std::endl;
  
  // Cleanup gemm
  if (gemm_workspace) hipFree(gemm_workspace);
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
  hipblasLtDestroy(hipblaslt_handle);
}

int main(int argc, char** argv) {
  std::cout << "\n========================================" << std::endl;
  std::cout << "ROCm Conv and Gemm Combined Test" << std::endl;
  std::cout << "========================================\n" << std::endl;
  
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
