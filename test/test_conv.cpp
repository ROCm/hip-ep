// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

/**
 * Test for ROCm Conv custom op using MIOpen
 */

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <miopen/miopen.h>
#include <vector>
#include <iostream>

// Test MIOpen convolution directly (without ORT)
TEST(RocmConvTest, MiopenDirectConv) {
  // Check GPU availability
  int device_count = 0;
  hipError_t err = hipGetDeviceCount(&device_count);
  if (err != hipSuccess || device_count == 0) {
    GTEST_SKIP() << "No AMD GPU available, skipping test";
  }

  // Initialize MIOpen
  miopenHandle_t handle;
  ASSERT_EQ(miopenCreate(&handle), miopenStatusSuccess);

  hipStream_t stream;
  ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
  ASSERT_EQ(miopenSetStream(handle, stream), miopenStatusSuccess);

  // Define dimensions: Input [1, 3, 8, 8], Weight [16, 3, 3, 3], Output [1, 16, 8, 8]
  const int N = 1, C = 3, H = 8, W = 8;
  const int K = 16, R = 3, S = 3;
  const int pad = 1, stride = 1, dilation = 1;

  // Create tensor descriptors
  miopenTensorDescriptor_t input_desc, weight_desc, output_desc;
  ASSERT_EQ(miopenCreateTensorDescriptor(&input_desc), miopenStatusSuccess);
  ASSERT_EQ(miopenCreateTensorDescriptor(&weight_desc), miopenStatusSuccess);
  ASSERT_EQ(miopenCreateTensorDescriptor(&output_desc), miopenStatusSuccess);

  ASSERT_EQ(miopenSet4dTensorDescriptor(input_desc, miopenFloat, N, C, H, W), miopenStatusSuccess);
  ASSERT_EQ(miopenSet4dTensorDescriptor(weight_desc, miopenFloat, K, C, R, S), miopenStatusSuccess);

  // Create convolution descriptor
  miopenConvolutionDescriptor_t conv_desc;
  ASSERT_EQ(miopenCreateConvolutionDescriptor(&conv_desc), miopenStatusSuccess);
  ASSERT_EQ(miopenInitConvolutionDescriptor(conv_desc, miopenConvolution,
                                             pad, pad, stride, stride, dilation, dilation),
            miopenStatusSuccess);

  // Compute output dimensions
  int out_n, out_c, out_h, out_w;
  ASSERT_EQ(miopenGetConvolutionForwardOutputDim(conv_desc, input_desc, weight_desc,
                                                  &out_n, &out_c, &out_h, &out_w),
            miopenStatusSuccess);

  EXPECT_EQ(out_n, N);
  EXPECT_EQ(out_c, K);
  EXPECT_EQ(out_h, H);  // Same padding
  EXPECT_EQ(out_w, W);

  ASSERT_EQ(miopenSet4dTensorDescriptor(output_desc, miopenFloat, out_n, out_c, out_h, out_w),
            miopenStatusSuccess);

  std::cout << "Output shape: [" << out_n << ", " << out_c << ", " << out_h << ", " << out_w << "]"
            << std::endl;

  // Allocate device memory
  float *d_input, *d_weight, *d_output;
  size_t input_size = N * C * H * W * sizeof(float);
  size_t weight_size = K * C * R * S * sizeof(float);
  size_t output_size = out_n * out_c * out_h * out_w * sizeof(float);

  ASSERT_EQ(hipMalloc(&d_input, input_size), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_weight, weight_size), hipSuccess);
  ASSERT_EQ(hipMalloc(&d_output, output_size), hipSuccess);

  // Initialize host data
  std::vector<float> h_input(N * C * H * W, 1.0f);
  std::vector<float> h_weight(K * C * R * S, 0.1f);

  ASSERT_EQ(hipMemcpy(d_input, h_input.data(), input_size, hipMemcpyHostToDevice), hipSuccess);
  ASSERT_EQ(hipMemcpy(d_weight, h_weight.data(), weight_size, hipMemcpyHostToDevice), hipSuccess);

  // Get workspace size
  size_t workspace_size;
  ASSERT_EQ(miopenConvolutionForwardGetWorkSpaceSize(handle, weight_desc, input_desc,
                                                      conv_desc, output_desc, &workspace_size),
            miopenStatusSuccess);

  void* workspace = nullptr;
  if (workspace_size > 0) {
    ASSERT_EQ(hipMalloc(&workspace, workspace_size), hipSuccess);
  }

  std::cout << "Workspace size: " << workspace_size << " bytes" << std::endl;

  // Find algorithm
  miopenConvAlgoPerf_t perf_results[4];
  int algo_count;
  ASSERT_EQ(miopenFindConvolutionForwardAlgorithm(handle, input_desc, d_input,
                                                   weight_desc, d_weight,
                                                   conv_desc, output_desc, d_output,
                                                   4, &algo_count, perf_results,
                                                   workspace, workspace_size, false),
            miopenStatusSuccess);

  EXPECT_GT(algo_count, 0);
  std::cout << "Found " << algo_count << " algorithms" << std::endl;

  // Execute convolution
  float alpha = 1.0f, beta = 0.0f;
  ASSERT_EQ(miopenConvolutionForward(handle, &alpha,
                                      input_desc, d_input,
                                      weight_desc, d_weight,
                                      conv_desc, perf_results[0].fwd_algo, &beta,
                                      output_desc, d_output,
                                      workspace, workspace_size),
            miopenStatusSuccess);

  // Sync and copy result
  ASSERT_EQ(hipStreamSynchronize(stream), hipSuccess);

  std::vector<float> h_output(out_n * out_c * out_h * out_w);
  ASSERT_EQ(hipMemcpy(h_output.data(), d_output, output_size, hipMemcpyDeviceToHost), hipSuccess);

  // Verify output (should be non-zero)
  bool has_nonzero = false;
  for (const auto& val : h_output) {
    if (val != 0.0f) {
      has_nonzero = true;
      break;
    }
  }
  EXPECT_TRUE(has_nonzero) << "Output should contain non-zero values";

  std::cout << "Output[0]: " << h_output[0] << std::endl;

  // Cleanup
  if (workspace) hipFree(workspace);
  hipFree(d_output);
  hipFree(d_weight);
  hipFree(d_input);
  miopenDestroyConvolutionDescriptor(conv_desc);
  miopenDestroyTensorDescriptor(output_desc);
  miopenDestroyTensorDescriptor(weight_desc);
  miopenDestroyTensorDescriptor(input_desc);
  hipStreamDestroy(stream);
  miopenDestroy(handle);
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
