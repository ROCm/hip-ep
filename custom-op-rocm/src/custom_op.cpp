// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "custom_op.hpp"
#include <glog/logging.h>
#include <iostream>
#include <stdexcept>
#include "google/protobuf/util/json_util.h"
#include "morphizen/env_config.hpp"

DEF_ENV_PARAM(MORPHIZEN_DEBUG_ROCM, "0")
DEF_ENV_PARAM(MORPHIZEN_GPU_TIMEOUT_MS, "5000")
DEF_ENV_PARAM(MORPHIZEN_GPU_WATCHDOG_ENABLED, "1")

#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)

namespace rocm_ep {

// Implementation of HipContext::sync_stream_with_timeout
TimeoutStatus HipContext::sync_stream_with_timeout(int timeout_ms) {
  ensure_initialized();
  if (!initialized_) {
    return TimeoutStatus::ERROR;
  }
  
  // Use environment variable default if not specified
  if (timeout_ms <= 0) {
    timeout_ms = ENV_PARAM(MORPHIZEN_GPU_TIMEOUT_MS);
  }
  
  MY_LOG(1) << "[HipContext] Synchronizing stream with " << timeout_ms << "ms timeout...";
  return WaitStreamWithTimeout(stream_, timeout_ms);
}

RocmCustomOp::RocmCustomOp(
    std::shared_ptr<const vaip_core::PassContext> context,
    const std::shared_ptr<vaip_core::MetaDefProto>& meta_def,
    onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  
  MY_LOG(2) << "[ROCm CustomOp] Constructor called";
  
  // Get JSON params attached by the pass
  auto rocm_json_str = get_meta_def_param();
  MY_LOG(2) << "[ROCm CustomOp] Received JSON params: " << rocm_json_str;
  MY_LOG(1) << "[ROCm CustomOp] Received JSON params: " << rocm_json_str;
  
  // Parse JSON to RocmParamProto
  auto status = google::protobuf::util::JsonStringToMessage(
      rocm_json_str, &rocm_proto_);
  if (!status.ok()) {
    LOG(ERROR) << "[ROCm CustomOp] Failed to parse JSON params: " << status.ToString();
    throw std::runtime_error("Failed to parse RocmParamProto: " + status.ToString());
  }

  MY_LOG(1) << "[ROCm CustomOp] Created for op_type: " << rocm_proto_.op_type();
  
  // Load cached weights from PassContext
  MY_LOG(2) << "[ROCm CustomOp] Loading cached weights...";
  LoadCachedWeights();
  MY_LOG(2) << "[ROCm CustomOp] Cached weights loaded successfully";
}

void RocmCustomOp::LoadCachedWeights() {
  const auto& op_type = rocm_proto_.op_type();
  auto pass_context = get_context();
  
  if (op_type == "conv") {
    const auto& params = rocm_proto_.conv_params();
    
    // Load weight file
    const auto& weight_file = params.weight_file_path();
    if (!weight_file.empty()) {
      int64_t weight_size = params.weight_file_size();
      MY_LOG(2) << "[ROCm CustomOp] Loading conv weight from: " << weight_file << " (" << weight_size << " bytes)";
      
      auto reader = pass_context->open_file_for_read(weight_file);
      if (reader) {
        weight_size_ = static_cast<size_t>(weight_size);
        host_weight_.resize(weight_size_ / sizeof(float));
        reader->fread(host_weight_.data(), weight_size_);
        MY_LOG(1) << "[ROCm CustomOp] Loaded weight: " << weight_file 
                  << " (" << host_weight_.size() << " floats)";
      } else {
        LOG(ERROR) << "[ROCm CustomOp] Failed to open weight file: " << weight_file;
      }
    }
    
    // Load bias file if present
    const auto& bias_file = params.bias_file_path();
    if (!bias_file.empty() && params.has_bias()) {
      int64_t bias_size = params.bias_file_size();
      MY_LOG(2) << "[ROCm CustomOp] Loading conv bias from: " << bias_file << " (" << bias_size << " bytes)";
      
      auto reader = pass_context->open_file_for_read(bias_file);
      if (reader) {
        bias_size_ = static_cast<size_t>(bias_size);
        host_bias_.resize(bias_size_ / sizeof(float));
        reader->fread(host_bias_.data(), bias_size_);
        MY_LOG(1) << "[ROCm CustomOp] Loaded bias: " << bias_file 
                  << " (" << host_bias_.size() << " floats)";
      } else {
        LOG(ERROR) << "[ROCm CustomOp] Failed to open bias file: " << bias_file;
      }
    }
  } else if (op_type == "gemm") {
    // TODO: Load GEMM weights if needed (B matrix for fused GEMM)
    MY_LOG(2) << "[ROCm CustomOp] GEMM weight loading not yet implemented";
  }
  
  weights_cached_ = true;
}

RocmCustomOp::~RocmCustomOp() {
  // Cleanup device memory
  if (d_weight_) hipFree(d_weight_);
  if (d_bias_) hipFree(d_bias_);
  if (workspace_) hipFree(workspace_);

  // Cleanup MIOpen descriptors
  if (input_desc_) miopenDestroyTensorDescriptor(input_desc_);
  if (weight_desc_) miopenDestroyTensorDescriptor(weight_desc_);
  if (output_desc_) miopenDestroyTensorDescriptor(output_desc_);
  if (conv_desc_) miopenDestroyConvolutionDescriptor(conv_desc_);

  // Cleanup hipBLASLt descriptors
  if (layout_a_) hipblasLtMatrixLayoutDestroy(layout_a_);
  if (layout_b_) hipblasLtMatrixLayoutDestroy(layout_b_);
  if (layout_c_) hipblasLtMatrixLayoutDestroy(layout_c_);
  if (matmul_desc_) hipblasLtMatmulDescDestroy(matmul_desc_);
}

void RocmCustomOp::Compute(const OrtApi* api, OrtKernelContext* context) const {
  const auto& op_type = rocm_proto_.op_type();
  MY_LOG(1) << "[ROCm CustomOp] Compute(" << op_type << ")";

  // Check if GPU is available before attempting execution
  auto& hip_ctx = HipContext::instance();
  if (!hip_ctx.is_initialized()) {
    LOG(ERROR) << "[ROCm CustomOp] HIP context not initialized - no AMD GPU available!";
    throw std::runtime_error("ROCm CustomOp: No AMD GPU available. HIP context initialization failed.");
  }

  if (op_type == "conv") {
    ExecuteConv(api, context);
  } else if (op_type == "gemm") {
    ExecuteGemm(api, context);
  } else {
    LOG(ERROR) << "[ROCm CustomOp] Unknown op_type: " << op_type;
    throw std::runtime_error("Unknown op_type: " + op_type);
  }
}

void RocmCustomOp::ExecuteConv(const OrtApi* api, OrtKernelContext* context) const {
  MY_LOG(1) << "[ROCm CustomOp] ExecuteConv (MIOpen)";
  
  auto& hip_ctx = HipContext::instance();
  auto miopen_handle = hip_ctx.miopen_handle();
  auto stream = hip_ctx.stream();

  const auto& params = rocm_proto_.conv_params();
  MY_LOG(2) << "[ROCm CustomOp] Conv: batch=" << params.batch_size()
            << ", C_in=" << params.in_channels() << ", C_out=" << params.out_channels()
            << ", H=" << params.in_height() << "x" << params.in_width()
            << " -> " << params.out_height() << "x" << params.out_width();

  // Get input tensor from ORT context
  const OrtValue* input_x = nullptr;
  api->KernelContext_GetInput(context, 0, &input_x);
  if (input_x == nullptr) {
    LOG(ERROR) << "[ROCm CustomOp] input_x is NULL!";
    throw std::runtime_error("ROCm CustomOp: Input tensor X is NULL");
  }

  // Get CPU data pointer for input
  float* h_input = nullptr;
  api->GetTensorMutableData(const_cast<OrtValue*>(input_x), (void**)&h_input);

  // Get output tensor
  std::vector<int64_t> output_shape = {
    params.batch_size(),
    params.out_channels(),
    params.out_height(),
    params.out_width()
  };
  
  // Validate output shape
  if (output_shape[0] <= 0 || output_shape[1] <= 0 || 
      output_shape[2] <= 0 || output_shape[3] <= 0) {
    LOG(ERROR) << "[ROCm CustomOp] Invalid output shape!";
    throw std::runtime_error("ROCm CustomOp: Invalid output shape. Conv params not properly populated by pass.");
  }
  
  OrtValue* output = nullptr;
  api->KernelContext_GetOutput(context, 0, output_shape.data(), output_shape.size(), &output);
  if (output == nullptr) {
    LOG(ERROR) << "[ROCm CustomOp] output is NULL!";
    throw std::runtime_error("ROCm CustomOp: Failed to get output tensor");
  }
  
  float* h_output = nullptr;
  api->GetTensorMutableData(output, (void**)&h_output);

  // Check cached weights
  if (host_weight_.empty()) {
    LOG(ERROR) << "[ROCm CustomOp] No cached weights available!";
    throw std::runtime_error("ROCm CustomOp Conv: No cached weights available");
  }

  // Calculate tensor sizes
  size_t input_size = static_cast<size_t>(params.batch_size()) * params.in_channels() * 
                      params.in_height() * params.in_width();
  size_t output_size = static_cast<size_t>(params.batch_size()) * params.out_channels() * 
                       params.out_height() * params.out_width();
  size_t input_bytes = input_size * sizeof(float);
  size_t output_bytes = output_size * sizeof(float);
  size_t weight_bytes = host_weight_.size() * sizeof(float);

  MY_LOG(2) << "[ROCm CustomOp] Input: " << input_size << " floats (" << input_bytes << " bytes)";
  MY_LOG(2) << "[ROCm CustomOp] Output: " << output_size << " floats (" << output_bytes << " bytes)";

  // =========================================================================
  // GPU Memory Management with Async Transfers
  // =========================================================================
  
  // Allocate device input buffer (mutable, reallocate if needed)
  static thread_local float* d_input = nullptr;
  static thread_local size_t d_input_size = 0;
  
  if (d_input == nullptr || d_input_size < input_bytes) {
    if (d_input) hipFree(d_input);
    hipError_t err = hipMalloc(&d_input, input_bytes);
    if (err != hipSuccess) {
      LOG(ERROR) << "[ROCm CustomOp] Failed to allocate device input: " << hipGetErrorString(err);
      throw std::runtime_error("Failed to allocate device input memory");
    }
    d_input_size = input_bytes;
    MY_LOG(1) << "[ROCm CustomOp] Allocated device input buffer: " << input_bytes << " bytes";
  }

  // Allocate device output buffer (mutable, reallocate if needed)
  static thread_local float* d_output = nullptr;
  static thread_local size_t d_output_size = 0;
  
  if (d_output == nullptr || d_output_size < output_bytes) {
    if (d_output) hipFree(d_output);
    hipError_t err = hipMalloc(&d_output, output_bytes);
    if (err != hipSuccess) {
      LOG(ERROR) << "[ROCm CustomOp] Failed to allocate device output: " << hipGetErrorString(err);
      throw std::runtime_error("Failed to allocate device output memory");
    }
    d_output_size = output_bytes;
    MY_LOG(1) << "[ROCm CustomOp] Allocated device output buffer: " << output_bytes << " bytes";
  }

  // Upload weights to device (lazy initialization - only once)
  if (d_weight_ == nullptr) {
    hipError_t err = hipMalloc(&d_weight_, weight_bytes);
    if (err != hipSuccess) {
      LOG(ERROR) << "[ROCm CustomOp] Failed to allocate device weight: " << hipGetErrorString(err);
      throw std::runtime_error("Failed to allocate device weight memory");
    }
    err = hipMemcpyAsync(d_weight_, host_weight_.data(), weight_bytes, hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
      LOG(ERROR) << "[ROCm CustomOp] Failed to upload weights: " << hipGetErrorString(err);
      throw std::runtime_error("Failed to copy weights to device");
    }
    MY_LOG(1) << "[ROCm CustomOp] Uploaded weights to GPU: " << weight_bytes << " bytes";
  }
  
  // Upload bias if present (lazy initialization)
  if (params.has_bias() && !host_bias_.empty() && d_bias_ == nullptr) {
    size_t bias_bytes = host_bias_.size() * sizeof(float);
    hipError_t err = hipMalloc(&d_bias_, bias_bytes);
    if (err != hipSuccess) {
      LOG(ERROR) << "[ROCm CustomOp] Failed to allocate device bias: " << hipGetErrorString(err);
      throw std::runtime_error("Failed to allocate device bias memory");
    }
    err = hipMemcpyAsync(d_bias_, host_bias_.data(), bias_bytes, hipMemcpyHostToDevice, stream);
    if (err != hipSuccess) {
      LOG(ERROR) << "[ROCm CustomOp] Failed to upload bias: " << hipGetErrorString(err);
      throw std::runtime_error("Failed to copy bias to device");
    }
    MY_LOG(1) << "[ROCm CustomOp] Uploaded bias to GPU: " << bias_bytes << " bytes";
  }

  // =========================================================================
  // Async Input Transfer: CPU -> GPU
  // =========================================================================
  MY_LOG(2) << "[ROCm CustomOp] Async copy input to GPU...";
  hipError_t err = hipMemcpyAsync(d_input, h_input, input_bytes, hipMemcpyHostToDevice, stream);
  if (err != hipSuccess) {
    LOG(ERROR) << "[ROCm CustomOp] Failed to copy input to device: " << hipGetErrorString(err);
    throw std::runtime_error("Failed to copy input to device");
  }

  // =========================================================================
  // MIOpen Tensor Descriptors
  // =========================================================================
  miopenTensorDescriptor_t input_desc, weight_desc, output_desc;
  miopenConvolutionDescriptor_t conv_desc;
  
  miopenCreateTensorDescriptor(&input_desc);
  miopenCreateTensorDescriptor(&weight_desc);
  miopenCreateTensorDescriptor(&output_desc);
  miopenCreateConvolutionDescriptor(&conv_desc);

  miopenSet4dTensorDescriptor(input_desc, miopenFloat,
                              params.batch_size(), params.in_channels(),
                              params.in_height(), params.in_width());
  
  miopenSet4dTensorDescriptor(weight_desc, miopenFloat,
                              params.out_channels(), params.in_channels(),
                              params.filter_height(), params.filter_width());
  
  miopenSet4dTensorDescriptor(output_desc, miopenFloat,
                              params.batch_size(), params.out_channels(),
                              params.out_height(), params.out_width());
  
  miopenInitConvolutionDescriptor(conv_desc, miopenConvolution,
                                  params.pad_h(), params.pad_w(),
                                  params.stride_h(), params.stride_w(),
                                  params.dilation_h(), params.dilation_w());

  // =========================================================================
  // Get Workspace Size and Allocate
  // =========================================================================
  size_t workspace_size = 0;
  miopenConvolutionForwardGetWorkSpaceSize(miopen_handle, weight_desc, input_desc,
                                           conv_desc, output_desc, &workspace_size);
  
  void* workspace_ptr = nullptr;
  if (workspace_size > 0) {
    if (workspace_ == nullptr || workspace_size_ < workspace_size) {
      if (workspace_) hipFree(workspace_);
      err = hipMalloc(&workspace_, workspace_size);
      if (err != hipSuccess) {
        miopenDestroyConvolutionDescriptor(conv_desc);
        miopenDestroyTensorDescriptor(output_desc);
        miopenDestroyTensorDescriptor(weight_desc);
        miopenDestroyTensorDescriptor(input_desc);
        LOG(ERROR) << "[ROCm CustomOp] Failed to allocate workspace: " << hipGetErrorString(err);
        throw std::runtime_error("Failed to allocate workspace");
      }
      workspace_size_ = workspace_size;
      MY_LOG(1) << "[ROCm CustomOp] Allocated workspace: " << workspace_size << " bytes";
    }
    workspace_ptr = workspace_;
  }

  // =========================================================================
  // Find Best Algorithm (uses GPU input/output buffers)
  // =========================================================================
  miopenConvAlgoPerf_t perf_results[4];
  int algo_count = 0;
  
  miopenStatus_t status = miopenFindConvolutionForwardAlgorithm(
      miopen_handle, input_desc, d_input,
      weight_desc, d_weight_,
      conv_desc, output_desc, d_output,
      4, &algo_count, perf_results,
      workspace_ptr, workspace_size, false);
  
  if (status != miopenStatusSuccess || algo_count == 0) {
    miopenDestroyConvolutionDescriptor(conv_desc);
    miopenDestroyTensorDescriptor(output_desc);
    miopenDestroyTensorDescriptor(weight_desc);
    miopenDestroyTensorDescriptor(input_desc);
    LOG(ERROR) << "[ROCm CustomOp] Failed to find convolution algorithm";
    throw std::runtime_error("Failed to find convolution algorithm");
  }
  
  MY_LOG(1) << "[ROCm CustomOp] Found " << algo_count << " algorithms, best time: " 
            << perf_results[0].time << " ms";

  // =========================================================================
  // Execute Convolution (GPU -> GPU)
  // =========================================================================
  float alpha = params.alpha();
  float beta = params.beta();
  
  status = miopenConvolutionForward(miopen_handle, &alpha,
                                    input_desc, d_input,
                                    weight_desc, d_weight_,
                                    conv_desc, perf_results[0].fwd_algo, &beta,
                                    output_desc, d_output,
                                    workspace_ptr, workspace_size);
  
  // Cleanup descriptors
  miopenDestroyConvolutionDescriptor(conv_desc);
  miopenDestroyTensorDescriptor(output_desc);
  miopenDestroyTensorDescriptor(weight_desc);
  miopenDestroyTensorDescriptor(input_desc);
  
  if (status != miopenStatusSuccess) {
    LOG(ERROR) << "[ROCm CustomOp] miopenConvolutionForward failed";
    throw std::runtime_error("miopenConvolutionForward failed");
  }
  
  MY_LOG(1) << "[ROCm CustomOp] Convolution executed on GPU";

  // =========================================================================
  // Async Output Transfer: GPU -> CPU
  // =========================================================================
  MY_LOG(2) << "[ROCm CustomOp] Async copy output from GPU...";
  err = hipMemcpyAsync(h_output, d_output, output_bytes, hipMemcpyDeviceToHost, stream);
  if (err != hipSuccess) {
    LOG(ERROR) << "[ROCm CustomOp] Failed to copy output from device: " << hipGetErrorString(err);
    throw std::runtime_error("Failed to copy output from device");
  }

  // =========================================================================
  // Synchronize Stream with Timeout Protection
  // =========================================================================
  MY_LOG(2) << "[ROCm CustomOp] Synchronizing stream...";
  auto timeout_status = hip_ctx.sync_stream_with_timeout();
  
  if (timeout_status == TimeoutStatus::TIMEOUT) {
    LOG(ERROR) << "[ROCm CustomOp] Conv operation TIMED OUT!";
    throw std::runtime_error("ROCm CustomOp Conv: GPU operation timed out");
  } else if (timeout_status == TimeoutStatus::ERROR) {
    LOG(ERROR) << "[ROCm CustomOp] Conv stream synchronization ERROR!";
    throw std::runtime_error("ROCm CustomOp Conv: Stream synchronization error");
  }
  
  MY_LOG(1) << "[ROCm CustomOp] Conv completed successfully";
}

void RocmCustomOp::ExecuteGemm(const OrtApi* api, OrtKernelContext* context) const {
  MY_LOG(1) << "[ROCm CustomOp] ExecuteGemm (hipBLASLt)";
  
  auto& hip_ctx = HipContext::instance();
  auto blaslt_handle = hip_ctx.hipblaslt_handle();
  auto stream = hip_ctx.stream();

  const auto& params = rocm_proto_.gemm_params();
  MY_LOG(2) << "[ROCm CustomOp] Gemm params - M=" << params.m()
            << ", N=" << params.n() << ", K=" << params.k();

  // Get input tensor from ORT context
  const OrtValue* input_a = nullptr;
  api->KernelContext_GetInput(context, 0, &input_a);
  if (input_a == nullptr) {
    LOG(ERROR) << "[ROCm CustomOp] input_a is NULL!";
    throw std::runtime_error("ROCm CustomOp: Input tensor A is NULL");
  }

  // Get CPU data pointer for input
  float* h_a = nullptr;
  api->GetTensorMutableData(const_cast<OrtValue*>(input_a), (void**)&h_a);

  // Get output tensor
  std::vector<int64_t> output_shape = {params.m(), params.n()};
  
  // Validate output shape
  if (output_shape[0] <= 0 || output_shape[1] <= 0) {
    LOG(ERROR) << "[ROCm CustomOp] Invalid output shape!";
    throw std::runtime_error("ROCm CustomOp: Invalid output shape. Gemm params not properly populated by pass.");
  }
  
  OrtValue* output = nullptr;
  api->KernelContext_GetOutput(context, 0, output_shape.data(), output_shape.size(), &output);
  if (output == nullptr) {
    LOG(ERROR) << "[ROCm CustomOp] output is NULL!";
    throw std::runtime_error("ROCm CustomOp: Failed to get output tensor");
  }
  
  float* h_output = nullptr;
  api->GetTensorMutableData(output, (void**)&h_output);

  // TODO: Full hipBLASLt GEMM implementation with async transfers
  // Similar to ExecuteConv:
  // 1. Allocate device buffers for A, B, D
  // 2. hipMemcpyAsync A from host to device
  // 3. Create matrix layouts and matmul descriptor
  // 4. Get algorithm heuristics
  // 5. Execute hipblasLtMatmul
  // 6. hipMemcpyAsync D from device to host
  // 7. Synchronize stream

  MY_LOG(2) << "[ROCm CustomOp] Synchronizing stream...";
  auto timeout_status = hip_ctx.sync_stream_with_timeout();
  
  if (timeout_status == TimeoutStatus::TIMEOUT) {
    LOG(ERROR) << "[ROCm CustomOp] Gemm operation TIMED OUT!";
    throw std::runtime_error("ROCm CustomOp Gemm: GPU operation timed out");
  } else if (timeout_status == TimeoutStatus::ERROR) {
    LOG(ERROR) << "[ROCm CustomOp] Gemm stream synchronization ERROR!";
    throw std::runtime_error("ROCm CustomOp Gemm: Stream synchronization error");
  }

  MY_LOG(1) << "[ROCm CustomOp] Gemm completed";
}

} // namespace rocm_ep
