// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "custom_op.hpp"
#include <glog/logging.h>
#include <stdexcept>
#include "google/protobuf/util/json_util.h"
#include "morphizen/env_config.hpp"

DEF_ENV_PARAM(MORPHIZEN_DEBUG_ROCM, "0")
#define MY_LOG(n) LOG_IF(INFO, ENV_PARAM(MORPHIZEN_DEBUG_ROCM) >= n)

namespace hip_ep {

RocmCustomOp::RocmCustomOp(
    std::shared_ptr<const vaip_core::PassContext> context,
    const std::shared_ptr<vaip_core::MetaDefProto>& meta_def,
    onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  
  // Get JSON params attached by the pass
  auto rocm_json_str = get_meta_def_param();
  MY_LOG(1) << "[ROCm CustomOp] Received JSON params: " << rocm_json_str;
  
  // Parse JSON to RocmParamProto
  auto status = google::protobuf::util::JsonStringToMessage(
      rocm_json_str, &rocm_proto_);
  if (!status.ok()) {
    LOG(ERROR) << "[ROCm CustomOp] Failed to parse JSON params: " << status.ToString();
    throw std::runtime_error("Failed to parse RocmParamProto: " + status.ToString());
  }

  MY_LOG(1) << "[ROCm CustomOp] Created for op_type: " << rocm_proto_.op_type();
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
  MY_LOG(2) << "[ROCm CustomOp] Compute() called";
  
  const auto& op_type = rocm_proto_.op_type();
  MY_LOG(2) << "[ROCm CustomOp] op_type = " << op_type;

  // Check if GPU is available before attempting execution
  MY_LOG(2) << "[ROCm CustomOp] Checking HIP context initialization...";
  auto& hip_ctx = HipContext::instance();
  if (!hip_ctx.is_initialized()) {
    LOG(ERROR) << "[ROCm CustomOp] HIP context not initialized - no AMD GPU available!";
    throw std::runtime_error("ROCm CustomOp: No AMD GPU available. HIP context initialization failed.");
  }
  MY_LOG(2) << "[ROCm CustomOp] HIP context is initialized";

  if (op_type == "conv") {
    MY_LOG(2) << "[ROCm CustomOp] Dispatching to ExecuteConv...";
    ExecuteConv(api, context);
  } else if (op_type == "gemm") {
    MY_LOG(2) << "[ROCm CustomOp] Dispatching to ExecuteGemm...";
    ExecuteGemm(api, context);
  } else {
    LOG(ERROR) << "[ROCm CustomOp] Unknown op_type: " << op_type;
    throw std::runtime_error("Unknown op_type: " + op_type);
  }
  MY_LOG(2) << "[ROCm CustomOp] Compute() completed";
}

void RocmCustomOp::ExecuteConv(const OrtApi* api, OrtKernelContext* context) const {
  MY_LOG(1) << "[ROCm CustomOp] ExecuteConv (MIOpen)";
  MY_LOG(2) << "[ROCm CustomOp] ExecuteConv() starting...";
  
  MY_LOG(2) << "[ROCm CustomOp] Getting HIP context...";
  auto& hip_ctx = HipContext::instance();
  
  MY_LOG(2) << "[ROCm CustomOp] Getting MIOpen handle...";
  auto miopen_handle = hip_ctx.miopen_handle();
  MY_LOG(2) << "[ROCm CustomOp] MIOpen handle = " << miopen_handle;
  
  MY_LOG(2) << "[ROCm CustomOp] Getting HIP stream...";
  auto stream = hip_ctx.stream();
  MY_LOG(2) << "[ROCm CustomOp] HIP stream = " << stream;

  const auto& params = rocm_proto_.conv_params();
  MY_LOG(2) << "[ROCm CustomOp] Conv params - batch=" << params.batch_size()
            << ", out_channels=" << params.out_channels()
            << ", out_height=" << params.out_height()
            << ", out_width=" << params.out_width();

  // Get input tensors from ORT context
  MY_LOG(2) << "[ROCm CustomOp] Getting input tensors...";
  
  // Get number of inputs to understand what's available
  size_t num_inputs = 0;
  api->KernelContext_GetInputCount(context, &num_inputs);
  MY_LOG(2) << "[ROCm CustomOp] Number of inputs: " << num_inputs;
  
  const OrtValue* input_x = nullptr;
  api->KernelContext_GetInput(context, 0, &input_x);
  MY_LOG(2) << "[ROCm CustomOp] input_x = " << input_x;
  
  if (input_x == nullptr) {
    LOG(ERROR) << "[ROCm CustomOp] input_x is NULL!";
    throw std::runtime_error("ROCm CustomOp: Input tensor X is NULL");
  }

  // Note: In fused VitisAI nodes, weights are typically constant initializers
  // that should be extracted during pass execution, not runtime inputs.
  // For now, we check if a second input exists.
  const OrtValue* input_w = nullptr;
  if (num_inputs > 1) {
    api->KernelContext_GetInput(context, 1, &input_w);
    MY_LOG(2) << "[ROCm CustomOp] input_w = " << input_w;
  } else {
    MY_LOG(2) << "[ROCm CustomOp] No weight tensor as runtime input (expected for fused nodes)";
    MY_LOG(2) << "[ROCm CustomOp] Weights should be cached from pass (not yet implemented)";
    // TODO: Weights should be extracted from the initializers during pass execution
    // and cached in this custom op. For now, we'll skip the actual computation.
  }

  // Get tensor data pointers (const_cast needed for legacy API)
  MY_LOG(2) << "[ROCm CustomOp] Getting tensor data pointers...";
  float* x_data = nullptr;
  api->GetTensorMutableData(const_cast<OrtValue*>(input_x), (void**)&x_data);
  MY_LOG(2) << "[ROCm CustomOp] x_data = " << x_data;
  
  float* w_data = nullptr;
  if (input_w != nullptr) {
    api->GetTensorMutableData(const_cast<OrtValue*>(input_w), (void**)&w_data);
    MY_LOG(2) << "[ROCm CustomOp] w_data = " << w_data;
  }

  // Get output tensor
  std::vector<int64_t> output_shape = {
    params.batch_size(),
    params.out_channels(),
    params.out_height(),
    params.out_width()
  };
  
  MY_LOG(2) << "[ROCm CustomOp] Requested output shape [" 
            << output_shape[0] << ", " << output_shape[1] << ", "
            << output_shape[2] << ", " << output_shape[3] << "]";
  
  // Validate output shape - if params weren't properly populated, skip computation
  if (output_shape[0] <= 0 || output_shape[1] <= 0 || 
      output_shape[2] <= 0 || output_shape[3] <= 0) {
    LOG(ERROR) << "[ROCm CustomOp] Invalid output shape detected!";
    LOG(ERROR) << "[ROCm CustomOp] conv_params were not properly populated by the pass.";
    LOG(ERROR) << "[ROCm CustomOp] This is a pass configuration issue, not a runtime error.";
    throw std::runtime_error("ROCm CustomOp: Invalid output shape. Conv params not properly populated by pass.");
  }
  
  MY_LOG(2) << "[ROCm CustomOp] Getting output tensor...";
  OrtValue* output = nullptr;
  api->KernelContext_GetOutput(context, 0, output_shape.data(), output_shape.size(), &output);
  MY_LOG(2) << "[ROCm CustomOp] output = " << output;
  
  if (output == nullptr) {
    LOG(ERROR) << "[ROCm CustomOp] output is NULL!";
    throw std::runtime_error("ROCm CustomOp: Failed to get output tensor");
  }
  
  float* y_data = nullptr;
  api->GetTensorMutableData(output, (void**)&y_data);
  MY_LOG(2) << "[ROCm CustomOp] y_data = " << y_data;

  // TODO: Full MIOpen convolution implementation
  // For now, this is a placeholder showing the structure
  
  // 1. Create tensor descriptors
  // 2. Create convolution descriptor
  // 3. Find best algorithm
  // 4. Allocate workspace
  // 5. Execute miopenConvolutionForward
  // 6. Synchronize stream

  MY_LOG(2) << "[ROCm CustomOp] Calling hipStreamSynchronize...";
  hipError_t sync_err = hipStreamSynchronize(stream);
  MY_LOG(2) << "[ROCm CustomOp] hipStreamSynchronize returned: " << sync_err 
            << " (" << hipGetErrorString(sync_err) << ")";
  
  MY_LOG(2) << "[ROCm CustomOp] Conv completed";
  MY_LOG(2) << "[ROCm CustomOp] ExecuteConv() completed";
}

void RocmCustomOp::ExecuteGemm(const OrtApi* api, OrtKernelContext* context) const {
  MY_LOG(1) << "[ROCm CustomOp] ExecuteGemm (hipBLASLt)";
  MY_LOG(2) << "[ROCm CustomOp] ExecuteGemm() starting...";

  MY_LOG(2) << "[ROCm CustomOp] Getting HIP context...";
  auto& hip_ctx = HipContext::instance();
  
  MY_LOG(2) << "[ROCm CustomOp] Getting hipBLASLt handle...";
  auto blaslt_handle = hip_ctx.hipblaslt_handle();
  MY_LOG(2) << "[ROCm CustomOp] hipBLASLt handle = " << blaslt_handle;
  
  MY_LOG(2) << "[ROCm CustomOp] Getting HIP stream...";
  auto stream = hip_ctx.stream();
  MY_LOG(2) << "[ROCm CustomOp] HIP stream = " << stream;

  const auto& params = rocm_proto_.gemm_params();
  MY_LOG(2) << "[ROCm CustomOp] Gemm params - M=" << params.m()
            << ", N=" << params.n()
            << ", K=" << params.k();

  // Get input tensors from ORT context
  MY_LOG(2) << "[ROCm CustomOp] Getting input tensors...";
  
  // Get number of inputs to understand what's available
  size_t num_inputs = 0;
  api->KernelContext_GetInputCount(context, &num_inputs);
  MY_LOG(2) << "[ROCm CustomOp] Number of inputs: " << num_inputs;
  
  const OrtValue* input_a = nullptr;
  api->KernelContext_GetInput(context, 0, &input_a);
  MY_LOG(2) << "[ROCm CustomOp] input_a = " << input_a;
  
  if (input_a == nullptr) {
    LOG(ERROR) << "[ROCm CustomOp] input_a is NULL!";
    throw std::runtime_error("ROCm CustomOp: Input tensor A is NULL");
  }
  
  const OrtValue* input_b = nullptr;
  if (num_inputs > 1) {
    api->KernelContext_GetInput(context, 1, &input_b);
    MY_LOG(2) << "[ROCm CustomOp] input_b = " << input_b;
  } else {
    MY_LOG(2) << "[ROCm CustomOp] No B tensor as runtime input (expected for fused nodes)";
  }

  // Get tensor data pointers (const_cast needed for legacy API)
  MY_LOG(2) << "[ROCm CustomOp] Getting tensor data pointers...";
  float* a_data = nullptr;
  api->GetTensorMutableData(const_cast<OrtValue*>(input_a), (void**)&a_data);
  MY_LOG(2) << "[ROCm CustomOp] a_data = " << a_data;
  
  float* b_data = nullptr;
  if (input_b != nullptr) {
    api->GetTensorMutableData(const_cast<OrtValue*>(input_b), (void**)&b_data);
    MY_LOG(2) << "[ROCm CustomOp] b_data = " << b_data;
  }

  // Get output tensor
  std::vector<int64_t> output_shape = {params.m(), params.n()};
  
  MY_LOG(2) << "[ROCm CustomOp] Requested output shape [" 
            << output_shape[0] << ", " << output_shape[1] << "]";
  
  // Validate output shape
  if (output_shape[0] <= 0 || output_shape[1] <= 0) {
    LOG(ERROR) << "[ROCm CustomOp] Invalid output shape detected!";
    LOG(ERROR) << "[ROCm CustomOp] gemm_params were not properly populated by the pass.";
    throw std::runtime_error("ROCm CustomOp: Invalid output shape. Gemm params not properly populated by pass.");
  }
  
  MY_LOG(2) << "[ROCm CustomOp] Getting output tensor...";
  OrtValue* output = nullptr;
  api->KernelContext_GetOutput(context, 0, output_shape.data(), output_shape.size(), &output);
  MY_LOG(2) << "[ROCm CustomOp] output = " << output;
  
  if (output == nullptr) {
    LOG(ERROR) << "[ROCm CustomOp] output is NULL!";
    throw std::runtime_error("ROCm CustomOp: Failed to get output tensor");
  }
  
  float* d_data = nullptr;
  api->GetTensorMutableData(output, (void**)&d_data);
  MY_LOG(2) << "[ROCm CustomOp] d_data = " << d_data;

  // TODO: Full hipBLASLt GEMM implementation
  // For now, this is a placeholder showing the structure
  
  // 1. Create matrix layouts
  // 2. Create matmul descriptor
  // 3. Get heuristics for best algorithm
  // 4. Allocate workspace
  // 5. Execute hipblasLtMatmul
  // 6. Synchronize stream

  MY_LOG(2) << "[ROCm CustomOp] Calling hipStreamSynchronize...";
  hipError_t sync_err = hipStreamSynchronize(stream);
  MY_LOG(2) << "[ROCm CustomOp] hipStreamSynchronize returned: " << sync_err 
            << " (" << hipGetErrorString(sync_err) << ")";

  MY_LOG(2) << "[ROCm CustomOp] Gemm completed";
  MY_LOG(2) << "[ROCm CustomOp] ExecuteGemm() completed";
}

} // namespace hip_ep
