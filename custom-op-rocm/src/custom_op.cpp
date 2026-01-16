// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "custom_op.hpp"
#include <glog/logging.h>
#include <stdexcept>

namespace rocm_ep {

RocmCustomOp::RocmCustomOp(
    std::shared_ptr<const vaip_core::PassContext> context,
    const std::shared_ptr<vaip_core::MetaDefProto>& meta_def,
    onnxruntime::Model* model)
    : CustomOpImp(context, meta_def, model) {
  
  // Parse the generic_param to get RocmParamProto
  // generic_param is a Map<string, string>, look for "rocm_param" key
  const auto& params = meta_def->generic_param();
  auto it = params.find("rocm_param");
  if (it != params.end()) {
    if (!rocm_proto_.ParseFromString(it->second)) {
      throw std::runtime_error("Failed to parse RocmParamProto");
    }
  }

  LOG(INFO) << "[ROCm CustomOp] Created for op_type: " << rocm_proto_.op_type();
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

  if (op_type == "conv") {
    ExecuteConv(api, context);
  } else if (op_type == "gemm") {
    ExecuteGemm(api, context);
  } else {
    throw std::runtime_error("Unknown op_type: " + op_type);
  }
}

void RocmCustomOp::ExecuteConv(const OrtApi* api, OrtKernelContext* context) const {
  LOG(INFO) << "[ROCm CustomOp] ExecuteConv (MIOpen)";
  
  auto& hip_ctx = HipContext::instance();
  auto miopen_handle = hip_ctx.miopen_handle();
  auto stream = hip_ctx.stream();

  const auto& params = rocm_proto_.conv_params();

  // Get input tensors from ORT context
  const OrtValue* input_x = nullptr;
  const OrtValue* input_w = nullptr;
  api->KernelContext_GetInput(context, 0, &input_x);
  api->KernelContext_GetInput(context, 1, &input_w);

  // Get tensor data pointers (const_cast needed for legacy API)
  float* x_data = nullptr;
  float* w_data = nullptr;
  api->GetTensorMutableData(const_cast<OrtValue*>(input_x), (void**)&x_data);
  api->GetTensorMutableData(const_cast<OrtValue*>(input_w), (void**)&w_data);

  // Get output tensor
  std::vector<int64_t> output_shape = {
    params.batch_size(),
    params.out_channels(),
    params.out_height(),
    params.out_width()
  };
  
  OrtValue* output = nullptr;
  api->KernelContext_GetOutput(context, 0, output_shape.data(), output_shape.size(), &output);
  
  float* y_data = nullptr;
  api->GetTensorMutableData(output, (void**)&y_data);

  // TODO: Full MIOpen convolution implementation
  // For now, this is a placeholder showing the structure
  
  // 1. Create tensor descriptors
  // 2. Create convolution descriptor
  // 3. Find best algorithm
  // 4. Allocate workspace
  // 5. Execute miopenConvolutionForward
  // 6. Synchronize stream

  hipStreamSynchronize(stream);
  LOG(INFO) << "[ROCm CustomOp] Conv completed";
}

void RocmCustomOp::ExecuteGemm(const OrtApi* api, OrtKernelContext* context) const {
  LOG(INFO) << "[ROCm CustomOp] ExecuteGemm (hipBLASLt)";

  auto& hip_ctx = HipContext::instance();
  auto blaslt_handle = hip_ctx.hipblaslt_handle();
  auto stream = hip_ctx.stream();

  const auto& params = rocm_proto_.gemm_params();

  // Get input tensors from ORT context
  const OrtValue* input_a = nullptr;
  const OrtValue* input_b = nullptr;
  api->KernelContext_GetInput(context, 0, &input_a);
  api->KernelContext_GetInput(context, 1, &input_b);

  // Get tensor data pointers (const_cast needed for legacy API)
  float* a_data = nullptr;
  float* b_data = nullptr;
  api->GetTensorMutableData(const_cast<OrtValue*>(input_a), (void**)&a_data);
  api->GetTensorMutableData(const_cast<OrtValue*>(input_b), (void**)&b_data);

  // Get output tensor
  std::vector<int64_t> output_shape = {params.m(), params.n()};
  
  OrtValue* output = nullptr;
  api->KernelContext_GetOutput(context, 0, output_shape.data(), output_shape.size(), &output);
  
  float* d_data = nullptr;
  api->GetTensorMutableData(output, (void**)&d_data);

  // TODO: Full hipBLASLt GEMM implementation
  // For now, this is a placeholder showing the structure
  
  // 1. Create matrix layouts
  // 2. Create matmul descriptor
  // 3. Get heuristics for best algorithm
  // 4. Allocate workspace
  // 5. Execute hipblasLtMatmul
  // 6. Synchronize stream

  hipStreamSynchronize(stream);
  LOG(INFO) << "[ROCm CustomOp] Gemm completed";
}

} // namespace rocm_ep
