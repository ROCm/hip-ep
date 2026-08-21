/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
// Schema-only registration for com.amd::QMoE. The input/output signature
// below matches AMD's
// amd/NVIDIA-Nemotron-3-Super-120B-A12B-fp16-int4-rtn-onnx-QMoE
// CUSTOM_OP_SCHEMA.md. This file only makes ORT able to load/Resolve graphs
// containing this op.
#include "../register_custom_op.hpp"         // HIPEP_REGISTER_CUSTOM_OP
#include "../schema_only_custom_op_base.hpp" // hipep::SchemaOnlyCustomOpBase

namespace {

struct QMoESchemaOp : hipep::SchemaOnlyCustomOpBase<QMoESchemaOp> {
  const char *GetName() const { return "QMoE"; }

  size_t GetInputTypeCount() const { return 15; }
  ONNXTensorElementDataType GetInputType(size_t i) const {
    // Only the 6 INT4-packed weight tensors (indices 1/3/5/7/9/11: fc1/fc2
    // expert weights, fc1/fc2 latent weights, shared_fc1/fc2 weights) are
    // uint8; the remaining 9 inputs (hidden_states, every *_scales tensor,
    // router_weight, correction_bias) are fp16.
    static constexpr bool kIsUint8Weight[15] = {
        false, true, false, true, false, true,  false, true,
        false, true, false, true, false, false, false};
    return kIsUint8Weight[i] ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                             : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  }

  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
  }

  // All 15 inputs are required (no optional inputs, unlike
  // com.microsoft::QMoE), so the Ort::CustomOpBase default
  // (INPUT_OUTPUT_REQUIRED for every index) is already correct and doesn't
  // need to be overridden here.
  //
  // InferOutputShape (moe_out == hidden_states' shape) and the
  // num_experts/activation_type/... node attributes are intentionally not
  // handled here: ORT accepts arbitrary attributes on a custom-op node
  // without a schema declaration, and leaving the output shape dynamic is
  // sufficient for ORT to Resolve a graph that only contains this op.
};

} // namespace

HIPEP_REGISTER_CUSTOM_OP(QMoESchemaOp, "com.amd")
