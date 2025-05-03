/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "./onnxruntime_api.hpp"

namespace vaip_core {

OrtStatusPtr xilinx_custom_op_shape_infer(const OrtCustomOp* op,
                                          OrtShapeInferContext* ort_ctx);

template <typename TKernel>
struct XilinxCustomOpBase
    : public Ort::CustomOpBase<XilinxCustomOpBase<TKernel>, TKernel> {
  std::string name_;
  bool is_single_output_;

  XilinxCustomOpBase(const std::string& name, bool is_single_output)
      : name_(name), is_single_output_(is_single_output) {

    OrtCustomOp::InferOutputShapeFn = xilinx_custom_op_shape_infer;
  }

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return std::make_unique<TKernel>(api, info).release();
  };

  const char* GetName() const { return name_.c_str(); };

  const char* GetExecutionProviderType() const {
    return "VitisAIExecutionProvider";
  };

  size_t GetInputTypeCount() const { return 1u; }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    // versions of the same operator define.
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };

  size_t GetOutputTypeCount() const { return 1u; };

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    // CHECK_EQ(index, 0u)
    //   << "'com.xilinx' domain's op not support multiple outputs ";
    // If 'type' is undefined, all types are allowed regardless of what
    // other versions of the same operator define.

    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  };

  int GetVariadicInputMinArity() const { return 0; }
  int GetVariadicOutputMinArity() const { return 0; }

  OrtCustomOpInputOutputCharacteristic
  GetInputCharacteristic(size_t /*index*/) const {
    // disable number of inputs checking , export_to_xir will checking
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }
  OrtCustomOpInputOutputCharacteristic
  GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }
  bool GetVariadicOutputHomogeneity() const { return false; }
  bool GetVariadicInputHomogeneity() const { return false; }
};

struct XilinxCustomKernel {
  XilinxCustomKernel(const OrtApi& api, const OrtKernelInfo* info) {}
  void Compute(OrtKernelContext* context);
};

struct XilinxCustomOp : XilinxCustomOpBase<XilinxCustomKernel> {
  VAIP_DLL_SPEC
  static std::unique_ptr<XilinxCustomOp> create(const std::string& name,
                                                bool is_single_output = true);
  VAIP_DLL_SPEC
  XilinxCustomOp(const std::string& name, bool is_single_output = true);
  VAIP_DLL_SPEC ~XilinxCustomOp();
};
} // namespace vaip_core
