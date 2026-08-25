/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

#include <string>

namespace hipep {

// Never invoked on a healthy path: hip-ep's GetCapability claims and fuses
// the node into a MetaDef before ORT would ever need a CPU kernel for it.
// Reaching Compute() means hip-ep failed to claim/fuse a node it advertised
// a schema for; this is a defensive guardrail, not a real implementation.
struct SchemaOnlyStubKernel {
  explicit SchemaOnlyStubKernel(const OrtKernelInfo *info) : info_(info) {}

  void Compute(OrtKernelContext * /*context*/) {
    ORT_CXX_API_THROW(
        "hip-ep custom op node '" + Ort::ConstKernelInfo(info_).GetNodeName() +
            "' was not claimed/fused by the AMD GPU EP; schema-only custom "
            "ops have no CPU kernel.",
        ORT_NOT_IMPLEMENTED);
  }

private:
  const OrtKernelInfo *info_;
};

using SchemaOnlyShapeInferFn =
    OrtStatusPtr(ORT_API_CALL *)(const OrtCustomOp *, OrtShapeInferContext *);

// One Ort::CustomOpBase subclass covers every schema-only op; kSchemaOnlyOps
// passes a per-op SchemaOnlyShapeInferFn (see schema_only_custom_op.cpp).
// We inherit CustomOpBase for CreateKernel/GetName/type hooks but do not use
// its static InferOutputShape path -- that is one callback per class, while we
// need one callback per registered op instance.
class SchemaOnlyCustomOp
    : public Ort::CustomOpBase<SchemaOnlyCustomOp, SchemaOnlyStubKernel> {
public:
  SchemaOnlyCustomOp(std::string name, SchemaOnlyShapeInferFn shape_infer_fn)
      : name_(std::move(name)) {
    // CustomOpBase ctor clears InferOutputShapeFn when no static
    // InferOutputShape exists. Assign on this OrtCustomOp instance only; ORT
    // calls it for nodes matching this op's domain/name, not other custom ops.
    OrtCustomOp::InferOutputShapeFn = shape_infer_fn;
  }

  void *CreateKernel(const OrtApi &, const OrtKernelInfo *info) const {
    return new SchemaOnlyStubKernel(info);
  }
  const char *GetName() const { return name_.c_str(); }

  size_t GetInputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetInputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }
  int GetVariadicInputMinArity() const { return 0; }
  bool GetVariadicInputHomogeneity() const { return false; }

  size_t GetOutputTypeCount() const { return 1; }
  ONNXTensorElementDataType GetOutputType(size_t) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC;
  }
  int GetVariadicOutputMinArity() const { return 0; }
  bool GetVariadicOutputHomogeneity() const { return false; }

private:
  std::string name_;
};

} // namespace hipep
