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
// Reaching Compute() means the node was not assigned to hipgpu -- either
// GetCapability declined it, or hipgpu was never selected for this session,
// which is possible because schemas must be registered at Model::Load, before
// any backend selection happens. This is a defensive guardrail that reports
// what is missing, not a real implementation.
struct SchemaOnlyStubKernel {
  // The OrtKernelInfo handle is only valid for the duration of CreateKernel:
  // ORT hands over the address of a stack local (OpKernelInfo built in
  // KernelRegistryManager::CreateKernel, reinterpret_cast in
  // CustomOpKernel's ctor), and unlike ORT's own OpKernel base class a custom
  // op gets no copy of it. Read the node name here or not at all.
  explicit SchemaOnlyStubKernel(const OrtKernelInfo *info)
      : node_name_(NodeNameOf(info)) {}

  void Compute(OrtKernelContext * /*context*/) {
    ORT_CXX_API_THROW(
        "hip-ep-registered custom op node '" + node_name_ +
            "' fell back to CPU: this op is implemented only as a GPU kernel "
            "in the hip-ep backend and has no CPU kernel.",
        ORT_NOT_IMPLEMENTED);
  }

private:
  // Kernel construction must not throw, so that an unclaimed node still
  // reaches Compute and reports ORT_NOT_IMPLEMENTED rather than surfacing as
  // an opaque session-initialization failure.
  static std::string NodeNameOf(const OrtKernelInfo *info) noexcept {
    try {
      return Ort::ConstKernelInfo(info).GetNodeName();
    } catch (...) {
      return "<unknown>";
    }
  }

  std::string node_name_;
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
