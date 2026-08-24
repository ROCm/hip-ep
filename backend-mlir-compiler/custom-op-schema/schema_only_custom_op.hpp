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

// A schema-only custom op only needs to let ORT load/Resolve a graph
// containing it -- there is no real kernel and no shape inference, so
// there is nothing to distinguish one op from another beyond its name,
// domain, and output element type. UNDEFINED+VARIADIC inputs tell ORT to
// skip input type/count validation, which is why a single concrete class
// parameterized by name and output type (rather than one CRTP subclass per
// op) is enough to cover every op in this bucket.
//
// The output element type can't default to UNDEFINED the same way: ORT's
// ONNX type/shape inference (Graph::UpdateShapeInference) requires a single
// concrete formal output type whenever the graph doesn't already carry an
// explicit type for that intermediate tensor (the common case) -- with
// UNDEFINED the formal type list expands to every tensor element type, none
// of which is picked, and Model::Load fails with "type inference failed"
// on the first real graph where the op's output isn't already typed.
class SchemaOnlyCustomOp
    : public Ort::CustomOpBase<SchemaOnlyCustomOp, SchemaOnlyStubKernel> {
public:
  SchemaOnlyCustomOp(std::string name, ONNXTensorElementDataType output_type)
      : name_(std::move(name)), output_type_(output_type) {}

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
  ONNXTensorElementDataType GetOutputType(size_t) const { return output_type_; }
  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t) const {
    return output_type_ == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
               ? OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC
               : OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }
  int GetVariadicOutputMinArity() const { return 0; }
  bool GetVariadicOutputHomogeneity() const { return false; }

private:
  std::string name_;
  ONNXTensorElementDataType output_type_;
};

} // namespace hipep
