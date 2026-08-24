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
// there is nothing to distinguish one op from another beyond its name and
// domain. UNDEFINED element type + VARIADIC input/output characteristic
// tell ORT to skip type and count validation entirely, which is why a
// single concrete class parameterized only by name (rather than one CRTP
// subclass per op) is enough to cover every op in this bucket.
class SchemaOnlyCustomOp
    : public Ort::CustomOpBase<SchemaOnlyCustomOp, SchemaOnlyStubKernel> {
public:
  explicit SchemaOnlyCustomOp(std::string name) : name_(std::move(name)) {}

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
