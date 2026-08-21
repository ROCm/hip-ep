/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#pragma once

#define ORT_API_MANUAL_INIT
#include <onnxruntime_cxx_api.h>
#undef ORT_API_MANUAL_INIT

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

// New ops inherit this (CRTP: pass their own type as Derived) and implement
// only GetName/GetInputTypeCount/GetInputType/GetOutputTypeCount/
// GetOutputType (optionally InferOutputShape / GetInputCharacteristic).
// CreateKernel/Compute are fixed here so op authors never touch kernel code.
template <typename Derived>
struct SchemaOnlyCustomOpBase
    : Ort::CustomOpBase<Derived, SchemaOnlyStubKernel> {
  void *CreateKernel(const OrtApi &, const OrtKernelInfo *info) const {
    return new SchemaOnlyStubKernel(info);
  }
};

} // namespace hipep
