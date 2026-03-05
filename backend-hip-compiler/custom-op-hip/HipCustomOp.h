/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CUSTOM_OP_H
#define HIP_CUSTOM_OP_H

#include "HipInferenceState.h"
#include "morphizen/morphizen.hpp"
#include <memory>
#include <string>
#include <vector>

namespace hip_compilation {

/// ORT CustomOp that loads and runs a compiled HIP model DLL.
/// Mirrors MlirCustomOp in backend-mlir-compiler but uses domain "HIP".
class HipCustomOp : public morphizen::CustomOpImp {
public:
  HipCustomOp(std::shared_ptr<const morphizen::PassContext> context,
              const std::shared_ptr<morphizen::MetaDefProto> &meta_def,
              onnxruntime::Model *model);

  void Compute(const OrtApi *api, OrtKernelContext *context) const override;

  // Output tensor descriptors (public for marshaling helpers)
  struct OutputMeta {
    std::string name;
    int rank;
    int elem_type;
    std::vector<int64_t> shape;
  };
private:
  std::vector<OutputMeta> output_metas_;
  std::unique_ptr<customop::HipInferenceState> inference_state_;
};

} // namespace hip_compilation

#endif
