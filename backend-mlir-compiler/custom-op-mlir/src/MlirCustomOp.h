/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MLIR_CUSTOM_OP_H
#define MLIR_CUSTOM_OP_H

#include "InferenceState.h"
#include "morphizen/morphizen.hpp"
#include <memory>
#include <optional>

namespace onnxruntime {
class Model;
}

namespace mlir_compilation {

// Custom Op implementation for MLIR-compiled models
// Loads artifacts from EPContext and executes inference
class MlirCustomOp : public morphizen::CustomOpImp {
public:
  MlirCustomOp(std::shared_ptr<const morphizen::PassContext> context,
               const std::shared_ptr<morphizen::MetaDefProto> &meta_def,
               onnxruntime::Model *model);

  ~MlirCustomOp() override = default;

  // Execute inference using loaded artifact
  void Compute(const OrtApi *api, OrtKernelContext *context) const override;

private:
  // Inference state owns the artifact (clear dependency)
  std::optional<customop::InferenceState> inference_state_;

  // Output shapes from metadata (populated in constructor)
  std::vector<std::vector<int64_t>> output_shapes_;
};

} // namespace mlir_compilation

#endif
