/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MLIR_CUSTOM_OP_H
#define MLIR_CUSTOM_OP_H

#include "InferenceState.h"
#include "metadata.pb.h"
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
  // Inference state owns the artifact (clear ownership via unique_ptr)
  std::unique_ptr<customop::InferenceState> inference_state_;

  // Metadata from EPContext (contains output shapes)
  mlir_metadata::Metadata metadata_;
};

} // namespace mlir_compilation

#endif
