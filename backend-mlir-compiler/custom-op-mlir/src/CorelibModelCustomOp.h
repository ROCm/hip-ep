/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- CorelibModelCustomOp.h - A whole model run by RyzenAI CoreLib ------===//
//
// The custom op for the CORELIBMODEL domain: a whole onnxruntime-genai decoder
// the level-1 pass handed to a CoreLib model (corelib_model.h) instead of
// compiling. Each Compute passes the graph inputs the model's forward takes
// (input_ids, inputs_embeds, ...: CoreLib's names, which are
// onnxruntime-genai's) at the position attention_mask implies, and writes the
// last row's logits. CoreLib keeps the KV cache, so past_key_values are not
// read and present.* not written. See docs/design/ryzenai-corelib-model.md.
//
//===----------------------------------------------------------------------===//

#ifndef CORELIB_MODEL_CUSTOM_OP_H
#define CORELIB_MODEL_CUSTOM_OP_H

#include "corelib_model.h"
#include "morphizen/morphizen.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace mlir_compilation {

class CorelibModelCustomOp : public morphizen::CustomOpImp {
public:
  CorelibModelCustomOp(std::shared_ptr<const morphizen::PassContext> context,
                       const std::shared_ptr<morphizen::MetaDefProto> &meta_def,
                       onnxruntime::Model *model);

  void Compute(const OrtApi *api, OrtKernelContext *context) const override;

private:
  std::shared_ptr<hipdnn_ep::corelib::ModelSession> session_;
  int mask_ = -1, logits_ = -1; // meta_def indices
  // The inputs forward takes: meta_def index and name.
  std::vector<std::pair<int, std::string>> inputs_;
  // present.* output index -> the past_key_values input whose shape it takes.
  std::vector<std::pair<int, int>> presents_;
  mutable std::vector<std::vector<uint8_t>> host_inputs_;
  mutable std::vector<float> logits_buf_;
  mutable std::vector<Ort::Float16_t> logits_fp16_;
  mutable bool warned_mask_ = false;
};

} // namespace mlir_compilation

#endif // CORELIB_MODEL_CUSTOM_OP_H
