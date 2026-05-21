/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef MLIR_CUSTOM_OP_H
#define MLIR_CUSTOM_OP_H

#include "InferenceState.h"
#include "metadata.pb.h"
#include "morphizen/morphizen.hpp"
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace onnxruntime {
class Model;
}

namespace mlir_compilation {

// Per-output dynamic-shape metadata cached in MlirCustomOp ctor. Populated
// from the FB-JSON metadata blob the EP parses out of the model.dll.
// `has_runtime_slot` is true iff at least one dim of this output is a
// RuntimeSlot leaf (Category C). For now we only support one slot per
// output (NonZero / Range / ConstantOfShape pattern); a future op that
// publishes two independent dims would need a multi-slot resolver path.
// Defined here (not in the .cpp) so MlirCustomOp's std::vector member can
// see the complete type at destructor instantiation time.
struct OutputDynamicInfo {
  bool has_runtime_slot = false;
  int32_t slot_id = -1;
  int32_t slot_dim_index = -1;
};

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

  // Dynamic-output-shape compute path. Used when at least one output of
  // this fused graph carries a RuntimeSlot dim (Category C). The two-phase
  // logic is implemented out-of-line for readability: (1) marshal inputs,
  // (2) try to pre-resolve every output shape from input shapes /
  // host-readable input values, (3) marshal resolved outputs normally and
  // un-resolved outputs as Category-C sentinels (data==null), (4) run
  // inference_compute, (5) stream-sync, (6) read each deferred slot, ctx
  // .GetOutput with the resolved shape, and D2H from the slot GPU buffer
  // into the freshly allocated OrtValue.
  void computeDynamic(OrtKernelContext *context) const;

private:
  // Inference state owns the artifact (clear ownership via unique_ptr)
  std::unique_ptr<customop::InferenceState> inference_state_;

  // Metadata from EPContext (contains output shapes)
  mlir_metadata::Metadata metadata_;

  // Maps compiler input index (= DLL input index) to ORT kernel context
  // input index. ORT's fused node may reorder inputs relative to the
  // compiler order; the mapping is derived from input_argument_indice.
  std::vector<int> input_index_map_;

  // Maps metadata output index (= DLL output index) to ORT kernel context
  // output index. Precomputed at construction to handle ordering differences
  // between the metadata (DLL-order) and the fused node (ORT-order).
  std::vector<int> output_index_map_;

  // Per-output dynamic-shape info derived from the FB metadata blob during
  // ctor. One entry per output; entries with `has_runtime_slot==false` are
  // pure static / Category A/B/D outputs and follow the legacy fast path.
  // Defined out-of-line so this header doesn't need to pull
  // DimSpecResolver.h.
  std::vector<OutputDynamicInfo> output_dyn_info_;
};

} // namespace mlir_compilation

#endif
