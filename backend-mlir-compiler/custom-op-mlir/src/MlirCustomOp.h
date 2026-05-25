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
//
// Two orthogonal flags describe the output's resolution requirements:
//
//   * `needs_pre_compute_resolve` -- the static metadata `shape` array
//     contains at least one INT64_MIN sentinel (= dynamic dim) AND the
//     output has a non-trivial DimSpec tree. Category B / Category D
//     outputs walk the spec tree pre-compute (with the EP's input
//     shapes + bytes) to resolve every dim to a concrete value, then
//     allocate the ORT OrtValue at the right size BEFORE running
//     main_graph. Even purely Category B outputs need this so the
//     runtime fill kernel writes into the correct buffer.
//
//   * `has_runtime_slot` -- at least one dim is a RuntimeSlot leaf
//     (Category C). The EP allocates a placeholder, defers the OrtValue
//     allocation until POST-compute when the wrap_* function has
//     published the actual dim into a slot, then D2H-copies the
//     wrapper-allocated GPU buffer into the freshly-sized OrtValue.
//
// Per-output dynamic-shape info. Supports any number of RuntimeSlot dims
// per output -- ConstantOfShape (Category C) publishes one slot per output
// axis, while NonZero / Range / etc. publish a single slot for their one
// dynamic dim. The buffer for Category C outputs is, by convention,
// published to `buffer_slot_id` (== the first RuntimeSlot we encountered
// while walking dim_specs; today this matches `slot_ids[0]` set by the
// wrap_* lowerings, but the resolver doesn't bake in that assumption).
// Defined here (not in the .cpp) so MlirCustomOp's std::vector member can
// see the complete type at destructor instantiation time.
struct OutputDynamicInfo {
  bool needs_pre_compute_resolve = false;
  bool has_runtime_slot = false;
  // For each output dim, the slot id that publishes it (or -1 if that dim
  // is not RuntimeSlot-driven). One entry per rank.
  std::vector<int32_t> dim_slot_ids;
  // Slot id from which the wrap_* publishes the output GPU buffer pointer.
  // -1 when the output has no RuntimeSlot dims.
  int32_t buffer_slot_id = -1;
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
