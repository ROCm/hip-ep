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
#include <vector>

namespace onnxruntime {
class Model;
}

namespace mlir_compilation {

// One grow-on-demand GPU scratch buffer for a host (CPU) graph output in
// output-allocator mode. Keeping the pointer and its capacity together (vs two
// parallel vectors) means the two can never fall out of sync on resize.
struct HostOutputScratch {
  void *ptr = nullptr; // device buffer the DLL writes into (nullptr = unset)
  size_t capacity = 0; // bytes currently allocated at ptr
};

// Custom Op implementation for MLIR-compiled models
// Loads artifacts from EPContext and executes inference
class MlirCustomOp : public morphizen::CustomOpImp {
public:
  MlirCustomOp(std::shared_ptr<const morphizen::PassContext> context,
               const std::shared_ptr<morphizen::MetaDefProto> &meta_def,
               onnxruntime::Model *model);

  // Defined out-of-line: frees the allocator-mode host-output GPU scratch.
  ~MlirCustomOp() override;

  // Execute inference using loaded artifact
  void Compute(const OrtApi *api, OrtKernelContext *context) const override;

private:
  // Output-allocator dispatch (2-arg inference_compute). Installs a per-Compute
  // callback that allocates each graph output in-graph, then performs any
  // host-output device->host copies. Selected in Compute() when
  // use_output_allocator_ is true.
  void compute_with_output_allocator(OrtKernelContext *context) const;

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

  // For each metadata output index, the compiler-input index of the matching
  // past_key_values input (or -1 if this output is not a `present.*` tensor).
  // Precomputed so the per-inference shape-override loop is O(1) per output
  // instead of an O(N×M) name-string scan.
  // TODO: replace the underlying name-based heuristic in
  // build_present_to_past_input_idx() by emitting explicit past↔present pairs
  // from the compiler (Level-1 pass walks GqaOp operands which carry the
  // pairing directly). Today's helper relies on the convention that present
  // outputs are named "present.N.{key,value}" and past inputs are named
  // "past_key_values.N.{key,value}"; an upstream rename would silently break
  // the share-buffer override (KV cache corruption with no crash).
  std::vector<int> present_to_past_input_idx_;

  // True when the compiled DLL was built in output-allocator mode (2-arg
  // inference_compute; graph outputs allocated via the EP callback). Read from
  // the embedded metadata in the ctor so it always matches the loaded DLL's
  // ABI -- even when the artifact came from a reused EPContext. Selects the
  // dispatch path in Compute().
  bool use_output_allocator_ = false;

  // Allocator-mode host-output GPU scratch: one device buffer per output index,
  // grow-on-demand, reused across Compute() and freed in the dtor. Only
  // populated when use_output_allocator_ is true and an output lands in host
  // (CPU) memory -- the DLL writes GPU results here and Compute() D2H-copies
  // into the ORT host buffer. mutable: Compute() is const but this is a runtime
  // cache, not observable state. Left empty (no allocation) in classic mode and
  // in mock builds.
  mutable std::vector<HostOutputScratch> host_out_scratch_;
};

} // namespace mlir_compilation

#endif
