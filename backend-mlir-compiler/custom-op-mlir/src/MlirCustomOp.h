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

  // Defined out-of-line: frees the allocator-mode host-output GPU scratch and
  // releases the lazy-allocated HIPDNN_EP_PERF event pair below
  // (hipEventDestroy lives in the hip header, which we keep out of this
  // header).
  ~MlirCustomOp() override;

  // Execute inference using loaded artifact
  void Compute(const OrtApi *api, OrtKernelContext *context) const override;

private:
  // Output-allocator dispatch (2-arg inference_compute). Installs a per-Compute
  // callback that allocates each graph output in-graph, then performs any
  // host-output device->host copies.
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

  // Host-output GPU scratch: one device buffer per output index,
  // grow-on-demand, reused across Compute() and freed in the dtor. Only
  // populated when an output lands in host (CPU) memory -- the DLL writes GPU
  // results here and Compute() D2H-copies into the ORT host buffer. mutable:
  // Compute() is const but this is a runtime cache, not observable state. Left
  // empty (no allocation) in mock builds.
  mutable std::vector<HostOutputScratch> host_out_scratch_;

  // HIPDNN_EP_PERF wall-clock event pair. Created lazily on the first
  // Compute() that observes perf_enabled(), then reused for the lifetime of
  // this MlirCustomOp instance and destroyed in the destructor. Stored as
  // void* to keep hip headers out of this header; the actual hipEvent_t type
  // is just a pointer typedef on amdhip64, so the round-trip is safe.
  // Mutable because Compute() is const but lazy-init writes these once.
  mutable void *ep_perf_ev_start_ = nullptr;
  mutable void *ep_perf_ev_stop_ = nullptr;
};

} // namespace mlir_compilation

#endif
