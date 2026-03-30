/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 *
 * Stub implementation of HipDNNGraph.
 *
 * The full implementation requires the hipDNN frontend SDK with a matching
 * FlatBuffers version. Until the SDK versions are aligned, all operations
 * return failure so the MLIR pipeline falls through to the standard
 * ConvertOnnxToHip path.
 */

#include "HipDNNGraph.h"

#include "mlir/IR/Region.h"

namespace hip::graph {

struct HipDNNGraphImpl {
  std::vector<int64_t> input_uids_;
  std::vector<int64_t> output_uids_;
  int64_t workspace_size_ = 0;
};

HipDNNGraph::HipDNNGraph(hipdnnHandle_t /*handle*/)
    : impl_(std::make_unique<HipDNNGraphImpl>()) {}

HipDNNGraph::~HipDNNGraph() = default;

Status HipDNNGraph::BuildFromOnnxMLIR(mlir::Region & /*region*/) {
  return Status::Failure(
      "hipDNN graph compilation unavailable (SDK version mismatch)");
}

Status HipDNNGraph::Compile() {
  return Status::Failure(
      "hipDNN graph compilation unavailable (SDK version mismatch)");
}

Status HipDNNGraph::Execute(hipdnnHandle_t /*handle*/,
                            std::unordered_map<int64_t, void *> & /*pack*/,
                            void * /*workspace*/) {
  return Status::Failure(
      "hipDNN graph execution unavailable (SDK version mismatch)");
}

llvm::ArrayRef<int64_t> HipDNNGraph::getInputUids() const {
  return impl_->input_uids_;
}

llvm::ArrayRef<int64_t> HipDNNGraph::getOutputUids() const {
  return impl_->output_uids_;
}

int64_t HipDNNGraph::getWorkspaceSize() const {
  return impl_->workspace_size_;
}

} // namespace hip::graph
