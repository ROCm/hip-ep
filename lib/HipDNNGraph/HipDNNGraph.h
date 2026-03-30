/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/Region.h"

#include <memory>
#include <string>
#include <unordered_map>

struct hipdnnHandle;
typedef hipdnnHandle *hipdnnHandle_t;

namespace hip::graph {

enum class StatusCode { kSuccess, kFailure };

class Status {
public:
  static Status Success() { return Status(); }

  static Status Failure(std::string message) {
    return Status(StatusCode::kFailure, std::move(message));
  }

  bool ok() const { return code_ == StatusCode::kSuccess; }
  bool failed() const { return code_ == StatusCode::kFailure; }
  const std::string &message() const { return message_; }

private:
  Status() : code_(StatusCode::kSuccess) {}
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}
  StatusCode code_;
  std::string message_;
};

struct HipDNNGraphImpl;

/// Encapsulates a hipDNN execution graph.
///
/// Ported from hipDNNEP's HipDNNGraph, adapted for onnx-hipdnn-ep:
///   - BuildFromOnnxMLIR() works with standard MLIR RankedTensorType
///   - Execute() takes raw GPU pointers + UIDs instead of OrtKernelContext
///   - UID accessors expose compile-time tensor IDs for IR embedding
class HipDNNGraph {
public:
  explicit HipDNNGraph(hipdnnHandle_t handle);
  ~HipDNNGraph();

  HipDNNGraph(const HipDNNGraph &) = delete;
  HipDNNGraph &operator=(const HipDNNGraph &) = delete;

  /// Build graph from an MLIR region containing onnx.* ops with standard
  /// RankedTensorType. Call Compile() after this succeeds.
  Status BuildFromOnnxMLIR(mlir::Region &region);

  /// Compile the built graph: validate, build operation graph, create
  /// execution plans, and determine workspace size.
  Status Compile();

  /// Execute with pre-built variant pack mapping UIDs to GPU pointers.
  Status Execute(hipdnnHandle_t handle,
                 std::unordered_map<int64_t, void *> &variant_pack,
                 void *workspace);

  /// UIDs assigned during Build(), baked into IR as attributes.
  llvm::ArrayRef<int64_t> getInputUids() const;
  llvm::ArrayRef<int64_t> getOutputUids() const;
  int64_t getWorkspaceSize() const;

private:
  std::unique_ptr<HipDNNGraphImpl> impl_;
};

} // namespace hip::graph
