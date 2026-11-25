/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- StateDonation.h - States updated in place ----------------*- C++ -*-===//
//
// A state (a KV cache) the step reads and writes back may be handed to the
// code as both: the code then writes only the rows it changes, into the
// caller's memory. That is safe when nothing can observe the old bytes
// afterwards, which ProveStateDonation.cpp proves from the graph.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_GGML_TRANSFORMS_STATEDONATION_H
#define HIP_DIALECT_GGML_TRANSFORMS_STATEDONATION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include <string>

namespace mlir::ggml {

/// Whether the step may write a state's result into its input's allocation:
/// when nothing can observe the old bytes afterwards.
struct StateDonation {
  bool inPlace = false;
  /// Why not, when not.
  std::string reason;
};

/// For each of `@main_graph`'s `ggml.state_pairs`, in order, whether it may
/// be updated in place. Fails on malformed state pairs or a graph the proof
/// cannot read.
FailureOr<SmallVector<StateDonation>> proveStateDonation(ModuleOp module);

} // namespace mlir::ggml

#endif // HIP_DIALECT_GGML_TRANSFORMS_STATEDONATION_H
