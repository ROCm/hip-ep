/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- RuntimeCallUtils.h - Runtime calls whose status leaves the graph ---===//
//
// A runtime wrapper returns an i32 status the compiled graph ignores by
// default. A frontend that needs a failure to reach its caller - so it can run
// the graph another way rather than use a wrong result - asks for it on the
// module (requireRuntimeStatus); a lowering then marks the calls whose status
// matters (markFallibleRuntimeCall), and convert-hip-to-llvm makes
// @main_graph return the first nonzero status.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_HIPTOLLVM_RUNTIMECALLUTILS_H
#define HIP_CONVERSION_HIPTOLLVM_RUNTIMECALLUTILS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hip {

/// On a module: its marked runtime calls' statuses leave @main_graph.
inline constexpr llvm::StringLiteral kPropagateRuntimeStatusAttrName =
    "hip.runtime.propagate_status";

/// On an LLVM call: a runtime call whose i32 status leaves @main_graph.
inline constexpr llvm::StringLiteral kFallibleRuntimeCallAttrName =
    "hip.runtime.fallible";

inline void requireRuntimeStatus(ModuleOp module) {
  module->setAttr(kPropagateRuntimeStatusAttrName,
                  UnitAttr::get(module.getContext()));
}

inline bool requiresRuntimeStatus(ModuleOp module) {
  return module && module->hasAttr(kPropagateRuntimeStatusAttrName);
}

/// Marks `call` fallible when its module asks for runtime statuses.
inline void markFallibleRuntimeCall(LLVM::CallOp call) {
  if (requiresRuntimeStatus(call->getParentOfType<ModuleOp>()))
    call->setAttr(kFallibleRuntimeCallAttrName,
                  UnitAttr::get(call.getContext()));
}

inline bool isFallibleRuntimeCall(LLVM::CallOp call) {
  return call->hasAttr(kFallibleRuntimeCallAttrName);
}

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_HIPTOLLVM_RUNTIMECALLUTILS_H
