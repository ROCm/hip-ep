/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_HIPTOLLVM_RUNTIMECALLUTILS_H
#define HIP_CONVERSION_HIPTOLLVM_RUNTIMECALLUTILS_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hip {

/// Marks an LLVM runtime call whose single i32 result must be propagated out
/// of the compiled graph. Out-of-tree conversion patterns use this marker to
/// opt in without teaching the core pass about their runtime symbol names.
inline constexpr llvm::StringLiteral kFallibleRuntimeCallAttrName =
    "hip.runtime.fallible";

/// Opts a module into fail-closed status handling for device-side memref.copy
/// and other runtime calls explicitly marked by their lowering. This is useful
/// for source pipelines that may introduce copies inside helpers: until helper
/// functions have a status ABI, such copies are rejected instead of silently
/// discarding runtime failures. The legacy attribute name is retained for
/// compatibility.
inline constexpr llvm::StringLiteral kRequireFallibleMemRefCopiesAttrName =
    "hip.runtime.require_fallible_memref_copies";

inline void requireFallibleMemRefCopies(ModuleOp module) {
  module->setAttr(kRequireFallibleMemRefCopiesAttrName,
                  UnitAttr::get(module.getContext()));
}

inline bool requiresFallibleMemRefCopies(ModuleOp module) {
  return module && static_cast<bool>(module->getAttrOfType<UnitAttr>(
                       kRequireFallibleMemRefCopiesAttrName));
}

inline LLVM::CallOp markFallibleRuntimeCall(LLVM::CallOp call) {
  call->setAttr(kFallibleRuntimeCallAttrName, UnitAttr::get(call.getContext()));
  return call;
}

inline bool isFallibleRuntimeCall(LLVM::CallOp call) {
  return static_cast<bool>(
      call->getAttrOfType<UnitAttr>(kFallibleRuntimeCallAttrName));
}

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_HIPTOLLVM_RUNTIMECALLUTILS_H
