/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ResolveExternConstants.cpp - Replace extern globals with hip.get_constant
//
// The --hip-resolve-extern-constants pass bridges the gap between compile-time
// constant externalization (which produces memref.global ops with
// hip.external_data attributes and no initial value) and LLVM lowering (which
// needs a runtime mechanism to provide constant data).
//
// Strategy:
//
// For each function that uses externalized constants via memref.get_global:
//   1. Get !hip.context from arg 0 (inserted by hip-add-context-arg).
//   2. Read the constant index from hip.external_data on the memref.global.
//   3. Replace memref.get_global with hip.get_constant(%ctx, %index).
//   4. Erase the now-unused memref.global ops.
//   5. Keep hip.constants_file for downstream metadata generation.
//
// The runtime is responsible for:
//   - Loading constants.bin via the FileSystem abstraction
//   - Uploading constants to GPU memory during initialization
//   - Returning GPU pointers via hipdnn_ep_constant_get(state, index)
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-resolve-extern-constants"

STATISTIC(NumGlobalsResolved, "Number of extern globals resolved to views");
STATISTIC(NumFuncsUpdated, "Number of functions receiving constants argument");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_RESOLVEEXTERNCONSTANTSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct ResolveExternConstantsPass
    : public impl::ResolveExternConstantsPassBase<ResolveExternConstantsPass> {
  void runOnOperation() override;
};

void ResolveExternConstantsPass::runOnOperation() {
  ModuleOp module = getOperation();

  // Collect extern globals that have hip.external_data.
  struct ExternGlobalInfo {
    memref::GlobalOp globalOp;
    int64_t index;
    int64_t offset;
    int64_t size;
  };
  SmallVector<ExternGlobalInfo> externGlobals;

  for (auto globalOp : module.getOps<memref::GlobalOp>()) {
    auto extDataAttr =
        globalOp->getAttrOfType<DictionaryAttr>("hip.external_data");
    if (!extDataAttr)
      continue;
    auto indexAttr = extDataAttr.getAs<IntegerAttr>("index");
    auto offsetAttr = extDataAttr.getAs<IntegerAttr>("offset");
    auto sizeAttr = extDataAttr.getAs<IntegerAttr>("size");
    if (!indexAttr || !offsetAttr || !sizeAttr) {
      globalOp.emitError("hip.external_data missing index, offset, or size");
      return signalPassFailure();
    }
    externGlobals.push_back(
        {globalOp, indexAttr.getInt(), offsetAttr.getInt(), sizeAttr.getInt()});
  }

  if (externGlobals.empty())
    return;

  // Index-based lookup avoids storing pointers into the SmallVector.
  DenseMap<StringRef, size_t> globalsByName;
  for (auto [idx, info] : llvm::enumerate(externGlobals))
    globalsByName[info.globalOp.getSymName()] = idx;

  // Replace memref.get_global with hip.get_constant in every function.
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    if (funcOp.isDeclaration())
      continue;

    // Get !hip.context from arg 0.
    auto &entry = funcOp.getBody().front();
    if (entry.getNumArguments() == 0)
      continue;
    Value ctxArg = entry.getArgument(0);
    if (!isa<hip::ContextType>(ctxArg.getType()))
      continue;

    SmallVector<memref::GetGlobalOp> getGlobalOps;
    funcOp.walk([&](memref::GetGlobalOp op) {
      if (globalsByName.contains(op.getName()))
        getGlobalOps.push_back(op);
    });

    if (!getGlobalOps.empty()) {
      ++NumFuncsUpdated;
      LLVM_DEBUG(llvm::dbgs()
                 << "  Resolving constants in @" << funcOp.getName() << "\n");
    }

    for (auto getGlobalOp : getGlobalOps) {
      ExternGlobalInfo &info =
          externGlobals[globalsByName[getGlobalOp.getName()]];
      OpBuilder builder(getGlobalOp);
      Location loc = getGlobalOp.getLoc();

      // Create index constant.
      Value indexVal =
          arith::ConstantOp::create(builder, loc, builder.getI64Type(),
                                    builder.getI64IntegerAttr(info.index));

      // Produce hip.get_constant with the original memref type so that all
      // downstream users see the same type they expected from the
      // memref.get_global.  The HipToLLVM lowering will emit the correct
      // GPU-pointer logic regardless of the declared address space.
      auto origMemRefType = getGlobalOp.getType();
      auto getConstOp = hip::GetConstantOp::create(builder, loc, origMemRefType,
                                                   ctxArg, indexVal);

      getGlobalOp.replaceAllUsesWith(getConstOp.getResult());
      getGlobalOp.erase();
      ++NumGlobalsResolved;
    }
  }

  // Clean up -- erase extern globals and strip module attribute.
  for (auto &info : externGlobals)
    info.globalOp.erase();

  // Keep hip.constants_file so GenerateInterface can embed the correct
  // constants-file name in the model metadata.
}

} // namespace
} // namespace hip
} // namespace mlir
