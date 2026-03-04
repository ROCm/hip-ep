/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ResolveExternConstants.cpp - Wire extern globals to buffer arg -----===//
//
// The --hip-resolve-extern-constants pass bridges the gap between compile-time
// constant externalization (which produces memref.global ops with
// hip.external_data attributes and no initial value) and LLVM lowering (which
// needs concrete data or a runtime mechanism to provide it).
//
// Strategy (mirrors the pool allocator pattern from --hip-pool-allocs):
//
// For each function that uses externalized constants via memref.get_global:
//   1. Add a new argument: %_constants : memref<?xi8>
//   2. Replace each memref.get_global @hip_ext_xxx with a memref.view into
//      %_constants at the byte offset recorded in hip.external_data.
//   3. Erase the now-unused memref.global ops.
//   4. Strip hip.constants_file from the module.
//
// The host (ORT EP or test harness) is responsible for:
//   - Loading the sidecar .constants.bin via hip_load_constants()
//   - Passing the resulting device pointer as the %_constants argument
//
// This keeps the MLIR pass simple (no LLVM dialect mixing) and lets
// --convert-memref-to-llvm handle everything naturally.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

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
    int64_t offset;
    int64_t size;
  };
  llvm::SmallVector<ExternGlobalInfo> externGlobals;
  llvm::DenseMap<StringRef, ExternGlobalInfo *> globalsByName;

  for (auto globalOp : module.getOps<memref::GlobalOp>()) {
    auto extDataAttr =
        globalOp->getAttrOfType<DictionaryAttr>("hip.external_data");
    if (!extDataAttr)
      continue;
    auto offsetAttr = extDataAttr.getAs<IntegerAttr>("offset");
    auto sizeAttr = extDataAttr.getAs<IntegerAttr>("size");
    if (!offsetAttr || !sizeAttr) {
      globalOp.emitError("hip.external_data missing offset or size");
      return signalPassFailure();
    }
    externGlobals.push_back(
        {globalOp, offsetAttr.getInt(), sizeAttr.getInt()});
  }

  if (externGlobals.empty())
    return;

  for (auto &info : externGlobals)
    globalsByName[info.globalOp.getSymName()] = &info;

  // Process each function that references extern globals.
  for (auto funcOp : module.getOps<func::FuncOp>()) {
    if (funcOp.isDeclaration())
      continue;

    // Find all memref.get_global ops that reference extern globals.
    llvm::SmallVector<memref::GetGlobalOp> getGlobalOps;
    funcOp.walk([&](memref::GetGlobalOp op) {
      if (globalsByName.count(op.getName()))
        getGlobalOps.push_back(op);
    });

    if (getGlobalOps.empty())
      continue;

    // Add %_constants : memref<?xi8> argument to the function.
    Block &entryBlock = funcOp.getBody().front();
    auto i8Type = IntegerType::get(module.getContext(), 8);
    auto constantsType = MemRefType::get({ShapedType::kDynamic}, i8Type);
    entryBlock.addArgument(constantsType, funcOp.getLoc());
    Value constantsArg = entryBlock.getArguments().back();

    // Update the function type to include the new argument.
    auto oldFuncType = funcOp.getFunctionType();
    llvm::SmallVector<Type> newInputTypes(oldFuncType.getInputs());
    newInputTypes.push_back(constantsType);
    funcOp.setFunctionType(FunctionType::get(module.getContext(),
                                             newInputTypes,
                                             oldFuncType.getResults()));

    // Extend argument attributes array to include the new argument.
    if (auto allArgAttrs = funcOp.getAllArgAttrs()) {
      llvm::SmallVector<Attribute> newArgAttrs(allArgAttrs.begin(),
                                               allArgAttrs.end());
      newArgAttrs.push_back(DictionaryAttr::get(module.getContext()));
      funcOp.setAllArgAttrs(newArgAttrs);
    }

    // Replace each memref.get_global with memref.view into the buffer.
    for (auto getGlobalOp : getGlobalOps) {
      ExternGlobalInfo *info = globalsByName[getGlobalOp.getName()];
      OpBuilder builder(getGlobalOp);
      Location loc = getGlobalOp.getLoc();

      Value offset = arith::ConstantOp::create(
          builder, loc, builder.getIndexAttr(info->offset));
      auto viewOp = memref::ViewOp::create(
          builder, loc, getGlobalOp.getType(), constantsArg, offset,
          /*sizes=*/ValueRange{});

      getGlobalOp.replaceAllUsesWith(viewOp.getResult());
      getGlobalOp.erase();
    }
  }

  // Erase the extern memref.global ops (now unused).
  for (auto &info : externGlobals)
    info.globalOp.erase();

  // Strip module-level hip.constants_file attribute.
  module->removeAttr("hip.constants_file");
}

} // namespace
} // namespace hip
} // namespace mlir
