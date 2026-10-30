/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LoopBodyToOutParams.cpp - Out-param ABI for outlined loop bodies ---===//
//
// Module pass that runs AFTER `one-shot-bufferize` and BEFORE the pool/lowering
// passes that consume the out-param ABI.
//
// Problem. `onnx-loop-outline` emits each `hip.loop` body as a private
// `func.func` named `*_loop_body_*`. After bufferization those helpers still
// return loop-carried memrefs via `func.return`, while `convert-hip-to-llvm`
// LoopLowering expects the out-param ABI: one extra memref argument per
// loop-carried value (`v_in` + `v_out`, tagged `{bufferize.result}`).
//
// `@main_graph`'s own outputs are handled by `hip-use-output-allocator`
// (`hip.alloc_output` + the EP callback); that path never touches the private
// outlined loop bodies. This loop-body out-param ABI is INTERNAL to the DLL
// and unrelated to the graph-entry output allocator ABI.
//
// Fix. Invoke MLIR's `promoteBufferResultsToOutParams` with
// `modifyPublicFunctions = false` and a name filter that selects only
// `*_loop_body_*` functions.
//
// Before:
//   func.func private @main_loop_body_n0(..., %v_in: memref<...>)
//       -> memref<...> {
//     %out = memref.alloc() : memref<...>
//     hip.add ... outs(%out)
//     return %out : memref<...>
//   }
// After:
//   func.func private @main_loop_body_n0(..., %v_in: memref<...>,
//                                        %v_out: memref<...>
//                                        {bufferize.result}) {
//     hip.add ... outs(%v_out)
//     return
//   }
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "hip-loop-body-to-out-params"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_LOOPBODYTOOUTPARAMSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

static bool isOutlinedControlFlowBody(func::FuncOp *func) {
  if (!func)
    return false;
  StringRef name = func->getName();
  return name.contains("_loop_body_") || name.contains("_if_then_") ||
         name.contains("_if_else_") || name.contains("_scan_body_");
}

static bool isOutlinedScanBody(func::FuncOp *func) {
  return func && func->getName().contains("_scan_body_");
}

static bool isOutlinedNonScanBody(func::FuncOp *func) {
  return isOutlinedControlFlowBody(func) && !isOutlinedScanBody(func);
}

static LogicalResult promoteScanBody(func::FuncOp func) {
  auto numStateAttr =
      func->getAttrOfType<IntegerAttr>("hip.scan_num_state_variables");
  if (!numStateAttr)
    return func.emitError("missing hip.scan_num_state_variables attribute");
  int64_t numStateValue = numStateAttr.getInt();
  if (numStateValue < 0)
    return func.emitError("Scan state result count must be non-negative");
  unsigned numState = static_cast<unsigned>(numStateValue);
  FunctionType oldType = func.getFunctionType();
  if (numState > oldType.getNumResults())
    return func.emitError("Scan state result count exceeds body result count");

  SmallVector<Type> resultTypes(oldType.getResults().begin(),
                                oldType.getResults().end());
  SmallVector<DictionaryAttr> resultAttrs;
  for (unsigned i = 0; i < oldType.getNumResults(); ++i)
    resultAttrs.push_back(
        DictionaryAttr::get(func.getContext(), func.getResultAttrs(i)));
  if (llvm::any_of(resultTypes,
                   [](Type type) { return !isa<MemRefType>(type); }))
    return func.emitError("expected all Scan body results to be memrefs");

  unsigned oldNumInputs = oldType.getNumInputs();
  SmallVector<Type> newInputTypes(oldType.getInputs().begin(),
                                  oldType.getInputs().end());
  newInputTypes.append(resultTypes.begin(), resultTypes.begin() + numState);
  newInputTypes.append(resultTypes);
  func.setType(FunctionType::get(func.getContext(), newInputTypes, {}));

  SmallVector<BlockArgument> stateScratch;
  for (unsigned i = 0; i < numState; ++i)
    stateScratch.push_back(
        func.front().addArgument(resultTypes[i], func.getLoc()));

  SmallVector<BlockArgument> outParams;
  for (auto [i, type] : llvm::enumerate(resultTypes)) {
    unsigned argIndex = oldNumInputs + numState + i;
    func.setArgAttrs(argIndex, resultAttrs[i]);
    func.setArgAttr(argIndex,
                    StringAttr::get(func.getContext(), "bufferize.result"),
                    UnitAttr::get(func.getContext()));
    outParams.push_back(func.front().addArgument(type, func.getLoc()));
  }

  SmallVector<func::ReturnOp> returns;
  func.walk([&](func::ReturnOp op) { returns.push_back(op); });
  for (func::ReturnOp returnOp : returns) {
    if (returnOp.getNumOperands() != resultTypes.size())
      return returnOp.emitError("Scan body return/result count mismatch");
    OpBuilder builder(returnOp);
    for (auto [i, pair] :
         llvm::enumerate(llvm::zip(returnOp.getOperands(), outParams))) {
      Value result = std::get<0>(pair);
      Value outParam = std::get<1>(pair);
      Operation *definingOp = result.getDefiningOp();
      if (i < numState &&
          isa_and_nonnull<bufferization::AllocationOpInterface>(definingOp)) {
        result.replaceAllUsesWith(stateScratch[i]);
        definingOp->erase();
        result = stateScratch[i];
      }
      bool canHoistScanOutput =
          i >= numState &&
          isa_and_nonnull<bufferization::AllocationOpInterface>(definingOp);
      if (canHoistScanOutput) {
        result.replaceAllUsesWith(outParam);
        definingOp->erase();
      } else {
        memref::CopyOp::create(builder, returnOp.getLoc(), result, outParam);
      }
    }
    func::ReturnOp::create(builder, returnOp.getLoc());
    returnOp.erase();
  }
  func->removeAttr("hip.scan_num_state_variables");
  return success();
}

struct LoopBodyToOutParamsPass
    : public impl::LoopBodyToOutParamsPassBase<LoopBodyToOutParamsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    SmallVector<ScanOp> scans;
    getOperation().walk([&](ScanOp scan) { scans.push_back(scan); });
    for (ScanOp scan : scans) {
      unsigned numState = scan.getNumStateVariables();
      if (!scan.getStateScratch().empty()) {
        if (scan.getStateScratch().size() != numState) {
          scan.emitOpError("state scratch count does not match state count");
          return signalPassFailure();
        }
        continue;
      }

      OpBuilder builder(scan);
      SmallVector<Value> scratch;
      for (Value stateInit : scan.getInits().take_front(numState)) {
        auto type = dyn_cast<MemRefType>(stateInit.getType());
        if (!type) {
          scan.emitOpError("expected memref state init after bufferization");
          return signalPassFailure();
        }
        SmallVector<Value> dynamicSizes;
        for (int64_t dim = 0; dim < type.getRank(); ++dim)
          if (type.isDynamicDim(dim))
            dynamicSizes.push_back(
                memref::DimOp::create(builder, scan.getLoc(), stateInit, dim));
        scratch.push_back(memref::AllocOp::create(builder, scan.getLoc(), type,
                                                  dynamicSizes));
      }
      scan.getStateScratchMutable().append(scratch);
    }

    bufferization::BufferResultsToOutParamsOpts opts;
    opts.hoistStaticAllocs = true;
    opts.hoistDynamicAllocs = true;
    opts.addResultAttribute = true;
    opts.modifyPublicFunctions = false;
    opts.filterFn = isOutlinedNonScanBody;

    if (failed(bufferization::promoteBufferResultsToOutParams(getOperation(),
                                                              opts)))
      return signalPassFailure();

    // Scan state inputs and outputs alias in the runtime ABI. Promote Scan
    // outputs normally, but copy state results only at the terminator after
    // every body read has been enqueued.
    SmallVector<func::FuncOp> scanBodies;
    getOperation().walk([&](func::FuncOp func) {
      if (isOutlinedScanBody(&func))
        scanBodies.push_back(func);
    });
    for (func::FuncOp func : scanBodies)
      if (failed(promoteScanBody(func)))
        return signalPassFailure();
  }
};

} // namespace
} // namespace hip
} // namespace mlir
