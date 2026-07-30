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

#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
         name.contains("_if_else_");
}

struct LoopBodyToOutParamsPass
    : public impl::LoopBodyToOutParamsPassBase<LoopBodyToOutParamsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    bufferization::BufferResultsToOutParamsOpts opts;
    opts.hoistStaticAllocs = true;
    opts.hoistDynamicAllocs = true;
    opts.addResultAttribute = true;
    opts.modifyPublicFunctions = false;
    opts.filterFn = isOutlinedControlFlowBody;

    if (failed(bufferization::promoteBufferResultsToOutParams(getOperation(),
                                                              opts)))
      signalPassFailure();
  }
};

} // namespace
} // namespace hip
} // namespace mlir
