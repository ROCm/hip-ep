/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LoopBodyToOutParams.cpp - Control-flow body result ABI ------------===//
//
// Module pass that runs AFTER `one-shot-bufferize` and BEFORE pool/lowering.
//
// Shape-changing loop carriers must return their final memref descriptors.
// Promoting them to out-params aliases v_in/v_out onto the immutable v_init
// descriptor and freezes dynamic extents. Loop bodies therefore retain ranked
// memref results. Returned allocations are redirected to hip.loop_alloc, whose
// per-invocation frame owns two high-water banks per carrier.
//
// `@main_graph`'s own outputs are handled by `hip-use-output-allocator`
// (`hip.alloc_output` + the EP callback); that path never touches the private
// outlined loop bodies. Loop descriptor returns are internal to the DLL and
// unrelated to the graph-entry output allocator ABI.
//
// If branch helpers keep using generic out-param promotion; their result shapes
// are not loop-carried and are outside this ABI.
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
//                                        %frame: !hip.loop_frame)
//       -> memref<...> {
//     %v_out = hip.loop_alloc(%frame) ... : memref<...>
//     hip.add ... outs(%v_out)
//     return %v_out : memref<...>
//   }
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "hip-loop-body-to-out-params"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_LOOPBODYTOOUTPARAMSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

static bool isOutlinedIfBody(func::FuncOp *func) {
  if (!func)
    return false;
  StringRef name = func->getName();
  return name.contains("_if_then_") || name.contains("_if_else_");
}

struct LoopBodyToOutParamsPass
    : public impl::LoopBodyToOutParamsPassBase<LoopBodyToOutParamsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<HipDialect, arith::ArithDialect, func::FuncDialect,
                    memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Conditional branches retain the established out-param ABI.
    bufferization::BufferResultsToOutParamsOpts opts;
    opts.hoistStaticAllocs = true;
    opts.hoistDynamicAllocs = true;
    opts.addResultAttribute = true;
    opts.modifyPublicFunctions = false;
    opts.filterFn = isOutlinedIfBody;

    if (failed(bufferization::promoteBufferResultsToOutParams(module, opts)))
      return signalPassFailure();

    // One-Shot has no tensor work to trigger on a zero-carrier loop.
    // Materialize its ownership token explicitly so lowering still uses the
    // real status/frame ABI instead of an empty aggregate or a result-less
    // call.
    SmallVector<LoopOp> zeroCarrierLoops;
    module.walk([&](LoopOp loop) {
      if (loop.getNumLoopCarried() == 0 && loop.getNumResults() == 0)
        zeroCarrierLoops.push_back(loop);
    });
    for (LoopOp loop : zeroCarrierLoops) {
      OpBuilder builder(loop);
      OperationState state(loop.getLoc(), loop->getName().getStringRef());
      state.addOperands(loop->getOperands());
      state.addTypes(LoopFrameType::get(module.getContext()));
      state.addAttributes(loop->getAttrs());
      state.addAttribute("descriptor_return", builder.getUnitAttr());
      state.propertiesAttr = loop->getPropertiesAsAttribute();
      Operation *replacement = builder.create(state);
      loop->replaceAllUsesWith(replacement->getResults());
      loop.erase();
    }

    WalkResult result = module.walk([&](LoopOp loop) {
      auto body = module.lookupSymbol<func::FuncOp>(loop.getBodyFuncAttr());
      if (!body || body.empty()) {
        loop.emitOpError("outlined body function is missing");
        return WalkResult::interrupt();
      }
      if (body.getNumArguments() == 0 ||
          !isa<LoopFrameType>(body.getArgumentTypes().back())) {
        body.emitOpError(
            "loop body must end its argument list with !hip.loop_frame");
        return WalkResult::interrupt();
      }

      unsigned numCarriers = loop.getNumLoopCarried();
      unsigned carrierResultStart = loop.getCondIsPassthrough() ? 1u : 2u;
      if (body.getNumResults() != carrierResultStart + numCarriers) {
        body.emitOpError("descriptor-return result count mismatch");
        return WalkResult::interrupt();
      }
      if (!body.getResultTypes().front().isInteger(32)) {
        body.emitOpError("descriptor-return ABI requires i32 status result #0");
        return WalkResult::interrupt();
      }
      if (body.getNumArguments() < 3 + numCarriers + 1) {
        body.emitOpError("descriptor-return argument count mismatch");
        return WalkResult::interrupt();
      }

      for (unsigned i = 0; i < numCarriers; ++i) {
        Type inputType = body.getArgument(3 + i).getType();
        Type resultType = body.getResultTypes()[carrierResultStart + i];
        auto inputMemref = dyn_cast<MemRefType>(inputType);
        auto resultMemref = dyn_cast<MemRefType>(resultType);
        if (!inputMemref || !resultMemref) {
          body.emitOpError("carrier #")
              << i << " must use ranked memref input/result types";
          return WalkResult::interrupt();
        }
        if (inputMemref != resultMemref) {
          body.emitOpError("carrier #")
              << i
              << " changes rank, element type, layout, or static "
                 "extents across the body";
          return WalkResult::interrupt();
        }
      }

      BufferViewFlowAnalysis aliases(body);
      SmallVector<std::pair<memref::AllocOp, unsigned>> carrierAllocs;
      bool invalid = false;
      body.walk([&](memref::AllocOp alloc) {
        std::optional<unsigned> carrierIndex;
        for (Value alias : aliases.resolve(alloc.getResult())) {
          for (OpOperand &use : alias.getUses()) {
            auto ret = dyn_cast<func::ReturnOp>(use.getOwner());
            if (!ret || use.getOperandNumber() < carrierResultStart)
              continue;
            unsigned resultIndex = use.getOperandNumber() - carrierResultStart;
            if (resultIndex >= numCarriers)
              continue;
            if (carrierIndex && *carrierIndex != resultIndex) {
              alloc.emitOpError(
                  "one allocation cannot back multiple loop carriers");
              invalid = true;
              return;
            }
            carrierIndex = resultIndex;
          }
        }
        if (carrierIndex)
          carrierAllocs.emplace_back(alloc, *carrierIndex);
      });
      if (invalid)
        return WalkResult::interrupt();

      OpBuilder builder(body.getContext());
      Value frame = body.getArguments().back();
      for (auto [alloc, carrierIndex] : carrierAllocs) {
        for (Operation *user : llvm::make_early_inc_range(alloc->getUsers()))
          if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
            dealloc.erase();
        builder.setInsertionPoint(alloc);
        auto loopAlloc = LoopAllocOp::create(
            builder, alloc.getLoc(), alloc.getType(), frame,
            alloc.getDynamicSizes(), builder.getI32IntegerAttr(carrierIndex));
        alloc.getResult().replaceAllUsesWith(loopAlloc.getResult());
        alloc.erase();
      }

      func::ReturnOp returnOp;
      body.walk([&](func::ReturnOp ret) { returnOp = ret; });
      if (!returnOp) {
        body.emitOpError("outlined loop body must have one func.return");
        return WalkResult::interrupt();
      }

      // A nested loop result or arbitrary body-produced descriptor belongs to
      // a different lifetime domain. Before publishing it as this body's next
      // carrier, materialize it into this frame/carrier's exact next bank.
      // Direct current pass-through and descriptors rooted at this carrier's
      // own hip.loop_alloc are already safe.
      BufferViewFlowAnalysis postRewriteAliases(body);
      for (unsigned carrier = 0; carrier < numCarriers; ++carrier) {
        unsigned resultSlot = carrierResultStart + carrier;
        Value returned = returnOp.getOperand(resultSlot);
        Value current = body.getArgument(3 + carrier);
        bool owned = returned == current;
        if (!owned) {
          body.walk([&](LoopAllocOp alloc) {
            if (owned ||
                static_cast<unsigned>(alloc.getCarrierIndex()) != carrier)
              return;
            for (Value alias : postRewriteAliases.resolve(alloc.getResult()))
              if (alias == returned) {
                owned = true;
                break;
              }
          });
        }
        if (owned)
          continue;

        auto type = cast<MemRefType>(returned.getType());
        builder.setInsertionPoint(returnOp);
        SmallVector<Value> dynamicSizes;
        for (int64_t dim = 0; dim < type.getRank(); ++dim) {
          if (!type.isDynamicDim(dim))
            continue;
          Value index =
              arith::ConstantIndexOp::create(builder, returnOp.getLoc(), dim);
          dynamicSizes.push_back(memref::DimOp::create(
              builder, returnOp.getLoc(), returned, index));
        }
        auto parentOwned = LoopAllocOp::create(
            builder, returnOp.getLoc(), type, frame, dynamicSizes,
            builder.getI32IntegerAttr(carrier));
        CopyOutputOp::create(builder, returnOp.getLoc(), body.getArgument(0),
                             returned, parentOwned.getResult());
        returnOp->setOperand(resultSlot, parentOwned.getResult());
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
} // namespace hip
} // namespace mlir
