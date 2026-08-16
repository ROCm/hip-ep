/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PromoteStridedHipOperands.cpp - Normalize DPS input layouts --------===//
//
// Materializes an identity-layout temporary for each bare-pointer consumer
// memref operand with a non-identity layout (a non-zero offset or
// non-contiguous strides).
//
// Why this pass exists
// --------------------
// HIP and plugin runtime wrappers commonly receive one bare pointer per memref
// operand (see extractContiguousMemRefPtr in HipToLLVMUtils.h), with no channel
// for the descriptor offset or strides. Passing a non-identity-layout memref
// directly to such a wrapper would address the parent buffer rather than the
// intended logical view.
//
// Rather than grow every wrapper signature and every backing library call
// site to accept (offset, strides[]), we promote upstream: insert a fresh
// identity-layout allocation, copy the source into it, and rewrite the
// consumer's operand. This mirrors upstream linalg::promoteSubViews and
// establishes that DPS-input memrefs have identity layout after this pass.
//
// DPS-init (output) operands
// --------------------------
// Inits are intentionally left untouched: promoting a write target requires
// copy-back and must preserve any read-before-write semantics. The production
// pipeline normally supplies identity-layout inits through fresh allocations,
// hip.alloc_output, or IdentityLayoutMap at function boundaries. Lowerings
// that extract a bare pointer rely on that independent output-side invariant.
//
// Loop captures
// -------------
// After bufferization, a hip.loop capture may be a strided view while its
// outlined body argument has the identity layout selected for function
// boundaries. Captures are read-only and cannot alias loop results, so one
// identity-layout copy per distinct capture is live only across the loop
// invocation. Loop-carried v_init operands are deliberately excluded: a
// zero-trip loop may return the borrowed seed descriptor.
//
// Pipeline placement
// ------------------
// Run between hip-optimize-memrefs and hip-pool-allocs (see Pipelines.cpp).
// PoolAllocs replaces every memref.alloc with a memref.view into the pool
// and erases deallocs whose target is a view, so the new transient buffers
// fold cleanly into the existing pool — no extra hipMalloc per inference.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-promote-strided-operands"

STATISTIC(NumOperandsPromoted,
          "Number of non-identity-layout operands promoted");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_PROMOTESTRIDEDHIPOPERANDSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Returns true iff \p type's layout is non-identity, i.e., the memref has
/// either a non-zero offset or non-contiguous strides. Identity-layout
/// memrefs (no layout attribute, or a strided<> attribute that happens to be
/// canonical) are reported as false.
static bool isNonIdentityMemRef(MemRefType type) {
  return !type.getLayout().isIdentity();
}

/// Builds a memref type with the same shape, element type, and memory space
/// as \p sourceType but with identity layout.
static MemRefType makeIdentityLayoutType(MemRefType sourceType) {
  return MemRefType::get(sourceType.getShape(), sourceType.getElementType(),
                         /*layout=*/MemRefLayoutAttrInterface{},
                         sourceType.getMemorySpace());
}

/// Collects SSA values for each dynamic dimension of \p source (in dim order).
/// Emits memref.dim ops at the current insertion point of \p builder.
static SmallVector<Value> collectDynamicSizes(OpBuilder &builder, Location loc,
                                              Value source, MemRefType type) {
  SmallVector<Value> sizes;
  for (int64_t i : llvm::seq<int64_t>(type.getRank())) {
    if (type.isDynamicDim(i))
      sizes.push_back(memref::DimOp::create(builder, loc, source, i));
  }
  return sizes;
}

/// Materializes an identity-layout copy of \p source immediately before
/// \p consumer. Returns the temporary so the caller can rewrite the operand and
/// place a paired deallocation.
///
/// Resulting IR (alloc + copy only):
///   %tmp = memref.alloc(%dyn0, %dyn1, ...) : memref<...identity...>
///   memref.copy %source, %tmp
///   <consumer rewritten to read %tmp>
static Value materializeIdentityLayoutCopy(Value source, Operation *consumer) {
  auto sourceType = cast<MemRefType>(source.getType());
  MemRefType identityType = makeIdentityLayoutType(sourceType);

  OpBuilder builder(consumer);
  Location loc = consumer->getLoc();

  SmallVector<Value> dynSizes =
      collectDynamicSizes(builder, loc, source, sourceType);

  auto allocOp = memref::AllocOp::create(builder, loc, identityType, dynSizes);
  memref::CopyOp::create(builder, loc, source, allocOp.getResult());

  ++NumOperandsPromoted;
  LLVM_DEBUG(llvm::dbgs() << "  promoted " << sourceType << " -> "
                          << identityType << " before " << consumer->getName()
                          << "\n");

  return allocOp.getResult();
}

/// Places paired deallocations immediately after \p consumer in allocation
/// order.
static void deallocateAfter(Operation *consumer, ValueRange temporaries) {
  Operation *anchor = consumer;
  OpBuilder builder(consumer);
  for (Value temporary : temporaries) {
    builder.setInsertionPointAfter(anchor);
    anchor = memref::DeallocOp::create(builder, consumer->getLoc(), temporary);
  }
}

struct LoopPromotionPlan {
  LoopOp loop;
  SmallVector<unsigned> captureIndices;
};

struct PromoteStridedHipOperandsPass
    : public impl::PromoteStridedHipOperandsPassBase<
          PromoteStridedHipOperandsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override;
};

void PromoteStridedHipOperandsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.empty())
    return;

  // Validate every loop plan before changing any IR. A malformed body ABI must
  // not leave earlier loops partially promoted.
  SymbolTableCollection symbolTable;
  SmallVector<LoopPromotionPlan> loopPlans;
  WalkResult loopValidation = funcOp.walk([&](LoopOp loop) {
    SmallVector<unsigned> candidates;
    for (auto [index, capture] : llvm::enumerate(loop.getCaptures())) {
      auto captureType = dyn_cast<MemRefType>(capture.getType());
      if (captureType && isNonIdentityMemRef(captureType))
        candidates.push_back(index);
    }
    if (candidates.empty())
      return WalkResult::advance();

    auto body = symbolTable.lookupNearestSymbolFrom<func::FuncOp>(
        loop, loop.getBodyFuncAttr());
    if (!body) {
      loop.emitOpError("cannot promote strided captures: body_func '")
          << loop.getBodyFunc() << "' does not reference a func.func";
      return WalkResult::interrupt();
    }

    unsigned numCarriers = loop.getNumLoopCarried();
    unsigned expectedArgs = 4 + numCarriers + loop.getCaptures().size();
    if (body.getNumArguments() != expectedArgs) {
      loop.emitOpError("cannot promote strided captures: body_func argument "
                       "count mismatch; expected ")
          << expectedArgs
          << " (context, iter, cond, carriers, captures, frame), "
          << "got " << body.getNumArguments();
      return WalkResult::interrupt();
    }

    LoopPromotionPlan plan{loop, {}};
    for (unsigned index : candidates) {
      auto sourceType = cast<MemRefType>(loop.getCaptures()[index].getType());
      unsigned bodyArgIndex = 3 + numCarriers + index;
      auto bodyType =
          dyn_cast<MemRefType>(body.getArgumentTypes()[bodyArgIndex]);
      if (!bodyType) {
        loop.emitOpError("cannot promote strided capture #")
            << index << ": body argument #" << bodyArgIndex
            << " must be a ranked memref";
        return WalkResult::interrupt();
      }
      if (!bodyType.getLayout().isIdentity()) {
        if (bodyType == sourceType)
          continue;
        loop.emitOpError("cannot promote strided capture #")
            << index << ": body argument #" << bodyArgIndex
            << " has an unsupported non-identity layout mismatch";
        return WalkResult::interrupt();
      }
      if (bodyType.getRank() != sourceType.getRank()) {
        loop.emitOpError("cannot promote strided capture #")
            << index << ": body argument rank " << bodyType.getRank()
            << " does not match capture rank " << sourceType.getRank();
        return WalkResult::interrupt();
      }
      if (bodyType.getElementType() != sourceType.getElementType()) {
        loop.emitOpError("cannot promote strided capture #")
            << index << ": body argument element type "
            << bodyType.getElementType()
            << " does not match capture element type "
            << sourceType.getElementType();
        return WalkResult::interrupt();
      }
      MemRefType promotedType = makeIdentityLayoutType(sourceType);
      if (bodyType.getShape() != promotedType.getShape()) {
        loop.emitOpError("cannot promote strided capture #")
            << index << ": body argument shape " << bodyType.getShape()
            << " does not match capture shape " << promotedType.getShape();
        return WalkResult::interrupt();
      }
      if (bodyType.getMemorySpace() != promotedType.getMemorySpace()) {
        loop.emitOpError("cannot promote strided capture #")
            << index << ": body argument memory space "
            << bodyType.getMemorySpace()
            << " does not match capture memory space "
            << promotedType.getMemorySpace();
        return WalkResult::interrupt();
      }
      plan.captureIndices.push_back(index);
    }
    if (!plan.captureIndices.empty())
      loopPlans.push_back(std::move(plan));
    return WalkResult::advance();
  });
  if (loopValidation.wasInterrupted())
    return signalPassFailure();

  for (LoopPromotionPlan &plan : loopPlans) {
    DenseMap<Value, Value> promoted;
    SmallVector<Value> temporaries;
    MutableOperandRange captures = plan.loop.getCapturesMutable();
    for (unsigned index : plan.captureIndices) {
      OpOperand &capture = captures[index];
      Value source = capture.get();
      auto [it, inserted] = promoted.try_emplace(source);
      if (inserted) {
        it->second =
            materializeIdentityLayoutCopy(source, plan.loop.getOperation());
        temporaries.push_back(it->second);
      }
      capture.set(it->second);
    }
    deallocateAfter(plan.loop.getOperation(), temporaries);
  }

  // Collect candidate consumers first so that rewriting (which inserts new
  // ops) does not invalidate the walk.
  SmallVector<DestinationStyleOpInterface> consumers;
  funcOp.walk(
      [&](DestinationStyleOpInterface dpsOp) { consumers.push_back(dpsOp); });

  for (DestinationStyleOpInterface dpsOp : consumers) {
    Operation *consumer = dpsOp.getOperation();
    SmallVector<Value> temporaries;
    for (OpOperand *input : dpsOp.getDpsInputOperands()) {
      auto type = dyn_cast<MemRefType>(input->get().getType());
      if (!type || !isNonIdentityMemRef(type))
        continue;
      Value temporary = materializeIdentityLayoutCopy(input->get(), consumer);
      input->set(temporary);
      temporaries.push_back(temporary);
    }

    deallocateAfter(consumer, temporaries);
  }
}

} // namespace
} // namespace hip
} // namespace mlir
