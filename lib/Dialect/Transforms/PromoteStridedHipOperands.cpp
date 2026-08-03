/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PromoteStridedHipOperands.cpp - Normalize DPS input layouts --------===//
//
// Materializes an identity-layout temporary for each DPS-input memref operand
// with a non-identity layout (a non-zero offset or non-contiguous strides).
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
// Pipeline placement
// ------------------
// Run between hip-optimize-memrefs and hip-pool-allocs (see Pipelines.cpp).
// PoolAllocs replaces every memref.alloc with a memref.view into the pool
// and erases deallocs whose target is a view, so the new transient buffers
// fold cleanly into the existing pool — no extra hipMalloc per inference.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-promote-strided-operands"

STATISTIC(NumOperandsPromoted,
          "Number of non-identity-layout DPS inputs promoted");

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

/// Materializes an identity-layout copy of \p input immediately before
/// \p consumer and rewrites the operand to use it. Returns the temporary so the
/// caller can place a paired deallocation.
///
/// Resulting IR (alloc + copy only):
///   %tmp = memref.alloc(%dyn0, %dyn1, ...) : memref<...identity...>
///   memref.copy %source, %tmp
///   <consumer rewritten to read %tmp>
static Value materializeIdentityLayoutCopy(OpOperand &input,
                                           Operation *consumer) {
  Value source = input.get();
  auto sourceType = cast<MemRefType>(source.getType());
  MemRefType identityType = makeIdentityLayoutType(sourceType);

  OpBuilder builder(consumer);
  Location loc = consumer->getLoc();

  SmallVector<Value> dynSizes =
      collectDynamicSizes(builder, loc, source, sourceType);

  auto allocOp = memref::AllocOp::create(builder, loc, identityType, dynSizes);
  memref::CopyOp::create(builder, loc, source, allocOp.getResult());

  input.set(allocOp.getResult());

  ++NumOperandsPromoted;
  LLVM_DEBUG(llvm::dbgs() << "  promoted " << sourceType << " -> "
                          << identityType << " before " << consumer->getName()
                          << "\n");

  return allocOp.getResult();
}

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
      temporaries.push_back(materializeIdentityLayoutCopy(*input, consumer));
    }

    // Emit deallocs in the same order as the allocations (TA, TB, ...).
    // Without anchoring, repeated `setInsertionPointAfter(consumer)` would
    // push each new dealloc directly after the consumer, reversing the
    // sequence.  Advancing the anchor preserves source order and makes the
    // IR easier to read in tests / dumps.
    Operation *anchor = consumer;
    OpBuilder builder(consumer);
    for (Value temporary : temporaries) {
      builder.setInsertionPointAfter(anchor);
      anchor =
          memref::DeallocOp::create(builder, consumer->getLoc(), temporary);
    }
  }
}

} // namespace
} // namespace hip
} // namespace mlir
