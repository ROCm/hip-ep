/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PromoteStridedHipOperands.cpp - Strided -> contiguous copy ---------===//
//
// Materializes a contiguous temporary buffer for any DPS-input memref operand
// of a hip.* op that has a non-identity layout (non-zero offset or
// non-contiguous strides).
//
// Why this pass exists
// --------------------
// The HIP runtime call ABI used by --convert-hip-to-llvm receives a single
// bare pointer per memref operand (see extractContiguousMemRefPtr in
// HipToLLVMUtils.h).  There is no parameter on the wrapper signatures for
// per-dimension strides or for a base offset.  When a hip op consumes a
// strided memref (e.g., the result of memref.subview from onnx.Split), the
// wrapper would otherwise read the parent buffer from its base, ignoring the
// slice — silently producing wrong results.
//
// Rather than grow every wrapper signature and every backing library call
// site to accept (offset, strides[]), we promote upstream: insert a fresh
// contiguous alloc, copy the strided source into it, and rewrite the
// consumer's operand.  This mirrors upstream linalg::promoteSubViews and
// preserves a clean "hip.* ops always see contiguous memrefs" invariant.
//
// DPS-init (output) operands
// --------------------------
// Outs are left untouched.  In this pipeline they always come from either:
//   (a) memref.alloc() introduced by --one-shot-bufferize (identity layout),
//       or
//   (b) function arguments forced to identity layout by IdentityLayoutMap at
//       function boundaries.
// Either way they are guaranteed contiguous by construction.
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
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-promote-strided-operands"

STATISTIC(NumOperandsPromoted,
          "Number of strided hip.* DPS-input operands promoted to contiguous");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_PROMOTESTRIDEDHIPOPERANDSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Returns true iff \p type's layout is non-identity, i.e., the memref has
/// either a non-zero offset or non-contiguous strides.  Identity-layout
/// memrefs (no layout attribute, or a strided<> attribute that happens to be
/// canonical) are reported as contiguous.
static bool isNonIdentityMemRef(MemRefType type) {
  return !type.getLayout().isIdentity();
}

/// Builds a memref type with the same shape, element type, and memory space
/// as \p src but with identity layout.
static MemRefType makeContiguousType(MemRefType src) {
  return MemRefType::get(src.getShape(), src.getElementType(),
                         /*layout=*/MemRefLayoutAttrInterface{},
                         src.getMemorySpace());
}

/// Collects SSA values for each dynamic dimension of \p src (in dim order).
/// Emits memref.dim ops at the current insertion point of \p builder.
static SmallVector<Value> collectDynamicSizes(OpBuilder &builder, Location loc,
                                              Value src, MemRefType type) {
  SmallVector<Value> sizes;
  for (int64_t i : llvm::seq<int64_t>(type.getRank())) {
    if (type.isDynamicDim(i))
      sizes.push_back(memref::DimOp::create(builder, loc, src, i));
  }
  return sizes;
}

/// Materializes a contiguous temporary for \p strided just before \p consumer
/// and rewrites \p use to point at it.  Returns the new contiguous buffer so
/// the caller can place a paired dealloc.
///
/// Resulting IR (alloc + copy only):
///   %tmp = memref.alloc(%dyn0, %dyn1, ...) : memref<...identity...>
///   memref.copy %strided, %tmp
///   <consumer rewritten to read %tmp>
static Value materializeContiguousCopy(OpOperand &use, Operation *consumer) {
  Value strided = use.get();
  auto stridedType = cast<MemRefType>(strided.getType());
  MemRefType contiguousType = makeContiguousType(stridedType);

  OpBuilder builder(consumer);
  Location loc = consumer->getLoc();

  SmallVector<Value> dynSizes =
      collectDynamicSizes(builder, loc, strided, stridedType);

  auto allocOp =
      memref::AllocOp::create(builder, loc, contiguousType, dynSizes);
  memref::CopyOp::create(builder, loc, strided, allocOp.getResult());

  use.set(allocOp.getResult());

  ++NumOperandsPromoted;
  LLVM_DEBUG(llvm::dbgs() << "  promoted " << stridedType << " -> "
                          << contiguousType << " before " << consumer->getName()
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
  funcOp.walk([&](DestinationStyleOpInterface dpsOp) {
    Operation *op = dpsOp.getOperation();
    if (op->getDialect() != op->getContext()->getLoadedDialect<HipDialect>())
      return;
    consumers.push_back(dpsOp);
  });

  for (DestinationStyleOpInterface dpsOp : consumers) {
    Operation *consumer = dpsOp.getOperation();
    SmallVector<Value> tmps;
    for (OpOperand *input : dpsOp.getDpsInputOperands()) {
      auto type = dyn_cast<MemRefType>(input->get().getType());
      if (!type || !isNonIdentityMemRef(type))
        continue;
      tmps.push_back(materializeContiguousCopy(*input, consumer));
    }

    // Emit deallocs in the same order as the allocations (TA, TB, ...).
    // Without anchoring, repeated `setInsertionPointAfter(consumer)` would
    // push each new dealloc directly after the consumer, reversing the
    // sequence.  Advancing the anchor preserves source order and makes the
    // IR easier to read in tests / dumps.
    Operation *anchor = consumer;
    OpBuilder builder(consumer);
    for (Value tmp : tmps) {
      builder.setInsertionPointAfter(anchor);
      anchor = memref::DeallocOp::create(builder, consumer->getLoc(), tmp);
    }
  }
}

} // namespace
} // namespace hip
} // namespace mlir
