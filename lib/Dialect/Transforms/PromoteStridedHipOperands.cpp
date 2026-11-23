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
// Ops that read strided inputs
// ----------------------------
// hip.reverse lowers to a strided copy that reads its input through the memref
// descriptor, so its input keeps its layout. Promoting it would add the very
// copy the op exists to avoid.
//
// hip.concat reads each input from one offset-aware pointer, so a contiguous
// view (one flat run with a descriptor offset) is read in place. Each strided
// input still needs a copy before the kernel. The concat kernel launches once
// per 16 non-empty inputs. When those copies plus the kernel launches are at
// least as many as one copy per input, the concat becomes one copy per input
// into its output slot, which is what insert_slice bufferizes to:
//
//   Before:
//     hip.concat(%ctx) ins(%a, %b : memref<64x64x1xf32, strided<[128, 2, 1]>>,
//                          memref<64x64x1xf32, strided<[128, 2, 1],
//                                                      offset: 1>>)
//                      outs(%out : memref<64x64x2xf32>) {axis = 2 : i64}
//   After:
//     %sa = memref.subview %out[0, 0, 0] [64, 64, 1] [1, 1, 1]
//     memref.copy %a, %sa
//     %sb = memref.subview %out[0, 0, 1] [64, 64, 1] [1, 1, 1]
//     memref.copy %b, %sb
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

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/BufferUtils.h"
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
STATISTIC(NumConcatsSplit,
          "Number of hip.concat ops split into per-input slot copies");

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

/// One flat run of elements: identity layout, or static strides that differ
/// from identity only by the descriptor offset.
static bool isContiguousView(MemRefType type) {
  if (type.getLayout().isIdentity())
    return true;
  SmallVector<int64_t> strides;
  int64_t offset;
  return type.hasStaticShape() &&
         succeeded(type.getStridesAndOffset(strides, offset)) &&
         isContiguousRowMajor(type.getShape(), strides);
}

/// Rewrites \p concat to one memref.copy per input into its output slot when
/// promoting its strided inputs would launch at least as many copies.
static bool splitStridedConcat(ConcatOp concat) {
  auto outputType = dyn_cast<MemRefType>(concat.getOutput().getType());
  if (!outputType || !outputType.hasStaticShape() || concat->getNumResults())
    return false;
  const int64_t axis = concat.getAxis();
  int64_t strided = llvm::count_if(concat.getInputs(), [](Value input) {
    return !isContiguousView(cast<MemRefType>(input.getType()));
  });
  int64_t inputs = concat.getInputs().size();
  // hip_concat launches once per 16 inputs with a non-zero width (its
  // CC_MAX_INPUTS window), so a promoted concat pays one launch per window.
  constexpr int64_t kInputsPerLaunch = 16;
  int64_t nonEmpty = llvm::count_if(concat.getInputs(), [axis](Value input) {
    return cast<MemRefType>(input.getType()).getDimSize(axis) != 0;
  });
  int64_t launches = std::max<int64_t>(1, (nonEmpty + kInputsPerLaunch - 1) /
                                              kInputsPerLaunch);
  if (strided == 0 || strided + launches < inputs)
    return false;
  if (!llvm::all_of(concat.getInputs(), [](Value input) {
        return cast<MemRefType>(input.getType()).hasStaticShape();
      }))
    return false;

  OpBuilder builder(concat);
  Location loc = concat.getLoc();
  int64_t rank = outputType.getRank();
  int64_t start = 0;
  for (Value input : concat.getInputs()) {
    auto inputType = cast<MemRefType>(input.getType());
    SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes, strides(rank, builder.getIndexAttr(1));
    for (int64_t extent : inputType.getShape())
      sizes.push_back(builder.getIndexAttr(extent));
    offsets[axis] = builder.getIndexAttr(start);
    Value slot = memref::SubViewOp::create(builder, loc, concat.getOutput(),
                                           offsets, sizes, strides);
    memref::CopyOp::create(builder, loc, input, slot);
    start += inputType.getDimSize(axis);
  }
  concat.erase();
  ++NumConcatsSplit;
  return true;
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
    if (isa<ReverseOp>(consumer))
      continue;
    auto concat = dyn_cast<ConcatOp>(consumer);
    if (concat && splitStridedConcat(concat))
      continue;
    SmallVector<Value> temporaries;
    for (OpOperand *input : dpsOp.getDpsInputOperands()) {
      auto type = dyn_cast<MemRefType>(input->get().getType());
      if (!type || !isNonIdentityMemRef(type))
        continue;
      if (concat && isContiguousView(type))
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
