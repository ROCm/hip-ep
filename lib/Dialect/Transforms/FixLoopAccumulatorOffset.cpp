/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- FixLoopAccumulatorOffset.cpp - iter-driven Concat accumulator -----===//
//
// In an outlined `hip.loop` body, out-param promotion + the LoopLowering
// trampoline alias each loop-carried `v_in` arg and its `{bufferize.result}`
// `v_out` out-param onto ONE immutable memref descriptor. So `memref.dim
// %v_in, %cN` is frozen at the v_init dim every iteration. A growing-Concat
// accumulator bufferizes to a self-copy subview (frozen dim in SIZES, offset 0)
// plus a chunk-append subview (frozen dim in the OFFSET); the frozen append
// offset makes every iteration overwrite the same slab, so only the last chunk
// survives.
//
// This pass rewrites each chunk-append subview OFFSET from the frozen
// `memref.dim` to the real per-iter chunk start: `seqlens_k[iter]`, read from
// the body's first `hip.gather(...) -> memref<1xi32>` via a SYNCHRONIZED
// `hip.readback_scalar` (a bare load races the async gather on true-device
// pools). Fallback for loops with no seqlens gather: `iter * chunk_size`
// (chunk_size = the subview SIZE for that dim). Only OFFSETs are touched, so
// the self-copy subview keeps copying the accumulated prefix in place.
//
// Before:
//   %dim = memref.dim %v_in, %c1                      // frozen at v_init dim
//   %sub = memref.subview %v_out[0, %dim, 0] [1, %chunk, W] [1, 1, 1]
// After:
//   %s   = hip.readback_scalar(%ctx, %gather_out : memref<1xi32>) -> i32
//   %off = arith.index_cast %s : i32 to index
//   %sub = memref.subview %v_out[0, %off, 0] [1, %chunk, W] [1, 1, 1]
//
// Assumes per-iter chunks tile the output axis in order (so seqlens_k[iter] is
// the chunk start) and v_init is pre-sized to full capacity (the canonical
// `Slice(x, k, k, axis) -> Loop` pattern). Runs after buffer-deallocation (the
// alias only exists post-out-param-promotion) and before the pool/hoist passes.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "hip-fix-loop-accumulator-offset"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_FIXLOOPACCUMULATOROFFSETPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// An outlined hip.loop body has arg0 = !hip.context, arg1 = memref<i64> (the
/// iter), and at least one `{bufferize.result}` out-param. Matched structurally
/// so the pass survives outliner renames.
static bool isOutlinedLoopBody(func::FuncOp fn) {
  if (fn.getNumArguments() < 4 || !isa<ContextType>(fn.getArgumentTypes()[0]))
    return false;
  auto iterTy = dyn_cast<MemRefType>(fn.getArgumentTypes()[1]);
  if (!iterTy || iterTy.getRank() != 0 ||
      !iterTy.getElementType().isInteger(64))
    return false;
  for (unsigned i : llvm::seq(0u, fn.getNumArguments()))
    if (fn.getArgAttr(i, "bufferize.result"))
      return true;
  return false;
}

/// Number of loop-carried-in args = `{bufferize.result}` out-params that are
/// not the scalar cond_out (rank-0 i1/i8 memref). Body signature: (ctx, iter,
/// cond_in, v_in[0..N), captures..., v_out[0..N) {bufferize.result}).
static unsigned numLoopCarriedIn(func::FuncOp fn) {
  unsigned n = 0;
  for (unsigned i : llvm::seq(0u, fn.getNumArguments())) {
    if (!fn.getArgAttr(i, "bufferize.result"))
      continue;
    auto memTy = dyn_cast<MemRefType>(fn.getArgumentTypes()[i]);
    bool isCondOut = memTy && memTy.getRank() == 0 &&
                     (memTy.getElementType().isInteger(1) ||
                      memTy.getElementType().isInteger(8));
    if (memTy && !isCondOut)
      ++n;
  }
  return n;
}

/// First `hip.gather` with a `memref<1xi32>` output (last DPS operand) reading
/// a captured arg (idx >= 3) -- the `seqlens_k[iter]` chunk-start gather. Null
/// for fixed-stride loops with no seqlens table.
static Operation *findStartGather(func::FuncOp fn, Block &entry) {
  Operation *result = nullptr;
  fn.walk([&](Operation *op) -> WalkResult {
    if (op->getName().getStringRef() != "hip.gather" ||
        op->getNumOperands() < 3)
      return WalkResult::advance();
    auto outTy = dyn_cast<MemRefType>(op->getOperands().back().getType());
    if (!outTy || outTy.getRank() != 1 ||
        !outTy.getElementType().isInteger(32) || outTy.getDimSize(0) != 1)
      return WalkResult::advance();
    for (Value opnd : op->getOperands()) {
      auto ba = dyn_cast<BlockArgument>(opnd);
      if (ba && ba.getOwner() == &entry && ba.getArgNumber() >= 3) {
        result = op;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return result;
}

struct FixLoopAccumulatorOffsetPass
    : public impl::FixLoopAccumulatorOffsetPassBase<
          FixLoopAccumulatorOffsetPass> {
  using FixLoopAccumulatorOffsetPassBase::FixLoopAccumulatorOffsetPassBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.empty() || !isOutlinedLoopBody(funcOp))
      return;

    Block &entry = funcOp.getBody().front();
    Location loc = funcOp.getLoc();
    Value ctx = funcOp.getArgument(0);
    Value iterArg = funcOp.getArgument(1);
    unsigned numLC = numLoopCarriedIn(funcOp);

    // A v_in arg is a ranked-memref block arg in [3, 3+numLC) that is not an
    // out-param. Captures (after the v_in slots) and out-params are excluded so
    // we only target dims of the frozen loop-carried descriptor.
    auto isVIn = [&](BlockArgument a) {
      unsigned i = a.getArgNumber();
      if (i < 3 || i >= 3 + numLC || funcOp.getArgAttr(i, "bufferize.result"))
        return false;
      auto m = dyn_cast<MemRefType>(a.getType());
      return m && m.getRank() >= 1;
    };

    // Frozen-descriptor dims: `memref.dim %v_in, <const>`.
    SmallVector<memref::DimOp> dimCandidates;
    funcOp.walk([&](memref::DimOp dimOp) {
      auto ba = dyn_cast<BlockArgument>(dimOp.getSource());
      if (ba && ba.getOwner() == &entry && isVIn(ba) &&
          dimOp.getConstantIndex())
        dimCandidates.push_back(dimOp);
    });
    if (dimCandidates.empty())
      return;

    // seqlens_k[iter], read back once via a synchronized hip.readback_scalar
    // placed right after the producing gather (dominates every chunk-append
    // subview, ordered after the async gather). Lazily materialized; null when
    // the body has no seqlens gather.
    Value cachedStartIdx;
    auto getSeqlensStartIdx = [&]() -> Value {
      if (cachedStartIdx)
        return cachedStartIdx;
      Operation *gather = findStartGather(funcOp, entry);
      if (!gather)
        return Value();
      OpBuilder b(gather);
      b.setInsertionPointAfter(gather);
      Value i32 = ReadbackScalarOp::create(b, loc, b.getI32Type(), ctx,
                                           gather->getOperands().back());
      cachedStartIdx =
          arith::IndexCastOp::create(b, loc, b.getIndexType(), i32);
      return cachedStartIdx;
    };

    // Fallback iter index (equal-chunk loops): synchronized readback of iter.
    Value cachedIterIdx;
    auto getIterIdx = [&]() -> Value {
      if (cachedIterIdx)
        return cachedIterIdx;
      OpBuilder b(&entry, entry.begin());
      Value i64 =
          ReadbackScalarOp::create(b, loc, b.getI64Type(), ctx, iterArg);
      cachedIterIdx = arith::IndexCastOp::create(b, loc, b.getIndexType(), i64);
      return cachedIterIdx;
    };

    for (memref::DimOp dimOp : dimCandidates) {
      // Snapshot users: we mutate operands below.
      SmallVector<memref::SubViewOp> subviews;
      for (Operation *user : dimOp->getUsers())
        if (auto sv = dyn_cast<memref::SubViewOp>(user))
          subviews.push_back(sv);

      for (memref::SubViewOp sv : subviews) {
        SmallVector<OpFoldResult> offsets = sv.getMixedOffsets();
        SmallVector<OpFoldResult> sizes = sv.getMixedSizes();
        for (size_t i : llvm::seq<size_t>(0, offsets.size())) {
          // Only rewrite when our frozen dim is this dim's OFFSET.
          Value offVal = dyn_cast<Value>(offsets[i]);
          if (offVal != dimOp.getResult())
            continue;

          // Prefer seqlens_k[iter]; else iter * chunk_size (chunk = SIZE[i]).
          Value newOff = getSeqlensStartIdx();
          if (!newOff) {
            OpBuilder b(sv);
            Value chunk = dyn_cast<Value>(sizes[i]);
            if (!chunk) {
              auto attr = dyn_cast<IntegerAttr>(cast<Attribute>(sizes[i]));
              if (!attr)
                continue;
              chunk = arith::ConstantIndexOp::create(b, loc, attr.getInt());
            }
            newOff = arith::MulIOp::create(b, loc, getIterIdx(), chunk);
          }

          // subview operands: source, dynamic offsets (dim order), sizes,
          // strides. Our offset is dynamic, so its operand index is 1 + (number
          // of dynamic offsets before dim i).
          ArrayRef<int64_t> staticOffsets = sv.getStaticOffsets();
          int operandIdx = 1;
          for (size_t j : llvm::seq<size_t>(0, i))
            if (ShapedType::isDynamic(staticOffsets[j]))
              ++operandIdx;
          sv->setOperand(operandIdx, newOff);
        }
      }
    }
  }
};

} // namespace
} // namespace hip
} // namespace mlir
