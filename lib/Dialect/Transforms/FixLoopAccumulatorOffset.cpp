/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- FixLoopAccumulatorOffset.cpp - iter-driven Concat-accumulator -----===//
//
// Replaces frozen Concat-accumulator offsets in outlined `hip.loop` body
// functions with iter-driven offsets. See the pass description in
// `include/hip/Dialect/Transforms/Passes.td` for the high-level problem /
// fix statement; this file documents the IR matcher in detail.
//
// PROBLEM (recap). `buffer-results-to-out-params` + the LoopLowering
// trampoline ("v_out_i == v_in_i (same buffer)") make the loop body an
// in-place writer: the loop-carried `v_in` arg and its `{bufferize.result}`
// `v_out` out-param alias ONE immutable memref descriptor. So
// `memref.dim %v_in, %cN` is frozen at the v_init descriptor's dim for every
// iteration. The canonical growing-Concat accumulator bufferizes to a
// self-copy subview (offset all-static-0, the frozen dim only in SIZES) plus
// a chunk-append subview (the frozen dim in OFFSET). Because the append
// offset is frozen, every iteration overwrites the same byte range and only
// the last chunk survives.
//
// FIX. Find every `memref.subview` whose OFFSET for dim N is
// `memref.dim %v_in, %cN`, and replace that offset with the real per-iter
// chunk start. The start is `seqlens_k[iter]`, recovered from the body's
// `hip.gather(%seqlens_capture, %iter) -> memref<1xi32>` and read back to the
// host with a SYNCHRONIZED `hip.readback_scalar` (D2H + stream sync). A bare
// `memref.load` of the gather output races the async gather kernel on targets
// where the pool is true device memory (it only accidentally works on
// UMA-mapped pools) -- the same correctness trap the trip-count readback and
// onnx.Range fixes avoid. When the body has no such gather (fixed-stride
// loops with no runtime seqlens table), fall back to `iter * chunk_size`,
// where `chunk_size` is the subview's SIZE for the same dim (the equal-chunk
// assumption). OFFSET operands only are touched -- the self-copy subview
// (frozen dim in SIZES, offset all-static-0) is never matched, so it keeps
// copying the accumulated-so-far prefix in place.
//
// Before:
//   %dim = memref.dim %arg_vin, %c1                 ; frozen at v_init dim
//   %sub = memref.subview %arg_vout[0, %dim, 0] [1, %chunk, W] [1, 1, 1]
//   memref.copy %src, %sub
//
// After:
//   %s_i32 = hip.readback_scalar(%ctx, %gather_out : memref<1xi32>) -> i32
//   %start = arith.index_cast %s_i32 : i32 to index
//   %sub = memref.subview %arg_vout[0, %start, 0] [1, %chunk, W] [1, 1, 1]
//   memref.copy %src, %sub
//
// ASSUMPTION (windowed-attention loops). The per-iter chunks tile the output
// sequence axis in order, so `seqlens_k[iter]` is exactly the chunk's start
// offset in v_out coordinates. The v_init buffer must already be sized to
// hold all chunks (true for the canonical `Slice(x, k, k, axis) -> Loop`
// capacity pattern).
//
// PIPELINE PLACEMENT. After buffer-deallocation, before `hip-optimize-memrefs`
// / `hip-materialize-host-scalars` / `hip-pool-allocs` (so the synthesized
// readback + index_cast are visible to CSE/canonicalize and to the pool/hoist
// analyses). The in-place-writer pattern only exists post-out-param-promotion,
// so the pass cannot run earlier.
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

#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "hip-fix-loop-accumulator-offset"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_FIXLOOPACCUMULATOROFFSETPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// True if `funcOp` looks like an outlined hip.loop body: arg0 is the
/// !hip.context, arg1 is `memref<i64>` (the iter), and at least one arg is
/// marked `{bufferize.result}` (the v_out out-param). We avoid a name-prefix
/// check so the pass is robust against future outlining-pass renames.
static bool isOutlinedLoopBody(func::FuncOp funcOp) {
  if (funcOp.getNumArguments() < 4)
    return false;
  if (!isa<ContextType>(funcOp.getArgumentTypes()[0]))
    return false;
  auto iterTy = dyn_cast<MemRefType>(funcOp.getArgumentTypes()[1]);
  if (!iterTy || iterTy.getRank() != 0 ||
      !iterTy.getElementType().isInteger(64))
    return false;
  for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i)
    if (funcOp.getArgAttr(i, "bufferize.result"))
      return true;
  return false;
}

/// Count loop-carried-in args: `{bufferize.result}`-marked args that are NOT a
/// scalar cond_out (rank-0 memref with i1/ui8/i8 element type). The body
/// signature is (ctx, iter, cond_in, v_in_0..N-1, captures..., v_out_0..N-1
/// {bufferize.result}); the loop-carried IN count equals the v_out count.
static unsigned numLoopCarriedIn(func::FuncOp funcOp) {
  unsigned numLC = 0;
  for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
    if (!funcOp.getArgAttr(i, "bufferize.result"))
      continue;
    auto memTy = dyn_cast<MemRefType>(funcOp.getArgumentTypes()[i]);
    if (!memTy)
      continue;
    Type elemTy = memTy.getElementType();
    if (memTy.getRank() == 0 && (elemTy.isInteger(1) || elemTy.isInteger(8)))
      continue; // cond_out
    ++numLC;
  }
  return numLC;
}

/// True if `arg` is a loop-carried-IN arg: a non-out-param block arg in
/// [3, 3 + numLoopCarriedIn) with a ranked memref type. The captures live
/// after the loop-carried-in slots, so this bounds-checks against the count.
static bool isLoopCarriedInArg(func::FuncOp funcOp, BlockArgument arg) {
  unsigned idx = arg.getArgNumber();
  if (idx < 3)
    return false; // ctx / iter / cond_in
  if (funcOp.getArgAttr(idx, "bufferize.result"))
    return false;
  if (idx >= 3 + numLoopCarriedIn(funcOp))
    return false; // capture or out-param
  auto memTy = dyn_cast<MemRefType>(arg.getType());
  return memTy && memTy.getRank() >= 1;
}

/// Find the first `hip.gather` whose output (last DPS operand) is
/// `memref<1xi32>` and whose data operand is a captured function arg
/// (argNumber >= 3). This is the canonical `seqlens_k[iter]` chunk-start
/// gather for a cu_seqlens-driven windowed-attention Loop. Returns null when
/// the body has no such gather (fixed-stride loops).
static Operation *findStartGather(func::FuncOp funcOp, Block &entry) {
  Operation *result = nullptr;
  funcOp.walk([&](Operation *op) -> WalkResult {
    if (op->getName().getStringRef() != "hip.gather")
      return WalkResult::advance();
    if (op->getNumOperands() < 3)
      return WalkResult::advance();
    Value out = op->getOperand(op->getNumOperands() - 1);
    auto outTy = dyn_cast<MemRefType>(out.getType());
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

    // Collect `memref.dim %v_in, %cN` candidates (constant dim index, source
    // is a loop-carried-in block arg). These are the frozen-descriptor dims.
    SmallVector<memref::DimOp> dimCandidates;
    funcOp.walk([&](memref::DimOp dimOp) {
      auto blockArg = dyn_cast<BlockArgument>(dimOp.getSource());
      if (!blockArg || blockArg.getOwner() != &entry)
        return;
      if (!isLoopCarriedInArg(funcOp, blockArg))
        return;
      if (!dimOp.getConstantIndex())
        return;
      dimCandidates.push_back(dimOp);
    });
    if (dimCandidates.empty())
      return;

    // The real per-iter chunk start = seqlens_k[iter], read back ONCE to the
    // host via a synchronized hip.readback_scalar inserted right after the
    // producing gather (so it dominates every chunk-append subview and is
    // correctly ordered after the async gather kernel). Lazily materialized.
    Value cachedStartIdx;
    auto getSeqlensStartIdx = [&]() -> Value {
      if (cachedStartIdx)
        return cachedStartIdx;
      Operation *gather = findStartGather(funcOp, entry);
      if (!gather)
        return Value();
      Value gatherOut = gather->getOperand(gather->getNumOperands() - 1);
      OpBuilder b(gather);
      b.setInsertionPointAfter(gather);
      Value i32Val =
          ReadbackScalarOp::create(b, loc, b.getI32Type(), ctx, gatherOut)
              .getResult();
      cachedStartIdx =
          arith::IndexCastOp::create(b, loc, b.getIndexType(), i32Val);
      return cachedStartIdx;
    };

    // Fallback iter index (equal-chunk loops with no seqlens table). Read the
    // iter scalar back synchronized, once.
    Value cachedIterIdx;
    auto getIterIdx = [&]() -> Value {
      if (cachedIterIdx)
        return cachedIterIdx;
      OpBuilder b(&entry, entry.begin());
      Value i64Val =
          ReadbackScalarOp::create(b, loc, b.getI64Type(), ctx, iterArg)
              .getResult();
      cachedIterIdx =
          arith::IndexCastOp::create(b, loc, b.getIndexType(), i64Val);
      return cachedIterIdx;
    };

    for (memref::DimOp dimOp : dimCandidates) {
      // Snapshot the subview users (we mutate operands below).
      SmallVector<memref::SubViewOp> subviewUsers;
      for (Operation *user : dimOp->getUsers())
        if (auto sv = dyn_cast<memref::SubViewOp>(user))
          subviewUsers.push_back(sv);

      for (memref::SubViewOp sv : subviewUsers) {
        SmallVector<OpFoldResult> offsets = sv.getMixedOffsets();
        SmallVector<OpFoldResult> sizes = sv.getMixedSizes();
        for (size_t i = 0; i < offsets.size(); ++i) {
          // Only rewrite when our frozen dim appears in this dim's OFFSET.
          Value offVal = dyn_cast<Value>(offsets[i]);
          if (!offVal || offVal != dimOp.getResult())
            continue;

          // Prefer the true seqlens_k[iter] start; fall back to iter*chunk.
          Value newOff = getSeqlensStartIdx();
          if (!newOff) {
            OpBuilder b(sv);
            Value chunkSize;
            if (auto cs = dyn_cast<Value>(sizes[i])) {
              chunkSize = cs;
            } else if (auto attr =
                           dyn_cast<IntegerAttr>(cast<Attribute>(sizes[i]))) {
              chunkSize = arith::ConstantIndexOp::create(b, loc, attr.getInt());
            } else {
              continue;
            }
            newOff = arith::MulIOp::create(b, loc, getIterIdx(), chunkSize);
          }

          // memref.subview operand layout: source, then DYNAMIC offsets in
          // dim order, then DYNAMIC sizes, then DYNAMIC strides. Locate the
          // operand index for dim i's (dynamic) offset by counting preceding
          // dynamic offsets.
          SmallVector<int64_t> staticOffsets(sv.getStaticOffsets());
          int operandIdx = 1; // skip source
          for (size_t j = 0; j < i; ++j)
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
