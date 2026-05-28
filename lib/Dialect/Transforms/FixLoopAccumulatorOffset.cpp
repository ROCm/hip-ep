/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- FixLoopAccumulatorOffset.cpp - iter-driven Concat-accumulator -----===//
//
// Replaces frozen-dim Concat-accumulator offsets in outlined `hip.loop` body
// functions with iter-driven offsets.
//
// PROBLEM. OneShotBufferize + BufferResultsToOutParams emits loop body funcs
// as in-place writers: the v_in (loop-carried in) and v_out (bufferize.result)
// arg pairs alias the SAME memref descriptor, and the trampoline
// (`LoopLowering.cpp::createOrGetTrampoline`, "v_out_i == v_in_i (same buffer,
// single-pass kernel safety)") passes that descriptor by value (expanded into
// separate scalars). MLIR memref descriptors are immutable SSA, so the body
// cannot mutate `sizes`; the trampoline reloads the same v_init descriptor
// every iteration. The body's typical Concat-grow accumulator pattern is:
//
//   %dim_13 = memref.dim %arg3, %c1                ; FROZEN at v_init's dim(1)
//   %subview1 = memref.subview %arg9[0, 0, 0] [..., %dim_13, ...] [1,1,1]
//   memref.copy %arg3, %subview1                   ; self-copy
//   %subview2 = memref.subview %arg9[0, %dim_13, 0] [1, %chunk, 1152] [1,1,1]
//   memref.copy %chunk_src, %subview2              ; append new chunk
//
// Every iteration `%dim_13` evaluates to v_init's dim(1). Every iteration
// writes the chunk to THE SAME byte offset. Only the LAST iteration's chunk
// survives. For Qwen 3.5 vision grid [2,8,8] with max_trip=2, cos vs CPU
// lands at ~1/max_trip ≈ 0.482.
//
// FIX. Find every `memref.subview` whose offset for dim N is `memref.dim
// %arg_v_in, %cN` (the v_in arg's own dim N). Replace that offset operand
// with `iter * chunk_size`, where `iter` is loaded from `arg1` (the body's
// iter arg) and `chunk_size` is the subview's size for the same dim N
// (which is the per-iteration chunk extent the body just computed).
//
// We INTENTIONALLY do not rewrite uses of `%dim_13` in SIZE positions: the
// first subview (the self-copy of arg3 into the in-place dest) has
// sizes=[dim_8, dim_13, dim_10] but offsets all-static-0 — it's not in our
// rewrite set. The chunk-append subview has dim_13 in OFFSETS only, so the
// rewrite affects exactly the offending offset and nothing else.
//
// Before:
//   %dim_13 = memref.dim %arg3, %c1                 ; (frozen at v_init dim)
//   ...
//   %sub = memref.subview %arg9[0, %dim_13, 0]
//                                       [1, %chunk, 1152] [1, 1, 1]
//   memref.copy %chunk_src, %sub
//
// After:
//   %iter_i64 = memref.load %arg1[]
//   %iter_idx = arith.index_cast %iter_i64 : i64 to index
//   %iter_offset = arith.muli %iter_idx, %chunk
//   %sub = memref.subview %arg9[0, %iter_offset, 0]
//                                       [1, %chunk, 1152] [1, 1, 1]
//   memref.copy %chunk_src, %sub
//
// ASSUMPTION (Qwen 3.5 vision and similar windowed-attention loops): chunks
// are equal-sized across iterations, so `iter * chunk_size` correctly
// computes the cumulative offset. For variable-chunk-size accumulators
// (uncommon), this pass would need extension to compute the prefix sum from
// a captured seqlens tensor instead.
//
// SAFETY REQUIREMENT. The v_init buffer MUST be sized large enough to hold
// max_trip chunks. With SliceConversion's upper-bound guard, the buffer is
// `data.dim(slice_axis)` rows = `max_trip * chunk_size` for the canonical
// `Slice(x, k, k, axis) -> Loop` pattern.
//
// PIPELINE PLACEMENT. After BufferResultsToOutParams (the in-place writer
// pattern only exists post-conversion) and after the buffer-deallocation
// cleanup; before OptimizeMemRefs and PoolAllocs (so the rewritten arith
// chain is visible to subsequent optimisation and hoist analysis).
//
//===----------------------------------------------------------------------===//
// ALTERNATIVE APPROACHES (DELETED THIS PASS? READ FIRST)
//===----------------------------------------------------------------------===//
//
// This pass is a SURGICAL POST-FIX layered on top of MLIR's standard
// BufferResultsToOutParams + in-place trampoline lowering. It is NOT the
// only valid solution. If you are tempted to delete this pass, please first
// adopt one of the alternatives below — otherwise the Qwen-style vision
// encoder cosine regression returns (cos ≈ 0.482 on Qwen 3.5 vision
// `test_vision_grid_small_matches_cpu_cosine`).
//
// Each alternative addresses the SAME ROOT CAUSE: the trampoline at
// `lib/Conversion/HipToLLVM/LoopLowering.cpp::createOrGetTrampoline` line
// "v_out_i == v_in_i (same buffer, single-pass kernel safety)" passes the
// v_in and v_out memref descriptors as the SAME by-value expanded scalars,
// so the body cannot mutate `sizes` across iterations and the body's
// `memref.dim %arg_v_in, %c1` stays frozen at the v_init descriptor's dim
// for every iteration. The four root-cause-equivalent options:
//
// Alt 1. SEPARATE v_out BUFFER (cleanest, biggest refactor).
//   Modify `createOrGetTrampoline` to allocate a fresh memref for v_out
//   each iteration instead of aliasing v_in. After the body returns,
//   write the new descriptor into the `lcArrayPtr[i]` slot. Next iter
//   loads the new descriptor as v_in. Final iter's descriptor becomes
//   the loop's result.
//   - Pros: matches MLIR scf.for / scf.while value-based semantics; no
//     special-case patterns; works for variable-chunk-size accumulators.
//   - Cons: trampoline needs to know v_out's size (unknown a priori for
//     dynamic-grow); also requires hip.loop's result type to be preserved
//     through bufferize + a new mechanism for downstream SSA uses to read
//     the final descriptor after the runtime call.
//
// Alt 2. SIDE CHANNEL FROM BODY (smallest signature change).
//   Add an extra `memref<i64>` arg to every loop body func (a pointer to
//   a single i64 slot). At the end of the body, store `dim_old + chunk`
//   to the slot. In `runLoopImpl`, after each body call, read the slot
//   and WRITE the new dim value into the appropriate `sizes[]` field of
//   the descriptor stored in `lcArrayPtr[i]`. Next iter's body sees the
//   updated descriptor.
//   - Pros: targets the exact mechanism (descriptor mutation between iters);
//     supports variable chunk sizes (body computes the prefix sum itself).
//   - Cons: body lowering must learn to emit the new store; requires
//     coordinated MLIR + runtime + body-signature changes; touches one
//     more cross-cutting concern.
//
// Alt 3. SPECIALIZE THE CONCAT-IN-LOOP-BODY LOWERING (this pass).
//   Recognise the in-place self-copy + chunk-append pattern in the
//   bufferized body IR and rewrite the chunk-append's offset to
//   `iter * subview.sizes[N]` (equivalent to "cumulative chunk size" when
//   chunks are equal-sized). Leave the self-copy alone.
//   - Pros: contained to one MLIR pass + a SliceConversion guard to size
//     the v_init buffer; no runtime / trampoline / signature changes.
//   - Cons: assumes equal-sized chunks across iterations (fine for
//     Qwen-style windowed attention, fails for variable-chunk-size
//     accumulators); coupled to the exact `memref.subview + memref.copy`
//     IR shape emitted by OneShotBufferize for the Concat lowering — a
//     future bufferize change that emits `memref.reinterpret_cast` + an
//     `arith.muli` offset chain instead would silently skip this pass.
//
// Alt 4. OPT LOOP BODIES OUT OF BufferResultsToOutParams.
//   Keep the loop body funcs returning a memref instead of writing through
//   a bufferize.result out-param. Body allocates its own buffer for the
//   Concat output and returns the descriptor. Trampoline captures the
//   returned descriptor into `lcArrayPtr[i]`.
//   - Pros: most "Mlir-idiomatic" — restores value-based dataflow.
//   - Cons: BufferResultsToOutParams is an upstream MLIR pass with no
//     per-function opt-out; would need either a custom mini-pass that
//     reverts its effect on loop body funcs (non-trivial inverse op)
//     or a fork of the pass with a function-filter callback. The
//     trampoline + LoopLowering also need adapting to receive the
//     returned descriptor.
//
// If you implement Alt 1, 2, or 4 above, you can delete this pass AND
// the SliceConversion upper-bound guard (the buffer doesn't need to be
// pre-sized when the body allocates fresh buffers per iter).

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "hip-fix-loop-accumulator-offset"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_FIXLOOPACCUMULATOROFFSETPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Returns true if the function looks like an outlined hip.loop body:
/// arg0 must be !hip.context, arg1 must be memref<i64> (iter), arg2 must be
/// memref<i1/ui8> (cond_in), and at least one arg must be marked
/// `bufferize.result` (the v_out). We avoid a name-prefix check so the pass
/// is robust against future outlining-pass renames.
static bool isOutlinedLoopBody(func::FuncOp funcOp) {
  if (funcOp.getNumArguments() < 4)
    return false;
  // arg1 must be memref<i64>.
  auto iterTy = dyn_cast<MemRefType>(funcOp.getArgumentTypes()[1]);
  if (!iterTy || iterTy.getRank() != 0 || !iterTy.getElementType().isInteger(64))
    return false;
  for (unsigned i = 0, e = funcOp.getNumArguments(); i < e; ++i) {
    if (funcOp.getArgAttr(i, "bufferize.result"))
      return true;
  }
  return false;
}

/// Count loop-carried-in args = bufferize.result-marked args whose memref
/// is NOT a scalar cond_out (rank-0 with i1/ui8 element type).
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

static bool isLoopCarriedInArg(func::FuncOp funcOp, BlockArgument arg) {
  unsigned idx = arg.getArgNumber();
  if (idx < 3)
    return false; // ctx / iter / cond_in
  if (funcOp.getArgAttr(idx, "bufferize.result"))
    return false;
  unsigned numLC = numLoopCarriedIn(funcOp);
  if (idx >= 3 + numLC)
    return false; // capture or out-param
  auto memTy = dyn_cast<MemRefType>(arg.getType());
  if (!memTy || memTy.getRank() < 1)
    return false;
  return true;
}

struct FixLoopAccumulatorOffsetPass
    : public impl::FixLoopAccumulatorOffsetPassBase<
          FixLoopAccumulatorOffsetPass> {
  using FixLoopAccumulatorOffsetPassBase::FixLoopAccumulatorOffsetPassBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (!isOutlinedLoopBody(funcOp))
      return;
    if (funcOp.empty())
      return;

    Block &entry = funcOp.getBody().front();
    Location loc = funcOp.getLoc();
    Value iterArg = funcOp.getArgument(1);

    // Cache the loaded iter index, inserted at the entry block's start.
    Value cachedIterIdx;
    auto getIterIdx = [&]() -> Value {
      if (cachedIterIdx)
        return cachedIterIdx;
      OpBuilder b(&entry, entry.begin());
      Value loaded = memref::LoadOp::create(b, loc, iterArg);
      cachedIterIdx = arith::IndexCastOp::create(
          b, loc, b.getIndexType(), loaded);
      return cachedIterIdx;
    };

    // Collect candidate `memref.dim %v_in_arg, %cN` ops.
    SmallVector<memref::DimOp> dimCandidates;
    funcOp.walk([&](memref::DimOp dimOp) {
      auto blockArg = dyn_cast<BlockArgument>(dimOp.getSource());
      if (!blockArg || blockArg.getOwner() != &entry)
        return;
      if (!isLoopCarriedInArg(funcOp, blockArg))
        return;
      auto cstIdx = dimOp.getConstantIndex();
      if (!cstIdx)
        return;
      dimCandidates.push_back(dimOp);
    });

    if (dimCandidates.empty())
      return;

    int rewriteCount = 0;
    for (memref::DimOp dimOp : dimCandidates) {
      // For each user that is a memref.subview where this dim appears in
      // OFFSETS (NOT sizes/strides), replace that offset operand with
      // iter * chunk_size where chunk_size = subview's size for the SAME
      // dim N. We only touch offsets — leaving uses in sizes alone keeps
      // the body's "self-copy" subview (sizes=[dim_8,dim_13,dim_10],
      // offsets=all-static-0) unaffected.
      //
      // memref.subview operand layout: source, offsets..., sizes...,
      // strides... where offsets/sizes/strides only include the DYNAMIC
      // entries (the static ones live in attributes). `getMixedOffsets`
      // returns an OpFoldResult per dim, with Values for dynamic offsets
      // and IntegerAttrs for static ones — that's what we need to figure
      // out which dim N our %dim_N corresponds to.

      // Walk a copy of the users since we'll mutate the IR.
      SmallVector<memref::SubViewOp> subviewUsers;
      for (Operation *user : dimOp->getUsers())
        if (auto sv = dyn_cast<memref::SubViewOp>(user))
          subviewUsers.push_back(sv);

      for (memref::SubViewOp sv : subviewUsers) {
        auto offsets = sv.getMixedOffsets();
        auto sizes = sv.getMixedSizes();
        // For each dim, check if the offset is OUR dim. If so, rewrite.
        for (size_t i = 0; i < offsets.size(); ++i) {
          Value offVal = dyn_cast<Value>(offsets[i]);
          if (!offVal || offVal != dimOp.getResult())
            continue;
          // Found it. The chunk_size is sizes[i].
          OpFoldResult sizeFR = sizes[i];
          Value chunkSize;
          OpBuilder b(sv);
          if (auto sv2 = dyn_cast<Value>(sizeFR)) {
            chunkSize = sv2;
          } else {
            auto attr = dyn_cast<IntegerAttr>(cast<Attribute>(sizeFR));
            if (!attr)
              continue;
            chunkSize =
                arith::ConstantIndexOp::create(b, loc, attr.getInt());
          }
          Value iterIdx = getIterIdx();
          Value newOff = arith::MulIOp::create(b, loc, iterIdx, chunkSize);

          // memref.subview's operand list is: source, then DYNAMIC offsets
          // in dim order, then DYNAMIC sizes, then DYNAMIC strides. To
          // find which operand index corresponds to dim i's offset, we
          // count preceding dynamic offsets.
          SmallVector<int64_t> staticOffsetsRaw(sv.getStaticOffsets());
          int operandIdx = 1; // skip source
          for (size_t j = 0; j < i; ++j) {
            if (ShapedType::isDynamic(staticOffsetsRaw[j]))
              ++operandIdx;
          }
          sv->setOperand(operandIdx, newOff);
          ++rewriteCount;
        }
      }
    }

    (void)rewriteCount;
  }
};

} // namespace

} // namespace hip
} // namespace mlir
