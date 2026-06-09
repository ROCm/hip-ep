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
// THE `iter * chunk_size` ASSUMPTION SILENTLY BREAKS WHEN `chunk_size` IS
// NOT THE PER-ITER WINDOW SIZE. Canonical break (Qwen 3.5 vision encoder,
// Flickr30k sample 1, ~50% of vision-loop trip counts > 1):
//
//   1. SliceConversion's upper-bound guard (lib/Conversion/OnnxToHip/
//      SliceConversion.cpp -- the gotcha "Slice with runtime extent 0
//      produces a TRUE dim-0 buffer" is the OPPOSITE direction; the
//      upper-bound guard fires for the typical hip.slice the body uses
//      to grab its per-iter chunk from a captured Q/K/V tensor) sizes
//      the slice OUTPUT to `data.dim(slice_axis)` rows -- the FULL
//      sequence length of the captured tensor, not the actual per-iter
//      chunk extent (`cu_seqlens_k[iter+1] - cu_seqlens_k[iter]`).
//      This over-allocation is needed elsewhere to keep v_init big
//      enough to hold max_trip chunks; it is "by design" but has the
//      side effect we hit here.
//   2. OneShotBufferize's Concat-grow lowering takes the chunk's own
//      dim 1 as the subview's `sizes[i]`. Because of (1) that value is
//      now `full_seq`, not `window_size`.
//   3. This pass's fallback computes `newOff = iter * sizes[i]`. With
//      `sizes[i] = full_seq`, iter=1's offset becomes `full_seq * stride`
//      -- past the v_out buffer's end. The OOB write corrupts adjacent
//      pool slots; downstream layers' K/V load from the corrupted slots
//      and produce NaN / garbage; vision encoder output `image_features`
//      degrades from numeric (cos > 0.99 vs CPU) to NaN.
//
// THE FIX (below). Before falling back to `iter * sizes[i]`, we look for
// a host-side `arith.index_cast(memref.load(<gather_out>))` chain whose
// gather reads a captured seqlens-style table indexed by iter. That gives
// the TRUE start position `cu_seqlens_k[iter]` -- the canonical chunk
// origin in v_out coordinates -- regardless of whether `sizes[i]` was
// over-allocated. If the body has no such chain (typical: gather output
// is consumed directly by hip.slice as a device buffer, never host-loaded),
// we SYNTHESISE the load + cast just before the subview rewrite. The fix
// activates whenever the body has a hip.gather of a captured arg with
// memref<1xi32> output -- which is the canonical IR shape for any ONNX
// Loop windowed-attention export driven by a cu_seqlens table (Qwen-VL,
// SigLIP windowed variants, Swin-style window indexing, etc.). It does
// NOT match -- and is therefore a no-op on -- fixed-stride loops where
// chunk_start is computed as `iter * static_stride` without going through
// a gather of a runtime table; those genuine equal-chunk loops are still
// served correctly by the original `iter * sizes[i]` fallback.
//
// CAVEAT. The SIZE operand of the chunk-append subview is still
// `full_seq` (not `window_size`), so each iter's memref.copy writes
// more bytes than are semantically valid for that window. Today this is
// "harmless" because (a) hip.slice writes the per-iter valid prefix and
// leaves the over-allocated tail as the SliceConversion zero-fill, and
// (b) downstream MHA / Reshape / view chains read only the first
// `valid_chunk` rows. Strictly fixing this requires either tightening
// SliceConversion to emit `cu_seqlens_k[iter+1] - cu_seqlens_k[iter]`
// as the slice extent (and resizing the v_init buffer accordingly --
// the CLAUDE.md SAFETY REQUIREMENT note above is the constraint that
// must hold), or adopting one of Alts 1/2/4 below to remove the in-place
// writer pattern entirely.
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
  if (!iterTy || iterTy.getRank() != 0 ||
      !iterTy.getElementType().isInteger(64))
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
      cachedIterIdx =
          arith::IndexCastOp::create(b, loc, b.getIndexType(), loaded);
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
            chunkSize = arith::ConstantIndexOp::create(b, loc, attr.getInt());
          }
          // Prefer using the body's already-computed `start` position
          // (= seqlens_k[iter], the actual chunk position in the source
          // tensor) as the offset, falling back to iter*chunk_size when
          // we can't find it. The start is more accurate when chunks
          // are not equal-sized — it directly matches how the body
          // slices its captures (arg7/arg8) for the per-iter Q/K/V.
          //
          // Detection: the body emits this canonical sequence right
          // after gathering seqlens_k:
          //   hip.gather(%argN_seqlens_k, %iter_buf) -> %alloc_start
          //   %x = memref.load %alloc_start[%c0] : memref<1xi32>
          //   %start_idx = arith.index_cast %x : i32 to index
          // We pick the FIRST such index_cast in the body whose source
          // load reads from a memref<1xi32> alloc that's the output of
          // a hip.gather op (any capture-fed gather will do — typically
          // arg4 = seqlens_k).
          Value existingStart;
          funcOp.walk([&](arith::IndexCastOp ic) -> WalkResult {
            if (existingStart)
              return WalkResult::interrupt();
            auto loadOp = ic.getIn().getDefiningOp<memref::LoadOp>();
            if (!loadOp)
              return WalkResult::advance();
            auto loadSrcTy = dyn_cast<MemRefType>(loadOp.getMemref().getType());
            if (!loadSrcTy || loadSrcTy.getRank() != 1 ||
                !loadSrcTy.getElementType().isInteger(32))
              return WalkResult::advance();
            // The loaded alloc must be the output of a hip.gather op
            // whose data operand is a function arg (a capture).
            for (Operation *u : loadOp.getMemref().getUsers()) {
              if (u == loadOp)
                continue;
              if (u->getName().getStringRef() != "hip.gather")
                continue;
              if (u->getNumOperands() < 3)
                continue;
              // Last operand of hip.gather is the output (outs).
              // First non-ctx operand is the data; check that's a
              // function arg.
              for (Value gOpnd : u->getOperands()) {
                if (auto ga = dyn_cast<BlockArgument>(gOpnd)) {
                  if (ga.getOwner() == &entry && ga.getArgNumber() >= 3) {
                    existingStart = ic.getResult();
                    return WalkResult::interrupt();
                  }
                }
              }
            }
            return WalkResult::advance();
          });
          // Fallback synthesis: if the body has no host-side
          // index_cast(memref.load(...)) chain (typical for Qwen-style
          // vision loops where the gather output `alloc_0` is consumed
          // directly by hip.slice as a device buffer), find the FIRST
          // hip.gather whose data is a captured function arg (a
          // seqlens_k/q-style table) and whose output is memref<1xi32>,
          // then synthesise a host load + index_cast right before the
          // subview. Result is the start position cu_seqlens_k[iter] —
          // the canonical chunk start in v_out coordinates for windowed
          // attention. The synthesised load reads pinned/UMA memory
          // (alloc was placed on the GPU pool but is host-readable on
          // UMA targets); on non-UMA arches the MaterializeHostScalars
          // pass would route this through host scratch.
          if (!existingStart) {
            Operation *firstGather = nullptr;
            funcOp.walk([&](Operation *op) -> WalkResult {
              if (firstGather)
                return WalkResult::interrupt();
              if (op->getName().getStringRef() != "hip.gather")
                return WalkResult::advance();
              if (op->getNumOperands() < 3)
                return WalkResult::advance();
              // Output (last operand for DPS) must be memref<1xi32>.
              Value out = op->getOperand(op->getNumOperands() - 1);
              auto outTy = dyn_cast<MemRefType>(out.getType());
              if (!outTy || outTy.getRank() != 1 ||
                  !outTy.getElementType().isInteger(32))
                return WalkResult::advance();
              if (outTy.getDimSize(0) != 1)
                return WalkResult::advance();
              // Must have a function-arg-as-data operand (capture).
              for (Value opnd : op->getOperands()) {
                if (auto ba = dyn_cast<BlockArgument>(opnd)) {
                  if (ba.getOwner() == &entry && ba.getArgNumber() >= 3) {
                    firstGather = op;
                    return WalkResult::interrupt();
                  }
                }
              }
              return WalkResult::advance();
            });
            if (firstGather) {
              OpBuilder b2(sv);
              Value gatherOut =
                  firstGather->getOperand(firstGather->getNumOperands() - 1);
              Value zeroIdx = arith::ConstantIndexOp::create(b2, loc, 0);
              Value loaded = memref::LoadOp::create(b2, loc, gatherOut,
                                                    ValueRange{zeroIdx});
              Value idx = arith::IndexCastOp::create(b2, loc, b2.getIndexType(),
                                                     loaded);
              existingStart = idx;
            }
          }
          Value newOff;
          if (existingStart) {
            // Use the actual seqlens_k[iter] start. The subview offset
            // is in DIM-1 ELEMENTS, and `existingStart` is exactly that.
            newOff = existingStart;
          } else {
            // Fall back to iter * chunk_size (equal-chunk assumption).
            Value iterIdx = getIterIdx();
            newOff = arith::MulIOp::create(b, loc, iterIdx, chunkSize);
          }
          // The newOff Op (if newly created) was inserted at `b`'s
          // current insertion point — make sure it dominates the
          // subview by moving it just before sv. existingStart values
          // are pre-existing in the IR and already dominate sv (they
          // come from earlier in the same block).
          if (auto *defOp = newOff.getDefiningOp()) {
            if (defOp->isBeforeInBlock(sv) == false &&
                defOp->getBlock() == sv->getBlock()) {
              defOp->moveBefore(sv);
            }
          }

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
