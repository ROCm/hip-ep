/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- AnnotateInputDimSlots.cpp - Tag consumer operands with slot ids ----===//
//
// Walks every op in `@main_graph`. For each operand whose producer's dim
// resolves to a `RuntimeSlot` leaf (per the per-op DimSpec system in
// HipShapeInterface), attaches a `hipdnn.input_dim_slots` attribute
// listing, per operand, the `(dim_index, slot_id)` pairs of every
// dynamic dim that resolves to a runtime-published slot.
//
// Why this exists
// ---------------
// After bufferize-to-out-params + pool-allocs, every dynamic-shape
// intermediate tensor is materialized as a memref view into an
// upper-bound pool buffer. The Hip op writing that buffer (e.g.
// `hip.nonzero` for the Category-C case) publishes the *true* runtime
// size and pointer into the slot table; the underlying memref
// descriptor, however, encodes the *upper-bound* size that pool-allocs
// reserved. Any consumer that reads `descriptor.sizes[d]` on a
// dynamic dim would silently iterate over upper-bound-many elements
// (writing garbage or reading uninitialized memory).
//
// This pass propagates the slot id at compile time so that the
// per-op HipToLLVM lowering can emit `hipdnn_ep_state_read_dim(state,
// slot_id)` for the affected dim instead of reading the descriptor.
// Consumer pointers are unchanged (the upper-bound buffer is correct;
// only the prefix `[0 .. published_dim)` is valid, but that is
// exactly what the kernel needs when given the right element count).
//
// Encoding
// --------
// Two attributes are attached on each annotated consumer op:
//
// 1. `hipdnn.input_dim_slots`: `ArrayAttr` with one entry per operand,
//    each entry an `ArrayAttr` of `DenseI32ArrayAttr` `[dim_idx,
//    slot_id]` pairs. Consumer SHAPE rewiring uses this — read the
//    runtime-published dim from the slot table instead of the
//    descriptor's sizes[d] (which holds the upper-bound pool
//    allocation).
//
// 2. `hipdnn.input_slot_buffers`: `DenseI32ArrayAttr` with one i32 per
//    operand; -1 means "use the descriptor's alignedPtr as usual",
//    otherwise the value is the slot id of the producer that
//    actually wrote the operand's data. Consumer POINTER rewiring
//    uses this — when set, the operand's data lives in the slot
//    publisher's exact-size buffer (`hipdnn_ep_state_read_buffer`),
//    NOT in the upper-bound DPS init.
//
// The distinction matters because slot publishers (e.g.
// `hip.nonzero`) allocate a SEPARATE exact-size buffer at runtime
// and ignore their upper-bound DPS init. Translucent propagators
// (e.g. `hip.transpose` with the slot-aware shape patch) write into
// their own upper-bound DPS init — its prefix is valid even though
// the descriptor's size encodes the upper bound. Only DIRECT
// consumers of slot publishers need pointer rewiring; everyone in
// the chain only needs shape rewiring.
//
// When NO operand has any slot-resolved dim, the dim_slots attribute
// is OMITTED. When no operand has a publisher producer, the
// input_slot_buffers attribute is omitted likewise.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeInterface.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/debug_log.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_ANNOTATEINPUTDIMSLOTSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Find the producing op for a memref operand value `v` AFTER
// bufferization. The standard pattern after pool-allocs is:
//
//     %buf  = memref.view %pool[%off][] : memref<2x?xi64>
//     hip.nonzero(%ctx) ins(%x) outs(%buf : memref<2x?xi64>)
//     hip.transpose(%ctx) ins(%buf : memref<2x?xi64>) outs(...)
//
// For the consumer (`hip.transpose`) we want to find `hip.nonzero` —
// not `memref.view` — because `output_dim_specs` / DimSpec builders
// live on the Hip op that writes the buffer, NOT on the alloc.
//
// Strategy: walk the use-list of `v`. The first Hip-dialect op
// (DestinationStyleOpInterface) that uses `v` as a DPS init operand
// (i.e. an output operand it writes into) is the producer. If no such
// op is found, fall back to `v.getDefiningOp()` (for the rare case of
// non-DPS or non-bufferized IR).
//
// Returns `nullptr` if no producer can be identified. Also writes
// the producer's result index (corresponding to `v`) into `*resultIdx`
// when non-null; for 0-result DPS ops (the post-bufferize norm) the
// result index is the position of `v` in the init operand list.
// Return the single-step view-source of `cursor` if its defining op is one
// of the view-style memref / tensor ops we treat as transparent. The
// returned value identifies the same underlying buffer (for memref ops)
// or the same tensor identity (for tensor ops). Returns `Value()` when
// the defining op is not a view -- callers can use that as a stop sentinel.
//
// Crucially, this includes `memref.view` and `memref.subview` even though
// those ops slice into a larger buffer at a non-zero offset. The walk is
// used to chase DPS writers: in the post-bufferize pool layout the writer
// (e.g. `hip.transpose`) and the reader (e.g. `hip.scatter_nd`) may share
// the same `memref.view` SSA value directly, share a chain of casts above
// it, or, in the rare aliasing case, the writer may use a different
// `memref.view` of the SAME pool offset. The first two cases are handled
// by the per-step search in `findMemRefWriter`; chasing through `view` /
// `subview` here only kicks in if no DPS writer was found at any closer
// step, which keeps the legacy aliasing fallback alive without
// incorrectly mistaking unrelated views of the same pool for the writer.
static Value stepThroughView(Value cursor) {
  if (!cursor)
    return Value();
  Operation *def = cursor.getDefiningOp();
  if (!def)
    return Value();
  if (auto cast = llvm::dyn_cast<memref::CastOp>(def))
    return cast.getSource();
  if (auto sub = llvm::dyn_cast<memref::SubViewOp>(def))
    return sub.getSource();
  if (auto view = llvm::dyn_cast<memref::ViewOp>(def))
    return view.getSource();
  if (auto reinterp = llvm::dyn_cast<memref::ReinterpretCastOp>(def))
    return reinterp.getSource();
  if (auto exp = llvm::dyn_cast<memref::ExpandShapeOp>(def))
    return exp.getSrc();
  if (auto coll = llvm::dyn_cast<memref::CollapseShapeOp>(def))
    return coll.getSrc();
  if (auto exp = llvm::dyn_cast<tensor::ExpandShapeOp>(def))
    return exp.getSrc();
  if (auto coll = llvm::dyn_cast<tensor::CollapseShapeOp>(def))
    return coll.getSrc();
  if (auto extract = llvm::dyn_cast<tensor::ExtractSliceOp>(def))
    return extract.getSource();
  return Value();
}

// Returns the first DPS hip-dialect user of `v` that writes into it (i.e.
// has `v` as one of its DPS init operands). Writes the matching init
// operand index into `*resultIdx`. Returns nullptr if no such user exists.
static Operation *findDirectDpsHipWriter(Value v, unsigned *resultIdx) {
  for (Operation *user : v.getUsers()) {
    auto dpsOp = llvm::dyn_cast<DestinationStyleOpInterface>(user);
    if (!dpsOp)
      continue;
    if (user->getDialect() !=
        user->getContext()->getLoadedDialect<HipDialect>())
      continue;
    auto inits = dpsOp.getDpsInits();
    for (auto [initIdx, initVal] : llvm::enumerate(inits)) {
      if (initVal == v) {
        if (resultIdx)
          *resultIdx = static_cast<unsigned>(initIdx);
        return user;
      }
    }
  }
  return nullptr;
}

Operation *findMemRefWriter(Value v, unsigned *resultIdx) {
  if (resultIdx)
    *resultIdx = 0;
  if (!v)
    return nullptr;
  // CRITICAL: search at EACH view step rather than walking all the way to
  // the underlying pool / alloc first. After bufferize + pool-allocs the
  // pool memref (`hip.get_pool`) is the source of every `memref.view`, and
  // the legacy "walk to the deepest source, then search users" strategy
  // saw only `memref.view` ops on the pool's user list -- no DPS writer.
  //
  // The Qwen embedding model exposes this: `hip.transpose` writes into
  // `%cast_5 = memref.cast %view_4` (where `%view_4 = memref.view %pool`),
  // and `hip.scatter_nd` reads the SAME `%cast_5` as its indices operand.
  // Searching `%cast_5.getUsers()` finds the transpose immediately; walking
  // back to `%pool` and searching ITS users misses the transpose entirely
  // and the scatter_nd's indices_shape[0] silently picks up the upper-bound
  // pool dim (524288) instead of the runtime-published NonZero slot (0).
  //
  // The per-step walk also subsumes the legacy aliasing fallback: if
  // writer and reader share a chain of casts above a single view, the
  // search finds the writer at the level where its DPS init operand
  // matches the cursor.
  Value cursor = v;
  while (cursor) {
    if (Operation *writer = findDirectDpsHipWriter(cursor, resultIdx))
      return writer;
    Value next = stepThroughView(cursor);
    if (!next || next == cursor)
      break;
    cursor = next;
  }
  // Fall back to the defining op of the deepest view-passthrough value
  // (tensor-mode IR or non-DPS producers).
  if (cursor)
    return cursor.getDefiningOp();
  return nullptr;
}

// Per-operand annotation gathered for one consumer op. `dim_slots`
// lists the (dim_idx, slot_id) pairs of every dynamic dim that
// resolves to a RuntimeSlot. `producer_slot_id` is the slot id of the
// op that actually WROTE the operand's buffer — set only when that
// op is a direct slot publisher (e.g. `hip.nonzero` itself, not an
// intermediate translucent op like `hip.transpose` that merely
// propagates the slot dim while writing its own DPS-init buffer).
//
// Why the distinction matters: consumer pointer rewiring requires
// `producer_slot_id` (the slot table holds the EXACT-size buffer the
// publisher allocated separately from its upper-bound DPS init).
// Consumer SHAPE rewiring uses `dim_slots` (the descriptor's dyn-dim
// sizes encode the upper-bound pool allocation, but the runtime
// dimension is what the slot publishes).
struct OperandSlotInfo {
  llvm::SmallVector<std::pair<int32_t, int32_t>, 2> dim_slots;
  int32_t producer_slot_id = -1;
};

// Returns the per-operand slot info. Empty `dim_slots` AND -1
// `producer_slot_id` for operands that have no dynamic dim or that
// resolve to no slot. Empty outer vector if `op` has no operands.
//
// Operand 0 of every Hip dialect op is the `!hip.context` arg by
// convention — it carries no shape and is excluded from the scan.
llvm::SmallVector<OperandSlotInfo, 4> collectInputDimSlots(Operation *op) {
  llvm::SmallVector<OperandSlotInfo, 4> perOperand(op->getNumOperands());
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    Value operand = op->getOperand(i);
    auto shaped = llvm::dyn_cast<ShapedType>(operand.getType());
    if (!shaped || !shaped.hasRank())
      continue;
    // Skip operands with no dynamic dims — they cannot carry slot info.
    bool anyDyn = false;
    for (int64_t d = 0; d < shaped.getRank(); ++d) {
      if (shaped.isDynamicDim(d)) {
        anyDyn = true;
        break;
      }
    }
    if (!anyDyn)
      continue;
    unsigned producerResultIdx = 0;
    Operation *producer = findMemRefWriter(operand, &producerResultIdx);
    if (!producer)
      continue;
    // If `producer` is itself a slot PUBLISHER — i.e. allocates an
    // EXACT-size output buffer at runtime and ignores its DPS-init
    // upper-bound — record the slot id so consumer lowerings can call
    // `hipdnn_ep_state_read_buffer(slot_id)` to get the real data
    // pointer instead of the (uninitialised) upper-bound DPS init.
    //
    // Critically, we ONLY honour DIRECT-publisher attributes
    // (`slot_id` for single-slot publishers like `hip.nonzero`, or
    // `slot_ids` for multi-axis publishers like `hip.constant_of_shape`).
    // We deliberately IGNORE `hipdnn.output_slot_ids` here:
    // `hipdnn.output_slot_ids` is set by `ReservePropagatorSlots` on
    // TRANSLUCENT propagators (e.g. `hip.transpose`, `hip.slice`) which
    // WROTE into their OWN DPS-init upper-bound buffer — the descriptor
    // pointer IS the right one and only the SHAPE needs slot lookup
    // (the dim_slots loop below). Confusing these two would route
    // consumers to a never-published slot buffer (segfault / "null
    // required argument") and silently corrupt the data path.
    auto pickProducerBufferSlot = [&](Operation *prod, unsigned r) -> int32_t {
      (void)r;
      if (auto arr = prod->getAttrOfType<DenseI32ArrayAttr>("slot_ids")) {
        for (int64_t d = 0; d < arr.size(); ++d) {
          int32_t s = arr.asArrayRef()[d];
          if (s >= 0)
            return s;
        }
        return -1;
      }
      if (auto slot = prod->getAttrOfType<IntegerAttr>("slot_id"))
        return (int32_t)slot.getInt();
      return -1;
    };
    int32_t prodSlot = pickProducerBufferSlot(producer, producerResultIdx);
    if (prodSlot >= 0)
      perOperand[i].producer_slot_id = prodSlot;
    // For each dynamic dim of the operand, query the DimSpec system.
    // If the spec root is a RuntimeSlot, record it. We deliberately
    // do NOT descend into arithmetic trees: the consumer-side
    // lowering only knows how to substitute a single slot read for
    // a single dim. A compound spec (e.g. mul(slot[0], slot[1]))
    // would need a more elaborate runtime evaluator (Phase 2.5).
    for (int64_t d = 0; d < shaped.getRank(); ++d) {
      if (!shaped.isDynamicDim(d))
        continue;
      DimSpec ds = shape_interface::getResultDimSpec(
          producer, producerResultIdx, (unsigned)d);
      if (ds.nodes().empty())
        continue;
      if (ds.root().kind != DimSpecKind::RuntimeSlot)
        continue;
      perOperand[i].dim_slots.push_back(
          {static_cast<int32_t>(d), ds.root().slot_id});
    }
  }
  return perOperand;
}

// Serialize the per-operand slot info into the `hipdnn.input_dim_slots`
// ArrayAttr. Each entry is itself an ArrayAttr of DenseI32ArrayAttr
// pairs `[dim_idx, slot_id]`. Returns null when EVERY operand entry
// is empty — caller should then skip the attribute set to keep IR
// uncluttered.
Attribute serializeInputDimSlots(
    MLIRContext *ctx, const llvm::SmallVector<OperandSlotInfo, 4> &perOperand) {
  bool anyNonEmpty = false;
  for (const auto &info : perOperand) {
    if (!info.dim_slots.empty()) {
      anyNonEmpty = true;
      break;
    }
  }
  if (!anyNonEmpty)
    return Attribute();
  Builder b(ctx);
  llvm::SmallVector<Attribute> outer;
  outer.reserve(perOperand.size());
  for (const auto &info : perOperand) {
    llvm::SmallVector<Attribute> inner;
    inner.reserve(info.dim_slots.size());
    for (auto [dimIdx, slotId] : info.dim_slots) {
      std::array<int32_t, 2> pair = {dimIdx, slotId};
      inner.push_back(b.getDenseI32ArrayAttr(pair));
    }
    outer.push_back(b.getArrayAttr(inner));
  }
  return b.getArrayAttr(outer);
}

// Serialize per-operand `producer_slot_id` into a DenseI32ArrayAttr
// (one i32 per operand; -1 = no slot publisher producer). Returns null
// when every entry is -1.
Attribute serializeInputSlotBuffers(
    MLIRContext *ctx, const llvm::SmallVector<OperandSlotInfo, 4> &perOperand) {
  bool anyNonNeg = false;
  for (const auto &info : perOperand) {
    if (info.producer_slot_id >= 0) {
      anyNonNeg = true;
      break;
    }
  }
  if (!anyNonNeg)
    return Attribute();
  Builder b(ctx);
  llvm::SmallVector<int32_t> values;
  values.reserve(perOperand.size());
  for (const auto &info : perOperand)
    values.push_back(info.producer_slot_id);
  return b.getDenseI32ArrayAttr(values);
}

class AnnotateInputDimSlotsPass
    : public impl::AnnotateInputDimSlotsPassBase<AnnotateInputDimSlotsPass> {
public:
  using impl::AnnotateInputDimSlotsPassBase<
      AnnotateInputDimSlotsPass>::AnnotateInputDimSlotsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main_graph");
    if (!mainFunc)
      return;
    MLIRContext *ctx = &getContext();

    auto *hipDialect = ctx->getLoadedDialect<HipDialect>();
    auto *memrefDialect = ctx->getLoadedDialect<memref::MemRefDialect>();
    mainFunc.walk([&](Operation *op) {
      // Annotate consumer ops whose lowering knows how to honour
      // `hipdnn.input_slot_buffers` / `hipdnn.input_dim_slots`. Today
      // that's every Hip-dialect op (via extractContiguousMemRefPtr-
      // WithSlot) plus `memref.copy` (special-cased in
      // MemRefCopyOpLowering so dynamic-shape Reshape outputs that
      // feed off a Cat-C publisher see the published slot pointer
      // instead of the elided 0-byte placeholder).
      bool isHipOp = op->getDialect() == hipDialect;
      bool isMemrefCopy = op->getDialect() == memrefDialect &&
                          op->getName().getStringRef() == "memref.copy";
      if (!isHipOp && !isMemrefCopy)
        return;
      // Skip the producer ops themselves (they already know their slot
      // via the `slot_id` attribute and DON'T consume slot-driven
      // dims). Without this guard, a Cat-C op whose own DPS init has
      // a dynamic dim that resolves to its OWN published slot would
      // get annotated with that slot — harmless but misleading and
      // wastes a runtime read.
      if (op->hasAttrOfType<IntegerAttr>("slot_id"))
        return;
      auto perOperand = collectInputDimSlots(op);
      if (Attribute dimsAttr = serializeInputDimSlots(ctx, perOperand))
        op->setAttr("hipdnn.input_dim_slots", dimsAttr);
      if (Attribute bufsAttr = serializeInputSlotBuffers(ctx, perOperand))
        op->setAttr("hipdnn.input_slot_buffers", bufsAttr);
    });
  }
};

} // namespace
} // namespace hip
} // namespace mlir
