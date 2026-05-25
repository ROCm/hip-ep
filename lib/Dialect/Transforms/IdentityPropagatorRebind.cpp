/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- IdentityPropagatorRebind.cpp - Erase identity propagators ----------===//
//
// Phase 4 of the slot-buffer-coalesce design
// (docs/design/slot-buffer-coalesce.md).
//
// For propagator ops whose runtime behaviour collapses to a copy of
// their input (e.g. `hip.transpose` with perm = [0..rank),
// `hip.cast` from f32 -> f32, `hip.slice` over the entire input),
// the kernel launch is wasted work. This pass:
//
//   1. Walks @main_graph in program order.
//   2. For each op with a registered identity predicate
//      (`shape_interface::isIdentityOp`), evaluates the predicate.
//   3. When the predicate is true:
//      a. Records the equivalence `output_slot_id -> input_slot_id`
//         when both sides have slots (so a follow-up Phase 3
//         coalescer collapses them into the upstream slot, or the
//         post-pass remap below does it inline).
//      b. RAUW the op's result with its input, then erases the op.
//   4. After the walk, applies the equivalence map to every slot id
//      reference in the module so consumer attrs / DimSpec leaves /
//      module-level metadata see the renumbered slot ids.
//
// When the input has NO slot (e.g. a propagator immediately
// downstream of a static-shape value), the op is still elided and
// its output slot survives unchanged (the upstream pass that
// reserved it will dead-code-eliminate later, OR the Phase 3
// coalescer will keep it as the only member of its bin and renumber
// to a contiguous range).
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeInterface.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdint>

#define DEBUG_TYPE "hip-identity-propagator-rebind"

STATISTIC(NumOpsElided, "Number of identity-propagator ops erased");
STATISTIC(NumSlotsRebound,
          "Number of output slot ids rebound to the upstream input slot");
STATISTIC(NumRefsRewritten,
          "Number of slot id references rewritten after rebinding");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_IDENTITYPROPAGATORREBINDPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers — read the producer's first published slot id for a value.
//===----------------------------------------------------------------------===//

// Try to find the slot id published by `v`'s producer for `v`'s
// dim 0 dynamic entry. We use dim 0 as the proxy because every
// translucent Cat-C propagator we erase here is single-result (the
// op classes covered by the identity predicate registry never
// produce more than one result), and the propagator's output slot
// grid lives on result index 0. Returns -1 when there's no slot.
//
// Walks both Phase 2's array-form attribute (`hipdnn.output_slot_ids`)
// and the legacy scalar / array `slot_id` / `slot_ids` attributes so
// we cover Cat-C publishers (NonZero, Range, ConstantOfShape) as
// well as translucent propagators reservation done by
// `ReservePropagatorSlotsPass`.
int32_t findProducerSlotForValue(Value v) {
  Operation *producer = v.getDefiningOp();
  if (!producer)
    return -1;
  // result index of `v` within `producer`.
  unsigned ri = 0;
  for (unsigned i = 0, e = producer->getNumResults(); i < e; ++i) {
    if (producer->getResult(i) == v) {
      ri = i;
      break;
    }
  }
  if (auto outer =
          producer->getAttrOfType<ArrayAttr>("hipdnn.output_slot_ids")) {
    if (ri < outer.size()) {
      if (auto pr = llvm::dyn_cast<DenseI32ArrayAttr>(outer[ri])) {
        for (int32_t s : pr.asArrayRef()) {
          if (s >= 0)
            return s;
        }
      }
    }
  }
  if (auto slotIds = producer->getAttrOfType<DenseI32ArrayAttr>("slot_ids")) {
    if (ri == 0) {
      for (int32_t s : slotIds.asArrayRef())
        if (s >= 0)
          return s;
    }
  }
  if (auto slot = producer->getAttrOfType<IntegerAttr>("slot_id")) {
    if (ri == 0)
      return (int32_t)slot.getInt();
  }
  return -1;
}

// Find the first slot id this op publishes (across its output_slot_ids
// grid or the legacy scalar / array attrs). Returns -1 when the op
// doesn't carry a slot.
int32_t findFirstOwnSlot(Operation *op) {
  if (auto outer = op->getAttrOfType<ArrayAttr>("hipdnn.output_slot_ids")) {
    for (Attribute resA : outer) {
      if (auto pr = llvm::dyn_cast<DenseI32ArrayAttr>(resA)) {
        for (int32_t s : pr.asArrayRef())
          if (s >= 0)
            return s;
      }
    }
  }
  if (auto slotIds = op->getAttrOfType<DenseI32ArrayAttr>("slot_ids")) {
    for (int32_t s : slotIds.asArrayRef())
      if (s >= 0)
        return s;
  }
  if (auto slot = op->getAttrOfType<IntegerAttr>("slot_id"))
    return (int32_t)slot.getInt();
  return -1;
}

//===----------------------------------------------------------------------===//
// Remap helpers — substitute every slot id on every slot-bearing attribute.
// (mirrors the helpers in SlotLifetimeCoalesce.cpp but kept local so the
// two passes stay independent.)
//===----------------------------------------------------------------------===//

Attribute remapDenseI32Array(MLIRContext *ctx, DenseI32ArrayAttr attr,
                             const DenseMap<int32_t, int32_t> &remap,
                             uint64_t &rewriteCount) {
  if (!attr)
    return attr;
  Builder b(ctx);
  SmallVector<int32_t> out;
  out.reserve(attr.size());
  bool changed = false;
  for (int32_t s : attr.asArrayRef()) {
    if (s >= 0) {
      auto it = remap.find(s);
      if (it != remap.end() && it->second != s) {
        out.push_back(it->second);
        ++rewriteCount;
        changed = true;
        continue;
      }
    }
    out.push_back(s);
  }
  if (!changed)
    return attr;
  return b.getDenseI32ArrayAttr(out);
}

Attribute remapInputDimSlotsAttr(MLIRContext *ctx, ArrayAttr attr,
                                 const DenseMap<int32_t, int32_t> &remap,
                                 uint64_t &rewriteCount) {
  if (!attr)
    return attr;
  Builder b(ctx);
  SmallVector<Attribute> outer;
  outer.reserve(attr.size());
  bool changed = false;
  for (Attribute opAttr : attr) {
    auto perOperand = llvm::dyn_cast<ArrayAttr>(opAttr);
    if (!perOperand) {
      outer.push_back(opAttr);
      continue;
    }
    SmallVector<Attribute> inner;
    inner.reserve(perOperand.size());
    for (Attribute pairAttr : perOperand) {
      auto pair = llvm::dyn_cast<DenseI32ArrayAttr>(pairAttr);
      if (!pair || pair.size() != 2) {
        inner.push_back(pairAttr);
        continue;
      }
      int32_t dim = pair.asArrayRef()[0];
      int32_t slot = pair.asArrayRef()[1];
      if (slot >= 0) {
        auto it = remap.find(slot);
        if (it != remap.end() && it->second != slot) {
          slot = it->second;
          ++rewriteCount;
          changed = true;
        }
      }
      std::array<int32_t, 2> repl = {dim, slot};
      inner.push_back(b.getDenseI32ArrayAttr(repl));
    }
    outer.push_back(b.getArrayAttr(inner));
  }
  if (!changed)
    return attr;
  return b.getArrayAttr(outer);
}

Attribute remapOutputSlotIdsAttr(MLIRContext *ctx, ArrayAttr attr,
                                 const DenseMap<int32_t, int32_t> &remap,
                                 uint64_t &rewriteCount) {
  if (!attr)
    return attr;
  Builder b(ctx);
  SmallVector<Attribute> outer;
  outer.reserve(attr.size());
  bool changed = false;
  for (Attribute resAttr : attr) {
    auto perResult = llvm::dyn_cast<DenseI32ArrayAttr>(resAttr);
    if (!perResult) {
      outer.push_back(resAttr);
      continue;
    }
    Attribute newAttr = remapDenseI32Array(ctx, perResult, remap, rewriteCount);
    if (newAttr != Attribute(perResult))
      changed = true;
    outer.push_back(newAttr);
  }
  if (!changed)
    return attr;
  return b.getArrayAttr(outer);
}

Attribute remapDimSpecArrayAttr(MLIRContext *ctx, ArrayAttr attr,
                                const DenseMap<int32_t, int32_t> &remap,
                                uint64_t &rewriteCount) {
  if (!attr)
    return attr;
  Builder b(ctx);
  SmallVector<Attribute> nodeAttrs;
  nodeAttrs.reserve(attr.size());
  bool changed = false;
  for (Attribute nodeAttr : attr) {
    auto fields = llvm::dyn_cast<DenseI64ArrayAttr>(nodeAttr);
    if (!fields || fields.size() < 8) {
      nodeAttrs.push_back(nodeAttr);
      continue;
    }
    SmallVector<int64_t> vals(fields.asArrayRef().begin(),
                              fields.asArrayRef().end());
    if (vals[0] == 3 && vals[5] >= 0) {
      auto it = remap.find((int32_t)vals[5]);
      if (it != remap.end() && it->second != vals[5]) {
        vals[5] = it->second;
        changed = true;
        ++rewriteCount;
      }
    }
    nodeAttrs.push_back(b.getDenseI64ArrayAttr(vals));
  }
  if (!changed)
    return attr;
  return b.getArrayAttr(nodeAttrs);
}

Attribute remapOutputDimSpecsAttr(MLIRContext *ctx, ArrayAttr attr,
                                  const DenseMap<int32_t, int32_t> &remap,
                                  uint64_t &rewriteCount) {
  if (!attr)
    return attr;
  Builder b(ctx);
  SmallVector<Attribute> resEntries;
  resEntries.reserve(attr.size());
  bool changed = false;
  for (Attribute resA : attr) {
    auto perResult = llvm::dyn_cast<ArrayAttr>(resA);
    if (!perResult) {
      resEntries.push_back(resA);
      continue;
    }
    SmallVector<Attribute> dimEntries;
    dimEntries.reserve(perResult.size());
    for (Attribute dimA : perResult) {
      auto perDim = llvm::dyn_cast<ArrayAttr>(dimA);
      if (!perDim) {
        dimEntries.push_back(dimA);
        continue;
      }
      Attribute newPerDim =
          remapDimSpecArrayAttr(ctx, perDim, remap, rewriteCount);
      if (newPerDim != Attribute(perDim))
        changed = true;
      dimEntries.push_back(newPerDim);
    }
    resEntries.push_back(b.getArrayAttr(dimEntries));
  }
  if (!changed)
    return attr;
  return b.getArrayAttr(resEntries);
}

//===----------------------------------------------------------------------===//
// Pass entry
//===----------------------------------------------------------------------===//

class IdentityPropagatorRebindPass
    : public impl::IdentityPropagatorRebindPassBase<
          IdentityPropagatorRebindPass> {
public:
  using impl::IdentityPropagatorRebindPassBase<
      IdentityPropagatorRebindPass>::IdentityPropagatorRebindPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    auto *hipDialect = ctx->getLoadedDialect<HipDialect>();
    if (!hipDialect)
      return;

    if (disableIdentityRebind)
      return;

    //
    // Phase A: collect ops to elide. Walk top-down so we visit
    // upstream ops before their downstream consumers. Each entry
    // records the op + its output slot id (or -1) + its input
    // slot id (or -1). We walk every `func.func` in the module so
    // the pass is usable both inside the production pipeline (where
    // the relevant func is `@main_graph`) and standalone from
    // `hip-mlir-opt` with test funcs of any name.
    //
    struct ToRebind {
      Operation *op;
      Value input;
      int32_t outSlot;
      int32_t inSlot;
    };
    SmallVector<ToRebind> toRebind;

    module.walk([&](Operation *op) {
      if (op->getDialect() != hipDialect)
        return;
      if (op->getNumResults() != 1)
        return;
      if (!shape_interface::isIdentityOp(op))
        return;
      // Identity propagators are single-input (the ctx operand at
      // index 0 is mandatory on every Hip_DpsOp). We treat operand
      // index 1 as "the input" which matches the slot-bearing
      // operand on every op class with a registered identity
      // predicate (transpose / cast / expand / slice / tile /
      // reduce_*). Defensive guard for unusual ops.
      if (op->getNumOperands() < 2)
        return;
      Value input = op->getOperand(1);
      if (input.getType() != op->getResult(0).getType()) {
        // The op's input and output MLIR types differ -- conservative
        // bail (e.g. some propagators advertise the same byte
        // semantics but differ in tensor/memref or memory space).
        // RAUW would type-check fail otherwise.
        return;
      }
      int32_t outSlot = findFirstOwnSlot(op);
      int32_t inSlot = findProducerSlotForValue(input);
      toRebind.push_back({op, input, outSlot, inSlot});
    });

    if (toRebind.empty())
      return;

    //
    // Phase B: build the output-slot -> input-slot remap. Only entries
    // where BOTH sides have a real slot id contribute (-1 cases are
    // left alone: the output keeps its own slot, the op is still
    // elided, and a later Phase 3 coalesce + the static-shape pool
    // packer will handle the now-zero-use slot).
    //
    DenseMap<int32_t, int32_t> remap;
    for (const ToRebind &r : toRebind) {
      if (r.outSlot < 0 || r.inSlot < 0)
        continue;
      if (r.outSlot == r.inSlot)
        continue;
      // Chase the chain: if outSlot was already remapped (e.g. the
      // chain has two identity ops in a row), follow it so the
      // final image points at the deepest input.
      int32_t dst = r.inSlot;
      while (true) {
        auto it = remap.find(dst);
        if (it == remap.end() || it->second == dst)
          break;
        dst = it->second;
      }
      remap[r.outSlot] = dst;
      ++NumSlotsRebound;
    }

    //
    // Phase C: erase the identity ops + RAUW their result with their
    // input. We do this in REVERSE order so an op whose result feeds
    // a later identity op gets erased after that downstream op has
    // already been rewritten -- otherwise the RAUW would update the
    // downstream op's operand to point at the upstream input, then
    // the downstream op would also try to erase itself, picking up
    // the new operand correctly. Either order works; reverse is
    // cleaner because it keeps the def-use graph valid at every
    // intermediate step.
    //
    for (auto it = toRebind.rbegin(); it != toRebind.rend(); ++it) {
      it->op->getResult(0).replaceAllUsesWith(it->input);
      it->op->erase();
      ++NumOpsElided;
    }

    //
    // Phase D: substitute the remap on every slot-bearing attribute
    // in the module. Re-uses the helper pattern from
    // SlotLifetimeCoalesce.cpp.
    //
    if (remap.empty())
      return;
    uint64_t rewrites = 0;
    module.walk([&](Operation *op) {
      if (auto a = op->getAttrOfType<IntegerAttr>("slot_id")) {
        int32_t s = (int32_t)a.getInt();
        if (s >= 0) {
          auto it = remap.find(s);
          if (it != remap.end() && it->second != s) {
            op->setAttr("slot_id", IntegerAttr::get(IntegerType::get(ctx, 32),
                                                    it->second));
            ++rewrites;
          }
        }
      }
      if (auto a = op->getAttrOfType<DenseI32ArrayAttr>("slot_ids")) {
        Attribute newAttr = remapDenseI32Array(ctx, a, remap, rewrites);
        if (newAttr != Attribute(a))
          op->setAttr("slot_ids", newAttr);
      }
      if (auto a = op->getAttrOfType<ArrayAttr>("hipdnn.output_slot_ids")) {
        Attribute newAttr = remapOutputSlotIdsAttr(ctx, a, remap, rewrites);
        if (newAttr != Attribute(a))
          op->setAttr("hipdnn.output_slot_ids", newAttr);
      }
      if (auto a = op->getAttrOfType<ArrayAttr>("output_dim_specs")) {
        Attribute newAttr = remapOutputDimSpecsAttr(ctx, a, remap, rewrites);
        if (newAttr != Attribute(a))
          op->setAttr("output_dim_specs", newAttr);
      }
      if (auto a = op->getAttrOfType<ArrayAttr>("hipdnn.input_dim_slots")) {
        Attribute newAttr = remapInputDimSlotsAttr(ctx, a, remap, rewrites);
        if (newAttr != Attribute(a))
          op->setAttr("hipdnn.input_dim_slots", newAttr);
      }
      if (auto a = op->getAttrOfType<DenseI32ArrayAttr>(
              "hipdnn.input_slot_buffers")) {
        Attribute newAttr = remapDenseI32Array(ctx, a, remap, rewrites);
        if (newAttr != Attribute(a))
          op->setAttr("hipdnn.input_slot_buffers", newAttr);
      }
    });
    if (auto a = module->getAttrOfType<ArrayAttr>("hipdnn.output_dim_specs")) {
      Attribute newAttr = remapOutputDimSpecsAttr(ctx, a, remap, rewrites);
      if (newAttr != Attribute(a))
        module->setAttr("hipdnn.output_dim_specs", newAttr);
    }
    NumRefsRewritten = rewrites;
  }
};

} // namespace
} // namespace hip
} // namespace mlir
