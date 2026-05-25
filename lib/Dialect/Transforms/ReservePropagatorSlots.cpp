/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReservePropagatorSlots.cpp - Reserve slots for Phase 2 propagators -===//
//
// Phase 2 of the slot-buffer-coalescing initiative
// (docs/design/slot-buffer-coalesce.md).
//
// Walks every `Hip_DpsOp` in `@main_graph` whose result has a dynamic dim
// that transitively depends on a Cat-C `RuntimeSlot` (i.e. a translucent
// propagator). Reserves one fresh slot id per such dim from the module-
// level `hipdnn.next_dyn_slot_id` counter, and records the per-result-
// per-dim slot map in the array-form attribute
// `hipdnn.output_slot_ids` (-1 means "no slot for this dim").
//
// Ops that already carry a slot_id / slot_ids / hipdnn.output_slot_ids
// (Cat-C publishers) are left alone. Ops whose dynamic dims resolve to
// non-RuntimeSlot DimSpecs (Cat-A / Cat-B / static / arithmetic over
// those) are also skipped -- they don't need a slot because their
// downstream consumer can resolve the dim from the operand-relative
// DimSpec via the existing AnnotateInputDimSlotsPass path.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeInterface.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-reserve-propagator-slots"

STATISTIC(NumPropagatorsAnnotated,
          "Number of translucent propagators given a hipdnn.output_slot_ids "
          "attribute");
STATISTIC(NumSlotsReserved,
          "Total number of slot ids reserved for propagator dynamic dims");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_RESERVEPROPAGATORSLOTSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Returns true when `ds` has at least one RuntimeSlot leaf anywhere in
// its expression tree.
bool dimSpecTransitivelyDependsOnRuntimeSlot(const DimSpec &ds) {
  if (ds.nodes().empty())
    return false;
  return ds.needsRuntimeSlot();
}

class ReservePropagatorSlotsPass
    : public impl::ReservePropagatorSlotsPassBase<ReservePropagatorSlotsPass> {
public:
  using impl::ReservePropagatorSlotsPassBase<
      ReservePropagatorSlotsPass>::ReservePropagatorSlotsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    auto *hipDialect = ctx->getLoadedDialect<HipDialect>();
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main_graph");
    if (!mainFunc)
      return;

    // Pull the current per-module slot counter (initialised by Cat-C
    // converters). Any propagator reservation we make extends this
    // counter.
    int32_t nextSlot = 0;
    if (auto a = module->getAttrOfType<IntegerAttr>("hipdnn.next_dyn_slot_id"))
      nextSlot = (int32_t)a.getInt();

    Builder b(ctx);

    mainFunc.walk([&](Operation *op) {
      // Limit scope to the hip dialect -- only Hip_DpsOps participate.
      if (op->getDialect() != hipDialect)
        return;
      // Cat-C publishers were already annotated at conversion time.
      if (op->hasAttr("slot_id") || op->hasAttr("slot_ids") ||
          op->hasAttr("hipdnn.output_slot_ids"))
        return;
      const unsigned numResults = op->getNumResults();
      if (numResults == 0)
        return;
      // Quick scan: any dynamic dim on any result?
      bool anyDyn = false;
      for (unsigned r = 0; r < numResults && !anyDyn; ++r) {
        auto rt = llvm::dyn_cast<ShapedType>(op->getResult(r).getType());
        if (!rt || !rt.hasRank())
          continue;
        for (int64_t d = 0; d < rt.getRank(); ++d) {
          if (rt.isDynamicDim(d)) {
            anyDyn = true;
            break;
          }
        }
      }
      if (!anyDyn)
        return;

      // Build the slot grid. For each result, allocate a
      // DenseI32ArrayAttr of length == rank, filled with -1 by default
      // and overwritten with a fresh slot id wherever the dim resolves
      // to a DimSpec containing a RuntimeSlot leaf.
      SmallVector<Attribute> outerGrid;
      outerGrid.reserve(numResults);
      bool anyReserved = false;
      for (unsigned r = 0; r < numResults; ++r) {
        auto rt = llvm::dyn_cast<ShapedType>(op->getResult(r).getType());
        unsigned rank = (rt && rt.hasRank()) ? (unsigned)rt.getRank() : 0;
        SmallVector<int32_t> perDim(rank, -1);
        for (unsigned d = 0; d < rank; ++d) {
          if (!rt.isDynamicDim(d))
            continue;
          DimSpec ds = shape_interface::getResultDimSpec(op, r, d);
          if (!dimSpecTransitivelyDependsOnRuntimeSlot(ds))
            continue;
          perDim[d] = nextSlot++;
          anyReserved = true;
        }
        outerGrid.push_back(b.getDenseI32ArrayAttr(perDim));
      }
      if (!anyReserved)
        return;

      op->setAttr("hipdnn.output_slot_ids", b.getArrayAttr(outerGrid));
      ++NumPropagatorsAnnotated;
      LLVM_DEBUG(llvm::dbgs()
                 << "  reserved propagator slot grid for " << *op << "\n");
    });

    // Persist the bumped counter so any later pass (e.g. another
    // converter run, lowering metadata emit) sees the new high water.
    int32_t prev = 0;
    if (auto a = module->getAttrOfType<IntegerAttr>("hipdnn.next_dyn_slot_id"))
      prev = (int32_t)a.getInt();
    if (nextSlot > prev) {
      module->setAttr("hipdnn.next_dyn_slot_id", b.getI32IntegerAttr(nextSlot));
      NumSlotsReserved += (uint64_t)(nextSlot - prev);
    }
  }
};

} // namespace
} // namespace hip
} // namespace mlir
