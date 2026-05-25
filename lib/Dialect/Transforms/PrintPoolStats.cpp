/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PrintPoolStats.cpp - Footprint diagnostic for slot-coalesce phases ===//
//
// Analysis-only pass that emits before/after slot-buffer-coalesce
// footprint metrics to stderr. Used by hand when tuning a model or as
// the validation pivot for the slot-buffer-coalesce design's per-phase
// regression assertions.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeInterface.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdint>

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_PRINTPOOLSTATSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

class PrintPoolStatsPass
    : public impl::PrintPoolStatsPassBase<PrintPoolStatsPass> {
public:
  using impl::PrintPoolStatsPassBase<
      PrintPoolStatsPass>::PrintPoolStatsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto &os = llvm::errs();

    int64_t poolBytes = -1;
    if (auto ps = module->getAttrOfType<IntegerAttr>("hipdnn.pool_size"))
      poolBytes = ps.getInt();
    int64_t dynSlotsCount = 0;
    if (auto ds =
            module->getAttrOfType<IntegerAttr>("hipdnn.dyn_dim_slots_count"))
      dynSlotsCount = ds.getInt();
    int64_t nextDynSlotId = 0;
    if (auto ds = module->getAttrOfType<IntegerAttr>("hipdnn.next_dyn_slot_id"))
      nextDynSlotId = ds.getInt();
    int64_t outputDimSpecCount = 0;
    if (auto outDS =
            module->getAttrOfType<ArrayAttr>("hipdnn.output_dim_specs")) {
      for (Attribute resA : outDS) {
        if (auto perResult = llvm::dyn_cast<ArrayAttr>(resA))
          outputDimSpecCount += (int64_t)perResult.size();
      }
    }

    // `@main_graph` is renamed to `@main_graph_internal` by
    // GenerateInterface; try both so the pass is useful at any
    // point in the pipeline.
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main_graph");
    if (!mainFunc)
      mainFunc = module.lookupSymbol<func::FuncOp>("main_graph_internal");
    if (!mainFunc) {
      os << "[pool-stats] (no @main_graph / @main_graph_internal found)\n"
         << "  static_pool_bytes = " << poolBytes << "\n"
         << "  dyn_dim_slots_count = " << dynSlotsCount << "\n"
         << "  next_dyn_slot_id = " << nextDynSlotId << "\n"
         << "  output_dim_specs.count = " << outputDimSpecCount << "\n";
      return;
    }

    int64_t identityHits = 0;
    int64_t publisherCount = 0;
    auto *hipDialect = module->getContext()->getLoadedDialect<HipDialect>();
    mainFunc.walk([&](Operation *op) {
      if (op == mainFunc.getOperation())
        return;
      if (op->getDialect() != hipDialect)
        return;
      if (op->getAttrOfType<IntegerAttr>("slot_id") ||
          op->getAttrOfType<DenseI32ArrayAttr>("slot_ids") ||
          op->getAttrOfType<ArrayAttr>("hipdnn.output_slot_ids"))
        ++publisherCount;
      if (shape_interface::isIdentityOp(op))
        ++identityHits;
    });

    os << "[pool-stats] func=main_graph\n"
       << "  static_pool_bytes = " << poolBytes << "\n"
       << "  dyn_dim_slots_count = " << dynSlotsCount << "\n"
       << "  next_dyn_slot_id = " << nextDynSlotId << "\n"
       << "  output_dim_specs.count = " << outputDimSpecCount << "\n"
       << "  identity_propagator_predicate_hits = " << identityHits << "\n"
       << "  slot_publisher_count = " << publisherCount << "\n";
  }
};

} // namespace
} // namespace hip
} // namespace mlir
