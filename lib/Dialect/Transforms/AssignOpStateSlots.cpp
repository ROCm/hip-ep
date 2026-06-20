/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- AssignOpStateSlots.cpp - Per-instance op-state slot assignment -----===//
//
// Identity half of the op-state-slots design (see
// docs/design/op-state-slots-design.md). Walks the module and gives EACH
// instance of an op implementing `OpStateOpInterface` its own dense slot
// `0..N-1`, stamping `hip.op_state_slot` on the op and recording the total
// count as the module attribute `hipdnn.num_op_state_slots`. Two operators of
// the same kind get two slots and (later) two independent state objects;
// per-class deduplication is intentionally NOT done here (it would be an
// optimization layered on top, not the default).
//
// Runs before --convert-hip-to-llvm so the slot index is available to both the
// lowering (threaded into each op's wrap_* call) and --generate-op-state-init
// (which consumes the count + per-op interface to build the init function).
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "assign-op-state-slots"

STATISTIC(NumSlotsAssigned,
          "Number of op-state slots assigned by --assign-op-state-slots");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_ASSIGNOPSTATESLOTSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct AssignOpStateSlotsPass
    : public impl::AssignOpStateSlotsPassBase<AssignOpStateSlotsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    Builder builder(&getContext());

    int32_t nextSlot = 0;
    // Pre-order module walk is deterministic, so slots follow program order:
    // one slot per stateful op instance, no per-class grouping.
    module.walk([&](Operation *op) {
      if (!isa<OpStateOpInterface>(op))
        return;
      op->setAttr("hip.op_state_slot", builder.getI32IntegerAttr(nextSlot));
      ++nextSlot;
      ++NumSlotsAssigned;
    });

    // Only stamp the count when there is at least one stateful op, so models
    // without any op-state ops are byte-identical to before this pass existed
    // (and --generate-op-state-init / generate-interface stay no-ops).
    if (nextSlot > 0)
      module->setAttr("hipdnn.num_op_state_slots",
                      builder.getI32IntegerAttr(nextSlot));
  }
};

} // namespace
} // namespace hip
} // namespace mlir
