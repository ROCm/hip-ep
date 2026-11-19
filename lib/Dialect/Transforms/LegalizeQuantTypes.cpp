/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip-legalize-quant-types: quantized element types -> `quant_types` attribute
// plus storage-integer tensors, so the IR can be bufferized.

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipQuantTypes.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

namespace mlir::hip {
#define GEN_PASS_DEF_LEGALIZEQUANTTYPESPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

bool hasQuantizedType(Operation *op) {
  auto quantized = [](Type type) { return isQuantized(type); };
  return llvm::any_of(op->getOperandTypes(), quantized) ||
         llvm::any_of(op->getResultTypes(), quantized);
}

struct LegalizeQuantTypesPass
    : public impl::LegalizeQuantTypesPassBase<LegalizeQuantTypesPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    Builder builder(&getContext());

    // 1. Record the element types on each quant-capable op, float or
    //    quantized, so lowering reads one place to pick the kernel.
    WalkResult result = func.walk([&](Operation *op) {
      auto quantOp = dyn_cast<HipQuantCapableOpInterface>(op);
      if (!quantOp) {
        if (!isa<StorageCastOp>(op) &&
            llvm::isa<DestinationStyleOpInterface>(op) &&
            hasQuantizedType(op)) {
          op->emitOpError("has quantized types but is not quant-capable");
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      }
      // Already legalized: the tensors no longer carry the quant types.
      if (op->hasAttr("quant_types"))
        return WalkResult::advance();

      // No `quant_types` yet, so these are the tensor element types.
      SmallVector<Type> types = quantOp.getOperandQuantTypes();
      if (hasQuantizedType(op) &&
          !quantOp.supportsQuantTypes(ArrayRef(types).drop_back(),
                                      types.back())) {
        // Known gap: the Q/DQ folds do not check kernels, so a typed op that
        // was not fused into one with a kernel ends here (docs/design/
        // quant-types.md, "Known gaps"). No fallback to DQ -> op -> Q.
        InFlightDiagnostic diag =
            op->emitOpError("has no kernel for its quantized types (");
        llvm::interleaveComma(ArrayRef(types).drop_back(), diag);
        diag << " -> " << types.back() << ")";
        return WalkResult::interrupt();
      }
      op->setAttr("quant_types", builder.getTypeArrayAttr(types));
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();

    // 2. Retype every quantized value to its storage integer.
    func.walk([&](Operation *op) {
      for (Value value : op->getResults())
        value.setType(toHipStorageType(value.getType()));
      for (Region &region : op->getRegions())
        for (BlockArgument arg : region.getArguments())
          arg.setType(toHipStorageType(arg.getType()));
    });

    // 3. Storage casts are now identities.
    func.walk([&](StorageCastOp cast) {
      cast.getResult().replaceAllUsesWith(cast.getInput());
      cast.erase();
    });

    // Postcondition: nothing quantized reaches bufferization.
    result = func.walk([&](Operation *op) {
      if (!hasQuantizedType(op))
        return WalkResult::advance();
      op->emitOpError("still has a quantized type after "
                      "hip-legalize-quant-types");
      return WalkResult::interrupt();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::hip
