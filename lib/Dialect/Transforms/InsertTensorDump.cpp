/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// InsertTensorDump Pass
//===----------------------------------------------------------------------===//
// Inserts hip.dump_tensor ops after every HIP compute op so that each output
// tensor is copied from GPU to host and saved as a NumPy .npy file.
//
// The pass runs after bufferization (all tensors are memrefs) and before
// convert-hip-to-llvm.  It is conditionally added to the pipeline when the
// dump_tensors compilation option is enabled.
//
// After bufferization, HIP DPS ops have zero SSA results — the output is
// written into the DPS init operand (an in-place memref).  We use the
// DestinationStyleOpInterface to find these output operands.
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_INSERTTENSORDUMPPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

static std::string buildDumpName(Operation *op, int index) {
  if (auto nameLoc = dyn_cast<NameLoc>(op->getLoc())) {
    std::string name = nameLoc.getName().str();
    if (!name.empty())
      return name;
  }
  if (auto fusedLoc = dyn_cast<FusedLoc>(op->getLoc())) {
    for (Location inner : fusedLoc.getLocations()) {
      if (auto nameLoc = dyn_cast<NameLoc>(inner)) {
        std::string name = nameLoc.getName().str();
        if (!name.empty())
          return name;
      }
    }
  }
  std::string mnemonic = op->getName().getStringRef().str();
  std::replace(mnemonic.begin(), mnemonic.end(), '.', '_');
  return mnemonic + "_" + std::to_string(index);
}

/// Collect the memref values that represent the "outputs" of a HIP compute op.
/// After bufferization DPS ops have no SSA results; the outputs are the DPS
/// init operands.  For non-DPS ops that still produce SSA results (e.g.
/// hip.get_constant) we fall back to getResults().
static SmallVector<Value> getOutputMemRefs(Operation *op) {
  SmallVector<Value> outputs;

  if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
    for (OpOperand &init : dps.getDpsInitsMutable()) {
      if (isa<MemRefType>(init.get().getType()))
        outputs.push_back(init.get());
    }
  }

  if (outputs.empty()) {
    for (Value result : op->getResults()) {
      if (isa<MemRefType>(result.getType()))
        outputs.push_back(result);
    }
  }
  return outputs;
}

class InsertTensorDumpPass
    : public impl::InsertTensorDumpPassBase<InsertTensorDumpPass> {
public:
  using InsertTensorDumpPassBase::InsertTensorDumpPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    int opIndex = 0;

    module.walk([&](Operation *op) {
      auto *dialect = op->getDialect();
      if (!dialect ||
          dialect->getNamespace() != hip::HipDialect::getDialectNamespace())
        return;

      if (isa<hip::AllocOp, hip::FreeOp, hip::GetConstantOp, hip::GetPoolOp,
              hip::YieldOp, hip::DumpTensorOp>(op))
        return;

      if (op->getNumOperands() == 0)
        return;
      Value ctx = op->getOperand(0);
      if (!isa<hip::ContextType>(ctx.getType()))
        return;

      std::string baseName = buildDumpName(op, opIndex++);
      SmallVector<Value> outputs = getOutputMemRefs(op);
      if (outputs.empty())
        return;

      OpBuilder builder(op->getContext());
      builder.setInsertionPointAfter(op);

      for (auto [idx, outVal] : llvm::enumerate(outputs)) {
        std::string name = baseName;
        if (outputs.size() > 1)
          name += "_out" + std::to_string(idx);

        hip::DumpTensorOp::create(builder, op->getLoc(), ctx, outVal,
                                  builder.getStringAttr(name),
                                  builder.getStringAttr(dumpTensorsDir));
      }
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createInsertTensorDumpPass(llvm::StringRef dir) {
  InsertTensorDumpPassOptions opts;
  opts.dumpTensorsDir = dir.str();
  return std::make_unique<InsertTensorDumpPass>(opts);
}

} // namespace hip
} // namespace mlir
