/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- GenerateOpStateInit.cpp - Emit the op-state init function ----------===//
//
// Construction half of the op-state-slots design (see
// docs/design/op-state-slots-design.md). Runs AFTER --assign-op-state-slots and
// BEFORE --convert-hip-to-llvm, while the HIP ops still exist, and emits a
// standalone LLVM function:
//
//   llvm.func @hipdnn_ep_op_states_init_fn(%state: !llvm.ptr) -> i32 {
//     %ok = llvm.call @hipdnn_ep_op_states_alloc(%state, N)   // alloc array
//     llvm.cond_br %ok, ^construct, ^fail   // skip construction if alloc
//     failed
//   ^construct:
//     // per stateful op, in slot order:
//     %st = <op.generateOpStateInit emits: call construct_<op>(%state, attrs)>
//     llvm.call @hipdnn_ep_op_state_set(%state, slot, %st)  // -> slot
//     llvm.return 0
//   ^fail:
//     llvm.return 1   // nothing constructed, so nothing to free
//   }
//
// --generate-interface later calls this from inference_init. Building the init
// here (not in generate-interface) is what lets each op contribute its own
// construction code via the interface: by the time generate-interface runs,
// the HIP ops are already lowered to wrap_* calls and the interface is gone.
// This mirrors the memory pool's count-then-consume shape.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/artifact_abi.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_GENERATEOPSTATEINITPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct GenerateOpStateInitPass
    : public impl::GenerateOpStateInitPassBase<GenerateOpStateInitPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto countAttr =
        module->getAttrOfType<IntegerAttr>("hipdnn.num_op_state_slots");
    if (!countAttr || countAttr.getInt() <= 0)
      return; // No stateful ops; nothing to generate.
    int64_t numSlots = countAttr.getInt();

    // Place every stateful op directly at its assigned slot. --assign-op-state-
    // slots (which sets hipdnn.num_op_state_slots, checked above) stamps every
    // OpStateOpInterface op with a dense slot in [0, numSlots), so indexing by
    // slot orders the emit deterministically without a sort. We guard only the
    // index itself: a stray slot would be an out-of-bounds write, whereas a
    // missing attr or gap can only come from a compiler bug and faults loudly.
    SmallVector<Operation *> bySlot(numSlots, nullptr);
    bool slotOutOfRange = false;
    module.walk([&](Operation *op) {
      if (!isa<OpStateOpInterface>(op))
        return;
      int64_t slot =
          op->getAttrOfType<IntegerAttr>("hip.op_state_slot").getInt();
      if (slot < 0 || slot >= numSlots) {
        op->emitError("hip.op_state_slot ")
            << slot << " is out of range [0, " << numSlots
            << "); inconsistent with hipdnn.num_op_state_slots";
        slotOutOfRange = true;
        return;
      }
      bySlot[slot] = op;
    });
    if (slotOutOfRange) {
      signalPassFailure();
      return;
    }

    MLIRContext *ctx = &getContext();
    OpBuilder builder(ctx);
    Location loc = module.getLoc();
    Type ptrType = LLVM::LLVMPointerType::get(ctx, 0);
    Type i8Type = builder.getI8Type();
    Type i32Type = builder.getI32Type();
    Type i64Type = builder.getI64Type();

    // Create the init function at module end.
    builder.setInsertionPointToEnd(module.getBody());
    auto fnType = LLVM::LLVMFunctionType::get(i32Type, {ptrType});
    auto initFn = LLVM::LLVMFuncOp::create(
        builder, loc, hipdnn::abi::kOpStatesInitFn, fnType);
    Block *entry = initFn.addEntryBlock(builder);
    builder.setInsertionPointToStart(entry);
    Value state = entry->getArgument(0);

    // _alloc / _set return bool (i8 in the C ABI). The declarations here must
    // match or the JIT call reads a garbage-extended result.
    FailureOr<LLVM::LLVMFuncOp> allocFn =
        LLVM::lookupOrCreateFn(builder, module, hipdnn::abi::kOpStatesAlloc,
                               {ptrType, i64Type}, i8Type);
    FailureOr<LLVM::LLVMFuncOp> setFn =
        LLVM::lookupOrCreateFn(builder, module, hipdnn::abi::kOpStateSet,
                               {ptrType, i32Type, ptrType}, i8Type);
    if (failed(allocFn) || failed(setFn)) {
      signalPassFailure();
      return;
    }

    // ok = op_states_alloc(state, numSlots).
    Value nVal = LLVM::ConstantOp::create(builder, loc, i64Type,
                                          builder.getI64IntegerAttr(numSlots));
    auto allocCall =
        LLVM::CallOp::create(builder, loc, *allocFn, ValueRange{state, nVal});
    Value zeroI8 = LLVM::ConstantOp::create(builder, loc, i8Type,
                                            builder.getI8IntegerAttr(0));
    Value allocFailed = LLVM::ICmpOp::create(
        builder, loc, LLVM::ICmpPredicate::eq, allocCall.getResult(), zeroI8);
    Value zeroI32 = LLVM::ConstantOp::create(builder, loc, i32Type,
                                             builder.getI32IntegerAttr(0));
    Value oneI32 = LLVM::ConstantOp::create(builder, loc, i32Type,
                                            builder.getI32IntegerAttr(1));

    // On alloc failure, return failure (1) WITHOUT running any construct_<op>:
    // the slot array is null, so _set would refuse to store (see op_state.cpp)
    // and every constructed OpState -- with the device resources / WeakStore
    // handles it owns -- would be dropped unreachable, leaking with its deletor
    // never run. inference_init maps the returned 1 to a failed session.
    Block *constructBlock = initFn.addBlock();
    Block *allocFailBlock = initFn.addBlock();
    LLVM::CondBrOp::create(builder, loc, allocFailed, allocFailBlock,
                           constructBlock);

    builder.setInsertionPointToStart(allocFailBlock);
    LLVM::ReturnOp::create(builder, loc, oneI32);

    // Success path: per stateful op (slot order), construct + store into its
    // slot, then report success (0).
    builder.setInsertionPointToStart(constructBlock);
    for (int64_t slot = 0; slot < numSlots; ++slot) {
      Operation *op = bySlot[slot];
      auto statefulOp = cast<OpStateOpInterface>(op);
      Value constructed = statefulOp.generateOpStateInit(builder, op->getLoc(),
                                                         state, (int32_t)slot);
      if (!constructed) {
        op->emitError("generateOpStateInit produced no value");
        signalPassFailure();
        return;
      }
      Value slotVal = LLVM::ConstantOp::create(
          builder, loc, i32Type, builder.getI32IntegerAttr((int32_t)slot));
      LLVM::CallOp::create(builder, loc, *setFn,
                           ValueRange{state, slotVal, constructed});
    }

    LLVM::ReturnOp::create(builder, loc, zeroI32);
  }
};

} // namespace
} // namespace hip
} // namespace mlir
