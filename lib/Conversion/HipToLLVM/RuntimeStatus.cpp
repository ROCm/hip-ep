/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "RuntimeStatus.h"
#include "HipToLLVMUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>
#include <utility>

namespace mlir::hip {
namespace {

bool isStatusReturningRuntimeSymbol(StringRef name) {
  if (name.starts_with("wrap_"))
    return true;
  return llvm::StringSwitch<bool>(name)
      .Cases({"hip_miopen_softmax", "hipdnn_graph_execute"}, true)
      .Cases({"hipdnn_ep_run_if", "hipdnn_ep_run_counted_loop"}, true)
      .Cases({"hipdnn_ep_run_loop", "hipdnn_ep_loop_frame_destroy"}, true)
      .Case("hipdnn_ep_copy_output", true)
      .Default(false);
}

bool hasStatusRecorderUse(LLVM::CallOp call) {
  if (call.getNumResults() != 1)
    return false;
  return llvm::any_of(call.getResult().getUsers(), [](Operation *user) {
    auto recorder = dyn_cast<LLVM::CallOp>(user);
    return recorder && recorder.getCallee() &&
           *recorder.getCallee() == kHipRecordStatus;
  });
}

} // namespace

LogicalResult recordAndVerifyRuntimeStatuses(ModuleOp module) {
  MLIRContext *ctx = module.getContext();
  Type i32Type = IntegerType::get(ctx, 32);
  Type ptrType = LLVM::LLVMPointerType::get(ctx, 0);
  SmallVector<LLVM::CallOp> statusCalls;

  WalkResult validation = module.walk([&](LLVM::CallOp call) -> WalkResult {
    std::optional<StringRef> callee = call.getCallee();
    if (!callee || !isStatusReturningRuntimeSymbol(*callee))
      return WalkResult::advance();
    if (call.getNumResults() != 1 || call.getResult().getType() != i32Type ||
        call.getArgOperands().empty() ||
        call.getArgOperands().front().getType() != ptrType) {
      call.emitOpError() << "status-bearing runtime call must have signature "
                            "(state, ...) -> i32";
      return WalkResult::interrupt();
    }
    statusCalls.push_back(call);
    return WalkResult::advance();
  });
  if (validation.wasInterrupted())
    return failure();

  SmallVector<std::pair<ModuleOp, LLVM::LLVMFuncOp>> recorders;
  for (LLVM::CallOp call : statusCalls) {
    ModuleOp owner = call->getParentOfType<ModuleOp>();
    auto existing = llvm::find_if(
        recorders, [&](const auto &entry) { return entry.first == owner; });
    if (existing != recorders.end())
      continue;
    OpBuilder declarationBuilder(ctx);
    declarationBuilder.setInsertionPointToStart(owner.getBody());
    FailureOr<LLVM::LLVMFuncOp> recorder =
        LLVM::lookupOrCreateFn(declarationBuilder, owner, kHipRecordStatus,
                               {ptrType, i32Type}, i32Type);
    if (failed(recorder))
      return failure();
    recorders.emplace_back(owner, *recorder);
  }

  for (LLVM::CallOp call : statusCalls) {
    if (hasStatusRecorderUse(call))
      continue;
    ModuleOp owner = call->getParentOfType<ModuleOp>();
    auto recorder = llvm::find_if(
        recorders, [&](const auto &entry) { return entry.first == owner; });
    assert(recorder != recorders.end() &&
           "recorder must exist for call module");
    OpBuilder builder(call);
    builder.setInsertionPointAfter(call);
    LLVM::CallOp::create(
        builder, call.getLoc(), recorder->second,
        ValueRange{call.getArgOperands().front(), call.getResult()});
  }

  WalkResult audit = module.walk([&](LLVM::CallOp call) -> WalkResult {
    if (call.getNumResults() != 1 || call.getResult().getType() != i32Type ||
        !call.getResult().use_empty())
      return WalkResult::advance();
    std::optional<StringRef> callee = call.getCallee();
    if (callee && *callee == kHipRecordStatus)
      return WalkResult::advance();
    InFlightDiagnostic diagnostic = call.emitOpError("unused i32 result from ");
    if (callee)
      diagnostic << "'" << *callee << "'";
    else
      diagnostic << "indirect call";
    diagnostic << "; consume data or propagate operation status";
    return WalkResult::interrupt();
  });
  return success(!audit.wasInterrupted());
}

} // namespace mlir::hip
