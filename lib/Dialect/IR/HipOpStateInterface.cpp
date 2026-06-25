/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipOpStateInterface.cpp - OpStateOpInterface + shared helper -------===//
//
// Provides the TableGen-generated `OpStateOpInterface` method bindings and the
// shared `emitOpStateConstruct` helper that most `generateOpStateInit` bodies
// reduce to. See docs/design/op-state-slots-design.md.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hip;

namespace mlir::hip {
#include "hip/Dialect/IR/HipOpStateInterface.cpp.inc"
} // namespace mlir::hip

namespace mlir {
namespace hip {

Value emitOpStateConstruct(OpBuilder &builder, Location loc, Value statePtr,
                           int32_t slot, StringRef ctorSymbol,
                           ArrayRef<int64_t> i64Args) {
  MLIRContext *ctx = builder.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(ctx, 0);
  Type i8Type = builder.getI8Type();
  Type i32Type = builder.getI32Type();
  Type i64Type = builder.getI64Type();

  // Construct signature: (RuntimeState*, i32 slot, i64 x N) -> i8. The
  // constructor builds its state and stores it into op_states[slot] itself (via
  // hipdnn_ep_op_state_set). The i8 result is vestigial (always 0) -- it keeps
  // a stable call signature but the init pass does not branch on it.
  SmallVector<Type> paramTypes;
  paramTypes.push_back(ptrType);
  paramTypes.push_back(i32Type);
  paramTypes.append(i64Args.size(), i64Type);

  auto module =
      builder.getInsertionBlock()->getParentOp()->getParentOfType<ModuleOp>();
  FailureOr<LLVM::LLVMFuncOp> ctorFn =
      LLVM::lookupOrCreateFn(builder, module, ctorSymbol, paramTypes, i8Type);
  if (failed(ctorFn))
    return Value();

  SmallVector<Value> args;
  args.push_back(statePtr);
  args.push_back(LLVM::ConstantOp::create(builder, loc, i32Type,
                                          builder.getI32IntegerAttr(slot)));
  for (int64_t v : i64Args)
    args.push_back(LLVM::ConstantOp::create(builder, loc, i64Type,
                                            builder.getI64IntegerAttr(v)));

  auto call = LLVM::CallOp::create(builder, loc, *ctorFn, args);
  return call.getResult();
}

} // namespace hip
} // namespace mlir
