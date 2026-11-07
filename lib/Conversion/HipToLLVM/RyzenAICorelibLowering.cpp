/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// RyzenAI CoreLib (NPU) call emission with GPU fallback.
//
// With --use-ryzenai-corelib, the MatMulNBits, RmsNorm and SkipRmsNorm
// lowerings route through emitCallWithCorelibFallback. The corelib wrapper
// takes the GPU wrapper's exact arguments plus the activation data type, and
// decides at runtime whether corelib has an NPU kernel for the shape. When it
// does not (or there is no NPU), it returns kCorelibNotHandled and the GPU
// wrapper runs instead, so the fallback is visible in the IR rather than
// hidden behind a cross-library function pointer: the corelib wrappers live in
// a native library, while the GPU wrappers live in the JIT-loaded runtime.
//
// Before:
//   hip.matmul_nbits(%h) ins(%a, %b, %scales) outs(%y) {...}
//
// After:
//   %rc = llvm.call @wrap_corelib_matmul_nbits(%state, %slot, ..., %dtype)
//   %gpu = llvm.icmp "eq" %rc, %c1 : i32
//   llvm.cond_br %gpu, ^gpu, ^cont
// ^gpu:
//   llvm.call @wrap_matmul_nbits(%state, %slot, ...)
//   llvm.br ^cont
// ^cont:
//   ...
//===----------------------------------------------------------------------===//

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {

LogicalResult
emitCallWithCorelibFallback(ConversionPatternRewriter &rewriter, Location loc,
                            ModuleOp module, StringRef gpuCallee,
                            StringRef corelibCallee, int64_t dataType,
                            ArrayRef<Type> paramTypes, ArrayRef<Value> args) {
  Type i32Type = rewriter.getI32Type();
  Type i64Type = rewriter.getI64Type();

  FailureOr<LLVM::LLVMFuncOp> gpuFn =
      LLVM::lookupOrCreateFn(rewriter, module, gpuCallee, paramTypes, i32Type);
  if (failed(gpuFn))
    return failure();

  if (corelibCallee.empty()) {
    LLVM::CallOp::create(rewriter, loc, *gpuFn, args);
    return success();
  }

  SmallVector<Type> corelibTypes(paramTypes);
  corelibTypes.push_back(i64Type);
  FailureOr<LLVM::LLVMFuncOp> corelibFn = LLVM::lookupOrCreateFn(
      rewriter, module, corelibCallee, corelibTypes, i32Type);
  if (failed(corelibFn))
    return failure();

  SmallVector<Value> corelibArgs(args);
  corelibArgs.push_back(LLVM::ConstantOp::create(
      rewriter, loc, i64Type, rewriter.getI64IntegerAttr(dataType)));
  Value rc =
      LLVM::CallOp::create(rewriter, loc, *corelibFn, corelibArgs).getResult();
  Value notHandled = LLVM::ConstantOp::create(
      rewriter, loc, i32Type, rewriter.getI32IntegerAttr(kCorelibNotHandled));
  Value useGpu = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::eq,
                                      rc, notHandled);

  Block *current = rewriter.getInsertionBlock();
  Block *cont = rewriter.splitBlock(current, rewriter.getInsertionPoint());
  Block *gpuBlock = rewriter.createBlock(cont);
  LLVM::CallOp::create(rewriter, loc, *gpuFn, args);
  LLVM::BrOp::create(rewriter, loc, cont);

  rewriter.setInsertionPointToEnd(current);
  LLVM::CondBrOp::create(rewriter, loc, useGpu, gpuBlock, cont);
  rewriter.setInsertionPointToStart(cont);
  return success();
}

} // namespace hip
} // namespace mlir
