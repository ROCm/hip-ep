/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- FusedMulAddLowering.cpp - HipToLLVM lowering for hip.fused_mul_add -===//
//
// Touchpoint E: ConvertOpToLLVMPattern for the plugin-contributed op
// hip.fused_mul_add. Emits a call to the runtime symbol wrap_fused_mul_add.
//
// Runtime signature (C ABI, defined in runtime/wrap_fused_mul_add.cpp):
//   int wrap_fused_mul_add(void* state,
//                          void* x,
//                          void* mul_operand,
//                          void* add_operand,
//                          void* output,
//                          int64_t num_elements,
//                          int64_t data_type)
//
// All memref operands must be contiguous by the time this runs
// (PromoteStridedHipOperands guarantees it for DPS-input memrefs).
//
//===----------------------------------------------------------------------===//

#include "plugins/fusion/FusedMulAddLowering.h"
#include "plugins/fusion/PluginOps.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

struct FusedMulAddOpLowering : public ConvertOpToLLVMPattern<FusedMulAddOp> {
  using ConvertOpToLLVMPattern<FusedMulAddOp>::ConvertOpToLLVMPattern;

  // Extract the aligned data pointer from a converted memref descriptor.
  Value extractPtr(Value desc, ConversionPatternRewriter &rewriter,
                   Location loc) const {
    Value ptr = MemRefDescriptor(desc).alignedPtr(rewriter, loc);
    auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
    if (ptrTy.getAddressSpace() != 0)
      ptr = LLVM::AddrSpaceCastOp::create(
          rewriter, loc,
          LLVM::LLVMPointerType::get(rewriter.getContext(), 0), ptr);
    return ptr;
  }

  // Compute total element count for a memref (static * dynamic dims).
  Value numElements(MemRefType type, Value desc,
                    ConversionPatternRewriter &rewriter, Location loc) const {
    Type i64 = rewriter.getI64Type();
    Value n = LLVM::ConstantOp::create(rewriter, loc, i64,
                                       rewriter.getI64IntegerAttr(1));
    for (int d : llvm::seq<int>(type.getRank())) {
      Value dim;
      if (type.isDynamicDim(d))
        dim = MemRefDescriptor(desc).size(rewriter, loc, d);
      else
        dim = LLVM::ConstantOp::create(rewriter, loc, i64,
                                       rewriter.getI64IntegerAttr(type.getDimSize(d)));
      n = LLVM::MulOp::create(rewriter, loc, n, dim);
    }
    return n;
  }

  // Map MLIR element type to HIPDNN_EP_DATATYPE_* enum value.
  int64_t dataTypeCode(Type elemType) const {
    if (elemType.isF32())  return 0;
    if (elemType.isF16())  return 1;
    if (elemType.isBF16()) return 2;
    if (elemType.isInteger(32)) return 3;
    if (elemType.isInteger(64)) return 4;
    if (elemType.isInteger(8))  return 5;
    if (elemType.isF64())  return 6;
    return -1;
  }

  LogicalResult
  matchAndRewrite(FusedMulAddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    int64_t dtCode = dataTypeCode(outputType.getElementType());
    if (dtCode < 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    // int wrap_fused_mul_add(state, x, mul_op, add_op, out, num_elem, dtype)
    SmallVector<Type, 7> paramTypes = {ptrType, ptrType, ptrType,
                                       ptrType, ptrType, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapFusedMulAdd,
                               paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    Value numElems = numElements(outputType, adaptor.getOutput(), rewriter, loc);

    SmallVector<Value, 7> args = {
        adaptor.getCtx(),
        extractPtr(adaptor.getX(), rewriter, loc),
        extractPtr(adaptor.getMulOperand(), rewriter, loc),
        extractPtr(adaptor.getAddOperand(), rewriter, loc),
        extractPtr(adaptor.getOutput(), rewriter, loc),
        numElems,
        LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                 rewriter.getI64IntegerAttr(dtCode))};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateFusedMulAddLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<FusedMulAddOpLowering>(converter);
}
