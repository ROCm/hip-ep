/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers hip.nonzero to a wrap_nonzero runtime call.
//
//   int wrap_nonzero(state, input, output,
//                    in_shape, rank, total_elements, k_max, data_type);
//
// in_shape is materialised as an LLVM alloca + per-element stores.  Both
// the input and the output (DPS init) are required to be ranked memrefs
// with static shapes -- we need `total_elements` and `k_max` at compile
// time to allocate the worst-case buffer in the parent pipeline.

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace hip {
namespace {

// Locally-defined symbol name so this file builds even if a sibling agent's
// edit drops kWrapNonzero from HipToLLVMUtils.h (the file gets reverted /
// regenerated frequently during multi-agent development).
inline constexpr const char *kLocalWrapNonzero = "wrap_nonzero";

struct NonzeroLowering : public ConvertOpToLLVMPattern<NonzeroOp> {
  using ConvertOpToLLVMPattern<NonzeroOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(NonzeroOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value inPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inType || !outType)
      return rewriter.notifyMatchFailure(
          op, "hip.nonzero lowering requires ranked memref operands");
    if (outType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "hip.nonzero output must be rank-2 (N, K)");

    int64_t rank = inType.getRank();

    int64_t dataType = getHipdnnDataType(inType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.nonzero unsupported input element type");

    auto inShape = getMemRefShape(inType, adaptor.getInput(), rewriter, loc);
    Value inShapePtr =
        materialiseValueArray(inShape, i64Type, ptrType, rewriter, loc);

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value rankConst = i64Const(rank);
    Value totalConst = computeNumElements(inType, adaptor.getInput(),
                                          rewriter, loc);
    Value kMaxConst = getMemRefDimSize(outType, 1, adaptor.getOutput(),
                                       rewriter, loc);
    Value dtypeConst = i64Const(dataType);

    // int wrap_nonzero(state, input, output,
    //                  in_shape, rank, total_elements, k_max, data_type);
    SmallVector<Type, 8> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kLocalWrapNonzero, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 8> args = {statePtr,   inPtr,      outPtr,    inShapePtr,
                                  rankConst,  totalConst, kMaxConst, dtypeConst};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateNonzeroLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<NonzeroLowering>(converter);
}

} // namespace hip
} // namespace mlir
