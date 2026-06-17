/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// ===== Custom HIP kernel ops ================================================

// hip.transpose(%handle, %dim0, %dim1) ins(%input) outs(%output)
//   -> hip_transpose(handle, input, output, rank, dim0, dim1, s0, s1, s2)
// Swaps the two specified dimensions. Pads shape to 3 dims (trailing 1s).
struct TransposeOpLowering : public ConvertOpToLLVMPattern<TransposeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();
    Type indexType = getIndexType();

    // (handle, input, output, rank, dim0, dim1, s0, s1, s2)
    SmallVector<Type> paramTypes = {ptrType,   ptrType,   ptrType,
                                    indexType, indexType, indexType,
                                    indexType, indexType, indexType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipTranspose, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    int rank = cast<MemRefType>(op.getInput().getType()).getRank();
    if (rank > 3)
      return op.emitOpError("hip.transpose lowering supports rank <= 3, got ")
             << rank;

    MemRefDescriptor inputDesc(adaptor.getInput());
    Value rankVal = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                             rewriter.getIndexAttr(rank));
    Value one = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                         rewriter.getIndexAttr(1));

    SmallVector<Value, 3> shape;
    for (int dimIdx : llvm::seq<int>(3))
      shape.push_back(dimIdx < rank ? inputDesc.size(rewriter, loc, dimIdx)
                                    : one);

    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        rankVal,
        adaptor.getDim0(),
        adaptor.getDim1(),
        shape[0],
        shape[1],
        shape[2]};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateTransposeLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<TransposeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
