/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.grid_sample -> wrap_grid_sample runtime call
//===----------------------------------------------------------------------===//
//
// Runtime ABI:
//   wrap_grid_sample(state, input, grid, output,
//                    data_type, N, C, inH, inW, outH, outW,
//                    mode, padding_mode, align_corners)
//   -> i32

struct GridSampleOpLowering : public ConvertOpToLLVMPattern<GridSampleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GridSampleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto gridType = dyn_cast<MemRefType>(op.getGrid().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inputType || !gridType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected memref operands");
    if (inputType.getRank() != 4 || gridType.getRank() != 4 ||
        outputType.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "expected 4-D memrefs");

    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType < 0 || (dataType > 2 && dataType != 6))
      return rewriter.notifyMatchFailure(
          op, "GridSample: only f16 / f32 / bf16 / f64 supported");

    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value gridPtr =
        extractContiguousMemRefPtr(adaptor.getGrid(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Value inputDesc = adaptor.getInput();
    Value gridDesc = adaptor.getGrid();

    auto createI64 = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value N = getMemRefDimSize(inputType, 0, inputDesc, rewriter, loc);
    Value C = getMemRefDimSize(inputType, 1, inputDesc, rewriter, loc);
    Value inH = getMemRefDimSize(inputType, 2, inputDesc, rewriter, loc);
    Value inW = getMemRefDimSize(inputType, 3, inputDesc, rewriter, loc);
    Value outH = getMemRefDimSize(gridType, 1, gridDesc, rewriter, loc);
    Value outW = getMemRefDimSize(gridType, 2, gridDesc, rewriter, loc);

    SmallVector<Type, 16> paramTypes;
    SmallVector<Value, 16> args;
    auto addPtr = [&](Value v) {
      paramTypes.push_back(ptrType);
      args.push_back(v);
    };
    auto addI64 = [&](Value v) {
      paramTypes.push_back(i64Type);
      args.push_back(v);
    };

    addPtr(statePtr);
    addPtr(inputPtr);
    addPtr(gridPtr);
    addPtr(outputPtr);
    addI64(createI64(dataType));
    addI64(N);
    addI64(C);
    addI64(inH);
    addI64(inW);
    addI64(outH);
    addI64(outW);
    addI64(createI64(op.getMode()));
    addI64(createI64(op.getPaddingMode()));
    addI64(createI64(op.getAlignCorners()));

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGridSample, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGridSampleLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns) {
  patterns.add<GridSampleOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
