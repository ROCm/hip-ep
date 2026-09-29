/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.resize -> wrap_resize runtime call
//===----------------------------------------------------------------------===//
//
// One input extent and one output extent per axis, up to kMaxRank.  The
// kernel resamples an axis whose extents differ and copies an axis whose
// extents match, so the dimension order is the layout.  Slots past `rank`
// are 1 and ignored.  Dynamic dims are read from the memref descriptor.
//
// Runtime ABI:
//   wrap_resize(state, input, output,
//               data_type, rank,
//               in0..4, out0..4,
//               mode, coord_transform, nearest_mode)
//   -> i32

constexpr int64_t kMaxRank = 5;

struct ResizeOpLowering : public ConvertOpToLLVMPattern<ResizeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ResizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64 = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    int64_t rank = inputType.getRank();
    if (rank < 1 || rank > kMaxRank || outputType.getRank() != rank)
      return rewriter.notifyMatchFailure(op, "expected rank in [1, 5]");

    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType < 0 || (dataType > 2 && dataType != 6))
      return rewriter.notifyMatchFailure(
          op, "Resize: only f16 / f32 / bf16 / f64 supported");

    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Value inputDesc = adaptor.getInput();
    Value outputDesc = adaptor.getOutput();

    SmallVector<Value, kMaxRank> inLens;
    SmallVector<Value, kMaxRank> outLens;
    for (int64_t i : llvm::seq<int64_t>(kMaxRank)) {
      if (i < rank) {
        inLens.push_back(
            getMemRefDimSize(inputType, i, inputDesc, rewriter, loc));
        outLens.push_back(
            getMemRefDimSize(outputType, i, outputDesc, rewriter, loc));
      } else {
        inLens.push_back(createI64(1));
        outLens.push_back(createI64(1));
      }
    }

    Value mode = createI64(op.getMode());
    Value coord = createI64(op.getCoordTransform());
    Value nearest = createI64(op.getNearestMode());
    Value rankV = createI64(rank);
    Value dataTypeV = createI64(dataType);

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
    addPtr(outputPtr);
    addI64(dataTypeV);
    addI64(rankV);
    for (Value extent : inLens)
      addI64(extent);
    for (Value extent : outLens)
      addI64(extent);
    addI64(mode);
    addI64(coord);
    addI64(nearest);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapResize, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateResizeLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<ResizeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
