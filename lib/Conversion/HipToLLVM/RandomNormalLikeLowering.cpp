/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

static Value buildShapeArray(MemRefType type, Value memref, Location loc,
                             ConversionPatternRewriter &rewriter) {
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  Type i32Type = rewriter.getI32Type();
  Type i64Type = rewriter.getI64Type();
  Value one = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                       rewriter.getI64IntegerAttr(1));
  int rank = type.getRank();
  int arrLen = std::max(rank, 1);
  auto arrType = LLVM::LLVMArrayType::get(i64Type, arrLen);
  Value shapeArr =
      LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
  for (int i = 0; i < rank; ++i) {
    Value dim = getMemRefDimSize(type, i, memref, rewriter, loc);
    Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                         rewriter.getI32IntegerAttr(i));
    Value elemPtr =
        LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, shapeArr, idx);
    LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
  }
  return shapeArr;
}

static int64_t f32Bits(const llvm::APFloat &value) {
  return static_cast<int64_t>(value.bitcastToAPInt().getZExtValue());
}

// hip.random_normal_like(ctx, input, output, mean, scale, seed?)
//   -> wrap_random_normal_like(state, output, rank, shape,
//          mean_bits, scale_bits, seed_bits, has_seed, data_type)
//
// The input memref is not passed. Its shape was copied onto the output
// init, and dynamic sizes are read from that output descriptor.
struct RandomNormalLikeOpLowering
    : public ConvertOpToLLVMPattern<RandomNormalLikeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RandomNormalLikeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    int rank = outputType.getRank();
    if (rank < 0 || rank > 8)
      return rewriter.notifyMatchFailure(op, "rank must be in [0, 8]");

    int64_t dataTypeEnum = getHipdnnDataType(outputType.getElementType());
    if (dataTypeEnum != HIPDNN_EP_DATATYPE_HALF &&
        dataTypeEnum != HIPDNN_EP_DATATYPE_BFLOAT16 &&
        dataTypeEnum != HIPDNN_EP_DATATYPE_FLOAT &&
        dataTypeEnum != HIPDNN_EP_DATATYPE_DOUBLE)
      return rewriter.notifyMatchFailure(
          op, "RandomNormalLike output must be f16, bf16, f32, or f64");

    int64_t seedBits = 0;
    int64_t hasSeed = 0;
    if (auto seedAttr = op.getSeedAttr()) {
      hasSeed = 1;
      seedBits = f32Bits(seedAttr.getValue());
    }

    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value shapeArr =
        buildShapeArray(outputType, adaptor.getOutput(), loc, rewriter);

    SmallVector<Type, 9> paramTypes = {ptrType, ptrType, i64Type,
                                       ptrType, i64Type, i64Type,
                                       i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapRandomNormalLike, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {
        adaptor.getCtx(),
        outputPtr,
        createI64Const(rank),
        shapeArr,
        createI64Const(f32Bits(op.getMeanAttr().getValue())),
        createI64Const(f32Bits(op.getScaleAttr().getValue())),
        createI64Const(seedBits),
        createI64Const(hasSeed),
        createI64Const(dataTypeEnum)};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateRandomNormalLikeLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<RandomNormalLikeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
