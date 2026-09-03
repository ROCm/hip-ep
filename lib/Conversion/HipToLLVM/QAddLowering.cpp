/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// Shared lowering for quantized elementwise ops:
//   hip.qadd / (future) qsub, qmul, qdiv
//     -> wrap_qelementwise(..., kind, ..., M_a, lhs_zp, M_b, rhs_zp, output_zp)
//
// M_a = s_a / s_out, M_b = s_b / s_out. Runtime uses
//   OUT = round(M_a * (A - z_a) [OP] M_b * (B - z_b)) + z_out

template <typename OpTy, HipdnnQElementwiseKind Kind>
struct QElementwiseLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    Type ptrType = this->getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto lhsType = cast<MemRefType>(op.getLhs().getType());
    auto rhsType = cast<MemRefType>(op.getRhs().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    auto createF32Const = [&](float v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(v));
    };

    // alloca a stack [max(rank, 1) x i64] array of dim sizes. max(rank, 1)
    // keeps rank-0 operands allocatable; the runtime ignores an empty shape.
    Value one = createI64Const(1);
    auto emitShapeArray = [&](MemRefType type, Value descriptor) -> Value {
      int rank = type.getRank();
      auto arrType = LLVM::LLVMArrayType::get(i64Type, std::max(rank, 1));
      Value arr =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
      for (int i = 0; i < rank; ++i) {
        Value dim = getMemRefDimSize(type, i, descriptor, rewriter, loc);
        Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                             rewriter.getI32IntegerAttr(i));
        Value elemPtr =
            LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, arr, idx);
        LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
      }
      return arr;
    };

    Value lhsShape = emitShapeArray(lhsType, adaptor.getLhs());
    Value rhsShape = emitShapeArray(rhsType, adaptor.getRhs());
    Value outShape = emitShapeArray(outputType, adaptor.getOutput());
    SmallVector<Type, 17> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, // state + 3 data ptrs
        i64Type,                            // kind
        ptrType, i64Type,                   // lhs_shape, lhs_rank
        ptrType, i64Type,                   // rhs_shape, rhs_rank
        ptrType, i64Type,                   // out_shape, out_rank
        i64Type,                            // data_type
        f32Type, i64Type,                   // M_a, lhs_zp
        f32Type, i64Type,                   // M_b, rhs_zp
        i64Type};                           // output_zp

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQElementwise, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    float outputScale = op.getOutputScale().convertToFloat();
    if (outputScale == 0.0f)
      return rewriter.notifyMatchFailure(op, "output_scale must be non-zero");
    float mA = op.getLhsScale().convertToFloat() / outputScale;
    float mB = op.getRhsScale().convertToFloat() / outputScale;

    SmallVector<Value, 17> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getLhs(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getRhs(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        createI64Const(Kind),
        lhsShape,
        createI64Const(lhsType.getRank()),
        rhsShape,
        createI64Const(rhsType.getRank()),
        outShape,
        createI64Const(outputType.getRank()),
        createI64Const(dataType),
        createF32Const(mA),
        createI64Const(op.getLhsZp()),
        createF32Const(mB),
        createI64Const(op.getRhsZp()),
        createI64Const(op.getOutputZp())};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct QAddOpLowering
    : public QElementwiseLowering<QAddOp,
                                  HipdnnQElementwiseKind::kQElementwiseAdd> {
  using QElementwiseLowering::QElementwiseLowering;
};

} // namespace

void populateQAddLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<QAddOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
