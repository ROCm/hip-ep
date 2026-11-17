/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

#include "hip/Dialect/IR/HipQuantTypes.h"

namespace mlir {
namespace hip {
namespace {

// Everything wrap_gemm needs, from hip.gemm or a float hip.matmul with a bias.
struct GemmSpec {
  Value a, b, c, y;               // converted operands; c may be null
  MemRefType aType, bType, cType; // cType null without C
  float alpha = 1.0f, beta = 1.0f;
  int64_t transA = 0, transB = 0;
  // A is [..., M, K] with transA == 0 and rank > 2: its leading dims fold into
  // the rows, so the kernel sees [prod(leading) * M, K]. Set for hip.matmul.
  bool foldLeadingDims = false;
};

static std::optional<GemmSpec> getSpec(GemmOp op, GemmOp::Adaptor adaptor) {
  if (isQuantizedOp(op))
    return std::nullopt; // wrap_qgemm (QGemmLowering.cpp)
  GemmSpec spec;
  spec.a = adaptor.getInputA();
  spec.b = adaptor.getInputB();
  spec.c = adaptor.getInputC();
  spec.y = adaptor.getOutput();
  spec.aType = cast<MemRefType>(op.getInputA().getType());
  spec.bType = cast<MemRefType>(op.getInputB().getType());
  if (op.getInputC())
    spec.cType = cast<MemRefType>(op.getInputC().getType());
  spec.alpha = op.getAlpha().convertToFloat();
  spec.beta = op.getBeta().convertToFloat();
  spec.transA = op.getTransA();
  spec.transB = op.getTransB();
  return spec;
}

// A float hip.matmul with a row bias is a gemm over [rows, K] x [K, N] + [N].
// Its op state is a GemmState for the same reason (MatmulOp's
// generateOpStateInit).
static std::optional<GemmSpec> getSpec(MatmulOp op, MatmulOp::Adaptor adaptor) {
  if (!op.getBias() || isQuantizedOp(op))
    return std::nullopt;
  GemmSpec spec;
  spec.a = adaptor.getA();
  spec.b = adaptor.getB();
  spec.c = adaptor.getBias();
  spec.y = adaptor.getOutput();
  spec.aType = cast<MemRefType>(op.getA().getType());
  spec.bType = cast<MemRefType>(op.getB().getType());
  spec.cType = cast<MemRefType>(op.getBias().getType());
  spec.transA = op.getTransA();
  spec.transB = op.getTransB();
  spec.foldLeadingDims = true;
  return spec;
}

// -> wrap_gemm(state, slot, A, B, C, Y, M, N, K, alpha, beta, transA, transB,
//              typeCode, cDim0, cDim1)
template <typename OpTy>
struct GemmLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<GemmSpec> spec = getSpec(op, adaptor);
    if (!spec)
      return rewriter.notifyMatchFailure(op, "not a float gemm");

    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    Type ptrType = this->getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    MemRefType AType = spec->aType;
    MemRefType BType = spec->bType;
    Type elemType = AType.getElementType();
    int64_t typeCode;
    if (elemType.isF16())
      typeCode = 0;
    else if (elemType.isF32())
      typeCode = 1;
    else if (elemType.isF64())
      typeCode = 2;
    else if (elemType.isBF16())
      typeCode = 3;
    else
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    // A.shape = transA ? [K, M] : [M, K]; B.shape = transB ? [N, K] : [K, N].
    // A rank > 2 A (matmul with a bias) folds its leading dims into M.
    int64_t aRank = AType.getRank();
    if (BType.getRank() != 2 ||
        (aRank > 2 && (!spec->foldLeadingDims || spec->transA)))
      return rewriter.notifyMatchFailure(
          op, "needs a rank-2 B, and transA = 0 for a rank > 2 A");
    Value M, K;
    if (aRank > 2) {
      M = getMemRefDimSize(AType, 0, spec->a, rewriter, loc);
      for (int64_t i : llvm::seq<int64_t>(1, aRank - 1))
        M = LLVM::MulOp::create(
            rewriter, loc, M,
            getMemRefDimSize(AType, i, spec->a, rewriter, loc));
      K = getMemRefDimSize(AType, aRank - 1, spec->a, rewriter, loc);
    } else {
      M = getMemRefDimSize(AType, spec->transA ? 1 : 0, spec->a, rewriter, loc);
      K = getMemRefDimSize(AType, spec->transA ? 0 : 1, spec->a, rewriter, loc);
    }
    Value N =
        getMemRefDimSize(BType, spec->transB ? 0 : 1, spec->b, rewriter, loc);

    // C's shape normalized to 2D [cDim0, cDim1] for broadcast support.
    // Absent -> [0, 0], scalar -> [1, 1], [X] -> [1, X], [X, Y] -> [X, Y].
    Value cDim0, cDim1;
    if (spec->cType) {
      MemRefType CType = spec->cType;
      int cRank = CType.getRank();
      if (cRank == 0) {
        cDim0 = createI64Const(1);
        cDim1 = createI64Const(1);
      } else if (cRank == 1) {
        cDim0 = createI64Const(1);
        cDim1 = getMemRefDimSize(CType, 0, spec->c, rewriter, loc);
      } else {
        cDim0 = getMemRefDimSize(CType, 0, spec->c, rewriter, loc);
        cDim1 = getMemRefDimSize(CType, 1, spec->c, rewriter, loc);
      }
    } else {
      cDim0 = createI64Const(0);
      cDim1 = createI64Const(0);
    }

    SmallVector<Type, 16> paramTypes = {
        ptrType, // state
        i32Type, // op_state_slot
        ptrType, // A
        ptrType, // B
        ptrType, // C (nullable)
        ptrType, // output
        i64Type, // M
        i64Type, // N
        i64Type, // K
        f32Type, // alpha
        f32Type, // beta
        i64Type, // transA
        i64Type, // transB
        i64Type, // typeCode
        i64Type, // cDim0
        i64Type, // cDim1
    };
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGemm, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 16> args = {
        adaptor.getCtx(),
        getOpStateSlotValue(op, rewriter, loc),
        extractContiguousMemRefPtr(spec->a, rewriter, loc),
        extractContiguousMemRefPtr(spec->b, rewriter, loc),
        extractOptionalMemRefPtr(spec->c, rewriter, loc),
        extractContiguousMemRefPtr(spec->y, rewriter, loc),
        M,
        N,
        K,
        LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                 rewriter.getF32FloatAttr(spec->alpha)),
        LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                 rewriter.getF32FloatAttr(spec->beta)),
        createI64Const(spec->transA),
        createI64Const(spec->transB),
        createI64Const(typeCode),
        cDim0,
        cDim1};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGemmLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<GemmLowering<GemmOp>, GemmLowering<MatmulOp>>(converter);
}

} // namespace hip
} // namespace mlir
