/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers hip.scatter_nd to a wrap_scatter_nd runtime call.
//
//   int wrap_scatter_nd(state, data, indices, updates, output,
//                       data_shape, data_rank,
//                       indices_shape, indices_rank,
//                       data_type, indices_type, reduction);
//
// data and indices_shape are materialised as LLVM allocas + per-element
// stores, in the same style as Tier3CompareLowering / Tier6Lowering.

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace hip {
namespace {

// Locally-defined symbol name so this file builds even if a sibling agent's
// edit drops kWrapScatterNd from HipToLLVMUtils.h.
inline constexpr const char *kLocalWrapScatterNd = "wrap_scatter_nd";

static Value materialiseI64Array(ArrayRef<int64_t> vals, Type i64Type,
                                 Type ptrType,
                                 ConversionPatternRewriter &rewriter,
                                 Location loc) {
  auto arrayType = LLVM::LLVMArrayType::get(i64Type, vals.size());
  Value one = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                       rewriter.getI64IntegerAttr(1));
  Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrayType, one,
                                        /*alignment=*/8);
  for (size_t i = 0; i < vals.size(); ++i) {
    Value v = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                       rewriter.getI64IntegerAttr(vals[i]));
    SmallVector<LLVM::GEPArg, 2> indices = {0, static_cast<int32_t>(i)};
    Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrType, arrayType,
                                        alloca, indices);
    LLVM::StoreOp::create(rewriter, loc, v, elemPtr);
  }
  return alloca;
}

struct ScatterNdLowering : public ConvertOpToLLVMPattern<ScatterNdOp> {
  using ConvertOpToLLVMPattern<ScatterNdOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ScatterNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value dataPtr = extractMemRefPtr(adaptor.getData(), rewriter, loc);
    Value indicesPtr = extractMemRefPtr(adaptor.getIndices(), rewriter, loc);
    Value updatesPtr = extractMemRefPtr(adaptor.getUpdates(), rewriter, loc);
    Value outPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto dataType = dyn_cast<MemRefType>(op.getData().getType());
    auto indicesType = dyn_cast<MemRefType>(op.getIndices().getType());
    auto updatesType = dyn_cast<MemRefType>(op.getUpdates().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!dataType || !indicesType || !updatesType || !outType)
      return rewriter.notifyMatchFailure(
          op, "hip.scatter_nd lowering requires ranked memref operands");

    int64_t dataRank = dataType.getRank();
    int64_t indicesRank = indicesType.getRank();
    if (indicesRank < 1)
      return rewriter.notifyMatchFailure(
          op, "hip.scatter_nd indices must have rank >= 1");
    if (!indicesType.isDynamicDim(indicesRank - 1)) {
      int64_t coordLen = indicesType.getDimSize(indicesRank - 1);
      if (coordLen <= 0 || coordLen > dataRank)
        return rewriter.notifyMatchFailure(
            op,
            "hip.scatter_nd indices innermost dim must be in [1, data_rank]");
    }

    int64_t dataDtype = getHipdnnDataType(dataType.getElementType());
    if (dataDtype < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.scatter_nd unsupported data element type");
    int64_t indicesDtype = getHipdnnDataType(indicesType.getElementType());
    if (indicesDtype < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.scatter_nd unsupported indices element type");

    auto dataShape = getMemRefShape(dataType, adaptor.getData(), rewriter, loc);
    auto indicesShape = getMemRefShape(indicesType, adaptor.getIndices(),
                                       rewriter, loc);
    Value dataShapePtr =
        materialiseValueArray(dataShape, i64Type, ptrType, rewriter, loc);
    Value indicesShapePtr =
        materialiseValueArray(indicesShape, i64Type, ptrType, rewriter, loc);

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value dataRankConst = i64Const(dataRank);
    Value indicesRankConst = i64Const(indicesRank);
    Value dataDtypeConst = i64Const(dataDtype);
    Value indicesDtypeConst = i64Const(indicesDtype);
    Value reductionConst = i64Const(op.getReduction());

    // int wrap_scatter_nd(state, data, indices, updates, output,
    //                     data_shape, data_rank,
    //                     indices_shape, indices_rank,
    //                     data_type, indices_type, reduction);
    SmallVector<Type, 12> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                        ptrType, ptrType, i64Type, ptrType,
                                        i64Type, i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kLocalWrapScatterNd, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 12> args = {
        statePtr,        dataPtr,         indicesPtr,
        updatesPtr,      outPtr,          dataShapePtr,
        dataRankConst,   indicesShapePtr, indicesRankConst,
        dataDtypeConst,  indicesDtypeConst, reductionConst};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateScatterNdLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<ScatterNdLowering>(converter);
}

} // namespace hip
} // namespace mlir
