/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers Tier 6 hip ops to wrap_* runtime calls:
//   - hip.pad -> wrap_pad
//
// Other Tier 6 ops (expand, conv_transpose, resize, range) live in
// their own lowering files (TODO).

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace hip {
namespace {

// Materialise a small int64 array as an LLVM alloca + per-element stores
// and return the alloca pointer.  Mirrors the helper in
// Tier3CompareLowering.cpp.
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

// Compute row-major element strides for `shape`.
static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  int64_t rank = shape.size();
  SmallVector<int64_t> strides(rank, 1);
  for (int64_t d = rank - 2; d >= 0; --d)
    strides[d] = strides[d + 1] * shape[d + 1];
  return strides;
}

//===----------------------------------------------------------------------===//
// hip.pad
//===----------------------------------------------------------------------===//

struct PadLowering : public ConvertOpToLLVMPattern<PadOp> {
  using ConvertOpToLLVMPattern<PadOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    Value statePtr = adaptor.getCtx();
    Value inPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inType || !outType || !inType.hasStaticShape() ||
        !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "hip.pad lowering requires static shapes");
    if (inType.getRank() != outType.getRank())
      return rewriter.notifyMatchFailure(
          op, "hip.pad input/output rank mismatch");

    int64_t rank = inType.getRank();
    SmallVector<int64_t> inShape(inType.getShape().begin(),
                                 inType.getShape().end());
    SmallVector<int64_t> outShape(outType.getShape().begin(),
                                  outType.getShape().end());
    SmallVector<int64_t> inStrides = computeRowMajorStrides(inShape);
    SmallVector<int64_t> outStrides = computeRowMajorStrides(outShape);

    // pads_begin / pads_end attributes have rank entries each.
    auto padsBeginAttr = op.getPadsBegin();
    auto padsEndAttr = op.getPadsEnd();
    if ((int64_t)padsBeginAttr.size() != rank ||
        (int64_t)padsEndAttr.size() != rank)
      return rewriter.notifyMatchFailure(
          op, "hip.pad pads_begin/pads_end length mismatch with rank");

    SmallVector<int64_t> padsBegin;
    padsBegin.reserve(rank);
    for (Attribute a : padsBeginAttr)
      padsBegin.push_back(cast<IntegerAttr>(a).getValue().getSExtValue());

    int64_t dataType = getHipdnnDataType(outType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.pad unsupported element type");

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value rankConst = i64Const(rank);
    Value padsBeginLen = i64Const(rank);
    Value modeConst = i64Const(op.getMode());
    Value dtypeConst = i64Const(dataType);
    Value valueConst = LLVM::ConstantOp::create(
        rewriter, loc, f32Type,
        rewriter.getF32FloatAttr(op.getValue().convertToFloat()));

    Value inShapePtr =
        materialiseI64Array(inShape, i64Type, ptrType, rewriter, loc);
    Value inStridesPtr =
        materialiseI64Array(inStrides, i64Type, ptrType, rewriter, loc);
    Value outShapePtr =
        materialiseI64Array(outShape, i64Type, ptrType, rewriter, loc);
    Value outStridesPtr =
        materialiseI64Array(outStrides, i64Type, ptrType, rewriter, loc);
    Value padsBeginPtr =
        materialiseI64Array(padsBegin, i64Type, ptrType, rewriter, loc);

    // int wrap_pad(state, input, output,
    //              in_shape, in_strides_elems,
    //              out_shape, out_strides_elems,
    //              rank,
    //              pads_begin, pads_begin_len,
    //              data_type, mode, value);
    SmallVector<Type, 13> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, ptrType, ptrType,
        i64Type, ptrType, i64Type, i64Type, i64Type, f32Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapPad, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 13> args = {statePtr,    inPtr,        outPtr,
                                   inShapePtr,  inStridesPtr, outShapePtr,
                                   outStridesPtr, rankConst,  padsBeginPtr,
                                   padsBeginLen, dtypeConst,  modeConst,
                                   valueConst};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateTier6LoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<PadLowering>(converter);
}

} // namespace hip
} // namespace mlir
