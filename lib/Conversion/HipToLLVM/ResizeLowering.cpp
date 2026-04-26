/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers hip.resize -> wrap_resize runtime call.
//
// We pass:
//   - state, input ptr, output ptr (memref aligned ptrs cast to !llvm.ptr)
//   - in_shape / in_strides_elems / out_shape / out_strides_elems arrays
//     (alloca + store, length == rank)
//   - rank, data_type, mode, coord_xform (i64) + cubic_coeff_a (f32)
//
// tf_crop_and_resize (4) is rejected with notifyMatchFailure -- Kokoro
// doesn't use it and the runtime kernel falls back to half_pixel for
// safety if it ever reaches it.

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace hip {
namespace {

// Local copy of the wrap_* runtime symbol name.  Carried here so this
// translation unit doesn't depend on a particular line in
// HipToLLVMUtils.h, which has been getting reverted by sibling-agent
// merges in flight.  Must match the canonical name exported by
// `lib/Runtime/real/resize.cpp::wrap_resize`.
constexpr const char *kLocalWrapResize = "wrap_resize";

// File-local copy of the helper from Tier3CompareLowering.cpp /
// Tier6Lowering.cpp -- materialise a small int64 array as an LLVM alloca
// + per-element stores and return the alloca pointer.  Keeping it local
// avoids a header churn while sibling lowerings are still landing.
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

// Compute row-major element strides for `shape`.  Last axis has stride 1.
static SmallVector<int64_t> computeRowMajorStrides(ArrayRef<int64_t> shape) {
  int64_t rank = shape.size();
  SmallVector<int64_t> strides(rank, 1);
  for (int64_t d = rank - 2; d >= 0; --d)
    strides[d] = strides[d + 1] * shape[d + 1];
  return strides;
}

//===----------------------------------------------------------------------===//
// hip.resize
//===----------------------------------------------------------------------===//

struct ResizeLowering : public ConvertOpToLLVMPattern<ResizeOp> {
  using ConvertOpToLLVMPattern<ResizeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ResizeOp op, OpAdaptor adaptor,
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
    if (!inType || !outType)
      return rewriter.notifyMatchFailure(
          op, "hip.resize lowering requires ranked memref operands");
    if (inType.getRank() != outType.getRank())
      return rewriter.notifyMatchFailure(
          op, "hip.resize input/output rank mismatch");
    if (inType.getRank() < 2)
      return rewriter.notifyMatchFailure(
          op, "hip.resize requires rank >= 2 (resize over last 2 dims)");

    int64_t mode = op.getMode();
    int64_t coordXform = op.getCoordTransform();
    if (mode < 0 || mode > 2)
      return rewriter.notifyMatchFailure(
          op, "hip.resize mode must be 0/1/2 (nearest/linear/cubic)");
    if (coordXform == 4)
      return rewriter.notifyMatchFailure(
          op,
          "hip.resize tf_crop_and_resize (4) not implemented; needs ROI input");
    if (coordXform < 0 || coordXform > 4)
      return rewriter.notifyMatchFailure(
          op, "hip.resize coord_transform out of range");

    int64_t dataType = getHipdnnDataType(outType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.resize unsupported element type (need f32/f16/bf16)");

    int64_t rank = inType.getRank();

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    auto inShape = getMemRefShape(inType, adaptor.getInput(), rewriter, loc);
    auto outShape = getMemRefShape(outType, adaptor.getOutput(), rewriter, loc);

    SmallVector<Value> inStrides(rank);
    Value acc = i64Const(1);
    for (int64_t d = rank - 1; d >= 0; --d) {
      inStrides[d] = acc;
      acc = LLVM::MulOp::create(rewriter, loc, acc, inShape[d]);
    }
    SmallVector<Value> outStrides(rank);
    acc = i64Const(1);
    for (int64_t d = rank - 1; d >= 0; --d) {
      outStrides[d] = acc;
      acc = LLVM::MulOp::create(rewriter, loc, acc, outShape[d]);
    }

    Value rankConst = i64Const(rank);
    Value dtypeConst = i64Const(dataType);
    Value modeConst = i64Const(mode);
    Value coordConst = i64Const(coordXform);
    Value cubicConst = LLVM::ConstantOp::create(
        rewriter, loc, f32Type,
        rewriter.getF32FloatAttr(op.getCubicCoeffA().convertToFloat()));

    Value inShapePtr =
        materialiseValueArray(inShape, i64Type, ptrType, rewriter, loc);
    Value inStridesPtr =
        materialiseValueArray(inStrides, i64Type, ptrType, rewriter, loc);
    Value outShapePtr =
        materialiseValueArray(outShape, i64Type, ptrType, rewriter, loc);
    Value outStridesPtr =
        materialiseValueArray(outStrides, i64Type, ptrType, rewriter, loc);

    // int wrap_resize(state, input, output,
    //                 in_shape, in_strides_elems,
    //                 out_shape, out_strides_elems,
    //                 rank, data_type, mode, coord_xform, cubic_coeff_a);
    SmallVector<Type, 12> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, ptrType,
        ptrType, i64Type, i64Type, i64Type, i64Type, f32Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kLocalWrapResize, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 12> args = {statePtr,      inPtr,        outPtr,
                                   inShapePtr,    inStridesPtr, outShapePtr,
                                   outStridesPtr, rankConst,    dtypeConst,
                                   modeConst,     coordConst,   cubicConst};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateResizeLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<ResizeLowering>(converter);
}

} // namespace hip
} // namespace mlir
