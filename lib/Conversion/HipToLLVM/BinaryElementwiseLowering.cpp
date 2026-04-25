/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers hip.binary_elementwise (Div / Pow) to wrap_elementwise_binary in
// the runtime bitcode.  We stack-allocate three i64 arrays of length `rank`
// for the output shape and the per-axis lhs / rhs strides (in elements;
// 0 = broadcast on that axis).  All shapes here are static (the conversion
// pass doesn't accept dynamic shapes), so the strides can be folded at
// compile time.
//
// The compute follows ONNX numpy-style broadcasting:
//   - left-pad lower-rank shapes with 1
//   - per-axis stride = product(trailing_dims) when extent > 1, else 0

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace hip {
namespace {

/// Compute element strides (innermost-first then unchanged) and broadcast
/// mask for an input shape against the output shape.  Returns an array of
/// stride values, length == out_rank.  A returned stride of 0 means
/// "broadcast over that axis".
static SmallVector<int64_t> computeBroadcastStrides(ArrayRef<int64_t> inShape,
                                                    ArrayRef<int64_t> outShape) {
  int64_t rank = outShape.size();
  SmallVector<int64_t> padded(rank, 1);
  int64_t offset = rank - inShape.size();
  for (size_t i = 0; i < inShape.size(); ++i)
    padded[offset + i] = inShape[i];

  // Row-major contiguous strides of the input (after left-padding with 1s).
  SmallVector<int64_t> strides(rank, 0);
  int64_t acc = 1;
  for (int64_t d = rank - 1; d >= 0; --d) {
    strides[d] = padded[d] == 1 ? 0 : acc;
    acc *= padded[d];
  }
  return strides;
}

/// Stack-allocate an i64 array of length `vals.size()`, store each value,
/// return the LLVM pointer to the first element.
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

struct BinaryElementwiseLowering
    : public ConvertOpToLLVMPattern<BinaryElementwiseOp> {
  using ConvertOpToLLVMPattern<BinaryElementwiseOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(BinaryElementwiseOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value lhsPtr = extractMemRefPtr(adaptor.getLhs(), rewriter, loc);
    Value rhsPtr = extractMemRefPtr(adaptor.getRhs(), rewriter, loc);
    Value outPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
    auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
    if (!outType || !lhsType || !rhsType)
      return rewriter.notifyMatchFailure(
          op, "hip.binary_elementwise expects ranked memref operands");
    if (!outType.hasStaticShape() || !lhsType.hasStaticShape() ||
        !rhsType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "hip.binary_elementwise lowering requires static shapes");

    int64_t dataType = getHipdnnDataType(outType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.binary_elementwise unsupported element type");

    int64_t rank = outType.getRank();
    SmallVector<int64_t> outShape(outType.getShape().begin(),
                                  outType.getShape().end());
    SmallVector<int64_t> lhsStrides =
        computeBroadcastStrides(lhsType.getShape(), outShape);
    SmallVector<int64_t> rhsStrides =
        computeBroadcastStrides(rhsType.getShape(), outShape);

    int64_t numElements = 1;
    for (int64_t d : outShape)
      numElements *= d;

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value numElems = i64Const(numElements);
    Value dtype = i64Const(dataType);
    Value kind = i64Const(static_cast<int64_t>(op.getKind()));
    Value rankConst = i64Const(rank);

    Value outShapePtr =
        materialiseI64Array(outShape, i64Type, ptrType, rewriter, loc);
    Value lhsStridePtr =
        materialiseI64Array(lhsStrides, i64Type, ptrType, rewriter, loc);
    Value rhsStridePtr =
        materialiseI64Array(rhsStrides, i64Type, ptrType, rewriter, loc);

    // int wrap_elementwise_binary(RuntimeState*, void* lhs, void* rhs,
    //                             void* out, int64 num_elements,
    //                             int64 data_type, int64 kind, int64 rank,
    //                             const int64* out_shape,
    //                             const int64* lhs_strides_elems,
    //                             const int64* rhs_strides_elems)
    SmallVector<Type, 11> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                        i64Type, i64Type, i64Type, i64Type,
                                        ptrType, ptrType, ptrType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapElementwiseBinary, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 11> args = {statePtr,    lhsPtr,       rhsPtr,
                                   outPtr,      numElems,     dtype,
                                   kind,        rankConst,    outShapePtr,
                                   lhsStridePtr, rhsStridePtr};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateBinaryElementwiseLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<BinaryElementwiseLowering>(converter);
}

} // namespace hip
} // namespace mlir
