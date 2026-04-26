/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers hip.binary_elementwise (Div / Pow) to wrap_elementwise_binary in
// the runtime bitcode.  Shapes and broadcast strides are read from memref
// descriptors at runtime, supporting both static and dynamic dimensions.
//
// Broadcasting follows ONNX numpy-style semantics:
//   - left-pad lower-rank shapes with 1
//   - per-axis stride = product(trailing_dims) when extent > 1, else 0

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace hip {
namespace {

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

    int64_t dataType = getHipdnnDataType(outType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.binary_elementwise unsupported element type");

    int64_t rank = outType.getRank();

    auto outShape = getMemRefShape(outType, adaptor.getOutput(), rewriter, loc);
    auto lhsShape = getMemRefShape(lhsType, adaptor.getLhs(), rewriter, loc);
    auto rhsShape = getMemRefShape(rhsType, adaptor.getRhs(), rewriter, loc);

    auto lhsStrides =
        computeBroadcastStridesSSA(lhsShape, outShape, rewriter, loc);
    auto rhsStrides =
        computeBroadcastStridesSSA(rhsShape, outShape, rewriter, loc);

    Value numElems = computeNumElements(outType, adaptor.getOutput(),
                                        rewriter, loc);

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value dtype = i64Const(dataType);
    Value kind = i64Const(static_cast<int64_t>(op.getKind()));
    Value rankConst = i64Const(rank);

    Value outShapePtr =
        materialiseValueArray(outShape, i64Type, ptrType, rewriter, loc);
    Value lhsStridePtr =
        materialiseValueArray(lhsStrides, i64Type, ptrType, rewriter, loc);
    Value rhsStridePtr =
        materialiseValueArray(rhsStrides, i64Type, ptrType, rewriter, loc);

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
