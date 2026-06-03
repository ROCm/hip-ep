/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.miopen.add / hip.mul  (element-wise binary ops)
// Lowered call: hip_miopen_{add,mul}(handle, A_ptr, B_ptr, C_ptr, numA, numB)
// numA/numB are computed as the product of all memref dimensions for each
// operand.  When numB == 1 (scalar broadcast), the runtime broadcasts B
// over all elements of A.
template <typename OpTy>
struct MiopenBinaryOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
  const char *funcName;

  MiopenBinaryOpLowering(const LLVMTypeConverter &converter, const char *name)
      : ConvertOpToLLVMPattern<OpTy>(converter), funcName(name) {}

  Value computeNumElements(MemRefType type, Value descriptor,
                           ConversionPatternRewriter &rewriter,
                           Location loc) const {
    Type indexType = this->getIndexType();
    int rank = type.getRank();
    Value num = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                         rewriter.getIndexAttr(1));
    for (int dimIdx : llvm::seq<int>(rank))
      num = LLVM::MulOp::create(
          rewriter, loc, num,
          MemRefDescriptor(descriptor).size(rewriter, loc, dimIdx));
    return num;
  }

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    Type voidType = this->getVoidType();
    Type ptrType = this->getPtrType();
    Type indexType = this->getIndexType();

    SmallVector<Type> paramTypes = {ptrType, ptrType,   ptrType,
                                    ptrType, indexType, indexType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, funcName, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    auto aType = cast<MemRefType>(op.getA().getType());
    auto bType = cast<MemRefType>(op.getB().getType());
    Value numA = computeNumElements(aType, adaptor.getA(), rewriter, loc);
    Value numB = computeNumElements(bType, adaptor.getB(), rewriter, loc);

    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getA(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getB(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getC(), rewriter, loc),
        numA,
        numB};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// Unified lowering for elementwise binary ops (hip.mul, hip.add, ...)
//   -> wrap_miopenOpTensor(state, lhs_ptr, rhs_ptr, out_ptr,
//       lhs_n, lhs_c, lhs_h, lhs_w,
//       rhs_n, rhs_c, rhs_h, rhs_w,
//       out_n, out_c, out_h, out_w,
//       data_type, tensor_op)
//
// Full 4D shapes are passed to enable MIOpen-native broadcasting.
// E.g. Add(memref<1x128x32xf16>, memref<32xf16>) passes:
//   lhs=[1,1,128,32], rhs=[1,1,1,32], out=[1,1,128,32]
// MIOpen broadcasts rhs dims that are 1 against lhs automatically.
//
// NOTE: The 4D shape passing is a workaround for MIOpen's
// miopenSetNdTensorDescriptorWithLayout API. Will be replaced when hipdnn
// elementwise support is available.
template <typename OpTy, HipdnnTensorOp tensorOpEnum>
struct ElementwiseOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    Type ptrType = this->getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto lhsType = cast<MemRefType>(op.getLhs().getType());
    auto rhsType = cast<MemRefType>(op.getRhs().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    if (lhsType.getRank() > 4 || rhsType.getRank() > 4 ||
        outputType.getRank() > 4)
      return rewriter.notifyMatchFailure(
          op, "rank > 4 unsupported by MIOpen 4D descriptor API");

    auto lhsDims =
        extractShape4D(lhsType, adaptor.getLhs(), rewriter, loc, i64Type);
    auto rhsDims =
        extractShape4D(rhsType, adaptor.getRhs(), rewriter, loc, i64Type);
    auto outDims =
        extractShape4D(outputType, adaptor.getOutput(), rewriter, loc, i64Type);

    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    // 18 params: state + 3 data ptrs + 12 shape dims + data_type + tensor_op
    SmallVector<Type, 18> paramTypes(4, ptrType);
    paramTypes.append(14, i64Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMiopenOpTensor, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 18> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getLhs(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getRhs(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc)};
    args.append(lhsDims.begin(), lhsDims.end());
    args.append(rhsDims.begin(), rhsDims.end());
    args.append(outDims.begin(), outDims.end());
    args.push_back(createI64Const(dataType));
    args.push_back(createI64Const(tensorOpEnum));

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.sub(handle, lhs, rhs, output)
//   -> wrap_miopenTensorOp(state, lhs, rhs, output, num_lhs, num_rhs,
//                          data_type, tensor_op=0)
// Supports both static and dynamic shapes (computes num_lhs, num_rhs at
// runtime).
struct SubOpLowering : public ConvertOpToLLVMPattern<SubOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    // Helper to create i64 constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Helper to compute num_elements for a memref (static or dynamic)
    auto computeNumElements = [&](MemRefType type, Value descriptor) -> Value {
      Value num = createI64Const(1);
      MemRefDescriptor desc(descriptor);
      for (auto dimIdx : llvm::seq<int64_t>(type.getRank())) {
        Value dimSize;
        if (type.isDynamicDim(dimIdx)) {
          dimSize = desc.size(rewriter, loc, dimIdx);
        } else {
          dimSize = createI64Const(type.getDimSize(dimIdx));
        }
        num = LLVM::MulOp::create(rewriter, loc, num, dimSize);
      }
      return num;
    };

    Value statePtr = adaptor.getCtx();
    Value lhsPtr = extractContiguousMemRefPtr(adaptor.getLhs(), rewriter, loc);
    Value rhsPtr = extractContiguousMemRefPtr(adaptor.getRhs(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Compute num_elements (supports dynamic shapes)
    Value numElementsVal = computeNumElements(outputType, adaptor.getOutput());

    unsigned elementSizeBytes =
        outputType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elementSizeBytes);

    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       ptrType, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapElementwiseSub, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr,  lhsPtr,         rhsPtr,
                                  outputPtr, numElementsVal, elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateElementwiseLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<ElementwiseOpLowering<MulOp, kTensorOpMul>,
               ElementwiseOpLowering<AddOp, kTensorOpAdd>,
               ElementwiseOpLowering<MinOp, kTensorOpMin>,
               ElementwiseOpLowering<MaxOp, kTensorOpMax>, SubOpLowering>(
      converter);
  patterns.insert<MiopenBinaryOpLowering<MiopenAddOp>>(converter, kMiopenAdd);
}

} // namespace hip
} // namespace mlir
