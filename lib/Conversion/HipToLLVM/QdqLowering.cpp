/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.quantize_linear   -> wrap_quantize_linear   (17 params)
// hip.dequantize_linear -> wrap_dequantize_linear (15 params)
//===----------------------------------------------------------------------===//

// Stack-allocates i64[max(rank, 1)] so rank-0 still has a buffer, reading
// dynamic dims from the descriptor rather than the kDynamic sentinel.
Value emitShapeArray(Type ptrType, MemRefType type, Value descriptor, Value one,
                     ConversionPatternRewriter &rewriter, Location loc) {
  Type i32Type = rewriter.getI32Type();
  Type i64Type = rewriter.getI64Type();
  int rank = type.getRank();
  auto arrType = LLVM::LLVMArrayType::get(i64Type, std::max(rank, 1));
  Value arr = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
  for (auto i : llvm::seq<int64_t>(0, rank)) {
    Value dim = getMemRefDimSize(type, static_cast<unsigned>(i), descriptor,
                                 rewriter, loc);
    Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                         rewriter.getI32IntegerAttr(i));
    Value elemPtr =
        LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, arr, idx);
    LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
  }
  return arr;
}

LogicalResult isDataTypeSupported(Operation *op, Type quantElem, Type floatElem,
                                  Type scaleElem,
                                  ConversionPatternRewriter &rewriter) {
  // supported: int8, uint8, int16, uint16, float32, float16
  auto isQuantStorage = [](Type t) {
    return t.isUnsignedInteger(8) || t.isSignedInteger(8) ||
           t.isSignlessInteger(8) || t.isUnsignedInteger(16) ||
           t.isSignedInteger(16) || t.isSignlessInteger(16);
  };
  auto isFloat = [](Type t) { return t.isF32() || t.isF16(); };

  if (!isQuantStorage(quantElem))
    return rewriter.notifyMatchFailure(
        op, "unsupported quantized element type; expected i8/ui8/i16/ui16");
  if (!isFloat(floatElem) || !isFloat(scaleElem))
    return rewriter.notifyMatchFailure(
        op, "unsupported float element type; expected f32/f16");
  return success();
}

// Value width the runtime should assume for the quantized side. A packed
// 4-bit tensor keeps its i8/ui8 element type -- which carries only the
// signedness -- and its logical shape, so the width has to travel separately.
// Anything but 8-bit storage under the flag means the producer and this
// lowering disagree, which would hand the runtime a stride nothing agreed on.
FailureOr<int64_t> resolveQuantBits(Operation *op, Type quantElem,
                                    bool packedInt4,
                                    ConversionPatternRewriter &rewriter) {
  unsigned bits = quantElem.getIntOrFloatBitWidth();
  if (!packedInt4)
    return static_cast<int64_t>(bits);
  if (bits != 8)
    return rewriter.notifyMatchFailure(
        op, "packed_int4 requires 8-bit storage for the quantized side");
  return static_cast<int64_t>(4);
}

struct QuantizeLinearOpLowering
    : public ConvertOpToLLVMPattern<QuantizeLinearOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QuantizeLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto scaleType = cast<MemRefType>(op.getScale().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    Type quantElem = outputType.getElementType();
    if (failed(isDataTypeSupported(op, quantElem, inputType.getElementType(),
                                   scaleType.getElementType(), rewriter)))
      return failure();

    FailureOr<int64_t> outputBits =
        resolveQuantBits(op, quantElem, op.getPackedInt4(), rewriter);
    if (failed(outputBits))
      return failure();

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value one = createI64Const(1);

    SmallVector<Type, 17> paramTypes = {
        ptrType,                   // state
        ptrType, ptrType, ptrType, // input, scale, zero_point (nullable)
        ptrType,                   // output
        ptrType, i64Type,          // input_shape, input_rank
        ptrType, i64Type,          // scale_shape, scale_rank
        i64Type, i64Type,          // axis, block_size
        i64Type, i64Type,          // precision, saturate
        i64Type, i64Type, i64Type, // input_dtype, scale_dtype, output_dtype
        i64Type,                   // output_bits
    };
    SmallVector<Value, 17> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getScale(), rewriter, loc),
        extractOptionalMemRefPtr(adaptor.getZeroPoint(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        emitShapeArray(ptrType, inputType, adaptor.getInput(), one, rewriter,
                       loc),
        createI64Const(inputType.getRank()),
        emitShapeArray(ptrType, scaleType, adaptor.getScale(), one, rewriter,
                       loc),
        createI64Const(scaleType.getRank()),
        createI64Const(op.getAxis()),
        createI64Const(op.getBlockSize()),
        createI64Const(op.getPrecision()),
        createI64Const(op.getSaturate()),
        createI64Const(getHipdnnDataType(inputType.getElementType())),
        createI64Const(getHipdnnDataType(scaleType.getElementType())),
        createI64Const(getHipdnnDataType(quantElem)),
        createI64Const(*outputBits),
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQuantizeLinear, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct DequantizeLinearOpLowering
    : public ConvertOpToLLVMPattern<DequantizeLinearOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DequantizeLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto scaleType = cast<MemRefType>(op.getScale().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    Type quantElem = inputType.getElementType();
    if (failed(isDataTypeSupported(op, quantElem, outputType.getElementType(),
                                   scaleType.getElementType(), rewriter)))
      return failure();

    FailureOr<int64_t> inputBits =
        resolveQuantBits(op, quantElem, op.getPackedInt4(), rewriter);
    if (failed(inputBits))
      return failure();

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value one = createI64Const(1);

    SmallVector<Type, 15> paramTypes = {
        ptrType,                   // state
        ptrType, ptrType, ptrType, // input, scale, zero_point (nullable)
        ptrType,                   // output
        ptrType, i64Type,          // input_shape, input_rank
        ptrType, i64Type,          // scale_shape, scale_rank
        i64Type, i64Type,          // axis, block_size
        i64Type, i64Type, i64Type, // input_dtype, scale_dtype, output_dtype
        i64Type,                   // input_bits
    };
    SmallVector<Value, 15> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getScale(), rewriter, loc),
        extractOptionalMemRefPtr(adaptor.getZeroPoint(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        emitShapeArray(ptrType, inputType, adaptor.getInput(), one, rewriter,
                       loc),
        createI64Const(inputType.getRank()),
        emitShapeArray(ptrType, scaleType, adaptor.getScale(), one, rewriter,
                       loc),
        createI64Const(scaleType.getRank()),
        createI64Const(op.getAxis()),
        createI64Const(op.getBlockSize()),
        createI64Const(getHipdnnDataType(quantElem)),
        createI64Const(getHipdnnDataType(scaleType.getElementType())),
        createI64Const(getHipdnnDataType(outputType.getElementType())),
        createI64Const(*inputBits),
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapDequantizeLinear, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQdqLoweringPatterns(const LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns) {
  patterns.insert<QuantizeLinearOpLowering, DequantizeLinearOpLowering>(
      converter);
}

} // namespace hip
} // namespace mlir
