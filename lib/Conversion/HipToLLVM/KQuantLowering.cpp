/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// hip.quantize_q8_1, hip.skip_rms_norm_q8_1, hip.kquant_matmul,
// hip.kquant_matmul_q8_1 and hip.kquant_swiglu_q8_1 to their runtime wrappers
// (lib/Runtime/real/kquant.cpp). Row-major operands go as bare pointers; the
// leading dimensions are folded into a batch count.

#include "HipToLLVMUtils.h"

#include "hip/Dialect/IR/HipKQuant.h"

namespace mlir {
namespace hip {
namespace {

Value createI64(ConversionPatternRewriter &rewriter, Location loc,
                int64_t value) {
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI64Type(),
                                  rewriter.getI64IntegerAttr(value));
}

Value createFormatId(ConversionPatternRewriter &rewriter, Location loc,
                     StringRef format) {
  return createI64(rewriter, loc,
                   kquant::getKernelFormatId(*kquant::parseFormat(format)));
}

SmallVector<Value> getDimensions(MemRefType type, Value descriptor,
                                 ConversionPatternRewriter &rewriter,
                                 Location loc) {
  SmallVector<Value> dimensions;
  for (int64_t dimension : llvm::seq<int64_t>(type.getRank()))
    dimensions.push_back(
        getMemRefDimSize(type, dimension, descriptor, rewriter, loc));
  return dimensions;
}

/// The rows (dimension -2) and the product of the dimensions before them.
std::pair<Value, Value> getRowsAndBatch(MemRefType type, Value descriptor,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc) {
  const int64_t rank = type.getRank();
  Value rows = getMemRefDimSize(type, rank - 2, descriptor, rewriter, loc);
  Value batch = createI64(rewriter, loc, 1);
  for (int64_t dimension : llvm::seq<int64_t>(rank - 2)) {
    batch = LLVM::MulOp::create(
        rewriter, loc, batch,
        getMemRefDimSize(type, dimension, descriptor, rewriter, loc));
  }
  return {rows, batch};
}

/// Calls `name`, which takes the runtime state, `pointers` operands' data and
/// then `scalars`; a failure leaves the graph when the module asks for it.
LogicalResult callWrapper(Operation *op, StringRef name, Value ctx,
                          ValueRange pointers, ValueRange scalars,
                          ConversionPatternRewriter &rewriter) {
  Location loc = op->getLoc();
  ModuleOp module = op->getParentOfType<ModuleOp>();
  auto ptrType = LLVM::LLVMPointerType::get(rewriter.getContext());
  SmallVector<Value> arguments = {ctx};
  SmallVector<Type> parameterTypes = {ptrType};
  for (Value pointer : pointers) {
    arguments.push_back(extractContiguousMemRefPtr(pointer, rewriter, loc));
    parameterTypes.push_back(ptrType);
  }
  for (Value scalar : scalars) {
    arguments.push_back(scalar);
    parameterTypes.push_back(scalar.getType());
  }
  FailureOr<LLVM::LLVMFuncOp> function = LLVM::lookupOrCreateFn(
      rewriter, module, name, parameterTypes, rewriter.getI32Type());
  if (failed(function))
    return failure();
  auto call = LLVM::CallOp::create(rewriter, loc, *function, arguments);
  if (requiresFallibleMemRefCopies(module))
    markFallibleRuntimeCall(call);
  rewriter.eraseOp(op);
  return success();
}

struct QuantizeQ8_1OpLowering : public ConvertOpToLLVMPattern<QuantizeQ8_1Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QuantizeQ8_1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto [rows, batch] =
        getRowsAndBatch(cast<MemRefType>(op.getActivation().getType()),
                        adaptor.getActivation(), rewriter, loc);
    return callWrapper(op, kWrapQuantizeQ8_1, adaptor.getCtx(),
                       {adaptor.getActivation(), adaptor.getOutput()},
                       {rows, createI64(rewriter, loc, op.getK()), batch,
                        createI64(rewriter, loc, op.getSumMode()),
                        createI64(rewriter, loc, op.getRowsPerToken())},
                       rewriter);
  }
};

struct SkipRmsNormQ8_1OpLowering
    : public ConvertOpToLLVMPattern<SkipRmsNormQ8_1Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SkipRmsNormQ8_1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    // The wrapper checks every operand's extents against each other: the
    // input's, the skip's, gamma's length, the sum's and the Q8_1 output's.
    SmallVector<Value> scalars;
    auto appendLeadingDimensions = [&](Value value, Value descriptor) {
      SmallVector<Value> dimensions = getDimensions(
          cast<MemRefType>(value.getType()), descriptor, rewriter, loc);
      scalars.append(dimensions.begin(), dimensions.begin() + 3);
    };
    appendLeadingDimensions(op.getInput(), adaptor.getInput());
    appendLeadingDimensions(op.getSkip(), adaptor.getSkip());
    scalars.push_back(
        getMemRefDimSize(cast<MemRefType>(op.getGamma().getType()), 0,
                         adaptor.getGamma(), rewriter, loc));
    appendLeadingDimensions(op.getOutputs()[0], adaptor.getOutputs()[0]);
    appendLeadingDimensions(op.getOutputs()[1], adaptor.getOutputs()[1]);
    scalars.push_back(createI64(rewriter, loc, op.getK()));
    scalars.push_back(LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getF32Type(), op.getEpsilonAttr()));
    scalars.push_back(createI64(rewriter, loc, op.getSumMode()));
    scalars.push_back(createI64(rewriter, loc, op.getRowsPerToken()));
    return callWrapper(op, kWrapSkipRmsNormQ8_1, adaptor.getCtx(),
                       {adaptor.getInput(), adaptor.getSkip(),
                        adaptor.getGamma(), adaptor.getOutputs()[0],
                        adaptor.getOutputs()[1]},
                       scalars, rewriter);
  }
};

struct KQuantMatmulOpLowering : public ConvertOpToLLVMPattern<KQuantMatmulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(KQuantMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto [rows, batch] =
        getRowsAndBatch(cast<MemRefType>(op.getActivation().getType()),
                        adaptor.getActivation(), rewriter, loc);
    return callWrapper(op, kWrapKQuantMatmul, adaptor.getCtx(),
                       {adaptor.getActivation(), adaptor.getPackedWeights(),
                        adaptor.getOutput()},
                       {rows, createI64(rewriter, loc, op.getN()),
                        createI64(rewriter, loc, op.getK()), batch,
                        createFormatId(rewriter, loc, op.getFormat())},
                       rewriter);
  }
};

struct KQuantMatmulQ8_1OpLowering
    : public ConvertOpToLLVMPattern<KQuantMatmulQ8_1Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(KQuantMatmulQ8_1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto [rows, batch] =
        getRowsAndBatch(cast<MemRefType>(op.getQuantizedActivation().getType()),
                        adaptor.getQuantizedActivation(), rewriter, loc);
    return callWrapper(op, kWrapKQuantMatmulQ8_1, adaptor.getCtx(),
                       {adaptor.getQuantizedActivation(),
                        adaptor.getPackedWeights(), adaptor.getOutput()},
                       {rows, createI64(rewriter, loc, op.getN()),
                        createI64(rewriter, loc, op.getK()), batch,
                        createFormatId(rewriter, loc, op.getFormat()),
                        createI64(rewriter, loc, op.getSumMode()),
                        createI64(rewriter, loc, op.getRowsPerToken())},
                       rewriter);
  }
};

struct KQuantSwigluQ8_1OpLowering
    : public ConvertOpToLLVMPattern<KQuantSwigluQ8_1Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(KQuantSwigluQ8_1Op op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    SmallVector<Value> q8Dimensions =
        getDimensions(cast<MemRefType>(op.getQuantizedActivation().getType()),
                      adaptor.getQuantizedActivation(), rewriter, loc);
    SmallVector<Value> outputDimensions =
        getDimensions(cast<MemRefType>(op.getOutput().getType()),
                      adaptor.getOutput(), rewriter, loc);
    SmallVector<Value> scalars(q8Dimensions.begin(), q8Dimensions.begin() + 3);
    scalars.push_back(
        getMemRefDimSize(cast<MemRefType>(op.getGateWeights().getType()), 0,
                         adaptor.getGateWeights(), rewriter, loc));
    scalars.push_back(
        getMemRefDimSize(cast<MemRefType>(op.getUpWeights().getType()), 0,
                         adaptor.getUpWeights(), rewriter, loc));
    scalars.append(outputDimensions.begin(), outputDimensions.begin() + 3);
    scalars.push_back(createI64(rewriter, loc, op.getK()));
    scalars.push_back(createI64(rewriter, loc, op.getN()));
    scalars.push_back(createFormatId(rewriter, loc, op.getGateFormat()));
    scalars.push_back(createFormatId(rewriter, loc, op.getUpFormat()));
    scalars.push_back(createI64(rewriter, loc, op.getSumMode()));
    scalars.push_back(createI64(rewriter, loc, op.getRowsPerToken()));
    return callWrapper(op, kWrapKQuantSwigluQ8_1, adaptor.getCtx(),
                       {adaptor.getQuantizedActivation(),
                        adaptor.getGateWeights(), adaptor.getUpWeights(),
                        adaptor.getOutput()},
                       scalars, rewriter);
  }
};

// The cache shape gives C, G and D; the query B, S and H; the mask's batch is
// 1 or B. The mask's and the output's S go along for the wrapper to check
// against the query's.
struct TokenMajorGqaOpLowering
    : public ConvertOpToLLVMPattern<TokenMajorGqaOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TokenMajorGqaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto queryType = cast<MemRefType>(op.getQuery().getType());
    auto keyType = cast<MemRefType>(op.getKeyCache().getType());
    auto maskType = cast<MemRefType>(op.getMask().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    Value writtenRows = op.getWrittenRows();
    Value writtenRowsPointer =
        writtenRows
            ? extractContiguousMemRefPtr(adaptor.getWrittenRows(), rewriter,
                                         loc)
            : LLVM::ZeroOp::create(rewriter, loc, getPtrType()).getResult();
    Value writtenRowCount =
        writtenRows
            ? getMemRefDimSize(cast<MemRefType>(writtenRows.getType()), 3,
                               adaptor.getWrittenRows(), rewriter, loc)
            : createI64(rewriter, loc, 0);
    auto dim = [&](MemRefType type, unsigned dimension, Value descriptor) {
      return getMemRefDimSize(type, dimension, descriptor, rewriter, loc);
    };
    return callWrapper(
        op, kWrapTokenMajorGqa, adaptor.getCtx(),
        {adaptor.getQuery(), adaptor.getKeyCache(), adaptor.getValueCache(),
         adaptor.getMask(), adaptor.getOutput()},
        {dim(queryType, 0, adaptor.getQuery()),
         dim(queryType, 1, adaptor.getQuery()),
         dim(queryType, 2, adaptor.getQuery()),
         dim(keyType, 2, adaptor.getKeyCache()),
         dim(keyType, 1, adaptor.getKeyCache()),
         dim(queryType, 3, adaptor.getQuery()),
         dim(maskType, 0, adaptor.getMask()),
         dim(maskType, 2, adaptor.getMask()),
         dim(outputType, 1, adaptor.getOutput()),
         LLVM::ConstantOp::create(rewriter, loc, rewriter.getF32Type(),
                                  op.getScaleAttr()),
         writtenRowsPointer, writtenRowCount},
        rewriter);
  }
};

} // namespace

void populateKQuantLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<QuantizeQ8_1OpLowering, SkipRmsNormQ8_1OpLowering,
               KQuantMatmulOpLowering, KQuantMatmulQ8_1OpLowering,
               KQuantSwigluQ8_1OpLowering, TokenMajorGqaOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
