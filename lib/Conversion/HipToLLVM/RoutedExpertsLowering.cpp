// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "HipToLLVMUtils.h"

namespace mlir::hip {
namespace {
struct RoutedExpertsLowering : ConvertOpToLLVMPattern<RoutedExpertsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(RoutedExpertsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ptr = getPtrType();
    auto i64 = rewriter.getI64Type();
    auto constant = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto address = [&](Value value) {
      return extractContiguousMemRefPtr(value, rewriter, loc);
    };
    auto array = LLVM::LLVMArrayType::get(ptr, adaptor.getParameters().size());
    Value table =
        LLVM::AllocaOp::create(rewriter, loc, ptr, array, constant(1), 8);
    for (auto [i, value] : llvm::enumerate(adaptor.getParameters())) {
      Value entry =
          LLVM::GEPOp::create(rewriter, loc, ptr, ptr, table, constant(i));
      LLVM::StoreOp::create(rewriter, loc, address(value), entry);
    }
    auto inputType = cast<MemRefType>(op.getInput().getType());
    Value tokens = constant(1);
    for (int axis = 0; axis < 2; ++axis)
      tokens = LLVM::MulOp::create(
          rewriter, loc, tokens,
          getMemRefDimSize(inputType, axis, adaptor.getInput(), rewriter, loc));
    SmallVector<Type> types(6, ptr);
    types.append(6, i64);
    auto function = LLVM::lookupOrCreateFn(
        rewriter, op->getParentOfType<ModuleOp>(), "wrap_routed_experts", types,
        rewriter.getI32Type());
    if (failed(function))
      return failure();
    SmallVector<Value> args = {adaptor.getHandle(),
                               address(adaptor.getInput()),
                               address(adaptor.getIndices()),
                               address(adaptor.getGates()),
                               table,
                               address(adaptor.getOutput()),
                               tokens,
                               constant(inputType.getDimSize(2)),
                               constant(op.getIntermediateSize()),
                               constant(op.getNumExperts()),
                               constant(op.getK()),
                               constant(op.getBlockSize())};
    LLVM::CallOp::create(rewriter, loc, *function, args);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace
void populateRoutedExpertsLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns) {
  patterns.add<RoutedExpertsLowering>(converter);
}
} // namespace mlir::hip
