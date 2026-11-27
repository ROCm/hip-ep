/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace hip {
namespace {

// hip.winograd_conv(%ctx) ins(%x, %w[, %b]) outs(%y)
//   -> wrap_winograd_conv(state, slot, kernel_binary, kernel_name,
//                         block_size, grid_size, x, w, b|null, y, K, C,
//                         full_transform)
// The code object and its symbol become module-level globals, as for
// hip.rocmlir; the runtime builds the filter store from %w once per op state.
struct WinogradConvOpLowering : public ConvertOpToLLVMPattern<WinogradConvOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static Value globalString(ModuleOp module, RewriterBase &rewriter,
                            Location loc, StringRef baseName, StringRef value) {
    std::string name = baseName.str();
    unsigned suffix = 0;
    while (module.lookupSymbol(name))
      name = (baseName + "_" + Twine(suffix++)).str();
    return LLVM::createGlobalString(loc, rewriter, name, value,
                                    LLVM::Linkage::Internal);
  }

  LogicalResult
  matchAndRewrite(WinogradConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getKernelBinary().empty() || op.getKernelName().empty())
      return rewriter.notifyMatchFailure(
          op, "hip.winograd_conv has no built kernel");
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    auto i64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    auto weightsType = cast<MemRefType>(op.getWeights().getType());
    const int64_t K = weightsType.getDimSize(0), C = weightsType.getDimSize(1);

    // The ELF carries embedded NULs, so the global keeps its exact length; the
    // symbol is NUL-terminated for the C consumer.
    StringRef name = op.getKernelName();
    Value binary = globalString(module, rewriter, loc, (name + "_binary").str(),
                                op.getKernelBinary());
    std::string nameStr = name.str();
    nameStr.push_back('\0');
    Value funcName =
        globalString(module, rewriter, loc, (name + "_name").str(), nameStr);

    Value input = extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value weights =
        extractContiguousMemRefPtr(adaptor.getWeights(), rewriter, loc);
    Value bias =
        adaptor.getBias()
            ? extractContiguousMemRefPtr(adaptor.getBias(), rewriter, loc)
            : LLVM::ZeroOp::create(rewriter, loc, ptrType).getResult();
    // The output may be a packed concat slot: the kernel stores densely from
    // the first element, so pass alignedPtr + offset.
    Value output = extractMemRefDataPtr(
        adaptor.getOutput(), cast<MemRefType>(op.getOutput().getType()),
        getTypeConverter(), rewriter, loc);
    if (!output)
      return rewriter.notifyMatchFailure(op, "winograd_conv output pointer");

    SmallVector<Type, 13> paramTypes = {
        ptrType, i32Type, ptrType, ptrType, i64Type, i64Type, ptrType,
        ptrType, ptrType, ptrType, i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapWinogradConv, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();
    SmallVector<Value, 13> args = {adaptor.getCtx(),
                                   getOpStateSlotValue(op, rewriter, loc),
                                   binary,
                                   funcName,
                                   i64Const(op.getBlockSize()),
                                   i64Const(op.getGridSize()),
                                   input,
                                   weights,
                                   bias,
                                   output,
                                   i64Const(K),
                                   i64Const(C),
                                   i64Const(op.getFullTransform() ? 1 : 0)};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateWinogradConvLoweringPatterns(const LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns) {
  patterns.add<WinogradConvOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
