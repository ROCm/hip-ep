/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers hip.lstm to a single call into the runtime wrapper:
//
//   wrap_miopenRNNForwardInference(state,
//                                  x, seq_len, batch, input_size,
//                                  w, r, b,            // b nullable
//                                  hx, cx,             // hx/cx nullable
//                                  y, hy, cy,
//                                  hidden_size, direction, data_type)
//
// All shape information is pulled off the memref descriptors at lowering
// time so the wrapper signature stays uniform across f16/f32/bf16.

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// Self-contained symbol name so the file builds even if a sibling agent's
// edit drops kWrapMiopenRNNForwardInference from HipToLLVMUtils.h.
inline constexpr const char *kLstmWrapMiopenRNNForwardInference =
    "wrap_miopenRNNForwardInference";

struct LstmOpLowering : public ConvertOpToLLVMPattern<LstmOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LstmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    if (!inputType || inputType.getRank() != 3)
      return rewriter.notifyMatchFailure(
          op, "hip.lstm input must be a rank-3 memref (T, N, in_size)");

    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.lstm unsupported element type");

    Value statePtr = adaptor.getCtx();
    Value xPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value wPtr = extractMemRefPtr(adaptor.getWeights(), rewriter, loc);
    Value rPtr = extractMemRefPtr(adaptor.getRecurrence(), rewriter, loc);
    Value bPtr = extractOptionalMemRefPtr(adaptor.getBias(), rewriter, loc);
    Value hxPtr =
        extractOptionalMemRefPtr(adaptor.getInitialH(), rewriter, loc);
    Value cxPtr =
        extractOptionalMemRefPtr(adaptor.getInitialC(), rewriter, loc);
    Value yPtr = extractMemRefPtr(adaptor.getYOut(), rewriter, loc);
    Value hyPtr = extractMemRefPtr(adaptor.getYHOut(), rewriter, loc);
    Value cyPtr = extractMemRefPtr(adaptor.getYCOut(), rewriter, loc);

    Value seqLen =
        getMemRefDimSize(inputType, 0, adaptor.getInput(), rewriter, loc);
    Value batch =
        getMemRefDimSize(inputType, 1, adaptor.getInput(), rewriter, loc);
    Value inputSize =
        getMemRefDimSize(inputType, 2, adaptor.getInput(), rewriter, loc);

    Value hiddenSize = i64Const(op.getHiddenSize());
    Value direction = i64Const(op.getDirection());
    Value dtypeConst = i64Const(dataType);

    SmallVector<Type, 16> paramTypes = {
        ptrType, // state
        ptrType, // x
        i64Type, // seq_len
        i64Type, // batch
        i64Type, // input_size
        ptrType, // w
        ptrType, // r
        ptrType, // b (nullable)
        ptrType, // hx (nullable)
        ptrType, // cx (nullable)
        ptrType, // y
        ptrType, // hy
        ptrType, // cy
        i64Type, // hidden_size
        i64Type, // direction
        i64Type  // data_type
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module,
                               kLstmWrapMiopenRNNForwardInference, paramTypes,
                               i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 16> args = {
        statePtr,  xPtr,    seqLen,   batch,      inputSize, wPtr,
        rPtr,      bPtr,    hxPtr,    cxPtr,      yPtr,      hyPtr,
        cyPtr,     hiddenSize, direction, dtypeConst};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateLstmLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<LstmOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
