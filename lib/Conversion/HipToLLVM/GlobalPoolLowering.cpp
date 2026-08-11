/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.global_pool -> wrap_global_pool runtime call (avg / max / lp)
//===----------------------------------------------------------------------===//
//
// hip.global_pool(ctx) ins(%input  : memref<NxCxD1x...xDkxT, 1>)
//                      outs(%output : memref<NxCx1x...x1xT, 1>)
//                      {mode = M : i64, p = P : i64}
//
// ->
//
// llvm.call @wrap_global_pool(
//     state,
//     input_ptr,
//     output_ptr,
//     int64_t outer,        // N * C  — number of independent (n, c) slices
//     int64_t reduce_size,  // product of spatial dims (D_1 * ... * D_k)
//     int64_t data_type,    // HIPDNN_EP_DATATYPE_* enum
//     int64_t mode,         // HIPDNN_EP_GLOBAL_POOL_* enum (0=AVG, 1=MAX,
//     2=LP) int64_t p)            // LP norm exponent (only used when mode ==
//     LP)
// : (ptr, ptr, ptr, i64, i64, i64, i64, i64) -> i32
//
// `outer` and `reduce_size` are computed from the input descriptor so dynamic
// N / C / D_i all work. We deliberately compute them off the **input**
// descriptor — the output's spatial dims are static "1", so its dynamic dims
// are only N and C, which match the input's first two; reading those off the
// input means we don't depend on the output having any dynamic sizes
// materialised by the upstream tensor.empty / pool-alloc pipeline.
//
// `mode` selects the reduction kernel; `p` is read by the runtime only on the
// LP branch (it is forwarded unconditionally so the call signature stays
// fixed across modes). ONNX `p` is an int64 but on the runtime side we only
// rely on small positive values (>=1, validated upstream in OnnxToHip).

struct GlobalPoolOpLowering : public ConvertOpToLLVMPattern<GlobalPoolOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GlobalPoolOpLowering)
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GlobalPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inputType = cast<MemRefType>(op.getInput().getType());
    int64_t rank = inputType.getRank();
    if (rank < 3)
      return rewriter.notifyMatchFailure(
          op, "global_pool needs rank >= 3 (N, C, spatial...)");

    Value inputDesc = adaptor.getInput();

    // outer = N * C  — product of the first two dims.
    Value outer = createI64Const(1);
    for (int64_t d : {0, 1}) {
      Value dim = getMemRefDimSize(inputType, static_cast<unsigned>(d),
                                   inputDesc, rewriter, loc);
      outer = LLVM::MulOp::create(rewriter, loc, outer, dim);
    }

    // reduce_size = product of D_2 ... D_{rank-1}.
    Value reduceSize = createI64Const(1);
    for (int64_t d : llvm::seq<int64_t>(2, rank)) {
      Value dim = getMemRefDimSize(inputType, static_cast<unsigned>(d),
                                   inputDesc, rewriter, loc);
      reduceSize = LLVM::MulOp::create(rewriter, loc, reduceSize, dim);
    }

    Type elemType = inputType.getElementType();
    int64_t dataType = getHipdnnDataType(elemType);
    if (dataType < 0 || (dataType > 2 && dataType != 6)) {
      std::string msg;
      llvm::raw_string_ostream os(msg);
      os << "unsupported element type '" << elemType
         << "' for global_pool. Only f16, f32, bf16, f64 are supported";
      return rewriter.notifyMatchFailure(op, os.str());
    }
    Value dataTypeVal = createI64Const(dataType);

    Value modeVal = createI64Const(op.getMode());
    Value pVal = createI64Const(op.getP());

    SmallVector<Type, 8> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type, i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGlobalPool, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 8> args = {statePtr,   inputPtr,    outputPtr, outer,
                                  reduceSize, dataTypeVal, modeVal,   pVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGlobalPoolLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns) {
  patterns.add<GlobalPoolOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
