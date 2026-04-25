/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers hip.expand (ONNX Expand: numpy-style broadcast) to wrap_expand.
//
// At MLIR-time we know both the input and output shapes statically (the
// bufferize step uses the result type to size the destination memref),
// so the lowering simply materialises three i64 arrays --
//   - input shape       (length = in_rank)
//   - input row-major strides in elements (length = in_rank)
//   - output shape      (length = out_rank)
// -- on the stack and threads them through to wrap_expand along with
// the in/out ranks and the runtime data type.  The runtime wrapper
// right-aligns to the output rank and zeros out the broadcast strides
// before calling the kernel.
//
// Mirrors the call-site pattern of Tier3CompareLowering / SliceLowering
// for the i64 array materialisation.

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace hip {
namespace {

// Local copy of the helper used in Tier3CompareLowering / SliceLowering.
// Kept private so we don't fight the parallel Pad agent over a shared
// utility header right now; can be hoisted later if more lowerings need it.
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

struct ExpandLowering : public ConvertOpToLLVMPattern<ExpandOp> {
  using ConvertOpToLLVMPattern<ExpandOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ExpandOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inType || !outType)
      return rewriter.notifyMatchFailure(
          op, "hip.expand expects ranked memref operands");
    if (!inType.hasStaticShape() || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "hip.expand lowering requires static shapes");

    int64_t inRank = inType.getRank();
    int64_t outRank = outType.getRank();
    if (inRank > outRank)
      return rewriter.notifyMatchFailure(
          op, "hip.expand input rank exceeds output rank");

    int64_t dataType = getHipdnnDataType(outType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.expand unsupported element type");

    SmallVector<int64_t> inShape(inType.getShape().begin(),
                                 inType.getShape().end());
    SmallVector<int64_t> outShape(outType.getShape().begin(),
                                  outType.getShape().end());

    // Row-major contiguous element strides of the dense input tensor.
    SmallVector<int64_t> inStrides(inRank, 0);
    int64_t acc = 1;
    for (int64_t d = inRank - 1; d >= 0; --d) {
      inStrides[d] = acc;
      acc *= inShape[d];
    }

    // Validate broadcast: each (right-aligned) input dim must be 1 or
    // equal to the matching output dim.  Surface mismatches early
    // instead of producing silent garbage at runtime.
    int64_t rankDiff = outRank - inRank;
    for (int64_t d = 0; d < outRank; ++d) {
      int64_t src = d - rankDiff;
      int64_t inDim = (src < 0) ? 1 : inShape[src];
      int64_t outDim = outShape[d];
      if (inDim != 1 && inDim != outDim)
        return rewriter.notifyMatchFailure(
            op, "hip.expand: input dim must be 1 or equal to output dim");
    }

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value inShapePtr =
        materialiseI64Array(inShape, i64Type, ptrType, rewriter, loc);
    Value inStridePtr =
        materialiseI64Array(inStrides, i64Type, ptrType, rewriter, loc);
    Value outShapePtr =
        materialiseI64Array(outShape, i64Type, ptrType, rewriter, loc);
    Value inRankVal = i64Const(inRank);
    Value outRankVal = i64Const(outRank);
    Value dtypeVal = i64Const(dataType);

    // int wrap_expand(state, input, output, in_shape, in_strides_elems,
    //                 out_shape, in_rank, out_rank, data_type)
    SmallVector<Type, 9> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       ptrType, ptrType, i64Type, i64Type,
                                       i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapExpand, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {statePtr,    inputPtr,    outputPtr,
                                  inShapePtr,  inStridePtr, outShapePtr,
                                  inRankVal,   outRankVal,  dtypeVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateExpandLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<ExpandLowering>(converter);
}

} // namespace hip
} // namespace mlir
