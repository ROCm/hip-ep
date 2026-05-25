/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "DimSpecMaterializer.h"
#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.shape(ctx, x, y) {element_dim_specs}
//   -> wrap_shape(state, output_ptr, num_elements, host_values)
//
// element_dim_specs is an ArrayAttr with one inner ArrayAttr per output
// element; each inner attr is a flat DimSpec encoding (same wire format
// the per-op `output_dim_specs` uses everywhere else).
//
// At lowering time we walk each per-element DimSpec and materialise an
// i64 SSA value. Static leaves become llvm.constants; RuntimeSlot leaves
// emit a call to hipdnn_ep_state_read_dim(state, slot_id); arithmetic
// nodes become arith.{add,sub,mul,...} pairs. The materialised i64s are
// packed into a host-side stack buffer via llvm.alloca and handed to
// wrap_shape, which does a single hipMemcpyAsync H2D into the output GPU
// memref. Total cost per call: a handful of constant ops + at most one
// runtime function call per dynamic dim + one H2D for ≤ rank * 8 bytes.
struct ShapeOpLowering : public ConvertOpToLLVMPattern<ShapeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    auto outputType = dyn_cast<MemRefType>(op.getY().getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(
          op, "hip.shape lowering expects a ranked memref output");
    if (outputType.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "hip.shape output must be rank-1");
    if (!outputType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(
          op, "hip.shape output must have i64 elements");
    if (!outputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "hip.shape output length must be statically known");

    ArrayAttr elementDimSpecs = op.getElementDimSpecsAttr();
    if (!elementDimSpecs)
      return rewriter.notifyMatchFailure(
          op, "hip.shape requires element_dim_specs attribute");
    int64_t length = static_cast<int64_t>(elementDimSpecs.size());
    if (length != outputType.getDimSize(0))
      return rewriter.notifyMatchFailure(
          op, "element_dim_specs size does not match output length");

    Value statePtr = adaptor.getCtx();
    Value outputPtr = extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc);

    // Stack-alloca a host int64[length] and fill it by materialising each
    // DimSpec. We rely on the shared DimSpecMaterializer to emit the right
    // arith / llvm.call chain per node kind.
    Value bufLen = createI64Const(length);
    Value hostBuf =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type, bufLen,
                               /*alignment=*/8);

    DimSpecMaterializerCallbacks cbacks;
    cbacks.statePtr = statePtr;
    // InputDim leaves: hip.shape has a single tensor input (operand index
    // 1, after ctx), and ShapeConversion resolved input_index == 0 (the
    // single ONNX input) for every InputDim leaf. We map every InputDim
    // back to the input memref descriptor's sizes[]. Asserting on any
    // other input_index makes regressions surface here instead of as
    // bogus dim values at runtime.
    Value inputDesc = adaptor.getX();
    auto inputMemRefTy = dyn_cast<MemRefType>(op.getX().getType());
    cbacks.readInputDim = [&](unsigned input_index,
                              unsigned dim_index) -> Value {
      if (input_index != 0) {
        // Should never trigger for hip.shape; defensive fallback.
        return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                        rewriter.getI64IntegerAttr(-1));
      }
      if (inputMemRefTy && dim_index < inputMemRefTy.getRank() &&
          !inputMemRefTy.isDynamicDim(dim_index)) {
        // Static dim — emit the constant directly so DCE removes the
        // descriptor read.
        return LLVM::ConstantOp::create(
            rewriter, loc, i64Type,
            rewriter.getI64IntegerAttr(inputMemRefTy.getDimSize(dim_index)));
      }
      return MemRefDescriptor(inputDesc).size(rewriter, loc, dim_index);
    };
    // InputValueI64 leaves: not expected on the Shape op result path
    // (ONNX Shape has no value-typed inputs). Wire a null lambda so any
    // stray leaf surfaces as -1 in the runtime tracer rather than a
    // segfault. The materializer handles a null lambda by emitting an
    // i64 -1 constant — already correct fallback semantics.

    for (int64_t i = 0; i < length; ++i) {
      auto perElem = dyn_cast<ArrayAttr>(elementDimSpecs[i]);
      if (!perElem)
        return rewriter.notifyMatchFailure(
            op, "element_dim_specs entries must be ArrayAttr");
      DimSpec spec = DimSpec::parseFromArrayAttr(perElem);
      if (spec.nodes().empty())
        return rewriter.notifyMatchFailure(
            op, "element_dim_specs entry parsed to empty DimSpec");

      OpBuilder builder(rewriter);
      builder.setInsertionPoint(op);
      Value dimVal = materializeDimSpec(builder, loc, spec, cbacks);
      if (dimVal.getType() != i64Type) {
        // Defensive: the materializer is supposed to return i64, but a
        // stray cast layer means we'd silently store the wrong width
        // otherwise. Bail out loudly so any future regression surfaces
        // in lowering rather than at runtime.
        return rewriter.notifyMatchFailure(
            op, "materializeDimSpec returned non-i64 value");
      }
      Value idx = createI64Const(i);
      Value slot = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, hostBuf,
                                       ValueRange{idx});
      LLVM::StoreOp::create(rewriter, loc, dimVal, slot);
    }

    // int wrap_shape(RuntimeState*, void* output_gpu, int64_t num_elements,
    //                const int64_t* host_values)
    SmallVector<Type, 4> paramTypes = {ptrType, ptrType, i64Type, ptrType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapShape, paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    Value numElems = createI64Const(length);
    SmallVector<Value, 4> args = {statePtr, outputPtr, numElems, hostBuf};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateShapeLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<ShapeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
