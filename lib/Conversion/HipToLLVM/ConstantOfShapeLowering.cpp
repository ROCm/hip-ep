/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.constant_of_shape(ctx, shape, output) {fill_value, output_data_type,
//                                             output_dim_specs?, slot_ids?}
//
// Operand-provenance dispatch is encoded in the IR by the conversion:
//
//   * `slot_ids` present  -> Category C: shape tensor is GPU-resident
//                            (intermediate). Lower to wrap_constant_of_shape_dyn
//                            which D2Hs the shape vector, publishes the dims
//                            and an output buffer to the slots, and runs the
//                            fill kernel.
//
//   * `slot_ids` absent  -> Category B: EP has resolved the output shape
//                            and allocated the output OrtValue (the `output`
//                            memref operand carries the right size). Lower
//                            to wrap_constant_of_shape against that buffer.
//
// `output_data_type` is the runtime's HIPDNN_EP_DATATYPE_* enum; the
// runtime maps it internally to a hip_dtype_t for the kernel. The conversion
// is in `ConstantOfShapeConversion.cpp::mapElemTypeToHipdnnDtype`.

constexpr const char *kWrapConstantOfShape = "wrap_constant_of_shape";
constexpr const char *kWrapConstantOfShapeDyn = "wrap_constant_of_shape_dyn";

// Map HIPDNN_EP_DATATYPE_* enum (used by op attribute) to the HIP_DTYPE_*
// enum the runtime + kernel switch on. The runtime accepts the kernel-side
// dtype directly; we translate here once at lowering time so the runtime
// stays a thin shim. Mirrors hipDTypeFromHipdnn in NonZero / Range wrappers.
static int64_t hipdnnDtypeToHipDtype(int64_t hipdnn_dtype) {
  switch (hipdnn_dtype) {
  case 0: return 0;  // HIPDNN_EP_DATATYPE_FLOAT   -> HIP_DTYPE_FLOAT32
  case 1: return 1;  // HIPDNN_EP_DATATYPE_HALF    -> HIP_DTYPE_FLOAT16
  case 2: return 5;  // HIPDNN_EP_DATATYPE_BFLOAT16-> HIP_DTYPE_BFLOAT16
  case 3: return 3;  // HIPDNN_EP_DATATYPE_INT32   -> HIP_DTYPE_INT32
  case 4: return 2;  // HIPDNN_EP_DATATYPE_INT64   -> HIP_DTYPE_INT64
  case 5: return 7;  // HIPDNN_EP_DATATYPE_INT8    -> HIP_DTYPE_INT8
  case 6: return 4;  // HIPDNN_EP_DATATYPE_DOUBLE  -> HIP_DTYPE_FLOAT64
  case 7: return 7;  // HIPDNN_EP_DATATYPE_UINT8   -> HIP_DTYPE_INT8 (same width)
  default: return -1;
  }
}

struct ConstantOfShapeOpLowering
    : public ConvertOpToLLVMPattern<ConstantOfShapeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConstantOfShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    int64_t fill_value = op.getFillValue();
    int64_t output_dtype = op.getOutputDataType();
    int64_t hip_dtype = hipdnnDtypeToHipDtype(output_dtype);
    if (hip_dtype < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.constant_of_shape: unsupported output_data_type");

    Value statePtr = adaptor.getCtx();
    Value fillValueConst = createI64Const(fill_value);

    DenseI32ArrayAttr slotIdsAttr = op.getSlotIdsAttr();
    if (slotIdsAttr) {
      // ----- Category C: dyn variant -----
      // Build a stack-resident i32 slot_ids[] array and dispatch
      // wrap_constant_of_shape_dyn(state, shape_dev, shape_dtype, rank,
      //                            slot_ids, value_bits, output_dtype)
      auto shapeType = dyn_cast<MemRefType>(op.getShape().getType());
      if (!shapeType)
        return rewriter.notifyMatchFailure(
            op, "shape operand must be a ranked memref");
      int64_t shape_dtype_enum =
          getHipdnnDataType(shapeType.getElementType());
      int64_t shape_hip_dtype = hipdnnDtypeToHipDtype(shape_dtype_enum);
      if (shape_hip_dtype < 0)
        return rewriter.notifyMatchFailure(
            op, "unsupported shape tensor element type");

      Value shapePtr =
          extractContiguousMemRefPtr(adaptor.getShape(), rewriter, loc);
      Value shapeDtypeConst = createI64Const(shape_hip_dtype);

      llvm::ArrayRef<int32_t> slot_ids = slotIdsAttr.asArrayRef();
      int64_t rank = static_cast<int64_t>(slot_ids.size());
      Value rankConst = createI64Const(rank);
      Value outputDtypeConst = createI64Const(hip_dtype);

      // Stack-alloc int32_t[rank] and populate with the slot ids.
      Value slotsLen = LLVM::ConstantOp::create(
          rewriter, loc, i64Type, rewriter.getI64IntegerAttr(rank));
      Value slotsArr =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, i32Type, slotsLen,
                                 /*alignment=*/4);
      for (int64_t i = 0; i < rank; ++i) {
        Value idx = createI64Const(i);
        Value slot = LLVM::GEPOp::create(rewriter, loc, ptrType, i32Type,
                                         slotsArr, ValueRange{idx});
        Value sidVal = LLVM::ConstantOp::create(
            rewriter, loc, i32Type,
            rewriter.getI32IntegerAttr(slot_ids[i]));
        LLVM::StoreOp::create(rewriter, loc, sidVal, slot);
      }

      SmallVector<Type, 7> paramTypes = {ptrType, ptrType, i64Type, i64Type,
                                         ptrType, i64Type, i64Type};
      FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
          rewriter, module, kWrapConstantOfShapeDyn, paramTypes, i32Type);
      if (failed(funcOp))
        return failure();
      SmallVector<Value, 7> args = {statePtr,        shapePtr,
                                    shapeDtypeConst, rankConst,
                                    slotsArr,        fillValueConst,
                                    outputDtypeConst};
      LLVM::CallOp::create(rewriter, loc, *funcOp, args);
      rewriter.eraseOp(op);
      return success();
    }

    // ----- Category B: static variant -----
    // wrap_constant_of_shape(state, output_ptr, num_elements, value_bits,
    //                        hip_dtype)
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(
          op, "output operand must be a ranked memref");
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value numElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);
    Value outputDtypeConst = createI64Const(hip_dtype);

    SmallVector<Type, 5> paramTypes = {ptrType, ptrType, i64Type, i64Type,
                                       i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapConstantOfShape, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();
    SmallVector<Value, 5> args = {statePtr, outputPtr, numElements,
                                  fillValueConst, outputDtypeConst};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateConstantOfShapeLoweringPatterns(const LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns) {
  patterns.add<ConstantOfShapeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
