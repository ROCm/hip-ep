/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Shared MIOpen Activation Lowering Helper
//===----------------------------------------------------------------------===//
// Generic lowering for single-input/single-output MIOpen activation functions
// that call wrap_miopenActivationForward with an activation_mode parameter.
//
// Supports data type:
//   float32, float16, bfloat16
//
// Template parameters:
//   OpType: The HIP op type (e.g., SigmoidOp, SoftplusOp)
//   activationMode: The HIPDNN_EP_ACTIVATION_* constant
//
// Requirements:
//   - OpType must have: getCtx(), getX(), getY() accessors
//   - OpType must be a DPS op with single input (x) and single output (y)
template <typename OpType, int64_t activationMode>
static LogicalResult
lowerMiopenActivation(OpType op, typename OpType::Adaptor adaptor,
                      ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  ModuleOp module = op->template getParentOfType<ModuleOp>();
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  Type i32Type = rewriter.getI32Type();
  Type i64Type = rewriter.getI64Type();

  // Helper to create i64 constants
  auto createI64Const = [&](int64_t value) -> Value {
    return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                    rewriter.getI64IntegerAttr(value));
  };

  // Extract pointers
  Value statePtr = adaptor.getCtx();
  Value inputPtr = extractMemRefPtr(adaptor.getX(), rewriter, loc);
  Value outputPtr = extractMemRefPtr(adaptor.getY(), rewriter, loc);

  auto outputType = cast<MemRefType>(op.getY().getType());

  // Compute num_elements (supports dynamic shapes)
  Value numElements = createI64Const(1);
  MemRefDescriptor outputDesc(adaptor.getY());

  for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
    Value dimSize;
    if (outputType.isDynamicDim(dimIdx)) {
      dimSize = outputDesc.size(rewriter, loc, dimIdx);
    } else {
      dimSize = createI64Const(outputType.getDimSize(dimIdx));
    }
    numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
  }

  // Get data type enum (f32=0, f16=1, bf16=2)
  // MIOpen activations only support floating-point types
  Type elemType = outputType.getElementType();
  int64_t dataType = getHipdnnDataType(elemType);

  // Validate: only f32, f16, bf16 are supported (no integer types)
  if (dataType < 0 || dataType > 2) {
    std::string errorMsg;
    llvm::raw_string_ostream os(errorMsg);
    os << "unsupported element type '" << elemType
       << "' for MIOpen activation (mode=" << activationMode
       << "). Only f32, f16, and bf16 are supported";
    return rewriter.notifyMatchFailure(op, os.str());
  }

  Value dataTypeVal = createI64Const(dataType);
  Value activationModeVal = createI64Const(activationMode);

  // int wrap_miopenActivationForward(RuntimeState* state, void* input,
  //     void* output, int64_t num_elements, int64_t data_type,
  //     int64_t activation_mode)
  SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                     i64Type, i64Type, i64Type};

  FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
      rewriter, module, kWrapMiopenActivationForward, paramTypes, i32Type);
  if (failed(funcOp))
    return failure();

  SmallVector<Value, 6> args = {statePtr,    inputPtr,    outputPtr,
                                numElements, dataTypeVal, activationModeVal};

  LLVM::CallOp::create(rewriter, loc, *funcOp, args);
  rewriter.eraseOp(op);
  return success();
}

//===----------------------------------------------------------------------===//
// Individual Activation Lowering Patterns
//===----------------------------------------------------------------------===//

// hip.sigmoid(ctx, x, y)
//   -> wrap_miopenActivationForward(state, x, y, num_elements,
//                                    data_type, activation_mode=SIGMOID)
struct SigmoidOpLowering : public ConvertOpToLLVMPattern<SigmoidOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerMiopenActivation<SigmoidOp, kActivationSigmoid>(op, adaptor,
                                                                rewriter);
  }
};

// hip.softplus(ctx, x, y)
//   -> wrap_miopenActivationForward(state, x, y, num_elements,
//                                    data_type, activation_mode=SOFTPLUS)
// Supports both static and dynamic shapes (computes num_elements at runtime).
struct SoftplusOpLowering : public ConvertOpToLLVMPattern<SoftplusOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SoftplusOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerMiopenActivation<SoftplusOp, kActivationSoftplus>(op, adaptor,
                                                                  rewriter);
  }
};

// hip.silu(handle, input, output)
struct SiluOpLowering : public ConvertOpToLLVMPattern<SiluOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SiluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();

    SmallVector<Type> paramTypes(3, ptrType);
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipSilu, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {
        adaptor.getCtx(), extractMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc)};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.miopen.softmax(%handle) ins(%input) outs(%output)
//   -> hip_miopen_softmax(handle, input, output, rows, cols, data_type)
// Rank-generic: softmax over last dim. For 3D [B,S,D], rows = B*S, cols = D.
struct MiopenSoftmaxOpLowering
    : public ConvertOpToLLVMPattern<MiopenSoftmaxOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MiopenSoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();
    Type indexType = getIndexType();

    Type i64Type = rewriter.getI64Type();
    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType, indexType,
                                    indexType, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kMiopenSoftmax, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    int rank = cast<MemRefType>(op.getInput().getType()).getRank();
    MemRefDescriptor inputDesc(adaptor.getInput());

    // cols = last dim; rows = product of all other dims
    Value cols = inputDesc.size(rewriter, loc, rank - 1);
    Value rows = inputDesc.size(rewriter, loc, 0);
    for (int i = 1; i < rank - 1; i++)
      rows = LLVM::MulOp::create(rewriter, loc, rows,
                                 inputDesc.size(rewriter, loc, i));
    auto inputType = cast<MemRefType>(op.getInput().getType());
    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported softmax element type");
    Value dtypeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(dataType));

    SmallVector<Value> args = {
        adaptor.getCtx(), extractMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc), rows, cols,
        dtypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateActivationLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<SigmoidOpLowering, SoftplusOpLowering, SiluOpLowering,
               MiopenSoftmaxOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
