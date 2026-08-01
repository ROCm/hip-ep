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
  Value inputPtr = extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc);
  Value outputPtr = extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc);

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

  // int wrap_miopenActivationForward(RuntimeState* state, int op_state_slot,
  //     void* input, void* output, int64_t num_elements, int64_t data_type,
  //     int64_t activation_mode)
  SmallVector<Type, 7> paramTypes = {ptrType, i32Type, ptrType, ptrType,
                                     i64Type, i64Type, i64Type};

  FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
      rewriter, module, kWrapMiopenActivationForward, paramTypes, i32Type);
  if (failed(funcOp))
    return failure();

  SmallVector<Value, 7> args = {
      statePtr,         getOpStateSlotValue(op, rewriter, loc),
      inputPtr,         outputPtr,
      numElements,      dataTypeVal,
      activationModeVal};

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

// hip.tanh(ctx, x, y)
//   -> wrap_miopenActivationForward(state, x, y, num_elements,
//                                    data_type, activation_mode=TANH)
struct TanhOpLowering : public ConvertOpToLLVMPattern<TanhOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TanhOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerMiopenActivation<TanhOp, kActivationTanh>(op, adaptor,
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

// hip.gelu(ctx, input, output)
//   -> wrap_gelu(state, input, output, num_elements, data_type, approximate)
// Uses custom HIP kernel (hip_elementwise_gelu) instead of MIOpen.
// Supports static and dynamic shapes (computes num_elements at runtime).
// Supports data types: f32, f16, bf16, f64 (per ONNX Gelu spec).
// Supports approximate modes: "none" (erf) and "tanh".
struct GeluOpLowering : public ConvertOpToLLVMPattern<GeluOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GeluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
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
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Compute num_elements (supports dynamic shapes)
    Value numElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getOutput());

    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize;
      if (outputType.isDynamicDim(dimIdx)) {
        dimSize = outputDesc.size(rewriter, loc, dimIdx);
      } else {
        dimSize = createI64Const(outputType.getDimSize(dimIdx));
      }
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
    }

    // Get data type enum (f32=0, f16=1, bf16=2, i32=3, i64=4, i8=5, f64=6)
    Type elemType = outputType.getElementType();
    int64_t dataType = getHipdnnDataType(elemType);

    // Validate: only f32, f16, bf16, f64 are supported (per ONNX Gelu spec)
    if (dataType < 0 || (dataType > 2 && dataType != 6)) {
      std::string errorMsg;
      llvm::raw_string_ostream os(errorMsg);
      os << "unsupported element type '" << elemType
         << "' for GELU. Only f32, f16, bf16, and f64 are supported";
      return rewriter.notifyMatchFailure(op, os.str());
    }

    Value dataTypeVal = createI64Const(dataType);

    // Get approximate mode: "none" -> 0, "tanh" -> 1
    std::string approximateStr = op.getApproximate().str();
    int64_t approximateMode = (approximateStr == "tanh") ? 1 : 0;
    Value approximateModeVal = createI64Const(approximateMode);

    // int wrap_gelu(RuntimeState* state, void* input, void* output,
    //               int64_t num_elements, int64_t data_type, int64_t
    //               approximate)
    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGelu, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr,    inputPtr,    outputPtr,
                                  numElements, dataTypeVal, approximateModeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.leaky_relu(ctx, input, output)
//   -> wrap_leaky_relu(state, input, output, num_elements, data_type, alpha)
// Uses custom HIP kernel (hip_leaky_relu).
// Supports static and dynamic shapes (computes num_elements at runtime).
// Supports data types: f32, f16, f64.
struct LeakyReluOpLowering : public ConvertOpToLLVMPattern<LeakyReluOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LeakyReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f64Type = rewriter.getF64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Compute num_elements (supports dynamic shapes)
    Value numElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getOutput());

    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize;
      if (outputType.isDynamicDim(dimIdx)) {
        dimSize = outputDesc.size(rewriter, loc, dimIdx);
      } else {
        dimSize = createI64Const(outputType.getDimSize(dimIdx));
      }
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
    }

    Type elemType = outputType.getElementType();
    int64_t dataType = getHipdnnDataType(elemType);

    if (dataType != 0 && dataType != 1 && dataType != 6) {
      std::string errorMsg;
      llvm::raw_string_ostream os(errorMsg);
      os << "unsupported element type '" << elemType
         << "' for LeakyRelu. Only f32, f16, and f64 are supported";
      return rewriter.notifyMatchFailure(op, os.str());
    }

    Value dataTypeVal = createI64Const(dataType);

    double alphaVal = op.getAlpha().convertToDouble();
    Value alphaConst = LLVM::ConstantOp::create(
        rewriter, loc, f64Type, rewriter.getF64FloatAttr(alphaVal));

    // int wrap_leaky_relu(RuntimeState* state, void* input, void* output,
    //                     int64_t num_elements, int64_t data_type, double
    //                     alpha)
    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       i64Type, i64Type, f64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapLeakyRelu, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr,    inputPtr,    outputPtr,
                                  numElements, dataTypeVal, alphaConst};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
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
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc)};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.miopen.softmax(%handle) ins(%input) outs(%output)
//   -> hip_miopen_softmax(handle, input, output, rows, cols, elem_size_bytes)
// Rank-generic: softmax over last dim. For 3D [B,S,D], rows = B*S, cols = D.
// elem_size_bytes is derived from the MemRef element type (2=fp16, 4=fp32)
// and passed to the runtime so it can copy + dispatch with the right dtype.
// Previously this was not passed, causing fp32 Softmax inputs (e.g. Qwen VLM
// attention scores) to be misread as fp16, producing completely wrong outputs.
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

    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType, indexType,
                                    indexType, indexType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kMiopenSoftmax, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    auto inputMemrefType = cast<MemRefType>(op.getInput().getType());
    int rank = inputMemrefType.getRank();
    MemRefDescriptor inputDesc(adaptor.getInput());

    // cols = last dim; rows = product of all other dims
    Value cols = inputDesc.size(rewriter, loc, rank - 1);
    Value rows = inputDesc.size(rewriter, loc, 0);
    for (int i = 1; i < rank - 1; i++)
      rows = LLVM::MulOp::create(rewriter, loc, rows,
                                 inputDesc.size(rewriter, loc, i));

    // Determine element size in bytes from the MemRef element type.
    // fp16/bf16 = 2, fp32 = 4.  Fall back to 2 for any unrecognised type
    // to preserve backward-compatible behaviour with the fp16 softmax path.
    Type elemType = inputMemrefType.getElementType();
    int64_t elemSizeBytes = 2;
    if (elemType.isF32() || elemType.isInteger(32))
      elemSizeBytes = 4;
    else if (elemType.isF64() || elemType.isInteger(64))
      elemSizeBytes = 8;
    Value elemSize = LLVM::ConstantOp::create(
        rewriter, loc, indexType,
        rewriter.getIntegerAttr(indexType, elemSizeBytes));

    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc), rows,
        cols, elemSize};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateActivationLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns) {
  patterns.add<SigmoidOpLowering, TanhOpLowering, SoftplusOpLowering,
               GeluOpLowering, LeakyReluOpLowering, SiluOpLowering,
               MiopenSoftmaxOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
