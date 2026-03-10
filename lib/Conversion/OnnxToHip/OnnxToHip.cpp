/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// ONNX to HIP Dialect Conversion
//===----------------------------------------------------------------------===//
// This file implements conversion patterns from ONNX dialect operations
// (provided by onnx-mlir) to HIP dialect operations (using MIOpen).
//
// Design: tensor-first (no allocation inside patterns).
// Each pattern emits a tensor::EmptyOp as DPS init and returns
// hipOp.getResultTensors() so that one-shot-bufferize handles all allocation.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"

#include "morphizen-foundation/file_io.hpp"
#include "hip/Support/DiskFileSystem.h"
#include "compilation_options_generated.h"
#include "hip/debug_log.h"

// Include ONNX dialect operations from onnx-mlir
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrBuilder.hpp"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Constant Information Storage
//===----------------------------------------------------------------------===//

/// Alignment boundary for constants in constants.bin (bytes).
/// - Small constants (size < kConstantAlignment): packed tightly within
///   kConstantAlignment-byte chunks; no constant spans two chunks.
/// - Large constants (size >= kConstantAlignment): individually aligned to
///   kConstantAlignment bytes, padded with zeros at the end.
static constexpr size_t kConstantAlignment = 256;

struct ConstantInfo {
  int64_t globalIndex;           // Sequential index (0, 1, 2, ...)
  ElementsAttr value;            // Constant data (from onnx.Constant)
  Type elementType;              // Element type (f32, i64, etc.)
  SmallVector<int64_t, 4> shape; // Tensor shape (owned storage)
  size_t sizeInBytes;            // Total size in bytes
  size_t elementSizeInBytes;     // Size of one element (sizeof(float), etc.)
  size_t numElements;            // Total number of elements
  std::string name;              // Debug name (from operation location)
  size_t offset = 0;             // Byte offset in constants.bin (set by computeOffsets())

  // Default constructor (required by DenseMap)
  ConstantInfo()
      : globalIndex(-1), sizeInBytes(0), elementSizeInBytes(0), numElements(0) {
  }

  ConstantInfo(int64_t idx, ElementsAttr val, Type elemType,
               ArrayRef<int64_t> shp, size_t size, size_t elemSize,
               size_t numElems, StringRef debugName)
      : globalIndex(idx), value(val), elementType(elemType),
        shape(shp.begin(), shp.end()), sizeInBytes(size),
        elementSizeInBytes(elemSize), numElements(numElems),
        name(debugName.str()) {}
};

namespace {

//===----------------------------------------------------------------------===//
// Helper: get !hip.context from function arg 0
//===----------------------------------------------------------------------===//

static FailureOr<Value> getContextArg(Operation* op,
                                       PatternRewriter& rewriter) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp)
    return rewriter.notifyMatchFailure(op, "not inside a function");
  auto& entry = funcOp.getBody().front();
  if (entry.getNumArguments() == 0)
    return rewriter.notifyMatchFailure(op, "function has no arguments");
  Value ctx = entry.getArgument(0);
  if (!isa<hip::ContextType>(ctx.getType()))
    return rewriter.notifyMatchFailure(op,
                                       "first argument is not !hip.context");
  return ctx;
}

//===----------------------------------------------------------------------===//
// ONNX Constant → HIP Get Constant Conversion Pattern
//===----------------------------------------------------------------------===//

/// Convert onnx.Constant to hip.get_constant that retrieves pre-uploaded
/// constant from state, then wrap in bufferization.to_tensor so the result
/// is tensor-typed for downstream tensor-mode HIP ops.
struct ConstantToHipPattern : public OpRewritePattern<ONNXConstantOp> {
  const DenseMap<Value, ConstantInfo>& constantRegistry;

  ConstantToHipPattern(MLIRContext* context,
                       const DenseMap<Value, ConstantInfo>& registry)
      : OpRewritePattern(context), constantRegistry(registry) {}

  LogicalResult matchAndRewrite(ONNXConstantOp constantOp,
                                PatternRewriter& rewriter) const override {
    auto loc = constantOp.getLoc();

    // Look up this constant in the registry
    auto it = constantRegistry.find(constantOp.getResult());
    if (it == constantRegistry.end())
      return rewriter.notifyMatchFailure(constantOp,
                                         "constant not found in registry");

    const auto& info = it->second;

    // Get context from parent function's first argument
    auto ctxOrFailure = getContextArg(constantOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    // Create index constant
    Value index = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(info.globalIndex));

    // Build the memref type for hip.get_constant (GPU address space 1)
    auto tensorType = cast<TensorType>(constantOp.getResult().getType());
    auto memSpace =
        IntegerAttr::get(IntegerType::get(rewriter.getContext(), 64), 1);
    auto memrefType = MemRefType::get(tensorType.getShape(),
                                      tensorType.getElementType(),
                                      AffineMap(), memSpace);

    // hip.get_constant retrieves a pre-uploaded constant buffer (memref)
    auto getConstOp =
        rewriter.create<hip::GetConstantOp>(loc, memrefType, context, index);

    // Wrap in bufferization.to_tensor so downstream tensor-mode patterns work
    Value tensorVal = rewriter.create<bufferization::ToTensorOp>(
        loc, tensorType, getConstOp.getResult(),
        /*restrict=*/true, /*writable=*/false);

    rewriter.replaceOp(constantOp, tensorVal);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX Conv → HIP Conv (tensor-first)
//===----------------------------------------------------------------------===//

struct ConvToHipPattern : public OpRewritePattern<ONNXConvOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXConvOp convOp,
                                PatternRewriter& rewriter) const override {
    auto loc = convOp.getLoc();

    auto ctxOrFailure = getContextArg(convOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value X = convOp.getX();
    Value W = convOp.getW();
    Value B = convOp.getB();

    auto kernelShape = convOp.getKernelShape();
    if (!kernelShape)
      return rewriter.notifyMatchFailure(convOp, "missing kernel_shape");

    size_t spatialDims = kernelShape.value().size();

    auto strides = convOp.getStrides();
    ArrayAttr stridesAttr = strides ? strides.value()
                                    : rewriter.getI64ArrayAttr(
                                          SmallVector<int64_t>(spatialDims, 1));

    auto pads = convOp.getPads();
    ArrayAttr padsAttr =
        pads ? pads.value()
             : rewriter.getI64ArrayAttr(
                   SmallVector<int64_t>(spatialDims * 2, 0));

    auto dilations = convOp.getDilations();
    ArrayAttr dilationsAttr =
        dilations ? dilations.value()
                  : rewriter.getI64ArrayAttr(
                        SmallVector<int64_t>(spatialDims, 1));

    auto kernelShapeAttr = kernelShape.value();
    auto groupAttr = rewriter.getI64IntegerAttr(convOp.getGroup());

    auto resultTensorType =
        cast<RankedTensorType>(convOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(
        loc, resultTensorType.getShape(), resultTensorType.getElementType());

    SmallVector<Value> operands = {context, X, W};
    if (B)
      operands.push_back(B);
    operands.push_back(init);

    SmallVector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("kernel_shape", kernelShapeAttr));
    attrs.push_back(rewriter.getNamedAttr("strides", stridesAttr));
    attrs.push_back(rewriter.getNamedAttr("pads", padsAttr));
    attrs.push_back(rewriter.getNamedAttr("dilations", dilationsAttr));
    attrs.push_back(rewriter.getNamedAttr("group", groupAttr));

    auto hipOp = rewriter.create<hip::ConvOp>(
        loc, TypeRange{resultTensorType}, operands, attrs);
    rewriter.replaceOp(convOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX ReLU → HIP ReLU (tensor-first)
//===----------------------------------------------------------------------===//

struct ReluToHipPattern : public OpRewritePattern<ONNXReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXReluOp reluOp,
                                PatternRewriter& rewriter) const override {
    auto loc = reluOp.getLoc();
    auto ctxOrFailure = getContextArg(reluOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;
    Value X = reluOp.getX();
    auto resultType = cast<RankedTensorType>(reluOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto hipOp = rewriter.create<hip::ReluOp>(loc, TypeRange{resultType},
                                               context, X, init);
    rewriter.replaceOp(reluOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX Gemm → HIP Gemm (tensor-first)
//===----------------------------------------------------------------------===//

struct GemmToHipPattern : public OpRewritePattern<ONNXGemmOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXGemmOp gemmOp,
                                PatternRewriter& rewriter) const override {
    auto loc = gemmOp.getLoc();
    auto ctxOrFailure = getContextArg(gemmOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value A = gemmOp.getA();
    Value B = gemmOp.getB();
    Value C = gemmOp.getC();

    FloatAttr alphaAttr = gemmOp.getAlphaAttr();
    FloatAttr betaAttr = gemmOp.getBetaAttr();
    IntegerAttr transAAttr =
        rewriter.getI64IntegerAttr(gemmOp.getTransAAttr().getSInt());
    IntegerAttr transBAttr =
        rewriter.getI64IntegerAttr(gemmOp.getTransBAttr().getSInt());

    auto resultType = cast<RankedTensorType>(gemmOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto hipOp = rewriter.create<hip::GemmOp>(loc, TypeRange{resultType},
                                               context, A, B, C, init,
                                               transAAttr, transBAttr,
                                               alphaAttr, betaAttr);
    rewriter.replaceOp(gemmOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX MatMul → HIP MatMul (tensor-first)
//===----------------------------------------------------------------------===//

struct MatMulToHipPattern : public OpRewritePattern<ONNXMatMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXMatMulOp matmulOp,
                                PatternRewriter& rewriter) const override {
    auto loc = matmulOp.getLoc();
    auto ctxOrFailure = getContextArg(matmulOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value A = matmulOp.getA();
    Value B = matmulOp.getB();

    auto resultType = cast<RankedTensorType>(matmulOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto hipOp = rewriter.create<hip::MatMulOp>(loc, TypeRange{resultType},
                                                context, A, B, init);
    rewriter.replaceOp(matmulOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX Mul → HIP Mul (tensor-first)
//===----------------------------------------------------------------------===//

struct MulToHipPattern : public OpRewritePattern<ONNXMulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXMulOp mulOp,
                                PatternRewriter& rewriter) const override {
    auto loc = mulOp.getLoc();
    auto ctxOrFailure = getContextArg(mulOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value A = mulOp.getA();
    Value B = mulOp.getB();

    auto resultType = cast<RankedTensorType>(mulOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto hipOp = rewriter.create<hip::MulOp>(loc, TypeRange{resultType},
                                              context, A, B, init);
    rewriter.replaceOp(mulOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX Sub → HIP Sub (tensor-first)
//===----------------------------------------------------------------------===//

struct SubToHipPattern : public OpRewritePattern<ONNXSubOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXSubOp subOp,
                                PatternRewriter& rewriter) const override {
    auto loc = subOp.getLoc();
    auto ctxOrFailure = getContextArg(subOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value A = subOp.getA();
    Value B = subOp.getB();

    auto resultType = cast<RankedTensorType>(subOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto hipOp = rewriter.create<hip::SubOp>(loc, TypeRange{resultType},
                                              context, A, B, init);
    rewriter.replaceOp(subOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX Gather → HIP Gather (tensor-first)
//===----------------------------------------------------------------------===//

struct GatherToHipPattern : public OpRewritePattern<ONNXGatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXGatherOp gatherOp,
                                PatternRewriter& rewriter) const override {
    auto loc = gatherOp.getLoc();
    auto ctxOrFailure = getContextArg(gatherOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value data = gatherOp.getData();
    Value indices = gatherOp.getIndices();
    auto axisAttr = rewriter.getI64IntegerAttr(gatherOp.getAxis());

    auto resultType = cast<RankedTensorType>(gatherOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto hipOp = rewriter.create<hip::GatherOp>(loc, TypeRange{resultType},
                                                context, data, indices, init,
                                                axisAttr);
    rewriter.replaceOp(gatherOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX ReduceSum → HIP ReduceSum (tensor-first)
//===----------------------------------------------------------------------===//

struct ReduceSumToHipPattern : public OpRewritePattern<ONNXReduceSumOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXReduceSumOp reduceOp,
                                PatternRewriter& rewriter) const override {
    auto loc = reduceOp.getLoc();
    auto ctxOrFailure = getContextArg(reduceOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value data = reduceOp.getData();
    Value axes = reduceOp.getAxes();
    auto keepdimsAttr = rewriter.getI64IntegerAttr(reduceOp.getKeepdims());

    auto resultType = cast<RankedTensorType>(reduceOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto hipOp = rewriter.create<hip::ReduceSumOp>(
        loc, TypeRange{resultType}, context, data, axes, init, keepdimsAttr);
    rewriter.replaceOp(reduceOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX Cast → HIP Cast (tensor-first)
//===----------------------------------------------------------------------===//

struct CastToHipPattern : public OpRewritePattern<ONNXCastOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXCastOp castOp,
                                PatternRewriter& rewriter) const override {
    auto loc = castOp.getLoc();
    auto ctxOrFailure = getContextArg(castOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value input = castOp.getInput();

    // Map MLIR element type to ONNX DataType enum
    Type targetType = castOp.getTo();
    int64_t onnxDataType = 0;
    if (targetType.isF16())
      onnxDataType = 10;
    else if (targetType.isBF16())
      onnxDataType = 16;
    else if (targetType.isF32())
      onnxDataType = 1;
    else if (targetType.isF64())
      onnxDataType = 11;
    else if (targetType.isInteger(8))
      onnxDataType = 3;
    else if (targetType.isInteger(16))
      onnxDataType = 5;
    else if (targetType.isInteger(32))
      onnxDataType = 6;
    else if (targetType.isInteger(64))
      onnxDataType = 7;
    else if (targetType.isInteger(1))
      onnxDataType = 9;
    auto toAttr = rewriter.getI64IntegerAttr(onnxDataType);

    auto resultType = cast<RankedTensorType>(castOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto hipOp = rewriter.create<hip::CastOp>(loc, TypeRange{resultType},
                                               context, input, init, toAttr);
    rewriter.replaceOp(castOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX Sigmoid → HIP Sigmoid (tensor-first)
//===----------------------------------------------------------------------===//

struct SigmoidToHipPattern : public OpRewritePattern<ONNXSigmoidOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXSigmoidOp sigmoidOp,
                                PatternRewriter& rewriter) const override {
    auto loc = sigmoidOp.getLoc();
    auto ctxOrFailure = getContextArg(sigmoidOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value X = sigmoidOp.getX();
    auto resultType = cast<RankedTensorType>(sigmoidOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto hipOp = rewriter.create<hip::SigmoidOp>(loc, TypeRange{resultType},
                                                  context, X, init);
    rewriter.replaceOp(sigmoidOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GroupQueryAttention (com.microsoft) → HIP GQA (tensor-first)
//===----------------------------------------------------------------------===//

struct GroupQueryAttentionToHipPattern
    : public OpRewritePattern<ONNXCustomOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
                                PatternRewriter& rewriter) const override {
    if (customOp.getFunctionName() != "GroupQueryAttention")
      return failure();
    auto domainAttr = customOp->getAttrOfType<StringAttr>("domain_name");
    if (!domainAttr || domainAttr.getValue() != "com.microsoft")
      return failure();

    auto loc = customOp.getLoc();

    auto inputs = customOp.getInputs();
    if (inputs.size() != 9)
      return rewriter.notifyMatchFailure(customOp,
                                         "expected 9 inputs for GQA");
    if (customOp.getNumResults() != 3)
      return rewriter.notifyMatchFailure(customOp,
                                         "expected 3 outputs for GQA");

    auto ctxOrFailure = getContextArg(customOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value query = inputs[0];
    Value key = inputs[1];
    Value value = inputs[2];
    Value pastKey = inputs[3];
    Value pastValue = inputs[4];
    Value seqlensK = inputs[5];
    Value totalSeqLen = inputs[6];

    auto numHeadsAttr = rewriter.getI64IntegerAttr(
        customOp->getAttrOfType<IntegerAttr>("num_heads").getSInt());
    auto kvNumHeadsAttr = rewriter.getI64IntegerAttr(
        customOp->getAttrOfType<IntegerAttr>("kv_num_heads").getSInt());
    auto scaleAttr = customOp->getAttrOfType<FloatAttr>("scale");
    auto softcapAttr = customOp->getAttrOfType<FloatAttr>("softcap");
    auto doRotaryAttr = rewriter.getI64IntegerAttr(
        customOp->getAttrOfType<IntegerAttr>("do_rotary").getSInt());
    auto rotaryInterleavedAttr = rewriter.getI64IntegerAttr(
        customOp->getAttrOfType<IntegerAttr>("rotary_interleaved").getSInt());

    SmallVector<Type, 3> resultTypes;
    SmallVector<Value, 3> inits;
    for (unsigned i = 0; i < 3; ++i) {
      auto rt = cast<RankedTensorType>(customOp.getResult(i).getType());
      resultTypes.push_back(rt);
      inits.push_back(rewriter.create<tensor::EmptyOp>(
          loc, rt.getShape(), rt.getElementType()));
    }

    auto hipOp = rewriter.create<hip::GroupQueryAttentionOp>(
        loc, TypeRange{resultTypes}, context, query, key, value, pastKey,
        pastValue, seqlensK, totalSeqLen, inits[0], inits[1], inits[2],
        numHeadsAttr, kvNumHeadsAttr, scaleAttr, softcapAttr, doRotaryAttr,
        rotaryInterleavedAttr);
    rewriter.replaceOp(customOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RotaryEmbedding (com.microsoft) → HIP RotaryEmbedding (tensor-first)
//===----------------------------------------------------------------------===//

struct RotaryEmbeddingToHipPattern : public OpRewritePattern<ONNXCustomOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
                                PatternRewriter& rewriter) const override {
    if (customOp.getFunctionName() != "RotaryEmbedding")
      return failure();
    auto domainAttr = customOp->getAttrOfType<StringAttr>("domain_name");
    if (!domainAttr || domainAttr.getValue() != "com.microsoft")
      return failure();

    auto loc = customOp.getLoc();
    auto inputs = customOp.getInputs();
    if (inputs.size() != 4)
      return rewriter.notifyMatchFailure(
          customOp, "expected 4 inputs for RotaryEmbedding");
    if (customOp.getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          customOp, "expected 1 output for RotaryEmbedding");

    auto ctxOrFailure = getContextArg(customOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value input = inputs[0];
    Value positionIds = inputs[1];
    Value cosCache = inputs[2];
    Value sinCache = inputs[3];

    auto interleavedAttr = rewriter.getI64IntegerAttr(
        customOp->getAttrOfType<IntegerAttr>("interleaved").getSInt());

    int64_t numHeadsVal =
        customOp->getAttrOfType<IntegerAttr>("num_heads").getSInt();
    int64_t rotaryDimVal =
        customOp->getAttrOfType<IntegerAttr>("rotary_embedding_dim").getSInt();

    // ONNX com.microsoft.RotaryEmbedding: 0 means "infer from tensor shapes".
    //   cos_cache: [max_seq, rotary_dim/2] → rotary_dim = last_dim * 2
    //   input:     [batch, seq, hidden]     → num_heads = hidden / rotary_dim
    if (rotaryDimVal == 0) {
      auto cosCacheType =
          cast<ShapedType>(customOp.getInputs()[2].getType());
      if (cosCacheType.hasStaticShape() && cosCacheType.getRank() >= 2) {
        rotaryDimVal = cosCacheType.getShape().back() * 2;
      } else {
        return rewriter.notifyMatchFailure(
            customOp, "Cannot infer rotary_embedding_dim: "
                       "cos_cache must have static shape with rank >= 2");
      }
    }

    if (numHeadsVal == 0 && rotaryDimVal > 0) {
      auto inputType =
          cast<ShapedType>(customOp.getInputs()[0].getType());
      if (inputType.hasStaticShape() && inputType.getRank() >= 1) {
        int64_t hidden = inputType.getShape().back();
        numHeadsVal = hidden / rotaryDimVal;
      } else {
        return rewriter.notifyMatchFailure(
            customOp, "Cannot infer num_heads: "
                       "input must have static shape");
      }
    }

    auto numHeadsAttr = rewriter.getI64IntegerAttr(numHeadsVal);
    auto rotaryDimAttr = rewriter.getI64IntegerAttr(rotaryDimVal);

    auto resultType =
        cast<RankedTensorType>(customOp.getResult(0).getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto hipOp = rewriter.create<hip::RotaryEmbeddingOp>(
        loc, TypeRange{resultType}, context, input, positionIds, cosCache,
        sinCache, init, interleavedAttr, numHeadsAttr, rotaryDimAttr);
    rewriter.replaceOp(customOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SimplifiedLayerNormalization → HIP SimplifiedLayerNorm (tensor-first)
//===----------------------------------------------------------------------===//

struct SimplifiedLayerNormToHipPattern : public OpRewritePattern<ONNXCustomOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
                                PatternRewriter& rewriter) const override {
    if (customOp.getFunctionName() != "SimplifiedLayerNormalization")
      return failure();

    auto loc = customOp.getLoc();
    auto inputs = customOp.getInputs();
    if (inputs.size() != 2)
      return rewriter.notifyMatchFailure(
          customOp, "expected 2 inputs for SimplifiedLayerNormalization");

    auto ctxOrFailure = getContextArg(customOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value input = inputs[0];
    Value scale = inputs[1];

    auto epsilonAttr = customOp->getAttrOfType<FloatAttr>("epsilon");
    auto axisAttr = rewriter.getI64IntegerAttr(
        customOp->getAttrOfType<IntegerAttr>("axis").getSInt());
    auto stashTypeAttr = rewriter.getI64IntegerAttr(
        customOp->getAttrOfType<IntegerAttr>("stash_type").getSInt());

    auto resultType =
        cast<RankedTensorType>(customOp.getResult(0).getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());
    auto hipOp = rewriter.create<hip::SimplifiedLayerNormOp>(
        loc, TypeRange{resultType}, context, input, scale, init, axisAttr,
        epsilonAttr, stashTypeAttr);
    rewriter.replaceOp(customOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SkipSimplifiedLayerNormalization → HIP SkipSimplifiedLayerNorm (tensor-first)
//===----------------------------------------------------------------------===//

struct SkipSimplifiedLayerNormToHipPattern
    : public OpRewritePattern<ONNXCustomOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXCustomOp customOp,
                                PatternRewriter& rewriter) const override {
    if (customOp.getFunctionName() != "SkipSimplifiedLayerNormalization")
      return failure();
    auto domainAttr = customOp->getAttrOfType<StringAttr>("domain_name");
    if (!domainAttr || domainAttr.getValue() != "com.microsoft")
      return failure();

    auto loc = customOp.getLoc();
    auto inputs = customOp.getInputs();
    if (inputs.size() != 3)
      return rewriter.notifyMatchFailure(
          customOp, "expected 3 inputs for SkipSimplifiedLayerNormalization");

    auto ctxOrFailure = getContextArg(customOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value input = inputs[0];
    Value skip = inputs[1];
    Value gamma = inputs[2];

    auto epsilonAttr = customOp->getAttrOfType<FloatAttr>("epsilon");

    unsigned numResults = customOp.getNumResults();

    // Output 0: normalized output
    auto resultType0 =
        cast<RankedTensorType>(customOp.getResult(0).getType());
    Value init0 = rewriter.create<tensor::EmptyOp>(
        loc, resultType0.getShape(), resultType0.getElementType());

    // The kernel always computes the residual skip sum, so we always need
    // a skip-output buffer. When the ONNX op exposes it (outputs >= 2),
    // use the last output's type; otherwise reuse output 0's type as scratch.
    bool hasSkipOutput = numResults >= 2;
    unsigned skipOutIdx = hasSkipOutput ? numResults - 1 : 0;
    RankedTensorType resultTypeLast =
        hasSkipOutput
            ? cast<RankedTensorType>(customOp.getResult(skipOutIdx).getType())
            : resultType0;
    Value initLast = rewriter.create<tensor::EmptyOp>(
        loc, resultTypeLast.getShape(), resultTypeLast.getElementType());

    // HIP op always produces 2 results (DPS inits are fixed at offset 4,
    // length 2). The second result may be unused when numResults == 1.
    auto hipOp = rewriter.create<hip::SkipSimplifiedLayerNormOp>(
        loc, TypeRange{resultType0, resultTypeLast}, context, input, skip,
        gamma, init0, initLast, epsilonAttr);

    // Build replacement values matching the ONNX op's result count.
    SmallVector<Value> replacements;
    replacements.push_back(hipOp.getResultTensors()[0]); // result 0

    if (hasSkipOutput) {
      // Intermediate results (indices 1..skipOutIdx-1) may be NoneType or
      // unused.
      for (unsigned i = 1; i < skipOutIdx; ++i) {
        Type origType = customOp.getResult(i).getType();
        if (isa<NoneType>(origType)) {
          assert(customOp.getResult(i).use_empty() &&
                 "unexpected use of NoneType result");
          replacements.push_back(Value{});
          continue;
        }
        auto dummyType = cast<RankedTensorType>(origType);
        replacements.push_back(rewriter.create<tensor::EmptyOp>(
            loc, dummyType.getShape(), dummyType.getElementType()));
      }
      replacements.push_back(hipOp.getResultTensors()[1]); // last result
    }

    rewriter.replaceOp(customOp, replacements);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX MaxPool → HIP MaxPool (tensor-first)
//===----------------------------------------------------------------------===//

struct MaxPoolToHipPattern : public OpRewritePattern<ONNXMaxPoolSingleOutOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXMaxPoolSingleOutOp poolOp,
                                PatternRewriter& rewriter) const override {
    auto loc = poolOp.getLoc();
    auto ctxOrFailure = getContextArg(poolOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value X = poolOp.getX();

    auto kernelShape = poolOp.getKernelShape();
    size_t spatialDims = kernelShape.size();

    auto strides = poolOp.getStrides();
    ArrayAttr stridesAttr =
        strides ? strides.value()
                : rewriter.getI64ArrayAttr(
                      SmallVector<int64_t>(spatialDims, 1));

    auto pads = poolOp.getPads();
    ArrayAttr padsAttr =
        pads ? pads.value()
             : rewriter.getI64ArrayAttr(
                   SmallVector<int64_t>(spatialDims * 2, 0));

    auto resultType =
        cast<RankedTensorType>(poolOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());

    auto hipOp = rewriter.create<hip::MaxPoolOp>(
        loc, TypeRange{resultType}, context, X, init,
        kernelShape, stridesAttr, padsAttr);
    rewriter.replaceOp(poolOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX AveragePool → HIP AvgPool (tensor-first)
//===----------------------------------------------------------------------===//

struct AvgPoolToHipPattern : public OpRewritePattern<ONNXAveragePoolOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXAveragePoolOp poolOp,
                                PatternRewriter& rewriter) const override {
    auto loc = poolOp.getLoc();
    auto ctxOrFailure = getContextArg(poolOp, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value X = poolOp.getX();

    auto kernelShape = poolOp.getKernelShape();
    size_t spatialDims = kernelShape.size();

    auto strides = poolOp.getStrides();
    ArrayAttr stridesAttr =
        strides ? strides.value()
                : rewriter.getI64ArrayAttr(
                      SmallVector<int64_t>(spatialDims, 1));

    auto pads = poolOp.getPads();
    ArrayAttr padsAttr =
        pads ? pads.value()
             : rewriter.getI64ArrayAttr(
                   SmallVector<int64_t>(spatialDims * 2, 0));

    auto resultType =
        cast<RankedTensorType>(poolOp.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                  resultType.getElementType());

    auto hipOp = rewriter.create<hip::AvgPoolOp>(
        loc, TypeRange{resultType}, context, X, init,
        kernelShape, stridesAttr, padsAttr);
    rewriter.replaceOp(poolOp, hipOp.getResultTensors());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX Return → func.return passthrough (tensor-first)
//===----------------------------------------------------------------------===//
// In the tensor-first design, onnx.Return simply forwards its tensor operands
// to func.return. Bufferization + buffer-results-to-out-params will later
// transform the signature to destination-passing style.

struct OnnxReturnToFuncReturn : public OpRewritePattern<ONNXReturnOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ONNXReturnOp returnOp,
                                PatternRewriter& rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(returnOp,
                                                returnOp.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ONNX Function Identification Helper
//===----------------------------------------------------------------------===//

static bool isOnnxFunction(func::FuncOp funcOp) {
  auto funcType = funcOp.getFunctionType();

  bool hasTensorTypes =
      llvm::any_of(funcType.getInputs(),
                   [](Type t) { return isa<TensorType>(t); }) ||
      llvm::any_of(funcType.getResults(),
                   [](Type t) { return isa<TensorType>(t); });

  if (!hasTensorTypes)
    return false;

  bool hasOnnxOps = false;
  funcOp.walk([&](Operation* op) {
    if (auto* dialect = op->getDialect()) {
      if (isa<ONNXDialect>(dialect)) {
        hasOnnxOps = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return hasOnnxOps;
}

//===----------------------------------------------------------------------===//
// ONNX to HIP Conversion Pass (Module-Level)
//===----------------------------------------------------------------------===//

class ConvertOnnxToHipPass
    : public PassWrapper<ConvertOnnxToHipPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertOnnxToHipPass)

  explicit ConvertOnnxToHipPass(morphizen::FileSystem* fs,
                                const hip::compiler::CompilationOptionsT& compilationOptions)
      : fileSystem_(fs), compilationOptions_(compilationOptions) {}

  StringRef getArgument() const final { return "convert-onnx-to-hip"; }
  StringRef getDescription() const final {
    return "Convert ONNX dialect operations to HIP dialect operations "
           "(tensor-first; allocation done by one-shot-bufferize)";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<hip::HipDialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<tensor::TensorDialect>();
    registry.insert<bufferization::BufferizationDialect>();
    registry.insert<ONNXDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* context = &getContext();

    // Discover constants and assign global indices
    if (failed(discoverConstants(module))) {
      signalPassFailure();
      return;
    }

    // Compute byte offsets using the packing scheme (sort by size, pack small
    // constants into chunks, align large constants to kConstantAlignment).
    computeOffsets();

    if (!constantRegistry_.empty()) {
      // If no FileSystem was provided, fall back to DiskFileSystem (current
      // dir). This is a troubleshooting convenience only; production callers
      // should supply a FileSystem via createConvertOnnxToHipPass(fs) or
      // hip_compile_with_fs().
      std::unique_ptr<hip::DiskFileSystem> fallbackFs;
      morphizen::FileSystem* fs = fileSystem_;
      if (!fs) {
        COMPILER_DEBUG_LOG("[ONNX→HIP] WARNING: no FileSystem provided, "
                        "falling back to DiskFileSystem (current dir)\n");
        fallbackFs = std::make_unique<hip::DiskFileSystem>(".");
        fs = fallbackFs.get();
      }
      if (failed(writeConstantsToFileSystem(module, fs))) {
        signalPassFailure();
        return;
      }
    }
    if (failed(emitConstantSizesAttributes(module))) {
      signalPassFailure();
      return;
    }

    // Generate module metadata (BEFORE processing functions)
    // CRITICAL: Must capture original function signature before transformation
    if (failed(generateModuleMetadata(module))) {
      signalPassFailure();
      return;
    }

    // Apply tensor-first conversion patterns to each ONNX function
    for (auto func : module.getOps<func::FuncOp>()) {
      if (!isOnnxFunction(func))
        continue;

      RewritePatternSet patterns(context);
      patterns.add<ConstantToHipPattern>(context, constantRegistry_);
      patterns.add<ConvToHipPattern>(context);
      patterns.add<ReluToHipPattern>(context);
      patterns.add<GemmToHipPattern>(context);
      patterns.add<MatMulToHipPattern>(context);
      patterns.add<MulToHipPattern>(context);
      patterns.add<SubToHipPattern>(context);
      patterns.add<GatherToHipPattern>(context);
      patterns.add<ReduceSumToHipPattern>(context);
      patterns.add<CastToHipPattern>(context);
      patterns.add<SigmoidToHipPattern>(context);
      patterns.add<GroupQueryAttentionToHipPattern>(context);
      patterns.add<RotaryEmbeddingToHipPattern>(context);
      patterns.add<SimplifiedLayerNormToHipPattern>(context);
      patterns.add<SkipSimplifiedLayerNormToHipPattern>(context);
      patterns.add<MaxPoolToHipPattern>(context);
      patterns.add<AvgPoolToHipPattern>(context);
      patterns.add<OnnxReturnToFuncReturn>(context);

      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        signalPassFailure();
        return;
      }
    }

    // Clean up onnx.NoValue and onnx.EntryPoint ops
    SmallVector<Operation*> toErase;
    module.walk([&](Operation* op) {
      StringRef name = op->getName().getStringRef();
      if (name == "onnx.NoValue" && op->use_empty())
        toErase.push_back(op);
      else if (name == "onnx.EntryPoint")
        toErase.push_back(op);
    });
    for (Operation* op : toErase)
      op->erase();

    // Strip ONNX result/arg attributes from func.func operations so that
    // buffer-results-to-out-params can convert memref return values to
    // out-params. That pass refuses to move results that carry attributes.
    module.walk([&](func::FuncOp funcOp) {
      unsigned numResults = funcOp.getNumResults();
      if (numResults > 0) {
        SmallVector<DictionaryAttr> emptyResAttrs(
            numResults, DictionaryAttr::get(context));
        funcOp.setAllResultAttrs(emptyResAttrs);
      }
    });
  }

private:
  /// FileSystem for external constant storage (null = embedded mode)
  morphizen::FileSystem* fileSystem_ = nullptr;

  /// User-facing compilation options; provides constants_file name and other
  /// settings. Defaults to "constants.bin" when constants_file is empty.
  const hip::compiler::CompilationOptionsT& compilationOptions_;

  /// Constant registry: maps SSA values to their global constant indices
  DenseMap<Value, ConstantInfo> constantRegistry_;

  /// Total byte size of constants.bin, including all alignment padding.
  /// Set by computeOffsets(); used by writeConstantsToFileSystem().
  size_t totalFileSize_ = 0;

  /// Discover all onnx.Constant operations in the module and assign global
  /// indices
  LogicalResult discoverConstants(ModuleOp module) {
    int64_t nextIndex = 0;

    for (auto func : module.getOps<func::FuncOp>()) {
      if (!isOnnxFunction(func))
        continue;

      func.walk([&](Operation* op) {
        auto constantOp = dyn_cast<ONNXConstantOp>(op);
        if (!constantOp)
          return WalkResult::advance();

        auto valueAttr = constantOp.getValue();
        if (!valueAttr)
          return WalkResult::advance();

        auto rawAttr = dyn_cast<ElementsAttr>(valueAttr.value());
        if (!rawAttr)
          return WalkResult::advance();

        // onnx-mlir import produces DisposableElementsAttr (memory-buffer
        // backed); convert to DenseElementsAttr so writeConstantsToFileSystem
        // can access raw bytes via getRawData()/isSplat().
        auto elementsAttr =
            onnx_mlir::ElementsAttrBuilder::toDenseElementsAttr(rawAttr);
        if (!elementsAttr)
          return WalkResult::advance();

        auto tensorType = cast<TensorType>(constantOp.getResult().getType());
        auto elementType = tensorType.getElementType();
        auto shape = tensorType.getShape();

        int64_t numElements = 1;
        for (int64_t dim : shape) {
          if (dim <= 0)
            return WalkResult::advance();
          numElements *= dim;
        }

        size_t elementSize = elementType.getIntOrFloatBitWidth() / 8;
        size_t totalSize = numElements * elementSize;

        std::string debugName = "constant_" + std::to_string(nextIndex);
        if (auto nameLoc = dyn_cast<NameLoc>(constantOp.getLoc()))
          debugName = nameLoc.getName().str();

        ConstantInfo info(nextIndex, elementsAttr, elementType, shape,
                          totalSize, elementSize, numElements, debugName);
        constantRegistry_[constantOp.getResult()] = std::move(info);
        nextIndex++;

        return WalkResult::advance();
      });
    }

    if (nextIndex > 0) {
      COMPILER_DEBUG_LOG("[ONNX→HIP] Discovered " << nextIndex << " constants:\n");
      for (const auto& entry : constantRegistry_) {
        const auto& info = entry.second;
        llvm::errs() << "  [" << info.globalIndex << "] " << info.name
                     << " : shape=[";
        for (size_t i = 0; i < info.shape.size(); ++i) {
          if (i > 0)
            llvm::errs() << "x";
          llvm::errs() << info.shape[i];
        }
        llvm::errs() << "], size=" << info.sizeInBytes << " bytes\n";
      }
    }

    return success();
  }

  /// Compute byte offsets for each constant in constants.bin and set
  /// totalFileSize_. Offsets are assigned in size-ascending order so that
  /// small constants are packed together at the front.
  ///
  /// Algorithm (sort ascending by size, walk once):
  ///
  ///   currentOffset = 0, chunkUsed = 0
  ///
  ///   for each constant (sorted by size asc):
  ///     if size < kConstantAlignment:                 // small
  ///       if chunkUsed + size > kConstantAlignment:   // doesn't fit -> close chunk
  ///         currentOffset += (kConstantAlignment - chunkUsed)
  ///         chunkUsed = 0
  ///       assign offset = currentOffset
  ///       currentOffset += size
  ///       chunkUsed += size
  ///     else:                                         // large
  ///       if chunkUsed > 0:                           // close any open chunk
  ///         currentOffset += (kConstantAlignment - chunkUsed)
  ///         chunkUsed = 0
  ///       assign offset = currentOffset               // already aligned
  ///       currentOffset += ceil(size / kConstantAlignment) * kConstantAlignment
  ///
  /// Small constants are packed tightly within kConstantAlignment-byte chunks
  /// (no individual alignment, but no constant spans two chunks).
  /// Large constants each occupy ceil(size/kConstantAlignment)*kConstantAlignment
  /// bytes, kConstantAlignment-byte aligned, zero-padded at the end.
  void computeOffsets() {
    SmallVector<ConstantInfo*> sorted;
    sorted.reserve(constantRegistry_.size());
    for (auto& [val, info] : constantRegistry_)
      sorted.push_back(&info);
    llvm::sort(sorted, [](const ConstantInfo* a, const ConstantInfo* b) {
      return a->sizeInBytes < b->sizeInBytes;
    });

    size_t currentOffset = 0;
    size_t chunkUsed = 0;
    for (ConstantInfo* info : sorted) {
      if (info->sizeInBytes < kConstantAlignment) {
        // Small: pack into current chunk; start a new chunk if it doesn't fit.
        if (chunkUsed + info->sizeInBytes > kConstantAlignment) {
          currentOffset += (kConstantAlignment - chunkUsed);
          chunkUsed = 0;
        }
        info->offset = currentOffset;
        currentOffset += info->sizeInBytes;
        chunkUsed += info->sizeInBytes;
      } else {
        // Large: close any open chunk, then place at next aligned boundary.
        if (chunkUsed > 0) {
          currentOffset += (kConstantAlignment - chunkUsed);
          chunkUsed = 0;
        }
        info->offset = currentOffset;
        size_t alignedSize =
            ((info->sizeInBytes + kConstantAlignment - 1) / kConstantAlignment) *
            kConstantAlignment;
        currentOffset += alignedSize;
      }
    }
    totalFileSize_ = currentOffset;
  }

  /// Write all constants to FileSystem in offset order, zero-filling all
  /// alignment gaps. Output: "constants.bin" with layout determined by
  /// computeOffsets().
  LogicalResult writeConstantsToFileSystem(ModuleOp module,
                                           morphizen::FileSystem* fs) {
    if (constantRegistry_.empty()) {
      COMPILER_DEBUG_LOG("[ONNX→HIP] No constants to write (external mode)\n");
      return success();
    }

    // Sort by offset (ascending) — this is the layout order in the file.
    SmallVector<const ConstantInfo*> sorted;
    sorted.reserve(constantRegistry_.size());
    for (const auto& entry : constantRegistry_)
      sorted.push_back(&entry.second);
    llvm::sort(sorted, [](const ConstantInfo* a, const ConstantInfo* b) {
      return a->offset < b->offset;
    });

    const std::string constantsFilename =
        !compilationOptions_.constants_file.empty()
            ? compilationOptions_.constants_file
            : "constants.bin";
    auto writer = fs->create_writer_template(constantsFilename.c_str());
    if (!writer) {
      COMPILER_DEBUG_LOG("[ONNX→HIP] Failed to create " << constantsFilename
                   << " via FileSystem\n");
      return failure();
    }

    // Zero buffer for alignment padding gaps (at most kConstantAlignment bytes).
    const std::array<char, kConstantAlignment> zeros = {};

    auto writeZeros = [&](size_t n) -> LogicalResult {
      while (n > 0) {
        size_t chunk = std::min(n, kConstantAlignment);
        if (writer->fwrite(zeros.data(), chunk) != chunk) {
          COMPILER_DEBUG_LOG("[ONNX→HIP] Short write during zero-fill\n");
          return failure();
        }
        n -= chunk;
      }
      return success();
    };

    size_t writePos = 0;
    for (const ConstantInfo* info : sorted) {
      // Zero-fill any alignment gap before this constant.
      if (info->offset > writePos) {
        if (failed(writeZeros(info->offset - writePos)))
          return failure();
        writePos = info->offset;
      }

      auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(info->value);
      if (!denseAttr) {
        COMPILER_DEBUG_LOG("[ONNX→HIP] Constant [" << info->globalIndex
                     << "] is not DenseElementsAttr\n");
        return failure();
      }
      auto rawData = denseAttr.getRawData();

      if (denseAttr.isSplat()) {
        // Splat: MLIR stores only one element; expand via a 1 MB staging
        // buffer. Writing element-by-element would be extremely slow for large
        // tensors (e.g. 1 GB embedding table = 134M fwrite calls).
        const size_t kSplatChunk = 1024 * 1024;
        size_t elemSize = rawData.size();
        size_t bufSize =
            (std::min(info->sizeInBytes, kSplatChunk) / elemSize) * elemSize;
        std::vector<char> buf(bufSize);
        for (size_t i = 0; i < bufSize; i += elemSize)
          std::memcpy(buf.data() + i, rawData.data(), elemSize);
        size_t remaining = info->sizeInBytes;
        while (remaining > 0) {
          size_t toWrite = std::min(remaining, bufSize);
          if (writer->fwrite(buf.data(), toWrite) != toWrite) {
            COMPILER_DEBUG_LOG("[ONNX→HIP] Short write (splat) for constant ["
                         << info->globalIndex << "]\n");
            return failure();
          }
          remaining -= toWrite;
        }
      } else {
        if (writer->fwrite(rawData.data(), rawData.size()) != rawData.size()) {
          COMPILER_DEBUG_LOG("[ONNX→HIP] Short write for constant ["
                       << info->globalIndex << "]\n");
          return failure();
        }
      }
      writePos += info->sizeInBytes;

      COMPILER_DEBUG_LOG("[ONNX→HIP] Wrote constant [" << info->globalIndex
                   << "] " << info->name << " at offset=" << info->offset
                   << " (" << info->sizeInBytes << " bytes)\n");
    }

    // Zero-fill trailing alignment padding for the last large constant.
    if (writePos < totalFileSize_) {
      if (failed(writeZeros(totalFileSize_ - writePos)))
        return failure();
      writePos = totalFileSize_;
    }

    COMPILER_DEBUG_LOG("[ONNX→HIP] constants.bin total: " << writePos
                 << " bytes\n");
    return success();
  }

  /// Emit hipdnn.constant_sizes and hipdnn.constant_offsets module attributes.
  /// GenerateInterfacePass reads these to build the FlatBuffers metadata blob
  /// used by the runtime to locate each constant in constants.bin.
  LogicalResult emitConstantSizesAttributes(ModuleOp module) {
    if (constantRegistry_.empty()) {
      // No constants - no sizes attribute needed
      return success();
    }

    // Sort by globalIndex
    SmallVector<std::pair<int64_t, const ConstantInfo*>> sorted;
    sorted.reserve(constantRegistry_.size());
    for (const auto& entry : constantRegistry_)
      sorted.push_back({entry.second.globalIndex, &entry.second});
    llvm::sort(sorted,
               [](const auto& a, const auto& b) { return a.first < b.first; });

    // Build sizes and offsets arrays, both indexed by globalIndex.
    SmallVector<int64_t> sizes;
    SmallVector<int64_t> offsets;
    sizes.reserve(sorted.size());
    offsets.reserve(sorted.size());
    for (const auto& [idx, info] : sorted) {
      sizes.push_back(static_cast<int64_t>(info->sizeInBytes));
      offsets.push_back(static_cast<int64_t>(info->offset));
    }

    MLIRContext* ctx = module.getContext();
    module->setAttr("hipdnn.constant_sizes",
                    DenseI64ArrayAttr::get(ctx, sizes));
    module->setAttr("hipdnn.constant_offsets",
                    DenseI64ArrayAttr::get(ctx, offsets));

    for (size_t i = 0; i < sizes.size(); ++i)
      COMPILER_DEBUG_LOG("[ONNX→HIP]   [" << i << "] size=" << sizes[i]
                   << " offset=" << offsets[i] << "\n");

    return success();
  }

  /// Generate module metadata attributes required by GenerateInterfacePass.
  /// CRITICAL: Must be called BEFORE patterns transform signatures.
  LogicalResult generateModuleMetadata(ModuleOp module) {
    auto mainFunc = module.lookupSymbol<func::FuncOp>("main_graph");
    if (!mainFunc)
      return success();

    auto originalFuncType = mainFunc.getFunctionType();
    OpBuilder builder(module.getContext());

    int64_t inputCount = originalFuncType.getNumInputs();
    SmallVector<Attribute> inputShapes;
    SmallVector<int64_t> inputElementSizes;

    for (Type inputType : originalFuncType.getInputs()) {
      // Skip the !hip.context argument inserted by hip-add-context-arg.
      if (isa<hip::ContextType>(inputType)) {
        --inputCount;
        continue;
      }
      if (auto tensorType = dyn_cast<RankedTensorType>(inputType)) {
        SmallVector<int64_t> shape(tensorType.getShape().begin(),
                                   tensorType.getShape().end());
        inputShapes.push_back(builder.getDenseI64ArrayAttr(shape));
        inputElementSizes.push_back(
            tensorType.getElementType().getIntOrFloatBitWidth() / 8);
      } else {
        COMPILER_DEBUG_LOG("[ONNX→HIP] Warning: non-tensor input in @main_graph, "
                        "skipping metadata\n");
        return success();
      }
    }

    int64_t outputCount = originalFuncType.getNumResults();
    SmallVector<Attribute> outputShapes;
    SmallVector<int64_t> outputElementSizes;

    for (Type resultType : originalFuncType.getResults()) {
      if (auto tensorType = dyn_cast<RankedTensorType>(resultType)) {
        SmallVector<int64_t> shape(tensorType.getShape().begin(),
                                   tensorType.getShape().end());
        outputShapes.push_back(builder.getDenseI64ArrayAttr(shape));
        outputElementSizes.push_back(
            tensorType.getElementType().getIntOrFloatBitWidth() / 8);
      } else {
        COMPILER_DEBUG_LOG("[ONNX→HIP] Warning: non-tensor output in @main_graph, "
                        "skipping metadata\n");
        return success();
      }
    }

    module->setAttr("hipdnn.input_count",
                    builder.getI64IntegerAttr(inputCount));
    module->setAttr("hipdnn.input_shapes", builder.getArrayAttr(inputShapes));
    module->setAttr("hipdnn.input_element_sizes",
                    builder.getDenseI64ArrayAttr(inputElementSizes));
    module->setAttr("hipdnn.output_count",
                    builder.getI64IntegerAttr(outputCount));
    module->setAttr("hipdnn.output_shapes",
                    builder.getArrayAttr(outputShapes));
    module->setAttr("hipdnn.output_element_sizes",
                    builder.getDenseI64ArrayAttr(outputElementSizes));

    COMPILER_DEBUG_LOG("[ONNX→HIP] Generated module metadata: input_count="
                 << inputCount << " output_count=" << outputCount << "\n");

    return success();
  }

  /// Generate constant registry function for runtime initialization.
  LogicalResult generateConstantRegistry(ModuleOp module) {
    if (constantRegistry_.empty())
      return success();

    OpBuilder builder(module.getBodyRegion());
    auto loc = module.getLoc();
    auto* context = builder.getContext();

    COMPILER_DEBUG_LOG("[ONNX→HIP] Generating constant registry\n");

    auto ptrType = LLVM::LLVMPointerType::get(context);
    auto i64Type = builder.getI64Type();
    auto i1Type = builder.getI1Type();
    auto constantInfoType = LLVM::LLVMStructType::getLiteral(
        context, {ptrType, i64Type, i64Type, i64Type});
    auto constantRegistryType =
        LLVM::LLVMStructType::getLiteral(context, {ptrType, i64Type});
    auto constantInfoArrayType =
        LLVM::LLVMArrayType::get(constantInfoType, constantRegistry_.size());

    builder.setInsertionPointToEnd(module.getBody());

    auto constantsArrayGlobal = builder.create<LLVM::GlobalOp>(
        loc, constantInfoArrayType, /*isConstant=*/false,
        LLVM::Linkage::Internal, "__constant_info_array",
        builder.getZeroAttr(constantInfoArrayType), /*alignment=*/0,
        /*addr_space=*/0);

    auto registryGlobal = builder.create<LLVM::GlobalOp>(
        loc, constantRegistryType, /*isConstant=*/false,
        LLVM::Linkage::Internal, "__constant_registry",
        builder.getZeroAttr(constantRegistryType), /*alignment=*/0,
        /*addr_space=*/0);

    auto initFlagGlobal = builder.create<LLVM::GlobalOp>(
        loc, i1Type, /*isConstant=*/false, LLVM::Linkage::Internal,
        "__registry_initialized", builder.getBoolAttr(false), /*alignment=*/0,
        /*addr_space=*/0);

    auto funcType = LLVM::LLVMFunctionType::get(ptrType, {});
    auto funcOp = builder.create<LLVM::LLVMFuncOp>(
        loc, "get_constant_registry", funcType, LLVM::Linkage::External);

    Block* entryBlock = funcOp.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value initFlagAddr = builder.create<LLVM::AddressOfOp>(
        loc, ptrType, initFlagGlobal.getSymName());
    Value isInitialized =
        builder.create<LLVM::LoadOp>(loc, i1Type, initFlagAddr);

    Block* initBlock = funcOp.addBlock();
    Block* returnBlock = funcOp.addBlock();

    builder.create<LLVM::CondBrOp>(loc, isInitialized, returnBlock, initBlock);

    builder.setInsertionPointToStart(initBlock);

    Value arrayAddr = builder.create<LLVM::AddressOfOp>(
        loc, ptrType, constantsArrayGlobal.getSymName());

    size_t index = 0;
    for (const auto& entry : constantRegistry_) {
      const auto& info = entry.second;

      Value constInfo = builder.create<LLVM::UndefOp>(loc, constantInfoType);

      Value dataPtr =
          builder.create<LLVM::AddressOfOp>(loc, ptrType, info.name);
      constInfo =
          builder.create<LLVM::InsertValueOp>(loc, constInfo, dataPtr, 0);

      Value sizeBytes = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(info.sizeInBytes));
      constInfo =
          builder.create<LLVM::InsertValueOp>(loc, constInfo, sizeBytes, 1);

      Value elemSize = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(info.elementSizeInBytes));
      constInfo =
          builder.create<LLVM::InsertValueOp>(loc, constInfo, elemSize, 2);

      Value numElems = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(info.numElements));
      constInfo =
          builder.create<LLVM::InsertValueOp>(loc, constInfo, numElems, 3);

      Value indexVal = builder.create<LLVM::ConstantOp>(
          loc, i64Type, builder.getI64IntegerAttr(info.globalIndex));
      Value elemPtr = builder.create<LLVM::GEPOp>(
          loc, ptrType, constantInfoArrayType, arrayAddr,
          ArrayRef<LLVM::GEPArg>{0, indexVal});
      builder.create<LLVM::StoreOp>(loc, constInfo, elemPtr);
    }

    Value registryAddr = builder.create<LLVM::AddressOfOp>(
        loc, ptrType, registryGlobal.getSymName());

    Value registry = builder.create<LLVM::UndefOp>(loc, constantRegistryType);
    registry =
        builder.create<LLVM::InsertValueOp>(loc, registry, arrayAddr, 0);
    Value count = builder.create<LLVM::ConstantOp>(
        loc, i64Type, builder.getI64IntegerAttr(constantRegistry_.size()));
    registry = builder.create<LLVM::InsertValueOp>(loc, registry, count, 1);
    builder.create<LLVM::StoreOp>(loc, registry, registryAddr);

    Value trueVal =
        builder.create<LLVM::ConstantOp>(loc, i1Type, builder.getBoolAttr(true));
    builder.create<LLVM::StoreOp>(loc, trueVal, initFlagAddr);

    builder.create<LLVM::BrOp>(loc, returnBlock);

    builder.setInsertionPointToStart(returnBlock);
    Value globalRegistryAddr = builder.create<LLVM::AddressOfOp>(
        loc, ptrType, registryGlobal.getSymName());
    builder.create<LLVM::ReturnOp>(loc, globalRegistryAddr);

    llvm::errs() << "  Generated: get_constant_registry() with "
                 << constantRegistry_.size() << " constants (lazy init)\n";

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

namespace mlir {
namespace hip {

std::unique_ptr<Pass>
createConvertOnnxToHipPass(morphizen::FileSystem* fs,
                           const hip::compiler::CompilationOptionsT& compilationOptions) {
  return std::make_unique<ConvertOnnxToHipPass>(fs, compilationOptions);
}

std::unique_ptr<Pass> createConvertOnnxToHipPass() {
  static const hip::compiler::CompilationOptionsT kDefaultOptions{};
  return std::make_unique<ConvertOnnxToHipPass>(nullptr, kDefaultOptions);
}

} // namespace hip
} // namespace mlir
