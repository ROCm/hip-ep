/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHip.cpp - Convert ONNX dialect to HIP dialect (tensor DPS) ---===//
//
// Converts ONNX dialect IR into HIP dialect IR using destination-passing style
// (DPS) with tensor types.  ONNX ops are matched by name via the generic MLIR
// Operation API, so no onnx-mlir headers or libraries are required.
// Bufferization to memref is handled by a separate downstream pass.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "hip/Support/DiskFileSystem.h"
#include "morphizen-foundation/file_io.hpp"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244) // Conversion warnings in LLVM JSON.h
#endif
#include "llvm/Support/JSON.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "convert-onnx-to-hip"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_CONVERTONNXTOHIPPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Sanitize an arbitrary string (typically an ONNX node name) into a valid
/// MLIR bare identifier fragment.  Non-alphanumeric characters are replaced
/// with '_', consecutive underscores are collapsed, and leading/trailing
/// underscores are stripped.  Returns an empty string if the input yields
/// no usable characters.
static std::string sanitizeForMlirIdentifier(llvm::StringRef raw) {
  std::string sanitized;
  sanitized.reserve(raw.size());
  for (char c : raw) {
    if (std::isalnum(static_cast<unsigned char>(c)) || c == '_')
      sanitized.push_back(c);
    else
      sanitized.push_back('_');
  }
  // Collapse runs of underscores and trim leading/trailing ones.
  std::string result;
  result.reserve(sanitized.size());
  bool lastWasUnderscore = true; // suppress leading '_'
  for (char c : sanitized) {
    if (c == '_') {
      if (!lastWasUnderscore)
        result.push_back(c);
      lastWasUnderscore = true;
    } else {
      result.push_back(c);
      lastWasUnderscore = false;
    }
  }
  while (!result.empty() && result.back() == '_')
    result.pop_back();
  return result;
}

/// Create a tensor.empty for a DPS init operand.  Dynamic dimension sizes
/// are extracted from \p source using tensor.dim at each dynamic index.
/// Suitable for ops where the output shape aligns positionally with one input
/// (e.g., softmax, element-wise).
static mlir::Value createEmptyTensor(mlir::OpBuilder &builder,
                                     mlir::Location loc,
                                     mlir::RankedTensorType resultType,
                                     mlir::Value source) {
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank())) {
    if (resultType.isDynamicDim(dimIdx))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(builder, loc, source, dimIdx));
  }
  return mlir::tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes);
}

//===----------------------------------------------------------------------===//
// Constant externalization helpers
//===----------------------------------------------------------------------===//

static std::string elementTypeToString(mlir::Type elemType) {
  if (elemType.isF16())
    return "f16";
  else if (elemType.isBF16())
    return "bf16";
  else if (elemType.isF32())
    return "f32";
  else if (elemType.isF64())
    return "f64";
  else if (elemType.isInteger(8))
    return "i8";
  else if (elemType.isInteger(16))
    return "i16";
  else if (elemType.isInteger(32))
    return "i32";
  else if (elemType.isInteger(64))
    return "i64";
  else if (elemType.isInteger(1))
    return "i1";
  std::string result;
  llvm::raw_string_ostream os(result);
  elemType.print(os);
  return result;
}

static int64_t alignTo(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

/// Mutable state shared across calls to lowerOnnxConstants when
/// externalization is enabled.
struct ExternalizationState {
  std::unique_ptr<morphizen::FileWriter,
                  morphizen::FileSystem::Deleter<morphizen::FileWriter>>
      writer;
  llvm::json::Array manifestEntries;
  int64_t currentOffset = 0;
  int64_t constantIndex = 0;
  std::string binFileName;
  llvm::SmallVector<int64_t> constantSizes;
  llvm::SmallVector<int64_t> constantOffsets;
};

/// Lower onnx.Constant ops.
///
/// Small constants (below \p minNumElements or when externalization is
/// disabled) are converted to arith.constant -- bufferization later turns
/// them into memref.global + memref.get_global.
///
/// Large constants (at or above threshold, non-splat) are externalized:
/// their raw data is appended to the sidecar binary via \p extState, and
/// they are replaced by an extern memref.global (no initial value) carrying
/// a hip.external_data {offset, size} attribute, plus a
/// memref.get_global + bufferization.to_tensor bridge at the use site.
static mlir::LogicalResult lowerOnnxConstants(mlir::ModuleOp module,
                                              mlir::func::FuncOp funcOp,
                                              int64_t minNumElements,
                                              ExternalizationState *extState) {
  constexpr int64_t kAlignment = 64;

  llvm::SmallVector<mlir::Operation *> constants;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "onnx.Constant")
      constants.push_back(op);
  });

  for (mlir::Operation *constOp : constants) {
    auto valueAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
        constOp->getAttrOfType<mlir::ElementsAttr>("value"));
    if (!valueAttr) {
      return constOp->emitError(
          "unsupported onnx.Constant form (expected dense value attribute)");
    }

    bool shouldExternalize = extState && minNumElements > 0 &&
                             valueAttr.getNumElements() >= minNumElements;

    if (shouldExternalize) {
      auto tensorType = mlir::cast<mlir::RankedTensorType>(valueAttr.getType());
      auto memrefType = mlir::MemRefType::get(tensorType.getShape(),
                                              tensorType.getElementType());

      // Build a unique, MLIR-safe symbol name: prefix + sanitized node
      // name (if present) + monotonic index to guarantee uniqueness.
      std::string name = "hip_ext_constant_";
      if (auto nodeNameAttr =
              constOp->getAttrOfType<mlir::StringAttr>("onnx_node_name")) {
        std::string fragment =
            sanitizeForMlirIdentifier(nodeNameAttr.getValue());
        if (!fragment.empty())
          name += fragment + "_";
      }
      name += std::to_string(extState->constantIndex);

      // Pad binary file to alignment boundary.
      int64_t aligned = llvm::alignTo(extState->currentOffset, kAlignment);
      int64_t padding = aligned - extState->currentOffset;
      if (padding > 0) {
        llvm::SmallVector<char> zeros(padding, 0);
        extState->writer->fwrite(zeros.data(), padding);
        extState->currentOffset += padding;
      }
      int64_t entryOffset = extState->currentOffset;

      // Compute full tensor byte size from shape and element width.
      auto rawData = valueAttr.getRawData();
      int64_t elemBits = tensorType.getElementTypeBitWidth();
      int64_t byteSize = valueAttr.getNumElements() * ((elemBits + 7) / 8);

      if (valueAttr.isSplat()) {
        // Splat: MLIR stores only one element; expand via a staging
        // buffer to avoid per-element fwrite overhead on large tensors.
        constexpr size_t kSplatChunk = 1024 * 1024;
        size_t elemSize = rawData.size();
        size_t bufSize =
            (std::min(static_cast<size_t>(byteSize), kSplatChunk) / elemSize) *
            elemSize;
        std::vector<char> buf(bufSize);
        for (size_t i = 0; i < bufSize; i += elemSize)
          std::memcpy(buf.data() + i, rawData.data(), elemSize);
        size_t remaining = static_cast<size_t>(byteSize);
        while (remaining > 0) {
          size_t toWrite = std::min(remaining, bufSize);
          extState->writer->fwrite(buf.data(), toWrite);
          remaining -= toWrite;
        }
      } else {
        extState->writer->fwrite(rawData.data(), byteSize);
      }
      extState->currentOffset += byteSize;

      // Track sizes and offsets for module-level metadata.
      extState->constantSizes.push_back(byteSize);
      extState->constantOffsets.push_back(entryOffset);

      // Build JSON manifest entry.
      llvm::json::Array shapeArray;
      for (int64_t dim : tensorType.getShape())
        shapeArray.push_back(dim);
      llvm::json::Object entry;
      entry["name"] = name;
      entry["shape"] = std::move(shapeArray);
      entry["element_type"] = elementTypeToString(tensorType.getElementType());
      entry["offset"] = entryOffset;
      entry["size"] = byteSize;
      entry["alignment"] = kAlignment;
      extState->manifestEntries.push_back(std::move(entry));

      // Create extern memref.global at module scope.
      mlir::OpBuilder moduleBuilder(module.getBody(),
                                    module.getBody()->begin());
      auto externalDataAttr = moduleBuilder.getDictionaryAttr({
          moduleBuilder.getNamedAttr("index", moduleBuilder.getI64IntegerAttr(
                                                  extState->constantIndex)),
          moduleBuilder.getNamedAttr(
              "offset", moduleBuilder.getI64IntegerAttr(entryOffset)),
          moduleBuilder.getNamedAttr("size",
                                     moduleBuilder.getI64IntegerAttr(byteSize)),
      });
      auto globalOp = mlir::memref::GlobalOp::create(
          moduleBuilder, constOp->getLoc(), name,
          /*sym_visibility=*/moduleBuilder.getStringAttr("private"),
          /*type=*/memrefType,
          /*initial_value=*/nullptr,
          /*constant=*/false,
          /*alignment=*/moduleBuilder.getI64IntegerAttr(kAlignment));
      globalOp->setAttr("hip.external_data", externalDataAttr);

      // At the use site: memref.get_global + bufferization.to_tensor.
      mlir::OpBuilder builder(constOp);
      auto getGlobal = mlir::memref::GetGlobalOp::create(
          builder, constOp->getLoc(), memrefType, name);
      auto toTensor = mlir::bufferization::ToTensorOp::create(
          builder, constOp->getLoc(), tensorType, getGlobal.getResult(),
          /*restrict=*/builder.getUnitAttr(),
          /*writable=*/nullptr);
      constOp->getResult(0).replaceAllUsesWith(toTensor.getResult());
      constOp->erase();

      ++extState->constantIndex;
    } else {
      // Small / splat / externalization disabled: inline arith.constant.
      mlir::OpBuilder builder(constOp);
      auto arithConst = mlir::arith::ConstantOp::create(
          builder, constOp->getLoc(), valueAttr);
      constOp->getResult(0).replaceAllUsesWith(arithConst.getResult());
      constOp->erase();
    }
  }
  return mlir::success();
}

/// Replace onnx.Return terminators with func.return.
///
/// In onnx-mlir's own pipeline a dedicated StandardFuncReturnPass handles
/// this before lowering.  Since we bypass that pipeline we must do it
/// ourselves.
static void lowerOnnxReturns(mlir::func::FuncOp funcOp) {
  llvm::SmallVector<mlir::Operation *> returns;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "onnx.Return")
      returns.push_back(op);
  });

  for (mlir::Operation *returnOp : returns) {
    mlir::OpBuilder builder(returnOp);
    mlir::func::ReturnOp::create(builder, returnOp->getLoc(),
                                 returnOp->getOperands());
    returnOp->erase();
  }
}

/// Get !hip.context from function argument 0. Returns failure if the
/// function has no arguments or the first argument is not !hip.context.
static mlir::FailureOr<mlir::Value>
getContextArg(mlir::Operation *op, mlir::PatternRewriter &rewriter) {
  auto funcOp = op->getParentOfType<mlir::func::FuncOp>();
  if (!funcOp)
    return rewriter.notifyMatchFailure(op, "not inside a function");
  auto &entry = funcOp.getBody().front();
  if (entry.getNumArguments() == 0)
    return rewriter.notifyMatchFailure(op, "function has no arguments");
  mlir::Value ctx = entry.getArgument(0);
  if (!mlir::isa<mlir::hip::ContextType>(ctx.getType()))
    return rewriter.notifyMatchFailure(op,
                                       "first argument is not !hip.context");
  return ctx;
}

static mlir::LogicalResult convertComputeOps(mlir::func::FuncOp funcOp,
                                             mlir::MLIRContext *ctx);

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

/// onnx.MatMul -> hip.hipblaslt.matmul
struct MatMulToHip : public mlir::RewritePattern {
  MatMulToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.MatMul", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MatMulToHip::matchAndRewrite(mlir::Operation *op,
                             mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value a = op->getOperand(0);
  mlir::Value b = op->getOperand(1);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // MatMul: result[..., M, N] = A[..., M, K] @ B[..., K, N].
  // Batch and M dims come from A; N comes from B's last dim.
  llvm::SmallVector<mlir::Value> dynSizes;
  const int64_t rank = resultType.getRank();
  const auto bType = mlir::cast<mlir::RankedTensorType>(b.getType());
  for (int64_t dimIdx : llvm::seq<int64_t>(rank)) {
    if (!resultType.isDynamicDim(dimIdx))
      continue;
    if (dimIdx == rank - 1) {
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, b, bType.getRank() - 1));
    } else {
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, a, dimIdx));
    }
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);
  auto hipOp = mlir::hip::MatmulOp::create(rewriter, loc, resultType, context,
                                           a, b, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Transpose -> hip.transpose
struct TransposeToHip : public mlir::RewritePattern {
  TransposeToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Transpose", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
TransposeToHip::matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);

  auto permAttr = op->getAttrOfType<mlir::ArrayAttr>("perm");
  if (!permAttr)
    return op->emitOpError("hip.transpose requires explicit perm attribute");

  int64_t dim0 = -1, dim1 = -1;
  int64_t mismatchCount = 0;
  for (auto [permIdx, attr] : llvm::enumerate(permAttr)) {
    int64_t p = mlir::cast<mlir::IntegerAttr>(attr).getInt();
    if (p != static_cast<int64_t>(permIdx)) {
      ++mismatchCount;
      if (dim0 < 0)
        dim0 = static_cast<int64_t>(permIdx);
      else if (dim1 < 0)
        dim1 = static_cast<int64_t>(permIdx);
    }
  }
  if (mismatchCount != 2 || dim0 < 0 || dim1 < 0)
    return op->emitOpError("perm must swap exactly two dimensions");
  int64_t p0 = mlir::cast<mlir::IntegerAttr>(permAttr[dim0]).getInt();
  int64_t p1 = mlir::cast<mlir::IntegerAttr>(permAttr[dim1]).getInt();
  if (p0 != dim1 || p1 != dim0)
    return op->emitOpError("perm must swap exactly two dimensions");

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Transpose: output dim i corresponds to input dim perm[i].
  llvm::SmallVector<mlir::Value> dynSizes;
  for (auto [outDimIdx, attr] : llvm::enumerate(permAttr)) {
    if (resultType.isDynamicDim(outDimIdx)) {
      const int64_t srcDim = mlir::cast<mlir::IntegerAttr>(attr).getInt();
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, data, srcDim));
    }
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  mlir::Value d0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, dim0);
  mlir::Value d1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, dim1);

  auto hipOp = mlir::hip::TransposeOp::create(rewriter, loc, resultType,
                                              context, d0, d1, data, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Mul -> hip.mul
struct MulToHip : public mlir::RewritePattern {
  MulToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Mul", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MulToHip::matchAndRewrite(mlir::Operation *op,
                          mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value a = op->getOperand(0);
  mlir::Value b = op->getOperand(1);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Use the operand whose rank matches the result for dim extraction
  // (handles scalar * tensor broadcasting).
  auto aType = mlir::cast<mlir::RankedTensorType>(a.getType());
  mlir::Value source = (aType.getRank() == resultType.getRank()) ? a : b;
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, source);

  auto hipOp =
      mlir::hip::MulOp::create(rewriter, loc, resultType, context, a, b, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Softmax -> hip.miopen.softmax
struct SoftmaxToHip : public mlir::RewritePattern {
  SoftmaxToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Softmax", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
SoftmaxToHip::matchAndRewrite(mlir::Operation *op,
                              mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::MiopenSoftmaxOp::create(rewriter, loc, resultType,
                                                  context, input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Sigmoid -> hip.sigmoid
struct SigmoidToHip : public mlir::RewritePattern {
  SigmoidToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Sigmoid", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
SigmoidToHip::matchAndRewrite(mlir::Operation *op,
                              mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::SigmoidOp::create(rewriter, loc, resultType, context,
                                            input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Sub -> hip.sub
struct SubToHip : public mlir::RewritePattern {
  SubToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Sub", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
SubToHip::matchAndRewrite(mlir::Operation *op,
                          mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value lhs = op->getOperand(0);
  mlir::Value rhs = op->getOperand(1);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, lhs);
  auto hipOp = mlir::hip::SubOp::create(rewriter, loc, resultType, context, lhs,
                                        rhs, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Cast -> hip.cast
struct CastToHip : public mlir::RewritePattern {
  CastToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Cast", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
CastToHip::matchAndRewrite(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // Map MLIR element type to ONNX DataType enum
  mlir::Type targetType = resultType.getElementType();
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

  // Validate that we have a supported type
  if (onnxDataType == 0)
    return rewriter.notifyMatchFailure(op, "unsupported cast target type");

  auto toAttr = rewriter.getI64IntegerAttr(onnxDataType);

  auto hipOp = mlir::hip::CastOp::create(rewriter, loc, resultType, context,
                                         input, init, toAttr);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.ReduceSum -> hip.reduce_sum
struct ReduceSumToHip : public mlir::RewritePattern {
  ReduceSumToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceSum", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReduceSumToHip::matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, data);

  // Handle axes: can be operand (opset 13+) or attribute (opset < 13)
  mlir::Value axesOperand;
  if (op->getNumOperands() > 1) {
    // Axes provided as operand (opset 13+)
    axesOperand = op->getOperand(1);
  } else {
    // Axes provided as attribute - convert to constant tensor
    llvm::SmallVector<int64_t> axesVec;
    if (auto axesAttr = op->getAttrOfType<mlir::ArrayAttr>("axes")) {
      for (auto a : axesAttr)
        axesVec.push_back(mlir::cast<mlir::IntegerAttr>(a).getInt());
    } else {
      // Default: reduce all axes
      auto inputType = mlir::cast<mlir::RankedTensorType>(data.getType());
      for (int64_t i = 0; i < inputType.getRank(); ++i)
        axesVec.push_back(i);
    }

    // Create constant tensor for axes
    auto axesType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(axesVec.size())}, rewriter.getI64Type());
    auto axesAttr =
        mlir::DenseIntElementsAttr::get(axesType, llvm::ArrayRef(axesVec));
    axesOperand =
        mlir::arith::ConstantOp::create(rewriter, loc, axesType, axesAttr);
  }

  // Extract keepdims attribute (defaults to 1 in ONNX)
  int64_t keepdims = 1;
  if (auto keepdimsAttr = op->getAttrOfType<mlir::IntegerAttr>("keepdims")) {
    keepdims = keepdimsAttr.getSInt();
  }

  // Create hip.reduce_sum operation
  auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
  auto hipOp =
      mlir::hip::ReduceSumOp::create(rewriter, loc, resultType, context, data,
                                     axesOperand, init, keepdimsAttr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ONNX MatMulNBits -> HIP MatMulNBits (com.microsoft custom op)
//===----------------------------------------------------------------------===//

struct MatMulNBitsToHip : public mlir::RewritePattern {
  MatMulNBitsToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MatMulNBitsToHip::matchAndRewrite(mlir::Operation *op,
                                  mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "MatMulNBits") {
    return rewriter.notifyMatchFailure(op, "not a MatMulNBits custom op");
  }
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(op, "not a com.microsoft domain op");
  }

  mlir::Location loc = op->getLoc();

  if (op->getNumOperands() < 3) {
    return rewriter.notifyMatchFailure(
        op, "expected at least 3 inputs for MatMulNBits");
  }
  if (op->getNumResults() != 1) {
    return rewriter.notifyMatchFailure(op, "expected 1 output for MatMulNBits");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "failed to get context argument");
  }
  mlir::Value context = *ctxOrFailure;

  mlir::Value A = op->getOperand(0);
  mlir::Value B = op->getOperand(1);
  mlir::Value scales = op->getOperand(2);

  auto getOptionalInput = [&](unsigned idx) -> mlir::Value {
    if (idx >= op->getNumOperands()) {
      return mlir::Value{};
    }
    mlir::Value v = op->getOperand(idx);
    if (!v || mlir::isa<mlir::NoneType>(v.getType())) {
      return mlir::Value{};
    }
    return v;
  };
  mlir::Value zeroPoints = getOptionalInput(3);
  mlir::Value gIdx = getOptionalInput(4);
  mlir::Value bias = getOptionalInput(5);

  auto KAttr = rewriter.getI64IntegerAttr(
      op->getAttrOfType<mlir::IntegerAttr>("K").getSInt());
  auto NAttr = rewriter.getI64IntegerAttr(
      op->getAttrOfType<mlir::IntegerAttr>("N").getSInt());

  auto bitsIntAttr = op->getAttrOfType<mlir::IntegerAttr>("bits");
  auto bitsAttr =
      rewriter.getI64IntegerAttr(bitsIntAttr ? bitsIntAttr.getSInt() : 4);

  auto blockSizeAttr = rewriter.getI64IntegerAttr(
      op->getAttrOfType<mlir::IntegerAttr>("block_size").getSInt());

  auto accuracyIntAttr = op->getAttrOfType<mlir::IntegerAttr>("accuracy_level");
  auto accuracyLevelAttr = rewriter.getI64IntegerAttr(
      accuracyIntAttr ? accuracyIntAttr.getSInt() : 0);

  auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, rt, A);

  auto hipOp = mlir::hip::MatMulNBitsOp::create(
      rewriter, loc, mlir::TypeRange{rt}, context, A, B, scales, zeroPoints,
      gIdx, bias, init, KAttr, NAttr, bitsAttr, blockSizeAttr,
      accuracyLevelAttr);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ONNX QMoE -> HIP QMoE (com.microsoft custom op)
//===----------------------------------------------------------------------===//

struct QMoEToHip : public mlir::RewritePattern {
  QMoEToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
QMoEToHip::matchAndRewrite(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "QMoE") {
    return rewriter.notifyMatchFailure(op, "not a QMoE custom op");
  }
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(op, "not a com.microsoft domain op");
  }

  mlir::Location loc = op->getLoc();

  if (op->getNumOperands() < 7) {
    return rewriter.notifyMatchFailure(op,
                                       "expected at least 7 inputs for QMoE");
  }
  if (op->getNumResults() != 1) {
    return rewriter.notifyMatchFailure(op, "expected 1 output for QMoE");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "failed to get context argument");
  }
  mlir::Value context = *ctxOrFailure;

  auto getOptionalInput = [&](unsigned idx) -> mlir::Value {
    if (idx >= op->getNumOperands()) {
      return mlir::Value{};
    }
    mlir::Value v = op->getOperand(idx);
    if (!v || mlir::isa<mlir::NoneType>(v.getType())) {
      return mlir::Value{};
    }
    return v;
  };

  mlir::Value input = op->getOperand(0);
  mlir::Value routerProbs = op->getOperand(1);
  mlir::Value fc1Weights = op->getOperand(2);
  mlir::Value fc1Scales = op->getOperand(3);
  mlir::Value fc1Bias = getOptionalInput(4);
  mlir::Value fc2Weights = op->getOperand(5);
  mlir::Value fc2Scales = op->getOperand(6);
  mlir::Value fc2Bias = getOptionalInput(7);
  mlir::Value fc3Weights = getOptionalInput(8);
  mlir::Value fc3Scales = getOptionalInput(9);
  mlir::Value fc3Bias = getOptionalInput(10);
  mlir::Value fc1ZeroPoints = getOptionalInput(11);
  mlir::Value fc2ZeroPoints = getOptionalInput(12);
  mlir::Value fc3ZeroPoints = getOptionalInput(13);

  auto expertWeightBitsIntAttr =
      op->getAttrOfType<mlir::IntegerAttr>("expert_weight_bits");
  auto expertWeightBitsAttr = rewriter.getI64IntegerAttr(
      expertWeightBitsIntAttr ? expertWeightBitsIntAttr.getSInt() : 4);

  auto kIntAttr = op->getAttrOfType<mlir::IntegerAttr>("k");
  auto kAttr = rewriter.getI64IntegerAttr(kIntAttr ? kIntAttr.getSInt() : 1);

  auto blockSizeIntAttr = op->getAttrOfType<mlir::IntegerAttr>("block_size");
  auto blockSizeAttr = rewriter.getI64IntegerAttr(
      blockSizeIntAttr ? blockSizeIntAttr.getSInt() : 0);

  auto normIntAttr =
      op->getAttrOfType<mlir::IntegerAttr>("normalize_routing_weights");
  auto normalizeAttr =
      rewriter.getI64IntegerAttr(normIntAttr ? normIntAttr.getSInt() : 0);

  auto swigluFusionIntAttr =
      op->getAttrOfType<mlir::IntegerAttr>("swiglu_fusion");
  auto swigluFusionAttr = rewriter.getI64IntegerAttr(
      swigluFusionIntAttr ? swigluFusionIntAttr.getSInt() : 0);

  auto sparseIntAttr = op->getAttrOfType<mlir::IntegerAttr>("use_sparse_mixer");
  auto useSparseAttr =
      rewriter.getI64IntegerAttr(sparseIntAttr ? sparseIntAttr.getSInt() : 0);

  auto alphaFloatAttr = op->getAttrOfType<mlir::FloatAttr>("activation_alpha");
  auto activationAlphaAttr =
      alphaFloatAttr ? alphaFloatAttr : rewriter.getF32FloatAttr(0.0f);

  auto betaFloatAttr = op->getAttrOfType<mlir::FloatAttr>("activation_beta");
  auto activationBetaAttr =
      betaFloatAttr ? betaFloatAttr : rewriter.getF32FloatAttr(0.0f);

  auto limitFloatAttr = op->getAttrOfType<mlir::FloatAttr>("swiglu_limit");
  auto swigluLimitAttr =
      limitFloatAttr ? limitFloatAttr : rewriter.getF32FloatAttr(0.0f);

  auto activationTypeStrAttr =
      op->getAttrOfType<mlir::StringAttr>("activation_type");
  auto activationTypeAttr = activationTypeStrAttr
                                ? activationTypeStrAttr
                                : rewriter.getStringAttr("relu");

  auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, rt, input);

  auto hipOp = mlir::hip::QMoEOp::create(
      rewriter, loc, mlir::TypeRange{rt}, context, input, routerProbs,
      fc1Weights, fc1Scales, fc2Weights, fc2Scales, fc1Bias, fc2Bias,
      fc3Weights, fc3Scales, fc3Bias, fc1ZeroPoints, fc2ZeroPoints,
      fc3ZeroPoints, init, expertWeightBitsAttr, kAttr, blockSizeAttr,
      normalizeAttr, swigluFusionAttr, useSparseAttr, activationAlphaAttr,
      activationBetaAttr, swigluLimitAttr, activationTypeAttr);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

/// onnx.Conv -> hip.conv
struct ConvToHip : public mlir::RewritePattern {
  ConvToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Conv", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ConvToHip::matchAndRewrite(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  mlir::Value weights = op->getOperand(1);

  // ONNX Conv always has 3 operands, but bias can be onnx.NoValue (NoneType)
  bool hasBias = op->getNumOperands() > 2 &&
                 !mlir::isa<mlir::NoneType>(op->getOperand(2).getType());
  mlir::Value bias = hasBias ? op->getOperand(2) : nullptr;

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Extract attributes from onnx.Conv
  llvm::SmallVector<int64_t> kernelShape;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("kernel_shape")) {
    for (auto a : attr)
      kernelShape.push_back(mlir::cast<mlir::IntegerAttr>(a).getInt());
  }

  llvm::SmallVector<int64_t> strides;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("strides")) {
    for (auto a : attr)
      strides.push_back(mlir::cast<mlir::IntegerAttr>(a).getInt());
  } else {
    // Default strides = 1 for each spatial dimension
    strides.assign(kernelShape.size(), 1);
  }

  llvm::SmallVector<int64_t> pads;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("pads")) {
    for (auto a : attr)
      pads.push_back(mlir::cast<mlir::IntegerAttr>(a).getInt());
  } else {
    // Default pads = 0
    pads.assign(kernelShape.size() * 2, 0);
  }

  llvm::SmallVector<int64_t> dilations;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("dilations")) {
    for (auto a : attr)
      dilations.push_back(mlir::cast<mlir::IntegerAttr>(a).getInt());
  } else {
    // Default dilations = 1
    dilations.assign(kernelShape.size(), 1);
  }

  int64_t group = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("group"))
    group = attr.getInt();

  // Create output tensor
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank())) {
    if (resultType.isDynamicDim(dimIdx))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, input, dimIdx));
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  // Build attributes for hip.conv
  auto kernelShapeAttr = rewriter.getI64ArrayAttr(kernelShape);
  auto stridesAttr = rewriter.getI64ArrayAttr(strides);
  auto padsAttr = rewriter.getI64ArrayAttr(pads);
  auto dilationsAttr = rewriter.getI64ArrayAttr(dilations);
  auto groupAttr = rewriter.getI64IntegerAttr(group);

  // Build operands vector: context, input, weights, [bias], init
  llvm::SmallVector<mlir::Value> operands = {context, input, weights};
  if (bias)
    operands.push_back(bias);
  operands.push_back(init);

  // Build attributes
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("kernel_shape", kernelShapeAttr));
  attrs.push_back(rewriter.getNamedAttr("strides", stridesAttr));
  attrs.push_back(rewriter.getNamedAttr("pads", padsAttr));
  attrs.push_back(rewriter.getNamedAttr("dilations", dilationsAttr));
  attrs.push_back(rewriter.getNamedAttr("group", groupAttr));

  // Create hip.conv operation using generic builder
  auto hipOp = mlir::hip::ConvOp::create(
      rewriter, loc, mlir::TypeRange{resultType}, operands, attrs);

  rewriter.replaceOp(op, hipOp.getResult(0));
  return mlir::success();
}

/// onnx.Custom(SimplifiedLayerNormalization) -> hip.rms_norm
struct SimplifiedLayerNormToHip : public mlir::RewritePattern {
  SimplifiedLayerNormToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult SimplifiedLayerNormToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  // Check if this is SimplifiedLayerNormalization
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr ||
      funcNameAttr.getValue() != "SimplifiedLayerNormalization")
    return rewriter.notifyMatchFailure(
        op, "not a SimplifiedLayerNormalization operation");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // Check operands (should be 2: input and scale)
  if (op->getNumOperands() != 2)
    return rewriter.notifyMatchFailure(
        op, "expected 2 operands for SimplifiedLayerNormalization");

  mlir::Value input = op->getOperand(0);
  mlir::Value scale = op->getOperand(1);

  // Extract attributes
  auto epsilonAttr = op->getAttrOfType<mlir::FloatAttr>("epsilon");
  if (!epsilonAttr)
    return rewriter.notifyMatchFailure(op, "missing epsilon attribute");

  auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
  if (!axisAttr)
    return rewriter.notifyMatchFailure(op, "missing axis attribute");

  auto stashTypeAttr = op->getAttrOfType<mlir::IntegerAttr>("stash_type");
  if (!stashTypeAttr)
    return rewriter.notifyMatchFailure(op, "missing stash_type attribute");

  // Convert axis to i64
  auto axisI64Attr = rewriter.getI64IntegerAttr(axisAttr.getSInt());
  auto stashTypeI64Attr = rewriter.getI64IntegerAttr(stashTypeAttr.getSInt());

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Create init tensor
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // Create hip.rms_norm operation
  auto hipOp = mlir::hip::RmsNormOp::create(rewriter, loc, resultType, context,
                                            input, scale, init, axisI64Attr,
                                            epsilonAttr, stashTypeI64Attr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Custom(SkipSimplifiedLayerNormalization) -> hip.skip_rms_norm
struct SkipSimplifiedLayerNormToHip : public mlir::RewritePattern {
  SkipSimplifiedLayerNormToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult SkipSimplifiedLayerNormToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  // Check if this is SkipSimplifiedLayerNormalization
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr ||
      funcNameAttr.getValue() != "SkipSimplifiedLayerNormalization")
    return rewriter.notifyMatchFailure(
        op, "not a SkipSimplifiedLayerNormalization operation");

  // Check domain is "com.microsoft"
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op,
        "domain must be com.microsoft for SkipSimplifiedLayerNormalization");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // MS spec: 3-4 inputs (input, skip, gamma, [bias])
  size_t numOps = op->getNumOperands();
  if (numOps < 3 || numOps > 4)
    return rewriter.notifyMatchFailure(
        op, "SkipSimplifiedLayerNormalization expects 3-4 operands");

  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps)
      return nullptr;
    mlir::Value val = op->getOperand(idx);
    if (mlir::isa<mlir::NoneType>(val.getType()))
      return nullptr;
    return val;
  };

  // Input 1-3: required
  mlir::Value input = op->getOperand(0);
  mlir::Value skip = op->getOperand(1);
  mlir::Value gamma = op->getOperand(2);

  // Input 4: bias (optional)
  mlir::Value bias = getOptionalOperand(3);

  // Extract epsilon attribute
  auto epsilonAttr = op->getAttrOfType<mlir::FloatAttr>("epsilon");
  if (!epsilonAttr)
    return rewriter.notifyMatchFailure(op, "missing epsilon attribute");

  // MS spec outputs (1-4): output, [mean], [inv_std_var], [input_skip_bias_sum]
  // mean and inv_std_var are training-only; not modeled in HIP op.
  unsigned numResults = op->getNumResults();

  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, input);

  // Find input_skip_bias_sum: it's the last non-None result (index 1 or 3)
  bool hasSkipOutput = numResults >= 2;
  unsigned skipOutIdx = hasSkipOutput ? numResults - 1 : 0;

  // Check if the last result is actually a real tensor (not None)
  bool skipOutputIsReal = false;
  mlir::RankedTensorType skipOutputType;
  if (hasSkipOutput) {
    mlir::Type lastType = op->getResult(skipOutIdx).getType();
    if (!mlir::isa<mlir::NoneType>(lastType)) {
      skipOutputIsReal = true;
      skipOutputType = mlir::cast<mlir::RankedTensorType>(lastType);
    }
  }

  mlir::Value skipOutputInit = nullptr;
  if (skipOutputIsReal)
    skipOutputInit = createEmptyTensor(rewriter, loc, skipOutputType, input);

  // Build operands list for hip.skip_rms_norm
  mlir::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(input);
  operands.push_back(skip);
  operands.push_back(gamma);
  if (bias)
    operands.push_back(bias);
  operands.push_back(outputInit);
  if (skipOutputInit)
    operands.push_back(skipOutputInit);

  // Build result types
  mlir::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(outputType);
  if (skipOutputIsReal)
    resultTypes.push_back(skipOutputType);

  // Build attributes
  mlir::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("epsilon", epsilonAttr));

  // operand_segment_sizes for AttrSizedOperandSegments
  // [ctx(1), input(1), skip(1), gamma(1), bias(0|1),
  //  output(1), input_skip_bias_sum(0|1)]
  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(1); // ctx
  segmentSizes.push_back(1); // input
  segmentSizes.push_back(1); // skip
  segmentSizes.push_back(1); // gamma
  segmentSizes.push_back(bias ? 1 : 0);
  segmentSizes.push_back(1); // output
  segmentSizes.push_back(skipOutputInit ? 1 : 0);

  auto state = mlir::OperationState(loc, "hip.skip_rms_norm");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addTypes(resultTypes);
  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));

  auto hipOp = rewriter.create(state);

  // Map HIP results back to ONNX results
  llvm::SmallVector<mlir::Value> replacements;
  replacements.push_back(hipOp->getResult(0)); // output

  if (hasSkipOutput) {
    // Fill intermediate None results (mean, inv_std_var) with empty tensors
    for (unsigned i = 1; i < skipOutIdx; ++i) {
      mlir::Type origType = op->getResult(i).getType();
      if (mlir::isa<mlir::NoneType>(origType)) {
        replacements.push_back(mlir::Value{});
        continue;
      }
      auto dummyType = mlir::cast<mlir::RankedTensorType>(origType);
      replacements.push_back(mlir::tensor::EmptyOp::create(
          rewriter, loc, dummyType.getShape(), dummyType.getElementType()));
    }
    if (skipOutputIsReal)
      replacements.push_back(hipOp->getResult(1)); // input_skip_bias_sum
    else {
      auto dummyType = mlir::RankedTensorType::get({}, rewriter.getF32Type());
      replacements.push_back(mlir::tensor::EmptyOp::create(
          rewriter, loc, dummyType.getShape(), dummyType.getElementType()));
    }
  }

  rewriter.replaceOp(op, replacements);
  return mlir::success();
}

/// onnx.Custom(RotaryEmbedding) -> hip.rope
struct RotaryEmbeddingToHip : public mlir::RewritePattern {
  RotaryEmbeddingToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
RotaryEmbeddingToHip::matchAndRewrite(mlir::Operation *op,
                                      mlir::PatternRewriter &rewriter) const {
  // Check if this is RotaryEmbedding
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "RotaryEmbedding")
    return rewriter.notifyMatchFailure(op, "not a RotaryEmbedding operation");

  // Check domain is "com.microsoft"
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for RotaryEmbedding");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // Check operands (should be 4: input, position_ids, cos_cache, sin_cache)
  if (op->getNumOperands() != 4)
    return rewriter.notifyMatchFailure(
        op, "expected 4 operands for RotaryEmbedding");

  mlir::Value input = op->getOperand(0);
  mlir::Value positionIds = op->getOperand(1);
  mlir::Value cosCache = op->getOperand(2);
  mlir::Value sinCache = op->getOperand(3);

  // Extract attributes
  auto interleavedAttr = op->getAttrOfType<mlir::IntegerAttr>("interleaved");
  if (!interleavedAttr)
    return rewriter.notifyMatchFailure(op, "missing interleaved attribute");

  auto numHeadsAttr = op->getAttrOfType<mlir::IntegerAttr>("num_heads");
  if (!numHeadsAttr)
    return rewriter.notifyMatchFailure(op, "missing num_heads attribute");

  auto rotaryDimAttr =
      op->getAttrOfType<mlir::IntegerAttr>("rotary_embedding_dim");
  if (!rotaryDimAttr)
    return rewriter.notifyMatchFailure(
        op, "missing rotary_embedding_dim attribute");

  int64_t numHeadsVal = numHeadsAttr.getSInt();
  int64_t rotaryDimVal = rotaryDimAttr.getSInt();

  // ONNX com.microsoft.RotaryEmbedding: 0 means "infer from tensor shapes"
  //   cos_cache: [max_seq, rotary_dim/2] → rotary_dim = last_dim * 2
  //   input:     [batch, seq, hidden]     → num_heads = hidden / rotary_dim
  if (rotaryDimVal == 0) {
    auto cosCacheType =
        mlir::dyn_cast<mlir::RankedTensorType>(cosCache.getType());
    if (cosCacheType && cosCacheType.hasStaticShape() &&
        cosCacheType.getRank() >= 2) {
      rotaryDimVal = cosCacheType.getShape().back() * 2;
    } else {
      return rewriter.notifyMatchFailure(
          op, "Cannot infer rotary_embedding_dim: "
              "cos_cache must have static shape with rank >= 2");
    }
  }

  if (numHeadsVal == 0 && rotaryDimVal > 0) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (inputType && inputType.hasStaticShape() && inputType.getRank() >= 1) {
      int64_t hidden = inputType.getShape().back();
      numHeadsVal = hidden / rotaryDimVal;
    } else {
      return rewriter.notifyMatchFailure(op, "Cannot infer num_heads: "
                                             "input must have static shape");
    }
  }

  // Convert to i64 attributes (using inferred or original values)
  auto interleavedI64Attr =
      rewriter.getI64IntegerAttr(interleavedAttr.getSInt());
  auto numHeadsI64Attr = rewriter.getI64IntegerAttr(numHeadsVal);
  auto rotaryDimI64Attr = rewriter.getI64IntegerAttr(rotaryDimVal);

  // Should have 1 result: output
  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op,
                                       "expected 1 result for RotaryEmbedding");

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Create init tensor
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // Create hip.rope operation
  auto hipOp = mlir::hip::RopeOp::create(
      rewriter, loc, resultType, context, input, positionIds, cosCache,
      sinCache, init, interleavedI64Attr, numHeadsI64Attr, rotaryDimI64Attr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Custom(GroupQueryAttention) -> hip.gqa
struct GroupQueryAttentionToHip : public mlir::RewritePattern {
  GroupQueryAttentionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult GroupQueryAttentionToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  // Check if this is GroupQueryAttention
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "GroupQueryAttention")
    return rewriter.notifyMatchFailure(op,
                                       "not a GroupQueryAttention operation");

  // Check domain is "com.microsoft"
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for GroupQueryAttention");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // Support variable operand count (7-14 inputs as per MS spec)
  // Minimum 7: query, key, value, past_key, past_value, seqlens_k,
  // total_seq_len Maximum 14: + cos_cache, sin_cache, position_ids,
  // attention_bias, head_sink, k_scale, v_scale
  size_t numOps = op->getNumOperands();
  if (numOps < 7 || numOps > 14)
    return rewriter.notifyMatchFailure(
        op, "GroupQueryAttention expects 7-14 operands");

  // Helper: get optional operand (check for NoneType)
  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps)
      return nullptr; // Operand not provided (trailing optionals omitted)

    mlir::Value val = op->getOperand(idx);

    // Check if it's ONNX NoneType (optional input marked as omitted)
    if (mlir::isa<mlir::NoneType>(val.getType()))
      return nullptr;

    return val; // Valid operand
  };

  // === Extract Inputs (MS GQA spec order 1-14) ===

  // Input 1: query (required)
  mlir::Value query = op->getOperand(0);

  // Inputs 2-3: key/value (optional - packed QKV)
  mlir::Value key = getOptionalOperand(1);
  mlir::Value value = getOptionalOperand(2);

  // Inputs 4-5: past_key/past_value (optional - first inference)
  mlir::Value pastKey = getOptionalOperand(3);
  mlir::Value pastValue = getOptionalOperand(4);

  // Inputs 6-7: seqlens_k, total_seq_len (required)
  if (numOps < 7)
    return rewriter.notifyMatchFailure(op, "missing seqlens_k/total_seq_len");
  mlir::Value seqlensK = op->getOperand(5);
  mlir::Value totalSeqLen = op->getOperand(6);

  // Inputs 8-10: cos_cache, sin_cache, position_ids (optional - RoPE)
  mlir::Value cosCache = getOptionalOperand(7);
  mlir::Value sinCache = getOptionalOperand(8);
  mlir::Value positionIds = getOptionalOperand(9);

  // Input 11: attention_bias (optional - ALiBi etc.)
  mlir::Value attentionBias = getOptionalOperand(10);

  // Input 12: head_sink (optional - smooth softmax)
  mlir::Value headSink = getOptionalOperand(11);

  // Inputs 13-14: k_scale, v_scale (optional - quantization)
  mlir::Value kScale = getOptionalOperand(12);
  mlir::Value vScale = getOptionalOperand(13);

  // === Extract Attributes ===

  // Required attributes - extract value and recreate as signless i64
  auto numHeadsAttrOnnx = op->getAttrOfType<mlir::IntegerAttr>("num_heads");
  if (!numHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing num_heads attribute");
  auto numHeadsAttr =
      rewriter.getI64IntegerAttr(numHeadsAttrOnnx.getValue().getSExtValue());

  auto kvNumHeadsAttrOnnx =
      op->getAttrOfType<mlir::IntegerAttr>("kv_num_heads");
  if (!kvNumHeadsAttrOnnx)
    return rewriter.notifyMatchFailure(op, "missing kv_num_heads attribute");
  auto kvNumHeadsAttr =
      rewriter.getI64IntegerAttr(kvNumHeadsAttrOnnx.getValue().getSExtValue());

  // Optional attributes (with default values)
  auto getFloatAttr = [&](const char *name,
                          float defaultVal) -> mlir::FloatAttr {
    auto attr = op->getAttrOfType<mlir::FloatAttr>(name);
    return attr ? attr : rewriter.getF32FloatAttr(defaultVal);
  };

  auto getI64Attr = [&](const char *name,
                        int64_t defaultVal) -> mlir::IntegerAttr {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    // Convert ONNX signed integer to signless integer for HIP dialect
    return attr ? rewriter.getI64IntegerAttr(attr.getValue().getSExtValue())
                : rewriter.getI64IntegerAttr(defaultVal);
  };

  auto getStrAttr = [&](const char *name,
                        const char *defaultVal) -> mlir::StringAttr {
    auto attr = op->getAttrOfType<mlir::StringAttr>(name);
    return attr ? attr : rewriter.getStringAttr(defaultVal);
  };

  // Calculate default scale = 1/sqrt(head_size) per ONNX spec
  // Query shape: [batch_size, seq_len, num_heads * head_size]
  // head_size = (num_heads * head_size) / num_heads = query_dim_2 / num_heads
  auto queryType = mlir::cast<mlir::RankedTensorType>(query.getType());
  int64_t numHeads = numHeadsAttrOnnx.getValue().getSExtValue();
  float defaultScale = 1.0f;
  if (queryType.hasRank() && queryType.getRank() >= 3) {
    int64_t hiddenSize = queryType.getDimSize(2); // num_heads * head_size
    if (hiddenSize != mlir::ShapedType::kDynamic && numHeads > 0) {
      int64_t headSize = hiddenSize / numHeads;
      defaultScale = 1.0f / std::sqrt(static_cast<float>(headSize));
    }
  }

  auto scaleAttr = getFloatAttr("scale", defaultScale);
  auto doRotaryAttr = getI64Attr("do_rotary", 0);
  auto rotaryInterleavedAttr = getI64Attr("rotary_interleaved", 0);
  auto softcapAttr = getFloatAttr("softcap", 0.0f);
  auto localWindowSizeAttr = getI64Attr("local_window_size", -1);
  auto smoothSoftmaxAttr = getI64Attr("smooth_softmax", 0);
  auto qkOutputAttr = getI64Attr("qk_output", 0);
  auto kQuantTypeAttr = getStrAttr("k_quant_type", "NONE");
  auto vQuantTypeAttr = getStrAttr("v_quant_type", "NONE");
  auto kvCacheBitWidthAttr = getI64Attr("kv_cache_bit_width", 8);

  // === Check Outputs (3 or 4: output, present_key, present_value, [output_qk])
  // ===

  size_t numResults = op->getNumResults();
  if (numResults < 3 || numResults > 4)
    return rewriter.notifyMatchFailure(
        op, "GroupQueryAttention expects 3-4 results");

  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto presentKeyType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());
  auto presentValueType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(2).getType());

  mlir::RankedTensorType outputQkType = nullptr;
  if (numResults == 4)
    outputQkType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(3).getType());

  // === Create DPS init tensors ===

  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, query);
  mlir::Value presentKeyInit = createEmptyTensor(
      rewriter, loc, presentKeyType, pastKey ? pastKey : (key ? key : query));
  mlir::Value presentValueInit =
      createEmptyTensor(rewriter, loc, presentValueType,
                        pastValue ? pastValue : (value ? value : query));

  mlir::Value outputQkInit = nullptr;
  if (outputQkType)
    outputQkInit = createEmptyTensor(rewriter, loc, outputQkType, query);

  // === Create hip.gqa operation ===

  mlir::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(outputType);
  resultTypes.push_back(presentKeyType);
  resultTypes.push_back(presentValueType);
  if (outputQkType)
    resultTypes.push_back(outputQkType);

  // Build operands: context + inputs + outputs
  // Note: Only add non-null operands (optional ones may be nullptr)
  mlir::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(query);
  if (key)
    operands.push_back(key);
  if (value)
    operands.push_back(value);
  if (pastKey)
    operands.push_back(pastKey);
  if (pastValue)
    operands.push_back(pastValue);
  operands.push_back(seqlensK);
  operands.push_back(totalSeqLen);
  if (cosCache)
    operands.push_back(cosCache);
  if (sinCache)
    operands.push_back(sinCache);
  if (positionIds)
    operands.push_back(positionIds);
  if (attentionBias)
    operands.push_back(attentionBias);
  if (headSink)
    operands.push_back(headSink);
  if (kScale)
    operands.push_back(kScale);
  if (vScale)
    operands.push_back(vScale);
  operands.push_back(outputInit);
  operands.push_back(presentKeyInit);
  operands.push_back(presentValueInit);
  if (outputQkInit)
    operands.push_back(outputQkInit);

  // Build named attributes
  mlir::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("num_heads", numHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("kv_num_heads", kvNumHeadsAttr));
  attrs.push_back(rewriter.getNamedAttr("scale", scaleAttr));
  attrs.push_back(rewriter.getNamedAttr("do_rotary", doRotaryAttr));
  attrs.push_back(
      rewriter.getNamedAttr("rotary_interleaved", rotaryInterleavedAttr));
  attrs.push_back(rewriter.getNamedAttr("softcap", softcapAttr));
  attrs.push_back(
      rewriter.getNamedAttr("local_window_size", localWindowSizeAttr));
  attrs.push_back(rewriter.getNamedAttr("smooth_softmax", smoothSoftmaxAttr));
  attrs.push_back(rewriter.getNamedAttr("qk_output", qkOutputAttr));
  attrs.push_back(rewriter.getNamedAttr("k_quant_type", kQuantTypeAttr));
  attrs.push_back(rewriter.getNamedAttr("v_quant_type", vQuantTypeAttr));
  attrs.push_back(
      rewriter.getNamedAttr("kv_cache_bit_width", kvCacheBitWidthAttr));

  // Create operation using builder
  // We need to compute the operand_segment_sizes attribute for
  // AttrSizedOperandSegments
  auto state = mlir::OperationState(loc, "hip.gqa");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addTypes(resultTypes);

  // Add operand_segment_sizes for AttrSizedOperandSegments trait
  // Segments: [ctx(1), query(1), key(0|1), value(0|1), past_key(0|1),
  // past_value(0|1),
  //            seqlens_k(1), total_seq_len(1), cos_cache(0|1), sin_cache(0|1),
  //            position_ids(0|1), attention_bias(0|1), head_sink(0|1),
  //            k_scale(0|1), v_scale(0|1), output(1), present_key(1),
  //            present_value(1), output_qk(0|1)]
  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(1); // ctx
  segmentSizes.push_back(1); // query
  segmentSizes.push_back(key ? 1 : 0);
  segmentSizes.push_back(value ? 1 : 0);
  segmentSizes.push_back(pastKey ? 1 : 0);
  segmentSizes.push_back(pastValue ? 1 : 0);
  segmentSizes.push_back(1); // seqlens_k
  segmentSizes.push_back(1); // total_seq_len
  segmentSizes.push_back(cosCache ? 1 : 0);
  segmentSizes.push_back(sinCache ? 1 : 0);
  segmentSizes.push_back(positionIds ? 1 : 0);
  segmentSizes.push_back(attentionBias ? 1 : 0);
  segmentSizes.push_back(headSink ? 1 : 0);
  segmentSizes.push_back(kScale ? 1 : 0);
  segmentSizes.push_back(vScale ? 1 : 0);
  segmentSizes.push_back(1);                    // output
  segmentSizes.push_back(1);                    // present_key
  segmentSizes.push_back(1);                    // present_value
  segmentSizes.push_back(outputQkInit ? 1 : 0); // output_qk

  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));

  auto hipOp = rewriter.create(state);

  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ONNX Gather → HIP Gather
//===----------------------------------------------------------------------===//

struct GatherToHip : public mlir::RewritePattern {
  GatherToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Gather", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context argument");
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value data = op->getOperand(0);
    mlir::Value indices = op->getOperand(1);

    // Get axis attribute from ONNX Gather operation
    int64_t axis = op->getAttrOfType<mlir::IntegerAttr>("axis").getSInt();
    auto axisAttr = rewriter.getI64IntegerAttr(axis);

    // Get result type
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto dataType = mlir::cast<mlir::RankedTensorType>(data.getType());
    auto indicesType = mlir::cast<mlir::RankedTensorType>(indices.getType());

    // Normalize negative axis for dimension calculations only
    int64_t normalizedAxis = axis < 0 ? axis + dataType.getRank() : axis;

    // Create output tensor with dynamic shape support
    // Output shape: [data[0:axis], indices.shape, data[axis+1:]]
    llvm::SmallVector<mlir::Value> dynSizes;
    int64_t outDimIdx = 0;

    // Copy dimensions before axis from data
    for (auto i : llvm::seq<int64_t>(0, normalizedAxis)) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
      outDimIdx++;
    }
    // Copy all dimensions from indices
    for (auto i : llvm::seq<int64_t>(0, indicesType.getRank())) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, indices, i));
      outDimIdx++;
    }
    // Copy dimensions after axis from data
    for (auto i : llvm::seq<int64_t>(normalizedAxis + 1, dataType.getRank())) {
      if (outDimIdx < resultType.getRank() &&
          resultType.isDynamicDim(outDimIdx))
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, i));
      outDimIdx++;
    }

    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes);

    // Create hip.gather operation
    auto gatherOp = mlir::hip::GatherOp::create(
        rewriter, loc, resultType, context, data, indices, init, axisAttr);

    rewriter.replaceOp(op, gatherOp->getResult(0));
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// convertComputeOps implementation
//===----------------------------------------------------------------------===//

static mlir::LogicalResult convertComputeOps(mlir::func::FuncOp funcOp,
                                             mlir::MLIRContext *ctx) {
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<MatMulToHip>(ctx);
  patterns.add<TransposeToHip>(ctx);
  patterns.add<MulToHip>(ctx);
  patterns.add<SoftmaxToHip>(ctx);
  patterns.add<SigmoidToHip>(ctx);
  patterns.add<SubToHip>(ctx);
  patterns.add<CastToHip>(ctx);
  patterns.add<ReduceSumToHip>(ctx);
  patterns.add<GatherToHip>(ctx);
  patterns.add<ConvToHip>(ctx);
  patterns.add<SimplifiedLayerNormToHip>(ctx);
  patterns.add<SkipSimplifiedLayerNormToHip>(ctx);
  patterns.add<RotaryEmbeddingToHip>(ctx);
  patterns.add<GroupQueryAttentionToHip>(ctx);
  patterns.add<MatMulNBitsToHip>(ctx);
  patterns.add<QMoEToHip>(ctx);

  mlir::GreedyRewriteConfig config;
  config.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);

  if (mlir::failed(
          mlir::applyPatternsGreedily(funcOp, std::move(patterns), config)))
    return mlir::failure();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Module metadata generation
//===----------------------------------------------------------------------===//

/// Generate module metadata attributes required by GenerateInterfacePass.
/// Must be called BEFORE patterns transform function signatures.
static mlir::LogicalResult generateModuleMetadata(mlir::ModuleOp module) {
  auto mainFunc = module.lookupSymbol<mlir::func::FuncOp>("main_graph");
  if (!mainFunc) {
    module.emitError("expected @main_graph function for metadata generation");
    return mlir::failure();
  }

  auto originalFuncType = mainFunc.getFunctionType();
  mlir::OpBuilder builder(module.getContext());

  int64_t inputCount = originalFuncType.getNumInputs();
  llvm::SmallVector<mlir::Attribute> inputShapes;
  llvm::SmallVector<int64_t> inputElementSizes;

  for (mlir::Type inputType : originalFuncType.getInputs()) {
    if (mlir::isa<mlir::hip::ContextType>(inputType)) {
      --inputCount;
      continue;
    }
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(inputType)) {
      auto elemType = tensorType.getElementType();
      if (!elemType.isIntOrFloat()) {
        mainFunc.emitError("unsupported element type in @main_graph input: ")
            << elemType;
        return mlir::failure();
      }
      llvm::SmallVector<int64_t> shape(tensorType.getShape().begin(),
                                       tensorType.getShape().end());
      inputShapes.push_back(builder.getDenseI64ArrayAttr(shape));
      inputElementSizes.push_back(elemType.getIntOrFloatBitWidth() / 8);
    } else {
      mainFunc.emitError("non-tensor input type in @main_graph: ") << inputType;
      return mlir::failure();
    }
  }

  int64_t outputCount = originalFuncType.getNumResults();
  llvm::SmallVector<mlir::Attribute> outputShapes;
  llvm::SmallVector<int64_t> outputElementSizes;

  for (mlir::Type resultType : originalFuncType.getResults()) {
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
      auto elemType = tensorType.getElementType();
      if (!elemType.isIntOrFloat()) {
        mainFunc.emitError("unsupported element type in @main_graph output: ")
            << elemType;
        return mlir::failure();
      }
      llvm::SmallVector<int64_t> shape(tensorType.getShape().begin(),
                                       tensorType.getShape().end());
      outputShapes.push_back(builder.getDenseI64ArrayAttr(shape));
      outputElementSizes.push_back(elemType.getIntOrFloatBitWidth() / 8);
    } else {
      mainFunc.emitError("non-tensor output type in @main_graph: ")
          << resultType;
      return mlir::failure();
    }
  }

  module->setAttr("hipdnn.input_count", builder.getI64IntegerAttr(inputCount));
  module->setAttr("hipdnn.input_shapes", builder.getArrayAttr(inputShapes));
  module->setAttr("hipdnn.input_element_sizes",
                  builder.getDenseI64ArrayAttr(inputElementSizes));
  module->setAttr("hipdnn.output_count",
                  builder.getI64IntegerAttr(outputCount));
  module->setAttr("hipdnn.output_shapes", builder.getArrayAttr(outputShapes));
  module->setAttr("hipdnn.output_element_sizes",
                  builder.getDenseI64ArrayAttr(outputElementSizes));

  LLVM_DEBUG({
    llvm::dbgs() << "[convert-onnx-to-hip] module metadata:"
                 << " input_count=" << inputCount
                 << " input_shapes=" << builder.getArrayAttr(inputShapes)
                 << " input_element_sizes="
                 << builder.getDenseI64ArrayAttr(inputElementSizes)
                 << " output_count=" << outputCount
                 << " output_shapes=" << builder.getArrayAttr(outputShapes)
                 << " output_element_sizes="
                 << builder.getDenseI64ArrayAttr(outputElementSizes) << "\n";
  });

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConvertOnnxToHip Pass
//===----------------------------------------------------------------------===//

struct ConvertOnnxToHipPass
    : public impl::ConvertOnnxToHipPassBase<ConvertOnnxToHipPass> {
  using ConvertOnnxToHipPassBase::ConvertOnnxToHipPassBase;

  ConvertOnnxToHipPass(morphizen::FileSystem *fs, int64_t minNumElements)
      : fileSystem_(fs), fsMinNumElements_(minNumElements) {}

  void runOnOperation() override;

  morphizen::FileSystem *fileSystem_ = nullptr;
  int64_t fsMinNumElements_ = 0;
};

void ConvertOnnxToHipPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = module.getContext();

  // Set up externalization state if enabled.
  // When fileSystem_ is provided (e.g. EPContext from ORT), use it directly.
  // Otherwise fall back to DiskFileSystem rooted at externalizeOutputDir.
  std::unique_ptr<ExternalizationState> extState;
  std::unique_ptr<mlir::hip::DiskFileSystem> fallbackFs;
  morphizen::FileSystem *fs = fileSystem_;

  int64_t minElems =
      fileSystem_ ? fsMinNumElements_ : externalizeMinNumElements.getValue();

  if (!fs) {
    llvm::StringRef dirRef = externalizeOutputDir.getValue();
    std::string dir = dirRef.empty() ? "." : dirRef.str();
    fallbackFs = std::make_unique<mlir::hip::DiskFileSystem>(dir.c_str());
    fs = fallbackFs.get();
  }

  if (minElems > 0) {
    extState = std::make_unique<ExternalizationState>();

    std::string baseName = "model";
    if (auto sym = module->getAttrOfType<mlir::StringAttr>(
            mlir::SymbolTable::getSymbolAttrName()))
      baseName = sym.getValue().str();
    extState->binFileName = baseName + ".constants.bin";

    extState->writer =
        fs->create_writer_template(extState->binFileName.c_str());
    if (!extState->writer) {
      module.emitError("failed to open constants binary file via FileSystem: " +
                       extState->binFileName);
      return signalPassFailure();
    }
  }

  // Capture original function signatures as module metadata before lowering.
  if (mlir::failed(generateModuleMetadata(module)))
    return signalPassFailure();

  for (auto funcOp :
       llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
    if (funcOp.isDeclaration())
      continue;
    if (mlir::failed(
            lowerOnnxConstants(module, funcOp, minElems, extState.get())))
      return signalPassFailure();
    lowerOnnxReturns(funcOp);
    if (mlir::failed(convertComputeOps(funcOp, ctx)))
      return signalPassFailure();
  }

  // Clean up onnx.NoValue and onnx.EntryPoint ops
  llvm::SmallVector<mlir::Operation *> toErase;
  module.walk([&](mlir::Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "onnx.NoValue" && op->use_empty())
      toErase.push_back(op);
    else if (name == "onnx.EntryPoint")
      toErase.push_back(op);
  });
  for (auto *op : toErase)
    op->erase();

  // ONNX-MLIR attaches per-result attributes (e.g. "onnx_node_name") to
  // func.func results. The downstream buffer-results-to-out-params pass
  // skips any result that still carries attributes, leaving the function
  // signature unconverted and causing later lowering failures. Clear all
  // result attributes so every result is eligible for out-param conversion.
  module.walk([&](mlir::func::FuncOp funcOp) {
    unsigned numResults = funcOp.getNumResults();
    if (numResults > 0) {
      llvm::SmallVector<mlir::DictionaryAttr> emptyResAttrs(
          numResults, mlir::DictionaryAttr::get(ctx));
      funcOp.setAllResultAttrs(emptyResAttrs);
    }
  });

  // Finalize externalization: release writer, write JSON manifest, set module
  // attributes.
  if (extState && extState->constantIndex > 0) {
    extState->writer.reset();

    // Set hip.constants_file on the module so downstream passes/tools know
    // where the sidecar lives.
    module->setAttr("hip.constants_file",
                    mlir::StringAttr::get(ctx, extState->binFileName));

    // Emit hipdnn.constant_sizes and hipdnn.constant_offsets for the runtime.
    module->setAttr("hipdnn.constant_sizes",
                    mlir::DenseI64ArrayAttr::get(ctx, extState->constantSizes));
    module->setAttr(
        "hipdnn.constant_offsets",
        mlir::DenseI64ArrayAttr::get(ctx, extState->constantOffsets));

    // Derive base name again for JSON path.
    std::string baseName = "model";
    if (auto sym = module->getAttrOfType<mlir::StringAttr>(
            mlir::SymbolTable::getSymbolAttrName()))
      baseName = sym.getValue().str();
    std::string jsonPath = baseName + ".constants.json";

    llvm::json::Object manifest;
    manifest["version"] = 1;
    manifest["binary_file"] = extState->binFileName;
    manifest["num_constants"] = extState->constantIndex;
    manifest["total_bytes"] = extState->currentOffset;
    manifest["constants"] = std::move(extState->manifestEntries);

    auto jsonWriter = fs->create_writer_template(jsonPath.c_str());
    if (!jsonWriter) {
      module.emitError("failed to open constants manifest via FileSystem: " +
                       jsonPath);
      return signalPassFailure();
    }
    std::string jsonStr;
    llvm::raw_string_ostream jsonOs(jsonStr);
    jsonOs << llvm::formatv("{0:2}", llvm::json::Value(std::move(manifest)));
    jsonWriter->fwrite(jsonStr.data(), jsonStr.size());

    LLVM_DEBUG(llvm::dbgs() << "externalized " << extState->constantIndex
                            << " constants (" << extState->currentOffset
                            << " bytes) to " << extState->binFileName << "\n");
  } else if (extState) {
    extState->writer.reset();
  }
}

} // namespace

std::unique_ptr<mlir::Pass>
createConvertOnnxToHipPass(morphizen::FileSystem *fs, int64_t minNumElements) {
  return std::make_unique<ConvertOnnxToHipPass>(fs, minNumElements);
}

} // namespace hip
} // namespace mlir
