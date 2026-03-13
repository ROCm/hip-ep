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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
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
  std::unique_ptr<llvm::raw_fd_ostream> binFile;
  llvm::json::Array manifestEntries;
  int64_t currentOffset = 0;
  int64_t constantIndex = 0;
  std::string binFileName;
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
                             !valueAttr.isSplat() &&
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
        extState->binFile->write(zeros.data(), padding);
        extState->currentOffset += padding;
      }
      int64_t entryOffset = extState->currentOffset;

      // Write raw bytes to the sidecar binary.
      auto rawData = valueAttr.getRawData();
      int64_t byteSize = static_cast<int64_t>(rawData.size());
      extState->binFile->write(rawData.data(), byteSize);
      extState->currentOffset += byteSize;

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
  auto hipOp = mlir::hip::HipblasltMatmulOp::create(rewriter, loc, resultType,
                                                    context, a, b, init);
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
        rewriter.create<mlir::arith::ConstantOp>(loc, axesType, axesAttr);
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
  auto hipOp = rewriter.create<mlir::hip::ConvOp>(
      loc, mlir::TypeRange{resultType}, operands, attrs);

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

  // Check operands (should be 3: input, skip, gamma)
  if (op->getNumOperands() != 3)
    return rewriter.notifyMatchFailure(
        op, "expected 3 operands for SkipSimplifiedLayerNormalization");

  mlir::Value input = op->getOperand(0);
  mlir::Value skip = op->getOperand(1);
  mlir::Value gamma = op->getOperand(2);

  // Extract epsilon attribute
  auto epsilonAttr = op->getAttrOfType<mlir::FloatAttr>("epsilon");
  if (!epsilonAttr)
    return rewriter.notifyMatchFailure(op, "missing epsilon attribute");

  // Should have 2 results: output and skip_output
  if (op->getNumResults() != 2)
    return rewriter.notifyMatchFailure(
        op, "expected 2 results for SkipSimplifiedLayerNormalization");

  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto skipOutputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(1).getType());

  // Create init tensors
  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, input);
  mlir::Value skipOutputInit =
      createEmptyTensor(rewriter, loc, skipOutputType, input);

  // Create hip.skip_rms_norm operation
  auto hipOp = mlir::hip::SkipRmsNormOp::create(
      rewriter, loc, {outputType, skipOutputType}, context, input, skip, gamma,
      outputInit, skipOutputInit, epsilonAttr);

  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

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
  patterns.add<ConvToHip>(ctx);
  patterns.add<SimplifiedLayerNormToHip>(ctx);
  patterns.add<SkipSimplifiedLayerNormToHip>(ctx);

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
  void runOnOperation() override;
};

void ConvertOnnxToHipPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext *ctx = module.getContext();

  // Set up externalization state if enabled.
  std::unique_ptr<ExternalizationState> extState;
  llvm::StringRef dirRef = externalizeOutputDir.getValue();
  std::string dir = dirRef.empty() ? "." : dirRef.str();
  if (externalizeMinNumElements > 0) {
    extState = std::make_unique<ExternalizationState>();

    // Derive base name from module symbol or default.
    std::string baseName = "model";
    if (auto sym = module->getAttrOfType<mlir::StringAttr>(
            mlir::SymbolTable::getSymbolAttrName()))
      baseName = sym.getValue().str();
    extState->binFileName = baseName + ".constants.bin";

    std::string binPath = dir + "/" + extState->binFileName;
    std::error_code binEC;
    extState->binFile = std::make_unique<llvm::raw_fd_ostream>(
        binPath, binEC, llvm::sys::fs::OF_None);
    if (binEC) {
      module.emitError("failed to open constants binary file: " + binPath +
                       " (" + binEC.message() + ")");
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
    if (mlir::failed(lowerOnnxConstants(
            module, funcOp, externalizeMinNumElements, extState.get())))
      return signalPassFailure();
    lowerOnnxReturns(funcOp);
    if (mlir::failed(convertComputeOps(funcOp, ctx)))
      return signalPassFailure();
  }

  llvm::SmallVector<mlir::Operation *> entryPoints;
  for (auto &op : module.getBody()->getOperations()) {
    if (op.getName().getStringRef() == "onnx.EntryPoint")
      entryPoints.push_back(&op);
  }
  for (auto *ep : entryPoints)
    ep->erase();

  // Finalize externalization: close binary, write JSON manifest, set module
  // attribute.
  if (extState && extState->constantIndex > 0) {
    extState->binFile->close();

    // Set hip.constants_file on the module so downstream passes/tools know
    // where the sidecar lives.
    module->setAttr("hip.constants_file",
                    mlir::StringAttr::get(ctx, extState->binFileName));

    // Derive base name again for JSON path.
    std::string baseName = "model";
    if (auto sym = module->getAttrOfType<mlir::StringAttr>(
            mlir::SymbolTable::getSymbolAttrName()))
      baseName = sym.getValue().str();
    std::string jsonPath = dir + "/" + baseName + ".constants.json";

    llvm::json::Object manifest;
    manifest["version"] = 1;
    manifest["binary_file"] = extState->binFileName;
    manifest["num_constants"] = extState->constantIndex;
    manifest["total_bytes"] = extState->currentOffset;
    manifest["constants"] = std::move(extState->manifestEntries);

    std::error_code ec;
    llvm::raw_fd_ostream jsonFile(jsonPath, ec);
    if (ec) {
      module.emitError("failed to open constants manifest: " + jsonPath + " (" +
                       ec.message() + ")");
      return signalPassFailure();
    }
    jsonFile << llvm::formatv("{0:2}", llvm::json::Value(std::move(manifest)));
    jsonFile.close();

    module.emitRemark("externalized ")
        << extState->constantIndex << " constants (" << extState->currentOffset
        << " bytes) to " << dir + "/" + extState->binFileName;
  } else if (extState) {
    // No constants qualified -- clean up empty file.
    extState->binFile->close();
  }
}

} // namespace

} // namespace hip
} // namespace mlir
