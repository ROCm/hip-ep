/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- TorchToHipUtils.h - Shared helpers for Torch-to-HIP patterns ------===//
//
// Shared utility functions and forward declarations used by per-operator
// conversion files for the Torch dialect -> HIP dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_TORCHTOHIP_UTILS_H
#define HIP_CONVERSION_TORCHTOHIP_UTILS_H

#include "hip/Conversion/TorchToHip/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

#define DEBUG_TYPE "convert-torch-to-hip"

namespace mlir {
namespace hip {

//===----------------------------------------------------------------------===//
// Torch-specific helpers
//===----------------------------------------------------------------------===//

/// Extract a constant integer value from a torch.constant.int defining op.
/// Returns std::nullopt if the value is not a constant integer.
inline std::optional<int64_t> getTorchConstantInt(mlir::Value val) {
  auto *defOp = val.getDefiningOp();
  if (!defOp || defOp->getName().getStringRef() != "torch.constant.int")
    return std::nullopt;
  auto valueAttr = defOp->getAttrOfType<mlir::IntegerAttr>("value");
  if (!valueAttr)
    return std::nullopt;
  return valueAttr.getValue().getSExtValue();
}

/// Extract a constant boolean value from a torch.constant.bool defining op.
inline std::optional<bool> getTorchConstantBool(mlir::Value val) {
  auto *defOp = val.getDefiningOp();
  if (!defOp || defOp->getName().getStringRef() != "torch.constant.bool")
    return std::nullopt;
  auto valueAttr = defOp->getAttrOfType<mlir::BoolAttr>("value");
  if (!valueAttr)
    return std::nullopt;
  return valueAttr.getValue();
}

/// Extract a list of constant integers from a torch.prim.ListConstruct op.
/// Returns std::nullopt if any element is not a constant integer.
inline std::optional<llvm::SmallVector<int64_t>>
getTorchConstantIntList(mlir::Value val) {
  auto *defOp = val.getDefiningOp();
  if (!defOp || defOp->getName().getStringRef() != "torch.prim.ListConstruct")
    return std::nullopt;
  llvm::SmallVector<int64_t> result;
  for (mlir::Value elem : defOp->getOperands()) {
    auto intVal = getTorchConstantInt(elem);
    if (!intVal)
      return std::nullopt;
    result.push_back(*intVal);
  }
  return result;
}

/// Check if a value is produced by torch.constant.none (absent optional).
inline bool isTorchNone(mlir::Value val) {
  auto *defOp = val.getDefiningOp();
  return defOp && defOp->getName().getStringRef() == "torch.constant.none";
}

//===----------------------------------------------------------------------===//
// Shared helpers (mirrored from OnnxToHipUtils.h)
//===----------------------------------------------------------------------===//

/// Create a tensor.empty for a DPS init operand. Dynamic dimension sizes
/// are extracted from \p source using tensor.dim at each dynamic index.
inline mlir::Value createEmptyTensorForTorch(mlir::OpBuilder &builder,
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

/// Get !hip.context from function argument 0. Returns failure if the
/// function has no arguments or the first argument is not !hip.context.
inline mlir::FailureOr<mlir::Value>
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

//===----------------------------------------------------------------------===//
// Pattern population functions (one per operator file)
//===----------------------------------------------------------------------===//

void populateTorchMatMulConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext *ctx);
void populateTorchConvConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *ctx);
void populateTorchElementwiseConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);
void populateTorchActivationConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);
void populateTorchCastConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *ctx);
void populateTorchGatherConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext *ctx);
void populateTorchReduceConversionPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext *ctx);
void populateTorchReshapeConversionPatterns(mlir::RewritePatternSet &patterns,
                                            mlir::MLIRContext *ctx);
void populateTorchTransposeConversionPatterns(mlir::RewritePatternSet &patterns,
                                              mlir::MLIRContext *ctx);
void populateTorchNormConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *ctx);
void populateTorchSliceCatConversionPatterns(mlir::RewritePatternSet &patterns,
                                              mlir::MLIRContext *ctx);
void populateTorchGqaConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *ctx);
void populateTorchMatMulNBitsConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx);
void populateTorchQMoEConversionPatterns(mlir::RewritePatternSet &patterns,
                                          mlir::MLIRContext *ctx);
void populateTorchMiscConversionPatterns(mlir::RewritePatternSet &patterns,
                                          mlir::MLIRContext *ctx);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_TORCHTOHIP_UTILS_H
