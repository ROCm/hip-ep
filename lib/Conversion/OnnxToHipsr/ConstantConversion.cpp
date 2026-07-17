/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ConstantConversion.cpp - onnx.Constant -> hipsr.constant ----------===//
//
// Converts `onnx.Constant` into the hipsr constant forms. Matched by name via
// the generic Operation API (no onnx-mlir headers), consistent with the rest
// of convert-onnx-to-hipsr. Four branches, keyed only on the storage form of
// the incoming onnx.Constant -- no size threshold (that policy is layered on
// later, in externalization):
//
//   inline value + rank-0 scalar     -> arith.constant <scalar>
//   inline value + rank>=1           -> hipsr.constant {value}      : tensor<>
//   location == "*/_ORT_MEM_ADDR_/*" -> hipsr.constant {mem_source} : tensor<>
//   location == <other path>         -> hipsr.constant {file_source}: tensor<>
//
// The result stays a ranked tensor (pre-bufferization); the broadened
// hipsr.constant result type (Hipsr_TensorOrDeviceMemRef) means no
// memref->tensor bridge is needed here.
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrConstantOp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hipsr {

namespace {

// The ORT bridge tags in-memory (zero-copy) constants with this location; any
// other location string is an absolute file path. Mirrors kOrtMemAddrTag in
// OnnxToHip.cpp.
constexpr ::llvm::StringLiteral kOrtMemAddrTag = "*/_ORT_MEM_ADDR_/*";

struct ConstantOpLowering : public ::mlir::RewritePattern {
  explicit ConstantOpLowering(::mlir::MLIRContext *ctx)
      : ::mlir::RewritePattern("onnx.Constant", /*benefit=*/1, ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto tensorType =
        ::llvm::dyn_cast<::mlir::RankedTensorType>(op->getResult(0).getType());
    if (!tensorType) {
      return rewriter.notifyMatchFailure(op, "non-ranked result type");
    }

    // Inline dense value.
    if (auto valueAttr = op->getAttrOfType<::mlir::ElementsAttr>("value")) {
      // Rank-0 scalar stays a compile-time arith.constant (keeps the tensor<>
      // result type; the elements attr is TypedAttr-compatible).
      if (tensorType.getRank() == 0) {
        rewriter.replaceOpWithNewOp<::mlir::arith::ConstantOp>(
            op, ::llvm::cast<::mlir::TypedAttr>(valueAttr));
        return ::mlir::success();
      }

      // rank>=1 -> hipsr.constant {value}.
      rewriter.replaceOpWithNewOp<ConstantOp>(
          op, /*result=*/tensorType, /*value=*/valueAttr,
          /*source=*/::mlir::Attribute(), /*offset=*/::mlir::IntegerAttr(),
          /*size=*/::mlir::IntegerAttr(), /*externalize=*/::mlir::BoolAttr());
      return ::mlir::success();
    }

    // External data: location + offset + size.
    auto locAttr = op->getAttrOfType<::mlir::StringAttr>("location");
    if (!locAttr) {
      return rewriter.notifyMatchFailure(
          op, "onnx.Constant has neither value nor location");
    }
    auto offsetAttr = op->getAttrOfType<::mlir::IntegerAttr>("offset");
    auto sizeAttr = op->getAttrOfType<::mlir::IntegerAttr>("size");
    if (!offsetAttr || !sizeAttr) {
      return rewriter.notifyMatchFailure(
          op, "onnx.Constant with location missing offset/size");
    }
    int64_t offset = offsetAttr.getInt();
    int64_t size = sizeAttr.getInt();

    ::mlir::Attribute source;
    if (locAttr.getValue() == kOrtMemAddrTag) {
      source = MemSourceAttr::get(op->getContext(), /*address=*/offset, size);
    } else {
      source = FileSourceAttr::get(op->getContext(), locAttr, offset, size);
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, /*result=*/tensorType, /*value=*/::mlir::ElementsAttr(), source,
        /*offset=*/::mlir::IntegerAttr(), /*size=*/::mlir::IntegerAttr(),
        /*externalize=*/::mlir::BoolAttr());
    return ::mlir::success();
  }
};

} // namespace

void populateOnnxToHipsrConstantPatterns(::mlir::RewritePatternSet &patterns) {
  patterns.add<ConstantOpLowering>(patterns.getContext());
}

} // namespace hipsr
} // namespace mlir
