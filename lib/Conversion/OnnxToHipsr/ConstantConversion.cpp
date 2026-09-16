/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir {
namespace hipsr {

namespace {

// The ORT bridge tags in-memory (zero-copy) constants with this location; any
// other location string is an absolute file path. Mirrors kOrtMemAddrTag in
// OnnxToHip.cpp.
constexpr ::llvm::StringLiteral kOrtMemAddrTag = "*/_ORT_MEM_ADDR_/*";

struct ConstantOpLowering
    : public ::mlir::OpConversionPattern<::mlir::onnx::ConstantOp> {
  ConstantOpLowering(const ::mlir::TypeConverter &typeConverter,
                     ::mlir::MLIRContext *ctx)
      : OpConversionPattern(ctx) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::onnx::ConstantOp op, OpAdaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto tensorType =
        ::llvm::dyn_cast<::mlir::RankedTensorType>(op.getOutput().getType());
    if (!tensorType) {
      return rewriter.notifyMatchFailure(op, "non-ranked result type");
    }

    // Inline dense value.
    if (auto valueAttr = ::llvm::dyn_cast_or_null<::mlir::ElementsAttr>(
            adaptor.getValueAttr())) {
      // Rank-0 scalar stays a compile-time arith.constant (keeps the tensor<>
      // result type; the elements attr is TypedAttr-compatible).
      if (tensorType.getRank() == 0) {
        rewriter.replaceOpWithNewOp<::mlir::arith::ConstantOp>(
            op, ::llvm::cast<::mlir::TypedAttr>(valueAttr));
        return ::mlir::success();
      }

      ::mlir::RankedTensorType resultType =
          tensorTypeInSpace(tensorType, MemorySpace::Device);
      rewriter.replaceOpWithNewOp<ConstantOp>(
          op, /*result=*/resultType, /*value=*/valueAttr,
          /*index=*/IntegerAttr(), /*offset=*/IntegerAttr(),
          /*size=*/IntegerAttr());
      return ::mlir::success();
    }

    // External data: location + offset + size.
    ::mlir::StringAttr locAttr = adaptor.getLocationAttr();
    if (!locAttr) {
      return rewriter.notifyMatchFailure(
          op, "onnx.Constant has neither value nor location");
    }
    ::mlir::IntegerAttr offsetAttr = adaptor.getOffsetAttr();
    ::mlir::IntegerAttr sizeAttr = adaptor.getSizeAttr();
    if (!offsetAttr || !sizeAttr) {
      return rewriter.notifyMatchFailure(
          op, "onnx.Constant with location missing offset/size");
    }
    int64_t offset = offsetAttr.getInt();
    int64_t size = sizeAttr.getInt();

    std::string key;
    ArrayRef<char> data;
    if (locAttr.getValue() == kOrtMemAddrTag) {
      key = ("mem|0x" + llvm::utohexstr(static_cast<uint64_t>(offset),
                                        /*LowerCase=*/true));
      data = {reinterpret_cast<const char *>(static_cast<uintptr_t>(offset)),
              static_cast<size_t>(size)};
    } else {
      auto *dialect = op->getContext()->getLoadedDialect<HipsrDialect>();
      llvm::MemoryBuffer *buf = dialect->getOrLoadFileMap(locAttr.getValue());
      if (!buf) {
        return rewriter.notifyMatchFailure(
            op, "cannot memory-map the external data file");
      }
      key = ("file|" + locAttr.getValue() + "|" + Twine(offset)).str();
      data = {buf->getBufferStart() + offset, static_cast<size_t>(size)};
    }

    ::mlir::RankedTensorType resultType =
        tensorTypeInSpace(tensorType, MemorySpace::Device);
    auto value = DenseResourceElementsAttr::get(
        resultType, key, UnmanagedAsmResourceBlob::allocateInferAlign(data));

    rewriter.replaceOpWithNewOp<ConstantOp>(
        op, /*result=*/resultType, value, /*index=*/IntegerAttr(),
        /*offset=*/IntegerAttr(), /*size=*/IntegerAttr());
    return ::mlir::success();
  }
};

} // namespace

void populateOnnxToHipsrConstantPatterns(
    const ::mlir::TypeConverter &typeConverter,
    ::mlir::RewritePatternSet &patterns) {
  patterns.add<ConstantOpLowering>(typeConverter, patterns.getContext());
}

} // namespace hipsr
} // namespace mlir
