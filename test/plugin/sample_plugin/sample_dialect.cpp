/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Minimal vendor-style dialect for the in-tree sample plugin, contributed via
// addDialectRegistration. It exists only to give the host CI coverage of the
// two plugin-dialect paths otherwise exercised only out-of-tree: the
// loadAllDialects() plugin loop (InitAllPasses.h) and the hasPromisedInterface
// guard in convert-hip-to-llvm (HipToLLVM.cpp). One nullary op whose
// ConvertToLLVMPatternInterface lowering just erases it -- the smallest shape
// touching both paths. A real vendor dialect would define its ops in ODS.

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/DialectConversion.h"

#include "sample_dialect.h"

using namespace mlir;

namespace {

/// Nullary op with no attributes -- enough to load the dialect and be walked by
/// convert-hip-to-llvm.
class SampleMarkerOp : public Op<SampleMarkerOp> {
public:
  using Op::Op;
  static StringRef getOperationName() { return "hip_ep_sample.marker"; }
  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SampleMarkerOp)
};

class SampleDialect : public Dialect {
public:
  explicit SampleDialect(MLIRContext *ctx)
      : Dialect(getDialectNamespace(), ctx, TypeID::get<SampleDialect>()) {
    addOperations<SampleMarkerOp>();
  }
  static StringRef getDialectNamespace() { return "hip_ep_sample"; }
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SampleDialect)
};

/// Stands in for a real op's lowering to a runtime call; only its dispatch
/// through the interface matters here, so it just erases the marker.
struct SampleMarkerLowering : public ConversionPattern {
  SampleMarkerLowering(const LLVMTypeConverter &converter, MLIRContext *ctx)
      : ConversionPattern(converter, SampleMarkerOp::getOperationName(),
                          /*benefit=*/1, ctx) {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> /*operands*/,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

/// Implement the interface to lower hip_ep_sample to LLVM. Attaching it (below)
/// -- not merely promising it -- is what makes the convert-hip-to-llvm guard
/// admit the dialect (hasPromisedInterface == false).
struct SampleConvertToLLVMInterface : public ConvertToLLVMPatternInterface {
  using ConvertToLLVMPatternInterface::ConvertToLLVMPatternInterface;

  void loadDependentDialects(MLIRContext *context) const final {
    context->loadDialect<LLVM::LLVMDialect>();
  }

  void populateConvertToLLVMConversionPatterns(
      ConversionTarget &target, LLVMTypeConverter &typeConverter,
      RewritePatternSet &patterns) const final {
    target.addIllegalDialect<SampleDialect>();
    patterns.add<SampleMarkerLowering>(typeConverter, patterns.getContext());
  }
};

} // namespace

void registerHipEpSampleDialect(mlir::DialectRegistry &registry) {
  registry.insert<SampleDialect>();
  // Attach (not promise) the lowering interface when the dialect loads, so it
  // is genuinely registered -- what the convert-hip-to-llvm guard expects.
  registry.addExtension(+[](mlir::MLIRContext *, SampleDialect *dialect) {
    dialect->addInterfaces<SampleConvertToLLVMInterface>();
  });
}
