/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHipsr.cpp - Convert ONNX dialect to the hipsr dialect --------===//
//
// Converts ONNX dialect IR into tensor-form hipsr dialect IR. The conversion
// target keeps helper computations inside hipsr.compute while allowing scalar
// constants as graph roots. After conversion, placeholder inputs are moved onto
// the shape graph.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipsrUtils.h"

#include "hip/Conversion/OnnxToHipsr/OnnxToHipsr.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Onnx/IR/OnnxOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

#ifdef ONNX_TO_HIPSR_PDL_FILE
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Parser/Parser.h"
#endif

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_CONVERTONNXTOHIPSRPASS
#include "hip/Conversion/Passes.h.inc"

namespace {

// Returns the shape-graph value for a placeholder input. A data result becomes
// the outs operand its producer writes into, which has the same shape.
Value resolveShapeGraphInput(Value input) {
  if (PlaceholderOp::isAllowedShapeGraphInput(input)) {
    return input;
  }
  // Block arguments are allowed, so anything left is a result.
  auto result = cast<OpResult>(input);
  OperandRange destinations = getHipsrDestinationOperands(result.getOwner());
  if (result.getResultNumber() >= destinations.size()) {
    return input;
  }
  return destinations[result.getResultNumber()];
}

// onnx.NoValue stands in for an omitted optional operand, so it only becomes
// dead once the conversion of its consumer drops that operand. Sweeping it up
// therefore belongs after the conversion.
void eraseDeadNoValue(ModuleOp module) {
  SmallVector<onnx::NoValueOp> dead;
  module.walk([&](onnx::NoValueOp op) {
    if (op->use_empty()) {
      dead.push_back(op);
    }
  });
  for (onnx::NoValueOp op : dead) {
    op.erase();
  }
}

// Placeholder inputs form the shape graph, not the data graph.
// PlaceholderOp::verify reports any input this cannot fix.
void rewirePlaceholderInputs(ModuleOp module) {
  module.walk([](PlaceholderOp placeholder) {
    SmallVector<Value> resolvedInputs =
        llvm::map_to_vector(placeholder.getInputs(), resolveShapeGraphInput);
    placeholder.getInputsMutable().assign(resolvedInputs);
  });
}

struct ConvertOnnxToHipsrPass
    : impl::ConvertOnnxToHipsrPassBase<ConvertOnnxToHipsrPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    TypeConverter converter;
    converter.addConversion([](Type type) { return type; });
    converter.addConversion([](RankedTensorType type) -> Type {
      // Leave rank-0 scalars (compile-time host roots lowered to
      // arith.constant) and tensors that already name a space (e.g. host
      // shapes) untouched.
      if (type.getRank() == 0 || type.getEncoding()) {
        return type;
      }
      return tensorTypeInSpace(type, MemorySpace::Device);
    });

    ConversionTarget target(getContext());
    target.addIllegalDialect<onnx::OnnxDialect>();
    // The consumer's conversion drops the operand this stands for, so the
    // placeholder has to survive until then.
    target.addLegalOp<onnx::NoValueOp>();
    target.addLegalDialect<HipsrDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<arith::ConstantOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](func::ReturnOp op) { return converter.isLegal(op); });
    // Helper operations are legal in the hipsr region that owns them: a compute
    // body or a placeholder's shape region.
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      return op->getParentOfType<ComputeOp>() != nullptr ||
             op->getParentOfType<PlaceholderOp>() != nullptr;
    });

    RewritePatternSet patterns(&getContext());

#ifdef ONNX_TO_HIPSR_PDL_FILE
    // Register native constraint for PDLL patterns
    patterns.getPDLPatterns().registerConstraintFunction(
        "GetHipsrContext",
        [](PatternRewriter &rewriter, Operation *op) -> FailureOr<Value> {
          return getHipsrContextArg(op, rewriter);
        });

    // Register native rewrites for PDLL patterns
    // These handle the actual op creation since PDLL can't easily create enum attributes
    patterns.getPDLPatterns().registerRewriteFunction(
        "RewriteCast",
        [](PatternRewriter &rewriter, Operation *op) -> Operation * {
          if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return nullptr;
          FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
          if (failed(ctx))
            return nullptr;

          Location loc = op->getLoc();
          Value input = op->getOperand(0);
          auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
          if (!resultType)
            return nullptr;

          Value init = rewriter.create<PlaceholderOp>(
              loc, TypeRange{resultType}, *ctx, ValueRange{input},
              PlaceholderType::Normal).getResult(0);

          return rewriter.create<CastOp>(loc, TypeRange{resultType}, *ctx, input, init);
        });

    patterns.getPDLPatterns().registerRewriteFunction(
        "RewriteMatMul",
        [](PatternRewriter &rewriter, Operation *op) -> Operation * {
          if (op->getNumOperands() != 2 || op->getNumResults() != 1)
            return nullptr;
          FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
          if (failed(ctx))
            return nullptr;

          Location loc = op->getLoc();
          Value lhs = op->getOperand(0);
          Value rhs = op->getOperand(1);
          auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
          if (!resultType)
            return nullptr;

          Value init = PlaceholderOp::create(
              rewriter, loc, TypeRange{resultType}, *ctx, ValueRange{lhs, rhs},
              PlaceholderType::Normal).getResult(0);

          return MatMulOp::create(rewriter, loc, TypeRange{resultType}, *ctx, lhs, rhs, init);
        });

    patterns.getPDLPatterns().registerRewriteFunction(
        "RewriteExpand",
        [](PatternRewriter &rewriter, Operation *op) -> Operation * {
          if (op->getNumOperands() != 2 || op->getNumResults() != 1)
            return nullptr;
          FailureOr<Value> ctx = getHipsrContextArg(op, rewriter);
          if (failed(ctx))
            return nullptr;

          Location loc = op->getLoc();
          Value data = op->getOperand(0);
          Value shape = op->getOperand(1);
          auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
          if (!resultType)
            return nullptr;

          Value init = PlaceholderOp::create(
              rewriter, loc, TypeRange{resultType}, *ctx, ValueRange{data, shape},
              PlaceholderType::Normal).getResult(0);

          return ExpandOp::create(rewriter, loc, TypeRange{resultType}, *ctx, data, shape, init);
        });

    // Load PDLL patterns from compiled bytecode
    OwningOpRef<ModuleOp> pdlModule =
        parseSourceFile<ModuleOp>(ONNX_TO_HIPSR_PDL_FILE, &getContext());

    if (pdlModule) {
      patterns.add(PDLPatternModule(std::move(pdlModule)));
    } else {
      llvm::errs() << "Warning: Failed to load PDLL patterns from "
                   << ONNX_TO_HIPSR_PDL_FILE << "\n";
      // Fall back to C++ patterns
      populateCastConversionPatterns(converter, patterns, &getContext());
      populateMatMulConversionPatterns(converter, patterns, &getContext());
      populateExpandConversionPatterns(converter, patterns, &getContext());
    }
#else
    // Use C++ patterns when PDLL not available
    populateCastConversionPatterns(converter, patterns, &getContext());
    populateMatMulConversionPatterns(converter, patterns, &getContext());
    populateExpandConversionPatterns(converter, patterns, &getContext());
#endif

    populateOnnxToHipsrConstantPatterns(converter, patterns);
    populateShapeConversionPatterns(converter, patterns, &getContext());
    populateReturnConversionPatterns(patterns, &getContext());
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    populateReturnOpTypeConversionPattern(patterns, converter);

    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    eraseDeadNoValue(module);
    // Runs after conversion so every producer has its outs operand.
    rewirePlaceholderInputs(module);
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
