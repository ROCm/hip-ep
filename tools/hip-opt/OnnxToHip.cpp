/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxToHip.cpp - Convert ONNX dialect to HIP dialect (tensor DPS) ---===//
//
// Converts ONNX dialect IR (from onnx-mlir) into HIP dialect IR using
// destination-passing style (DPS) with tensor types. Bufferization to memref
// is handled by a separate downstream pass.
//
//===----------------------------------------------------------------------===//

#include "HipDialect.h"
#include "HipPasses.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// onnx-mlir's libOMMlirDialects references globals from libOMCompilerOptions.
// We deliberately exclude libOMCompilerOptions because it registers a
// conflicting --allow-unregistered-dialect cl::opt. Stubs here satisfy the
// linker without pulling in that library.
namespace onnx_mlir {
bool disableMemRefPrefetch = false;
int64_t getZArchNum(const std::string& /*arch*/, const std::string /*cpu*/) {
  return 0;
}
}  // namespace onnx_mlir

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_CONVERTONNXTOHIPPASS
#include "HipPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Create a tensor.empty for a DPS init operand.  Dynamic dimension sizes
/// are extracted from \p source using tensor.dim at each dynamic index.
/// Suitable for ops where the output shape aligns positionally with one input
/// (e.g., softmax, element-wise).
static mlir::Value createEmptyTensor(mlir::OpBuilder& builder, mlir::Location loc,
                                     mlir::RankedTensorType resultType,
                                     mlir::Value source) {
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i = 0; i < resultType.getRank(); ++i) {
    if (resultType.isDynamicDim(i))
      dynSizes.push_back(mlir::tensor::DimOp::create(builder, loc, source, i));
  }
  return mlir::tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes);
}

/// Extract onnx.Constant ops into new function arguments.
static void extractWeights(mlir::func::FuncOp funcOp) {
  llvm::SmallVector<ONNXConstantOp> constants;
  funcOp.walk([&](ONNXConstantOp op) { constants.push_back(op); });

  for (ONNXConstantOp constOp : constants) {
    mlir::Type resultType = constOp.getResult().getType();
    unsigned argIdx = funcOp.getNumArguments();
    (void)funcOp.insertArgument(argIdx, resultType, mlir::DictionaryAttr(),
                                constOp.getLoc());
    mlir::BlockArgument newArg = funcOp.getArgument(argIdx);
    constOp.getResult().replaceAllUsesWith(newArg);
    constOp.erase();
  }
}

/// Insert hip.create_handle at function entry and hip.destroy_handle before
/// each return. Returns the handle value.
static mlir::Value insertHandleLifecycle(mlir::func::FuncOp funcOp,
                                         mlir::MLIRContext* ctx) {
  mlir::Block& entryBlock = funcOp.getBody().front();
  mlir::OpBuilder entryBuilder(ctx);
  entryBuilder.setInsertionPointToStart(&entryBlock);
  mlir::Value handle = mlir::hip::CreateHandleOp::create(
      entryBuilder, funcOp.getLoc(), mlir::hip::HandleType::get(ctx));

  funcOp.walk([&](mlir::func::ReturnOp retOp) {
    mlir::OpBuilder retBuilder(retOp);
    mlir::hip::DestroyHandleOp::create(retBuilder, retOp.getLoc(), handle);
  });

  return handle;
}

static mlir::LogicalResult convertComputeOps(mlir::func::FuncOp funcOp,
                                             mlir::MLIRContext* ctx,
                                             mlir::Value handle);

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

/// ONNXMatMulOp -> hip.hipblaslt.matmul
struct MatMulToHip : public mlir::OpRewritePattern<ONNXMatMulOp> {
  mlir::Value handle;
  MatMulToHip(mlir::MLIRContext* ctx, mlir::Value handle)
      : OpRewritePattern(ctx), handle(handle) {}

  mlir::LogicalResult matchAndRewrite(ONNXMatMulOp op,
                                      mlir::PatternRewriter& rewriter) const override;
};

mlir::LogicalResult MatMulToHip::matchAndRewrite(ONNXMatMulOp op,
                                                 mlir::PatternRewriter& rewriter) const {
  mlir::Location loc = op.getLoc();
  auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // MatMul: result[..., M, N] = A[..., M, K] @ B[..., K, N].
  // Batch and M dims come from A; N comes from B's last dim.
  llvm::SmallVector<mlir::Value> dynSizes;
  int64_t rank = resultType.getRank();
  auto bType = mlir::cast<mlir::RankedTensorType>(op.getB().getType());
  for (int64_t i = 0; i < rank; ++i) {
    if (!resultType.isDynamicDim(i))
      continue;
    if (i == rank - 1) {
      dynSizes.push_back(mlir::tensor::DimOp::create(
          rewriter, loc, op.getB(), bType.getRank() - 1));
    } else {
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, op.getA(), i));
    }
  }

  mlir::Value init = mlir::tensor::EmptyOp::create(
      rewriter, loc, resultType.getShape(), resultType.getElementType(),
      dynSizes);
  auto hipOp = mlir::hip::HipblasltMatmulOp::create(
      rewriter, loc, resultType, handle, op.getA(), op.getB(), init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// ONNXTransposeOp -> hip.transpose
struct TransposeToHip : public mlir::OpRewritePattern<ONNXTransposeOp> {
  mlir::Value handle;
  TransposeToHip(mlir::MLIRContext* ctx, mlir::Value handle)
      : OpRewritePattern(ctx), handle(handle) {}

  mlir::LogicalResult matchAndRewrite(ONNXTransposeOp op,
                                      mlir::PatternRewriter& rewriter) const override;
};

mlir::LogicalResult TransposeToHip::matchAndRewrite(
    ONNXTransposeOp op, mlir::PatternRewriter& rewriter) const {
  mlir::Location loc = op.getLoc();

  auto permAttr = op.getPerm();
  if (!permAttr)
    return op.emitOpError("hip.transpose requires explicit perm attribute");
  auto perm = *permAttr;

  int64_t dim0 = -1, dim1 = -1;
  for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
    int64_t p = mlir::cast<mlir::IntegerAttr>(perm[i]).getInt();
    if (p != i) {
      if (dim0 < 0)
        dim0 = i;
      else
        dim1 = i;
    }
  }
  if (dim0 < 0 || dim1 < 0)
    return op.emitOpError("perm must swap exactly two dimensions");

  auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Transpose: output dim i corresponds to input dim perm[i].
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i = 0; i < resultType.getRank(); ++i) {
    if (resultType.isDynamicDim(i)) {
      int64_t srcDim = mlir::cast<mlir::IntegerAttr>(perm[i]).getInt();
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, op.getData(), srcDim));
    }
  }

  mlir::Value init = mlir::tensor::EmptyOp::create(
      rewriter, loc, resultType.getShape(), resultType.getElementType(),
      dynSizes);

  mlir::Value d0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, dim0);
  mlir::Value d1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, dim1);

  auto hipOp = mlir::hip::TransposeOp::create(
      rewriter, loc, resultType, handle, d0, d1, op.getData(), init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// ONNXMulOp -> hip.miopen.mul
struct MulToHip : public mlir::OpRewritePattern<ONNXMulOp> {
  mlir::Value handle;
  MulToHip(mlir::MLIRContext* ctx, mlir::Value handle)
      : OpRewritePattern(ctx), handle(handle) {}

  mlir::LogicalResult matchAndRewrite(ONNXMulOp op,
                                      mlir::PatternRewriter& rewriter) const override;
};

mlir::LogicalResult MulToHip::matchAndRewrite(ONNXMulOp op,
                                              mlir::PatternRewriter& rewriter) const {
  mlir::Location loc = op.getLoc();
  auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Use the operand whose rank matches the result for dim extraction
  // (handles scalar * tensor broadcasting).
  auto aType = mlir::cast<mlir::RankedTensorType>(op.getA().getType());
  mlir::Value source =
      (aType.getRank() == resultType.getRank()) ? op.getA() : op.getB();
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, source);

  auto hipOp = mlir::hip::MiopenMulOp::create(
      rewriter, loc, resultType, handle, op.getA(), op.getB(), init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// ONNXSoftmaxOp -> hip.miopen.softmax
struct SoftmaxToHip : public mlir::OpRewritePattern<ONNXSoftmaxOp> {
  mlir::Value handle;
  SoftmaxToHip(mlir::MLIRContext* ctx, mlir::Value handle)
      : OpRewritePattern(ctx), handle(handle) {}

  mlir::LogicalResult matchAndRewrite(ONNXSoftmaxOp op,
                                      mlir::PatternRewriter& rewriter) const override;
};

mlir::LogicalResult SoftmaxToHip::matchAndRewrite(
    ONNXSoftmaxOp op, mlir::PatternRewriter& rewriter) const {
  mlir::Location loc = op.getLoc();
  auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, op.getInput());
  auto hipOp = mlir::hip::MiopenSoftmaxOp::create(
      rewriter, loc, resultType, handle, op.getInput(), init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// convertComputeOps implementation
//===----------------------------------------------------------------------===//

static mlir::LogicalResult convertComputeOps(mlir::func::FuncOp funcOp,
                                             mlir::MLIRContext* ctx,
                                             mlir::Value handle) {
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<MatMulToHip>(ctx, handle);
  patterns.add<TransposeToHip>(ctx, handle);
  patterns.add<MulToHip>(ctx, handle);
  patterns.add<SoftmaxToHip>(ctx, handle);

  mlir::GreedyRewriteConfig config;
  config.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);

  if (mlir::failed(mlir::applyPatternsGreedily(funcOp, std::move(patterns), config)))
    return mlir::failure();

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ConvertOnnxToHip Pass
//===----------------------------------------------------------------------===//

struct ConvertOnnxToHipPass
    : public impl::ConvertOnnxToHipPassBase<ConvertOnnxToHipPass> {
  void runOnOperation() override;
};

void ConvertOnnxToHipPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  mlir::MLIRContext* ctx = module.getContext();

  for (auto funcOp : llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
    extractWeights(funcOp);
    mlir::Value handle = insertHandleLifecycle(funcOp, ctx);
    if (mlir::failed(convertComputeOps(funcOp, ctx, handle)))
      return signalPassFailure();
  }

  for (auto ep : llvm::make_early_inc_range(module.getOps<ONNXEntryPointOp>()))
    ep.erase();
}

}  // namespace

}  // namespace hip
}  // namespace mlir
