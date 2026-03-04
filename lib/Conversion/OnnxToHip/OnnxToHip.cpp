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

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>

// onnx-mlir's libOMMlirDialects references globals from libOMCompilerOptions.
// We deliberately exclude libOMCompilerOptions because it registers a
// conflicting --allow-unregistered-dialect cl::opt. Stubs here satisfy the
// linker without pulling in that library.
namespace onnx_mlir {
bool disableMemRefPrefetch = false;
int64_t getZArchNum(const std::string & /*arch*/, const std::string /*cpu*/) {
  return 0;
}
} // namespace onnx_mlir

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_CONVERTONNXTOHIPPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Create a tensor.empty for a DPS init operand.  Dynamic dimension sizes
/// are extracted from \p source using tensor.dim at each dynamic index.
/// Suitable for ops where the output shape aligns positionally with one input
/// (e.g., softmax, element-wise).
static mlir::Value createEmptyTensor(mlir::OpBuilder &builder,
                                     mlir::Location loc,
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

//===----------------------------------------------------------------------===//
// Constant externalization helpers
//===----------------------------------------------------------------------===//

static std::string elementTypeToString(mlir::Type elemType) {
  if (elemType.isF16())
    return "f16";
  if (elemType.isBF16())
    return "bf16";
  if (elemType.isF32())
    return "f32";
  if (elemType.isF64())
    return "f64";
  if (elemType.isInteger(8))
    return "i8";
  if (elemType.isInteger(16))
    return "i16";
  if (elemType.isInteger(32))
    return "i32";
  if (elemType.isInteger(64))
    return "i64";
  if (elemType.isInteger(1))
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
  std::ofstream binFile;
  llvm::json::Array manifestEntries;
  int64_t currentOffset = 0;
  int64_t constantIndex = 0;
  std::string binFileName; // bare filename (no directory prefix)
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
static mlir::LogicalResult
lowerOnnxConstants(mlir::ModuleOp module, mlir::func::FuncOp funcOp,
                   int64_t minNumElements, ExternalizationState *extState) {
  constexpr int64_t kAlignment = 64;

  llvm::SmallVector<ONNXConstantOp> constants;
  funcOp.walk([&](ONNXConstantOp op) { constants.push_back(op); });

  for (ONNXConstantOp constOp : constants) {
    auto valueAttr =
        mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(constOp.getValueAttr());
    if (!valueAttr) {
      return constOp.emitError(
          "unsupported onnx.Constant form (expected dense value attribute)");
    }

    bool shouldExternalize =
        extState && minNumElements > 0 &&
        !valueAttr.isSplat() &&
        valueAttr.getNumElements() >= minNumElements;

    if (shouldExternalize) {
      auto tensorType = mlir::cast<mlir::RankedTensorType>(valueAttr.getType());
      auto memrefType = mlir::MemRefType::get(tensorType.getShape(),
                                              tensorType.getElementType());

      // Derive a name for the global symbol.
      std::string name;
      if (auto nodeNameAttr =
              constOp->getAttrOfType<mlir::StringAttr>("onnx_node_name"))
        name = "hip_ext_" + nodeNameAttr.getValue().str();
      else
        name = "hip_ext_constant_" +
               std::to_string(extState->constantIndex);

      // Pad binary file to alignment boundary.
      int64_t padding =
          alignTo(extState->currentOffset, kAlignment) - extState->currentOffset;
      if (padding > 0) {
        std::vector<char> zeros(padding, 0);
        extState->binFile.write(zeros.data(), padding);
        extState->currentOffset += padding;
      }
      int64_t entryOffset = extState->currentOffset;

      // Write raw bytes to the sidecar binary.
      auto rawData = valueAttr.getRawData();
      int64_t byteSize = static_cast<int64_t>(rawData.size());
      extState->binFile.write(rawData.data(), byteSize);
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
          moduleBuilder.getNamedAttr(
              "size", moduleBuilder.getI64IntegerAttr(byteSize)),
      });
      mlir::memref::GlobalOp::create(
          moduleBuilder, constOp.getLoc(), name,
          /*sym_visibility=*/moduleBuilder.getStringAttr("private"),
          /*type=*/memrefType,
          /*initial_value=*/nullptr,
          /*constant=*/false,
          /*alignment=*/moduleBuilder.getI64IntegerAttr(kAlignment));
      // Attach hip.external_data as a discardable attr on the global.
      auto globalOp = module.lookupSymbol<mlir::memref::GlobalOp>(name);
      globalOp->setAttr("hip.external_data", externalDataAttr);

      // At the use site: memref.get_global + bufferization.to_tensor.
      mlir::OpBuilder builder(constOp);
      auto getGlobal = mlir::memref::GetGlobalOp::create(
          builder, constOp.getLoc(), memrefType, name);
      auto toTensor = mlir::bufferization::ToTensorOp::create(
          builder, constOp.getLoc(), tensorType, getGlobal.getResult(),
          /*restrict=*/builder.getUnitAttr(),
          /*writable=*/nullptr);
      constOp.getResult().replaceAllUsesWith(toTensor.getResult());
      constOp.erase();

      ++extState->constantIndex;
    } else {
      // Small / splat / externalization disabled: inline arith.constant.
      mlir::OpBuilder builder(constOp);
      auto arithConst = mlir::arith::ConstantOp::create(
          builder, constOp.getLoc(), valueAttr);
      constOp.getResult().replaceAllUsesWith(arithConst.getResult());
      constOp.erase();
    }
  }
  return mlir::success();
}

/// Replace onnx.Return terminators with func.return.
///
/// In onnx-mlir's own pipeline a dedicated StandardFuncReturnPass handles
/// this before lowering.  Since we bypass that pipeline we must do it
/// ourselves.  This MUST run before insertHandleLifecycle(), which walks
/// func::ReturnOp to place hip.destroy_handle -- if only onnx.Return is
/// present the handle destroy would be silently skipped.
static void lowerOnnxReturns(mlir::func::FuncOp funcOp) {
  llvm::SmallVector<ONNXReturnOp> returns;
  funcOp.walk([&](ONNXReturnOp op) { returns.push_back(op); });

  for (ONNXReturnOp returnOp : returns) {
    mlir::OpBuilder builder(returnOp);
    mlir::func::ReturnOp::create(builder, returnOp.getLoc(),
                                 returnOp.getOperands());
    returnOp.erase();
  }
}

/// Insert hip.create_handle at function entry and hip.destroy_handle before
/// each return. Returns the handle value.
static mlir::Value insertHandleLifecycle(mlir::func::FuncOp funcOp,
                                         mlir::MLIRContext *ctx) {
  mlir::Block &entryBlock = funcOp.getBody().front();
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
                                             mlir::MLIRContext *ctx,
                                             mlir::Value handle);

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

/// ONNXMatMulOp -> hip.hipblaslt.matmul
struct MatMulToHip : public mlir::OpRewritePattern<ONNXMatMulOp> {
  mlir::Value handle;
  MatMulToHip(mlir::MLIRContext *ctx, mlir::Value handle)
      : OpRewritePattern(ctx), handle(handle) {}

  mlir::LogicalResult
  matchAndRewrite(ONNXMatMulOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MatMulToHip::matchAndRewrite(ONNXMatMulOp op,
                             mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op.getLoc();
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // MatMul: result[..., M, N] = A[..., M, K] @ B[..., K, N].
  // Batch and M dims come from A; N comes from B's last dim.
  llvm::SmallVector<mlir::Value> dynSizes;
  int64_t rank = resultType.getRank();
  auto bType = mlir::cast<mlir::RankedTensorType>(op.getB().getType());
  for (int64_t i = 0; i < rank; ++i) {
    if (!resultType.isDynamicDim(i))
      continue;
    if (i == rank - 1) {
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, op.getB(),
                                                     bType.getRank() - 1));
    } else {
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, op.getA(), i));
    }
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);
  auto hipOp = mlir::hip::HipblasltMatmulOp::create(
      rewriter, loc, resultType, handle, op.getA(), op.getB(), init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// ONNXTransposeOp -> hip.transpose
struct TransposeToHip : public mlir::OpRewritePattern<ONNXTransposeOp> {
  mlir::Value handle;
  TransposeToHip(mlir::MLIRContext *ctx, mlir::Value handle)
      : OpRewritePattern(ctx), handle(handle) {}

  mlir::LogicalResult
  matchAndRewrite(ONNXTransposeOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
TransposeToHip::matchAndRewrite(ONNXTransposeOp op,
                                mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op.getLoc();

  auto permAttr = op.getPerm();
  if (!permAttr)
    return op.emitOpError("hip.transpose requires explicit perm attribute");
  auto perm = *permAttr;

  int64_t dim0 = -1, dim1 = -1;
  int64_t mismatchCount = 0;
  for (int64_t i = 0; i < static_cast<int64_t>(perm.size()); ++i) {
    int64_t p = mlir::cast<mlir::IntegerAttr>(perm[i]).getInt();
    if (p != i) {
      ++mismatchCount;
      if (dim0 < 0)
        dim0 = i;
      else if (dim1 < 0)
        dim1 = i;
    }
  }
  if (mismatchCount != 2 || dim0 < 0 || dim1 < 0)
    return op.emitOpError("perm must swap exactly two dimensions");
  int64_t p0 = mlir::cast<mlir::IntegerAttr>(perm[dim0]).getInt();
  int64_t p1 = mlir::cast<mlir::IntegerAttr>(perm[dim1]).getInt();
  if (p0 != dim1 || p1 != dim0)
    return op.emitOpError("perm must swap exactly two dimensions");

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Transpose: output dim i corresponds to input dim perm[i].
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i = 0; i < resultType.getRank(); ++i) {
    if (resultType.isDynamicDim(i)) {
      int64_t srcDim = mlir::cast<mlir::IntegerAttr>(perm[i]).getInt();
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, op.getData(), srcDim));
    }
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  mlir::Value d0 = mlir::arith::ConstantIndexOp::create(rewriter, loc, dim0);
  mlir::Value d1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, dim1);

  auto hipOp = mlir::hip::TransposeOp::create(rewriter, loc, resultType, handle,
                                              d0, d1, op.getData(), init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// ONNXMulOp -> hip.miopen.mul
struct MulToHip : public mlir::OpRewritePattern<ONNXMulOp> {
  mlir::Value handle;
  MulToHip(mlir::MLIRContext *ctx, mlir::Value handle)
      : OpRewritePattern(ctx), handle(handle) {}

  mlir::LogicalResult
  matchAndRewrite(ONNXMulOp op, mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MulToHip::matchAndRewrite(ONNXMulOp op, mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op.getLoc();
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Use the operand whose rank matches the result for dim extraction
  // (handles scalar * tensor broadcasting).
  auto aType = mlir::cast<mlir::RankedTensorType>(op.getA().getType());
  mlir::Value source =
      (aType.getRank() == resultType.getRank()) ? op.getA() : op.getB();
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, source);

  auto hipOp = mlir::hip::MiopenMulOp::create(rewriter, loc, resultType, handle,
                                              op.getA(), op.getB(), init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// ONNXSoftmaxOp -> hip.miopen.softmax
struct SoftmaxToHip : public mlir::OpRewritePattern<ONNXSoftmaxOp> {
  mlir::Value handle;
  SoftmaxToHip(mlir::MLIRContext *ctx, mlir::Value handle)
      : OpRewritePattern(ctx), handle(handle) {}

  mlir::LogicalResult
  matchAndRewrite(ONNXSoftmaxOp op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
SoftmaxToHip::matchAndRewrite(ONNXSoftmaxOp op,
                              mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op.getLoc();
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init =
      createEmptyTensor(rewriter, loc, resultType, op.getInput());
  auto hipOp = mlir::hip::MiopenSoftmaxOp::create(rewriter, loc, resultType,
                                                  handle, op.getInput(), init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// convertComputeOps implementation
//===----------------------------------------------------------------------===//

static mlir::LogicalResult convertComputeOps(mlir::func::FuncOp funcOp,
                                             mlir::MLIRContext *ctx,
                                             mlir::Value handle) {
  mlir::RewritePatternSet patterns(ctx);
  patterns.add<MatMulToHip>(ctx, handle);
  patterns.add<TransposeToHip>(ctx, handle);
  patterns.add<MulToHip>(ctx, handle);
  patterns.add<SoftmaxToHip>(ctx, handle);

  mlir::GreedyRewriteConfig config;
  config.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);

  if (mlir::failed(
          mlir::applyPatternsGreedily(funcOp, std::move(patterns), config)))
    return mlir::failure();

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
  std::string dir =
      externalizeOutputDir.getValue().empty() ? "." : externalizeOutputDir.getValue();
  if (externalizeMinNumElements > 0) {
    extState = std::make_unique<ExternalizationState>();

    // Derive base name from module symbol or default.
    std::string baseName = "model";
    if (auto sym = module->getAttrOfType<mlir::StringAttr>(
            mlir::SymbolTable::getSymbolAttrName()))
      baseName = sym.getValue().str();
    extState->binFileName = baseName + ".constants.bin";

    std::string binPath = dir + "/" + extState->binFileName;
    extState->binFile.open(binPath, std::ios::binary | std::ios::trunc);
    if (!extState->binFile.is_open()) {
      module.emitError("failed to open constants binary file: " + binPath);
      return signalPassFailure();
    }
  }

  for (auto funcOp :
       llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
    if (funcOp.isDeclaration())
      continue;
    if (mlir::failed(lowerOnnxConstants(module, funcOp,
                                        externalizeMinNumElements,
                                        extState.get())))
      return signalPassFailure();
    lowerOnnxReturns(funcOp);
    mlir::Value handle = insertHandleLifecycle(funcOp, ctx);
    if (mlir::failed(convertComputeOps(funcOp, ctx, handle)))
      return signalPassFailure();
  }

  for (auto ep : llvm::make_early_inc_range(module.getOps<ONNXEntryPointOp>()))
    ep.erase();

  // Finalize externalization: close binary, write JSON manifest, set module
  // attribute.
  if (extState && extState->constantIndex > 0) {
    extState->binFile.close();

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
      module.emitError("failed to open constants manifest: " + jsonPath +
                       " (" + ec.message() + ")");
      return signalPassFailure();
    }
    jsonFile << llvm::formatv("{0:2}",
                              llvm::json::Value(std::move(manifest)));
    jsonFile.close();

    module.emitRemark("externalized ")
        << extState->constantIndex << " constants ("
        << extState->currentOffset << " bytes) to "
        << dir + "/" + extState->binFileName;
  } else if (extState) {
    // No constants qualified -- clean up empty file.
    extState->binFile.close();
  }
}

} // namespace

} // namespace hip
} // namespace mlir
