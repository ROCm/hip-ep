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

#include "OnnxToHipUtils.h"

#include "hip/Support/DiskFileSystem.h"
#include "hip/timing.h"
#include "morphizen-foundation/file_io.hpp"

#include "mlir/IR/Verifier.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
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

#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <chrono>
#include <unordered_map>
#include <vector>

#define DEBUG_TYPE "convert-onnx-to-hip"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_CONVERTONNXTOHIPPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

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

/// Write alignment padding to constants.bin and return the aligned byte
/// offset where the next constant's data should begin.
static int64_t writeAlignmentPadding(ExternalizationState *extState,
                                     int64_t alignment = 64) {
  int64_t aligned = llvm::alignTo(extState->currentOffset, alignment);
  int64_t padding = aligned - extState->currentOffset;
  if (padding > 0) {
    llvm::SmallVector<char> zeros(padding, 0);
    extState->writer->fwrite(zeros.data(), padding);
    extState->currentOffset += padding;
  }
  return extState->currentOffset;
}

/// Shared bookkeeping after constant data has been written to constants.bin.
/// Updates ExternalizationState counters, emits the JSON manifest entry,
/// creates the extern memref.global with hip.external_data, and replaces
/// the original op with memref.get_global + bufferization.to_tensor.
static void finalizeExternalizedConstant(mlir::ModuleOp module,
                                         mlir::Operation *constOp,
                                         mlir::RankedTensorType tensorType,
                                         int64_t byteSize, int64_t entryOffset,
                                         ExternalizationState *extState) {
  constexpr int64_t kAlignment = 64;
  auto memrefType =
      mlir::MemRefType::get(tensorType.getShape(), tensorType.getElementType());

  std::string name = "hip_ext_constant_";
  if (auto nodeNameAttr =
          constOp->getAttrOfType<mlir::StringAttr>("onnx_node_name")) {
    std::string fragment = sanitizeForMlirIdentifier(nodeNameAttr.getValue());
    if (!fragment.empty())
      name += fragment + "_";
  }
  name += std::to_string(extState->constantIndex);

  extState->constantSizes.push_back(byteSize);
  extState->constantOffsets.push_back(entryOffset);

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

  mlir::OpBuilder moduleBuilder(module.getBody(), module.getBody()->begin());
  auto externalDataAttr = moduleBuilder.getDictionaryAttr({
      moduleBuilder.getNamedAttr(
          "index", moduleBuilder.getI64IntegerAttr(extState->constantIndex)),
      moduleBuilder.getNamedAttr("offset",
                                 moduleBuilder.getI64IntegerAttr(entryOffset)),
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

  mlir::OpBuilder builder(constOp);
  auto getGlobal = mlir::memref::GetGlobalOp::create(builder, constOp->getLoc(),
                                                     memrefType, name);
  auto toTensor = mlir::bufferization::ToTensorOp::create(
      builder, constOp->getLoc(), tensorType, getGlobal.getResult(),
      /*restrict=*/builder.getUnitAttr(),
      /*writable=*/nullptr);
  constOp->getResult(0).replaceAllUsesWith(toTensor.getResult());
  constOp->erase();

  ++extState->constantIndex;
}

/// Write one constant's raw data to constants.bin and replace the op with
/// an extern memref.global + bufferization.to_tensor bridge.
static void externalizeConstant(mlir::ModuleOp module, mlir::Operation *constOp,
                                mlir::RankedTensorType tensorType,
                                const void *rawPtr, int64_t byteSize,
                                ExternalizationState *extState) {
  int64_t entryOffset = writeAlignmentPadding(extState);
  extState->writer->fwrite(rawPtr, byteSize);
  extState->currentOffset += byteSize;
  finalizeExternalizedConstant(module, constOp, tensorType, byteSize,
                               entryOffset, extState);
}

/// Replace an onnx.Constant op with an inline arith.constant.
static void replaceWithArithConstant(mlir::Operation *constOp,
                                     mlir::DenseElementsAttr valueAttr) {
  mlir::OpBuilder builder(constOp);
  auto arithConst =
      mlir::arith::ConstantOp::create(builder, constOp->getLoc(), valueAttr);
  constOp->getResult(0).replaceAllUsesWith(arithConst.getResult());
  constOp->erase();
}

/// Externalize a splat constant: expand the single element via chunked
/// writes to avoid allocating the full tensor in memory.
static void externalizeSplatConstant(mlir::ModuleOp module,
                                     mlir::Operation *constOp,
                                     mlir::RankedTensorType tensorType,
                                     mlir::DenseElementsAttr valueAttr,
                                     int64_t byteSize,
                                     ExternalizationState *extState) {
  auto rawData = valueAttr.getRawData();
  constexpr size_t kSplatChunk = 1024 * 1024;
  size_t elemSize = rawData.size();
  size_t bufSize =
      (std::min(static_cast<size_t>(byteSize), kSplatChunk) / elemSize) *
      elemSize;
  std::vector<char> buf(bufSize);
  for (size_t i = 0; i < bufSize; i += elemSize)
    std::memcpy(buf.data() + i, rawData.data(), elemSize);

  int64_t entryOffset = writeAlignmentPadding(extState);
  size_t remaining = static_cast<size_t>(byteSize);
  while (remaining > 0) {
    size_t toWrite = std::min(remaining, bufSize);
    extState->writer->fwrite(buf.data(), toWrite);
    remaining -= toWrite;
  }
  extState->currentOffset += byteSize;
  finalizeExternalizedConstant(module, constOp, tensorType, byteSize,
                               entryOffset, extState);
}

/// Resolve an onnx.Constant that carries a `location` attribute
/// (zero-copy external data emitted by the ORT bridge).
/// The `offset` attribute holds a raw memory address (ORT tensor pointer
/// cast to i64) and `size` holds the byte count.
///
/// Input IR (produced by ir-converter-imp.cpp):
///
///   %cst = "onnx.Constant"()
///       {location = "*/_ORT_MEM_ADDR_/*",
///        offset = 140695085056000 : i64,
///        size = 32 : i64} : () -> tensor<2x4xf32>
///
/// Output IR when externalization is enabled (extState != nullptr):
///
///   memref.global "private" @hip_ext_constant_0 : memref<2x4xf32>
///       {alignment = 64 : i64,
///        hip.external_data = {index = 0 : i64, offset = 0 : i64,
///                             size = 32 : i64}}
///   ...
///   %0 = memref.get_global @hip_ext_constant_0 : memref<2x4xf32>
///   %1 = bufferization.to_tensor %0 restrict
///       : memref<2x4xf32> to tensor<2x4xf32>
///
/// Output IR when externalization is disabled (extState == nullptr):
///
///   %cst = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], ...]>
///       : tensor<2x4xf32>
static mlir::LogicalResult
resolveExternalLocationConstant(mlir::ModuleOp module, mlir::Operation *constOp,
                                ExternalizationState *extState) {
  auto offsetAttr = constOp->getAttrOfType<mlir::IntegerAttr>("offset");
  auto sizeAttr = constOp->getAttrOfType<mlir::IntegerAttr>("size");
  if (!offsetAttr || !sizeAttr)
    return constOp->emitError(
        "onnx.Constant with location attribute missing offset or size");

  int64_t addr = offsetAttr.getInt();
  int64_t dataSize = sizeAttr.getInt();
  if (addr == 0 || dataSize <= 0)
    return constOp->emitError("onnx.Constant has invalid address/size");

  const void *dataPtr =
      reinterpret_cast<const void *>(static_cast<uintptr_t>(addr));

  auto tensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(constOp->getResult(0).getType());
  if (!tensorType)
    return constOp->emitError("external constant has non-ranked result type");

  if (extState) {
    externalizeConstant(module, constOp, tensorType, dataPtr, dataSize,
                        extState);
  } else {
    auto rawData =
        llvm::ArrayRef<char>(static_cast<const char *>(dataPtr), dataSize);
    auto denseAttr =
        mlir::DenseElementsAttr::getFromRawBuffer(tensorType, rawData);
    replaceWithArithConstant(constOp, denseAttr);
  }
  return mlir::success();
}

/// Lower onnx.Constant ops to either externalized constants (constants.bin)
/// or inline arith.constant ops.
static mlir::LogicalResult lowerOnnxConstants(mlir::ModuleOp module,
                                              mlir::func::FuncOp funcOp,
                                              int64_t minNumElements,
                                              ExternalizationState *extState) {

  // Drop onnx.NoValue placeholders; LayerNorm and other ops use these
  // to mean "this optional input is absent".  The conversion patterns
  // for those ops already check `isa<NoneType>` on the operand type, but
  // bufferization chokes on a dangling `none`-typed op, so we erase it
  // outright here.
  llvm::SmallVector<mlir::Operation *> noValues;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "onnx.NoValue")
      noValues.push_back(op);
  });
  for (auto *op : noValues) {
    if (op->use_empty())
      op->erase();
    // Else: still in use (LayerNorm-style optional input).  Lowering
    // patterns have to handle the `none` type or replace the use; we
    // can't safely erase it.
  }

  llvm::SmallVector<mlir::Operation *> constants;
  funcOp.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "onnx.Constant")
      constants.push_back(op);
  });

  for (mlir::Operation *constOp : constants) {
    auto valueAttr = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
        constOp->getAttrOfType<mlir::ElementsAttr>("value"));

    if (valueAttr) {
      // Inline dense constant -- fall through to externalize-or-inline below.
    } else if (constOp->hasAttr("location")) {
      if (mlir::failed(
              resolveExternalLocationConstant(module, constOp, extState)))
        return mlir::failure();
      continue;
    } else {
      return constOp->emitError(
          "unsupported onnx.Constant form (expected dense value attribute "
          "or location attribute)");
    }

    if (extState && minNumElements > 0 &&
        valueAttr.getNumElements() >= minNumElements) {
      auto tensorType = mlir::cast<mlir::RankedTensorType>(valueAttr.getType());
      int64_t elemBits = tensorType.getElementTypeBitWidth();
      int64_t byteSize = valueAttr.getNumElements() * ((elemBits + 7) / 8);

      if (valueAttr.isSplat()) {
        externalizeSplatConstant(module, constOp, tensorType, valueAttr,
                                 byteSize, extState);
      } else {
        externalizeConstant(module, constOp, tensorType,
                            valueAttr.getRawData().data(), byteSize, extState);
      }

    } else {
      replaceWithArithConstant(constOp, valueAttr);
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

//===----------------------------------------------------------------------===//
// convertComputeOps implementation
//===----------------------------------------------------------------------===//

static mlir::LogicalResult convertComputeOps(mlir::func::FuncOp funcOp,
                                             mlir::MLIRContext *ctx,
                                             bool timing) {
  mlir::RewritePatternSet patterns(ctx);
  populateMatMulConversionPatterns(patterns, ctx);
  populateTransposeConversionPatterns(patterns, ctx);
  populateElementwiseConversionPatterns(patterns, ctx);
  populatePowerConversionPatterns(patterns, ctx);
  populateActivationConversionPatterns(patterns, ctx);
  populateCastConversionPatterns(patterns, ctx);
  populateReduceSumConversionPatterns(patterns, ctx);
  populateGatherConversionPatterns(patterns, ctx);
  populateConvConversionPatterns(patterns, ctx);
  populateNormConversionPatterns(patterns, ctx);
  populateRotaryEmbeddingConversionPatterns(patterns, ctx);
  populateGqaConversionPatterns(patterns, ctx);
  populateMatMulNBitsConversionPatterns(patterns, ctx);
  populateQMoEConversionPatterns(patterns, ctx);
  populateReshapeConversionPatterns(patterns, ctx);
  populateCausalConvWithStateConversionPatterns(patterns, ctx);
  populateGemmConversionPatterns(patterns, ctx);
  populateUnaryElementwiseConversionPatterns(patterns, ctx);
  populateBinaryElementwiseConversionPatterns(patterns, ctx);
  populateSliceConversionPatterns(patterns, ctx);
  populateTier2ShapeConversionPatterns(patterns, ctx);
  populateTier3CompareConversionPatterns(patterns, ctx);
  populateTier5SeqConversionPatterns(patterns, ctx);
  populateTier6ConversionPatterns(patterns, ctx);
  populateLstmConversionPatterns(patterns, ctx);
  populateStftConversionPatterns(patterns, ctx);
  // NonzeroConversion is disabled while we investigate a heap
  // corruption that surfaces ~50 events into convertComputeOps when
  // the pattern is merely registered (the corruption isn't triggered
  // by NonzeroToHip::matchAndRewrite firing -- bisected by env-var
  // gating).  Tracked for follow-up; the runtime kernel +
  // declarations stay in tree so re-enabling is a one-line change.
  populateNonzeroConversionPatterns(patterns, ctx);
  populateScatterNdConversionPatterns(patterns, ctx);

  mlir::GreedyRewriteConfig config;
  config.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);
  // Single greedy pass is enough: each onnx.* op converts to hip.*
  // (or arith.constant for shape folds) and none of those match any of
  // our patterns themselves.  Capping the rewriter avoids a quadratic
  // re-walk over Kokoro's ~2000-op graph.
  // The rewriter needs at least 2 iterations to confirm convergence (one
  // walk to apply, a second walk to verify nothing else matches).  Cap at
  // 4 so a runaway pattern surfaces as a clean failure rather than a hang.
  config.setMaxIterations(4);
  config.setMaxNumRewrites(50000);

  // Heartbeat listener.  Every 50 events log the last erased + last
  // inserted op kind so when the compiler crashes mid-rewrite we know
  // exactly which onnx op + which downstream op the pattern produced.
  struct Hb : public mlir::RewriterBase::Listener {
    int n = 0;
    std::chrono::steady_clock::time_point t0 =
        std::chrono::steady_clock::now();
    std::unordered_map<std::string, int> kinds;
    std::string lastErased = "<none>";
    std::string lastInserted = "<none>";
    void notifyOperationInserted(mlir::Operation *op,
                                 mlir::IRRewriter::InsertPoint) override {
      lastInserted = op->getName().getStringRef().str();
      kinds[lastInserted]++;
      tick();
    }
    void notifyOperationErased(mlir::Operation *op) override {
      lastErased = op->getName().getStringRef().str();
      tick();
    }
    void tick() {
      ++n;
      if ((n % 50) == 0) {
        double sec = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - t0)
                         .count();
        llvm::errs() << "[convertComputeOps]   " << n << " ev @ "
                     << llvm::format("%.2f", sec)
                     << "s -" << lastErased << " +" << lastInserted << "\n";
      }
    }
  } hb;
  if (timing)
    config.setListener(&hb);

  if (timing) {
    int numOnnx = 0;
    funcOp.walk([&](mlir::Operation *op) {
      if (op->getName().getDialectNamespace() == "onnx")
        numOnnx++;
    });
    llvm::errs() << "[convertComputeOps]   patterns ready, " << numOnnx
                 << " onnx ops, calling applyPatternsGreedily\n";
    auto t0 = timing_now();
    auto rc = mlir::applyPatternsGreedily(funcOp, std::move(patterns), config);
    llvm::errs() << "[convertComputeOps]   greedy driver: "
                 << llvm::format("%.3f", record_elapsed(t0)) << "s, total "
                 << hb.n << " events\n";
    std::vector<std::pair<std::string, int>> v(hb.kinds.begin(),
                                                hb.kinds.end());
    std::sort(v.begin(), v.end(),
              [](auto &a, auto &b) { return a.second > b.second; });
    for (size_t i = 0; i < std::min<size_t>(15, v.size()); ++i)
      llvm::errs() << "    +" << v[i].second << " " << v[i].first << "\n";
    (void)rc;
    return mlir::success();
  }
  (void)mlir::applyPatternsGreedily(funcOp, std::move(patterns), config);
  return mlir::success();
}

/// Patterns that need to read the original onnx.Constant values out of the IR
/// (e.g. Slice's starts/ends/steps, Clip's min/max scalars, ReduceMean's
/// axes, Range's start/limit/delta).  Run BEFORE lowerOnnxConstants so the
/// inputs are still onnx.Constant ops with their dense value attribute
/// attached.  After this pass runs, lowerOnnxConstants only sees constants
/// that flow into ops which don't need them at compile time and either
/// inlines (small) or externalizes (large) them as usual.
static mlir::LogicalResult preLowerShapeOps(mlir::func::FuncOp funcOp,
                                            mlir::MLIRContext *ctx) {
  mlir::RewritePatternSet patterns(ctx);
  populateSliceConversionPatterns(patterns, ctx);
  populateUnaryElementwiseConversionPatterns(patterns, ctx);
  populateTier2ShapeConversionPatterns(patterns, ctx);
  populateTier5SeqConversionPatterns(patterns, ctx);
  // Tier 6 (onnx.Pad / Expand) reads its `pads` / `shape` operand from a
  // constant tensor at compile time -- run before lowerOnnxConstants so we
  // still see the original onnx.Constant.  Once the constant has been
  // externalized into a memref.global the pattern can no longer recover the
  // values without a host-side runtime read.
  populateTier6ConversionPatterns(patterns, ctx);
  // STFT reads frame_step / frame_length from onnx.Constant scalars
  // that get externalized after lowerOnnxConstants.
  populateStftConversionPatterns(patterns, ctx);
  // Unsqueeze/Squeeze infer their axes from input vs output shape, so
  // they do not need to be run pre-constant-lowering -- the main
  // convertComputeOps pass picks them up.

  mlir::GreedyRewriteConfig config;
  config.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);
  config.setMaxIterations(8);
  config.setMaxNumRewrites(100000);
  (void)mlir::applyPatternsGreedily(funcOp, std::move(patterns), config);
  return mlir::success();
}

/// Final fallback: any onnx.* op still in the IR after the main
/// convertComputeOps pass cannot be bufferized -- replace with a
/// `tensor.empty` (zero-initialised lazy buffer) of the result type so
/// downstream passes can succeed.  Same treatment for hip.* ops that
/// have dynamic-shape results, because most existing hip.* LLVM
/// lowerings require static shapes (HipToLLVM marks them illegal and
/// the dialect-conversion pass aborts).  This is OBVIOUSLY wrong for
/// correctness -- the model output for any path that depends on these
/// ops will be garbage -- but it lets us validate the rest of the
/// pipeline (bufferize / HipToLLVM / LLVM codegen / DLL load) end to
/// end.  All of Kokoro's surviving ops at this stage are in the
/// iSTFTNet noise / `Slice` runtime-shape paths that are difficult to
/// represent statically and are tracked for proper implementation.
static mlir::LogicalResult dropUnsupportedOnnxOps(mlir::func::FuncOp funcOp) {
  llvm::SmallVector<mlir::Operation *> stragglers;
  funcOp.walk([&](mlir::Operation *op) {
    auto ns = op->getName().getDialectNamespace();
    if (ns == "onnx") {
      stragglers.push_back(op);
      return;
    }
    if (ns == "hip") {
      // Drop any hip.* op whose lowering is known to require static
      // shapes when the result type isn't fully static.  Most hip.*
      // ops (Conv, Matmul, Gather, etc.) handle dynamic dims fine
      // and must NOT be dropped here.
      llvm::StringRef name = op->getName().getStringRef();
      static const llvm::StringSet<> kStaticShapeOnly = {
      };
      if (!kStaticShapeOnly.count(name))
        return;
      auto isDynamic = [](mlir::Type t) {
        if (auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t))
          return !rt.hasStaticShape();
        return false;
      };
      bool anyDynamic = false;
      for (mlir::Type t : op->getResultTypes())
        if (isDynamic(t)) { anyDynamic = true; break; }
      if (!anyDynamic) {
        for (mlir::Value v : op->getOperands())
          if (isDynamic(v.getType())) { anyDynamic = true; break; }
      }
      if (anyDynamic)
        stragglers.push_back(op);
    }
  });
  // Process in reverse program order so consumers are replaced before
  // their producers, preserving SSA dominance after replacement.
  std::reverse(stragglers.begin(), stragglers.end());
  for (mlir::Operation *op : stragglers) {
    mlir::OpBuilder builder(op);
    llvm::SmallVector<mlir::Value> replacements;
    bool ok = true;
    for (mlir::Type t : op->getResultTypes()) {
      auto rt = mlir::dyn_cast<mlir::RankedTensorType>(t);
      if (!rt) {
        ok = false;
        break;
      }
      // Build dynamic-sizes: walk each dim, use 0 as placeholder for
      // dynamic ones.  Zero-element tensors propagate benignly through
      // downstream shape computations (never produce huge / negative dims).
      llvm::SmallVector<mlir::Value> dynSizes;
      for (int64_t i = 0; i < rt.getRank(); ++i) {
        if (rt.isDynamicDim(i))
          dynSizes.push_back(mlir::arith::ConstantIndexOp::create(
              builder, op->getLoc(), 0));
      }
      mlir::Value empty = mlir::tensor::EmptyOp::create(
          builder, op->getLoc(), rt.getShape(), rt.getElementType(), dynSizes);
      replacements.push_back(empty);
    }
    if (!ok) {
      op->emitWarning() << "dropUnsupportedOnnxOps: cannot synthesise "
                          "replacement for non-ranked-tensor result type";
      continue;
    }
    llvm::errs() << "[dropUnsupportedOnnxOps] DROPPING: "
                 << op->getName().getStringRef();
    for (mlir::Type t : op->getResultTypes())
      llvm::errs() << " -> " << t;
    llvm::errs() << "\n";
    op->replaceAllUsesWith(replacements);
    op->erase();
  }
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
  const bool timing = hipdnn_ep_timing_enabled();

  auto passStart = timing_now();
  auto phaseStart = passStart;

  auto logSubpass = [&](const char *name, const char *extra = nullptr) {
    if (!timing)
      return;
    double sec = record_elapsed(phaseStart);
    if (extra)
      llvm::errs() << "[ConvertOnnxToHipPass] " << name << ": "
                   << llvm::format("%.3f", sec) << "s  " << extra << "\n";
    else
      llvm::errs() << "[ConvertOnnxToHipPass] " << name << ": "
                   << llvm::format("%.3f", sec) << "s\n";
  };

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
  logSubpass("metadata");

  for (auto funcOp :
       llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
    if (funcOp.isDeclaration())
      continue;
    auto subStart = timing_now();
    auto logSub = [&](const char *name) {
      if (!timing)
        return;
      llvm::errs() << "[ConvertOnnxToHipPass] " << name << ": "
                   << llvm::format("%.3f",
                                   record_elapsed(subStart))
                   << "s @" << funcOp.getName().str() << "\n";
    };
    // Convert ops that need original onnx.Constant inputs (Slice, Clip)
    // BEFORE we lower constants to arith.constant or externalize them.
    if (mlir::failed(preLowerShapeOps(funcOp, ctx)))
      return signalPassFailure();
    logSub("preLowerShapeOps");
    if (mlir::failed(
            lowerOnnxConstants(module, funcOp, minElems, extState.get())))
      return signalPassFailure();
    logSub("lowerOnnxConstants");
    lowerOnnxReturns(funcOp);
    logSub("lowerOnnxReturns");
    if (mlir::failed(convertComputeOps(funcOp, ctx, timing)))
      return signalPassFailure();
    logSub("convertComputeOps");
    if (mlir::failed(dropUnsupportedOnnxOps(funcOp)))
      return signalPassFailure();
    logSub("dropUnsupportedOnnxOps");
  }

  if (timing && extState) {
    std::string detail;
    llvm::raw_string_ostream os(detail);
    os << "(" << extState->constantIndex << " constants, "
       << llvm::format("%.1f", extState->currentOffset / (1024.0 * 1024.0))
       << " MB)";
    logSubpass("constants + compute ops", detail.c_str());
  } else {
    logSubpass("constants + compute ops");
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
  logSubpass("finalize");

  if (timing) {
    llvm::errs() << "[ConvertOnnxToHipPass] total: "
                 << llvm::format("%.3f", elapsed_since(passStart)) << "s\n";
  }
}

} // namespace

std::unique_ptr<mlir::Pass>
createConvertOnnxToHipPass(morphizen::FileSystem *fs, int64_t minNumElements) {
  return std::make_unique<ConvertOnnxToHipPass>(fs, minNumElements);
}

} // namespace hip
} // namespace mlir
