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

#include "hip/Conversion/OnnxToHip/ConstantsIO.h"
#include "hip/Support/DiskFileSystem.h"
#include "hip/timing.h"
#include "morphizen-foundation/file_io.hpp"

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

#include <chrono>
#include <cstring>
#include <fstream>
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
/// externalization is enabled. Constants are collected here during the
/// walk; the finalize step in runOnOperation emits either constants.bin
/// (skipDataWrite=false) via the shared ConstantsIO helper, or per-entry
/// source descriptors as module attrs (skipDataWrite=true) that
/// GenerateInterface bakes into __metadata_blob.ConstantInfo.source.
struct ExternalizationState {
  llvm::json::Array manifestEntries;
  int64_t currentOffset = 0;
  int64_t constantIndex = 0;
  std::string binFileName;
  llvm::SmallVector<int64_t> constantSizes;
  llvm::SmallVector<int64_t> constantOffsets;
  bool skipDataWrite = false;
  llvm::SmallVector<std::string> constantNames;

  // One descriptor per constant, in emission order. The descriptor encodes
  // one of three sources:
  //   * mem-addr / inline: `ptr` points at data owned elsewhere
  //     (DenseElementsAttr rawData held by MLIRContext) and stays valid
  //     through runOnOperation's end.
  //   * splat: `splatElemSize > 0` and `ptr` points at the single element
  //     bytes; the finalize emitter tile-expands on the fly.
  //   * file-ref: `filePath` is non-empty; `fileOffset` is the byte offset
  //     within that file. The runtime is expected to fread the data on
  //     demand into a small staging buffer instead of mmap'ing the whole
  //     external-data file. This is the path that avoids ORT mmap for
  //     multi-GB models.
  struct HostEntry {
    const void *ptr = nullptr;
    int64_t splatElemSize = 0;
    std::string filePath; // empty unless file-ref
    int64_t fileOffset = 0;
  };
  llvm::SmallVector<HostEntry> constantHostPtrs;
};

/// Advance `currentOffset` to the next aligned boundary. The actual
/// padding bytes are emitted by the finalize step (writeConstantsBin*);
/// in transfer-file mode padding is implicit in the recorded offsets.
static int64_t writeAlignmentPadding(ExternalizationState *extState,
                                     int64_t alignment = 64) {
  int64_t aligned = llvm::alignTo(extState->currentOffset, alignment);
  extState->currentOffset = aligned;
  return aligned;
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
  std::string onnxName;
  if (auto nodeNameAttr =
          constOp->getAttrOfType<mlir::StringAttr>("onnx_node_name")) {
    onnxName = nodeNameAttr.getValue().str();
    std::string fragment = sanitizeForMlirIdentifier(nodeNameAttr.getValue());
    if (!fragment.empty())
      name += fragment + "_";
  }
  // Initializers have their tensor name in "node.outputs" (the output NodeArg
  // name), not in "onnx_node_name" (the NodeProto.name which may be empty).
  if (onnxName.empty()) {
    if (auto outputsAttr =
            constOp->getAttrOfType<mlir::ArrayAttr>("node.outputs")) {
      if (outputsAttr.size() > 0) {
        if (auto strAttr =
                mlir::dyn_cast<mlir::StringAttr>(outputsAttr.getValue()[0]))
          onnxName = strAttr.getValue().str();
      }
    }
  }
  name += std::to_string(extState->constantIndex);

  extState->constantSizes.push_back(byteSize);
  extState->constantOffsets.push_back(entryOffset);
  extState->constantNames.push_back(onnxName);

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
  // Always collect layout + host pointer. The finalize step decides whether
  // to emit constants.bin (skipDataWrite=false) or module attrs encoding
  // the per-constant source (skipDataWrite=true). DenseElementsAttr raw
  // data / ORT mmap addresses
  // remain valid through runOnOperation, so recording the pointer here is
  // safe for both consumers.
  ExternalizationState::HostEntry entry;
  entry.ptr = rawPtr;
  extState->constantHostPtrs.push_back(std::move(entry));
  extState->currentOffset += byteSize;
  finalizeExternalizedConstant(module, constOp, tensorType, byteSize,
                               entryOffset, extState);
}

/// Record a file-ref entry: data lives on disk at (filePath, fileOffset)
/// and must be fread by the runtime into a staging buffer at upload time.
/// Avoids holding any of the data in host memory during compilation.
static void externalizeFileRefConstant(mlir::ModuleOp module,
                                       mlir::Operation *constOp,
                                       mlir::RankedTensorType tensorType,
                                       const std::string &filePath,
                                       int64_t fileOffset, int64_t byteSize,
                                       ExternalizationState *extState) {
  int64_t entryOffset = writeAlignmentPadding(extState);
  ExternalizationState::HostEntry entry;
  entry.filePath = filePath;
  entry.fileOffset = fileOffset;
  extState->constantHostPtrs.push_back(std::move(entry));
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

/// Externalize a splat constant: record only the single element bytes.
/// The finalize step (constants.bin emit OR transfer-file emit) tiles the
/// element value on the fly, so we never allocate a full-size buffer here.
/// DenseElementsAttr's raw data is owned by the MLIRContext and stays
/// valid through runOnOperation.
static void externalizeSplatConstant(mlir::ModuleOp module,
                                     mlir::Operation *constOp,
                                     mlir::RankedTensorType tensorType,
                                     mlir::DenseElementsAttr valueAttr,
                                     int64_t byteSize,
                                     ExternalizationState *extState) {
  auto rawData = valueAttr.getRawData();
  int64_t entryOffset = writeAlignmentPadding(extState);
  ExternalizationState::HostEntry entry;
  entry.ptr = rawData.data();
  entry.splatElemSize = static_cast<int64_t>(rawData.size());
  extState->constantHostPtrs.push_back(std::move(entry));
  extState->currentOffset += byteSize;
  finalizeExternalizedConstant(module, constOp, tensorType, byteSize,
                               entryOffset, extState);
}

/// Resolve an onnx.Constant that carries a `location` attribute (zero-copy
/// external data emitted by the ORT bridge). The `location` string selects
/// one of two semantics:
///
///   * "*/_ORT_MEM_ADDR_/*"  -> `offset` is a raw memory address (ORT tensor
///     pointer cast to i64) that the data lives at right now. Used for
///     inline raw_data and (legacy) ORT mmap-resolved tensors.
///   * <other string>        -> `offset` is a byte offset within the file
///     at the given absolute path. Data is NOT in memory; the runtime is
///     expected to fread it on demand. This is the path that avoids ORT
///     mmap'ing multi-GB external-data files into the system page cache.
///
/// Input IR (produced by ir-converter-imp.cpp), mem-addr form:
///
///   %cst = "onnx.Constant"()
///       {location = "*/_ORT_MEM_ADDR_/*",
///        offset = 140695085056000 : i64,
///        size = 32 : i64} : () -> tensor<2x4xf32>
///
/// File-ref form:
///
///   %cst = "onnx.Constant"()
///       {location = "C:/.../weights.data",
///        offset = 1048576 : i64,
///        size = 33554432 : i64} : () -> tensor<...>
///
/// Output IR when externalization is enabled (extState != nullptr) is the
/// same memref.global + bufferization.to_tensor bridge as in
/// externalizeConstant; the only difference is what the transfer-file
/// finalize step writes for this entry.
///
/// Output IR when externalization is disabled (extState == nullptr):
///
///   %cst = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], ...]>
///       : tensor<2x4xf32>
static constexpr llvm::StringLiteral kOrtMemAddrTag = "*/_ORT_MEM_ADDR_/*";

static mlir::LogicalResult
resolveExternalLocationConstant(mlir::ModuleOp module, mlir::Operation *constOp,
                                ExternalizationState *extState) {
  auto locAttr = constOp->getAttrOfType<mlir::StringAttr>("location");
  auto offsetAttr = constOp->getAttrOfType<mlir::IntegerAttr>("offset");
  auto sizeAttr = constOp->getAttrOfType<mlir::IntegerAttr>("size");
  if (!locAttr || !offsetAttr || !sizeAttr)
    return constOp->emitError(
        "onnx.Constant with location attribute missing location/offset/size");

  int64_t offsetVal = offsetAttr.getInt();
  int64_t dataSize = sizeAttr.getInt();
  if (dataSize <= 0)
    return constOp->emitError("onnx.Constant has invalid size");

  auto tensorType =
      mlir::dyn_cast<mlir::RankedTensorType>(constOp->getResult(0).getType());
  if (!tensorType)
    return constOp->emitError("external constant has non-ranked result type");

  llvm::StringRef location = locAttr.getValue();
  bool isMemAddr = (location == kOrtMemAddrTag);

  if (isMemAddr) {
    if (offsetVal == 0)
      return constOp->emitError("onnx.Constant mem-addr has null address");
    const void *dataPtr =
        reinterpret_cast<const void *>(static_cast<uintptr_t>(offsetVal));

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

  // File-reference path: location is an absolute file path, offset is the
  // byte offset within that file. Without externalization (offline / inline
  // arith.constant test path) we still have to materialize the bytes; do a
  // one-shot fread and feed them to DenseElementsAttr. The full streaming
  // benefit only kicks in along the externalize path where a downstream
  // consumer can stream tensor-by-tensor.
  if (extState) {
    externalizeFileRefConstant(module, constOp, tensorType, location.str(),
                               offsetVal, dataSize, extState);
    return mlir::success();
  }

  std::vector<char> buf(static_cast<size_t>(dataSize));
  std::ifstream ifs(location.str(), std::ios::binary);
  if (!ifs)
    return constOp->emitError("failed to open external data file: ")
           << location;
  ifs.seekg(offsetVal);
  ifs.read(buf.data(), dataSize);
  if (!ifs)
    return constOp->emitError("short read from external data file: ")
           << location;
  auto denseAttr = mlir::DenseElementsAttr::getFromRawBuffer(
      tensorType, llvm::ArrayRef<char>(buf.data(), buf.size()));
  replaceWithArithConstant(constOp, denseAttr);
  return mlir::success();
}

/// Lower onnx.Constant ops to either externalized constants (constants.bin)
/// or inline arith.constant ops.
static mlir::LogicalResult lowerOnnxConstants(mlir::ModuleOp module,
                                              mlir::func::FuncOp funcOp,
                                              int64_t minNumElements,
                                              ExternalizationState *extState) {

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
                                             mlir::MLIRContext *ctx) {
  mlir::RewritePatternSet patterns(ctx);
  populateMatMulConversionPatterns(patterns, ctx);
  populateTransposeConversionPatterns(patterns, ctx);
  populateElementwiseConversionPatterns(patterns, ctx);
  populatePowerConversionPatterns(patterns, ctx);
  populateActivationConversionPatterns(patterns, ctx);
  populateCastConversionPatterns(patterns, ctx);
  populateReduceSumConversionPatterns(patterns, ctx);
  populateGatherConversionPatterns(patterns, ctx);
  populateShapeConversionPatterns(patterns, ctx);
  populateConvConversionPatterns(patterns, ctx);
  populateNormConversionPatterns(patterns, ctx);
  populateRotaryEmbeddingConversionPatterns(patterns, ctx);
  populateGqaConversionPatterns(patterns, ctx);
  populateMultiHeadAttentionConversionPatterns(patterns, ctx);
  populateMatMulNBitsConversionPatterns(patterns, ctx);
  populateQMoEConversionPatterns(patterns, ctx);
  populateReshapeConversionPatterns(patterns, ctx);
  populateCausalConvWithStateConversionPatterns(patterns, ctx);
  populateGemmConversionPatterns(patterns, ctx);
  populateWhereConversionPatterns(patterns, ctx);
  populateLinearAttentionConversionPatterns(patterns, ctx);
  populateRangeConversionPatterns(patterns, ctx);
  populateEqualConversionPatterns(patterns, ctx);
  populateDivConversionPatterns(patterns, ctx);
  populateReduceMaxConversionPatterns(patterns, ctx);
  populateMinConversionPatterns(patterns, ctx);
  populateNotConversionPatterns(patterns, ctx);
  populateCosConversionPatterns(patterns, ctx);
  populateSinConversionPatterns(patterns, ctx);
  populateCumSumConversionPatterns(patterns, ctx);
  populatePadConversionPatterns(patterns, ctx);
  populateTileConversionPatterns(patterns, ctx);
  populateExpandConversionPatterns(patterns, ctx);
  populateReduceProdConversionPatterns(patterns, ctx);
  populateLessConversionPatterns(patterns, ctx);
  populateGatherNDConversionPatterns(patterns, ctx);
  populateSignConversionPatterns(patterns, ctx);
  populateModConversionPatterns(patterns, ctx);
  populateConstantOfShapeConversionPatterns(patterns, ctx);
  populateSliceConversionPatterns(patterns, ctx);
  populateScatterNDConversionPatterns(patterns, ctx);
  populateIdentityConversionPatterns(patterns, ctx);
  populateAndConversionPatterns(patterns, ctx);
  populateSizeConversionPatterns(patterns, ctx);
  populateNonZeroConversionPatterns(patterns, ctx);
  populateConcatConversionPatterns(patterns, ctx);
  populateGlobalPoolConversionPatterns(patterns, ctx);
  populateFlattenConversionPatterns(patterns, ctx);

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

  ConvertOnnxToHipPass(morphizen::FileSystem *fs, int64_t minNumElements,
                       bool skipConstantData = false)
      : fileSystem_(fs), fsMinNumElements_(minNumElements),
        skipConstantData_(skipConstantData) {}

  void runOnOperation() override;

  morphizen::FileSystem *fileSystem_ = nullptr;
  int64_t fsMinNumElements_ = 0;
  bool skipConstantData_ = false;
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
    extState->skipDataWrite = skipConstantData_;

    std::string baseName = "model";
    if (auto sym = module->getAttrOfType<mlir::StringAttr>(
            mlir::SymbolTable::getSymbolAttrName()))
      baseName = sym.getValue().str();
    extState->binFileName = baseName + ".constants.bin";
    // constants.bin is written in one streaming pass in finalize() via
    // writeConstantsBinToFileSystem, after all offsets are known; no
    // writer is opened here.
  }

  // NOTE: onnx.CastLike -> onnx.Cast + dead-type-donor function-argument
  // drop is handled by the standalone simplify-onnx pass, which must run
  // upstream of this one (see lib/Dialect/Transforms/Pipelines.cpp). We
  // capture metadata directly from the (already-simplified) signatures.
  if (mlir::failed(generateModuleMetadata(module)))
    return signalPassFailure();
  logSubpass("metadata");

  for (auto funcOp :
       llvm::make_early_inc_range(module.getOps<mlir::func::FuncOp>())) {
    if (funcOp.isDeclaration())
      continue;
    // Pre-lowering ONNX rewrites that must run BEFORE constants are
    // externalized:
    //   * Gather(Shape(x), const_idx) -> tensor.from_elements(tensor.dim),
    //     collapsing the dynseqlen runtime-shape arithmetic chain to a
    //     single 0-D / 1-element result (narrows the host-store-into-pool
    //     footprint the late `--hip-materialize-host-scalars` pass must
    //     redirect out of the GPU pool).
    //   * Inlined FastGelu primitive chain (Pow/Mul/Sum/Tanh) ->
    //     onnx.Gelu(approximate="tanh"), restoring the MorphiZen-supported
    //     form for ORT paths that inline the Gelu function body.
    //   * Projector/vision decompositions (patch-embed Conv-ND -> Gemm,
    //     AveragePool(kernel==stride) -> Reshape/Transpose/ReduceMean,
    //     Pow(x,c) -> Mul chain, broadcasting Div -> Mul(Reciprocal)).
    //     ProjectorOpsRewrites emits NEW `onnx.*` ops (Reshape, Gemm,
    //     ReduceMean, ...) that a subsequent round must visit (e.g. the
    //     AveragePool decomposition's emitted Reshape feeds the next
    //     round's ReduceMean handling), so the set is applied in a
    //     fixed-point loop until quiescence rather than a single pass.
    // All patterns are value-based and require the literal constants to
    // still be inline in `onnx.Constant` `value` attributes — once the
    // constants are externalized to memref.get_global the matchers break.
    // ExistingOps strictness is sufficient: the patterns either rewrite to
    // tensor.* (Gather) or emit `onnx.*` ops. FastGelu (-> onnx.Gelu) and
    // ReshapeShapeFold (roots on onnx.Reshape, only swaps its shape operand
    // in place; the re-visit fails the "operand1 is onnx.Shape" guard) are
    // convergent. ProjectorOpsRewrites emits NEW `onnx.*` ops (Reshape, Gemm,
    // ReduceMean, ...) that a subsequent round must visit (e.g. the
    // AveragePool decomposition's emitted Reshape feeds the next round's
    // ReduceMean handling), so the set is applied in a fixed-point loop until
    // quiescence rather than a single pass. Newly-emitted ops are given their
    // result types in-place at emission (constructed explicitly from the dims
    // the rewriter already knows), so no separate ONNX-level shape-inference
    // pass is run between rounds — the HIP-dialect `--hip-infer-shapes` pass
    // (pipeline tail, post-conversion) resolves any residual dynamic dims. A
    // tiny RewriterBase::Listener flips a flag on any IR mutation; the loop
    // breaks the first round that mutates nothing (capped at kMaxRounds as a
    // safety net).
    {
      struct ChangeFlagListener final : public mlir::RewriterBase::Listener {
        bool changed = false;
        void notifyOperationInserted(mlir::Operation *,
                                     mlir::OpBuilder::InsertPoint) override {
          changed = true;
        }
        void notifyOperationModified(mlir::Operation *) override {
          changed = true;
        }
        void notifyOperationReplaced(mlir::Operation *,
                                     mlir::ValueRange) override {
          changed = true;
        }
        void notifyOperationErased(mlir::Operation *) override {
          changed = true;
        }
      };
      constexpr int kMaxRounds = 4;
      bool quiesced = false;
      for (int round = 0; round < kMaxRounds; ++round) {
        mlir::RewritePatternSet preLoweringPatterns(ctx);
        populateGatherShapeFoldPatterns(preLoweringPatterns, ctx);
        populateReshapeShapeFoldPatterns(preLoweringPatterns, ctx);
        populateFastGeluFusionPatterns(preLoweringPatterns, ctx);
        populateProjectorOpsRewritePatterns(preLoweringPatterns, ctx);
        ChangeFlagListener listener;
        mlir::GreedyRewriteConfig preLoweringConfig;
        preLoweringConfig.setStrictness(
            mlir::GreedyRewriteStrictness::ExistingOps);
        preLoweringConfig.setListener(&listener);
        if (mlir::failed(mlir::applyPatternsGreedily(
                funcOp, std::move(preLoweringPatterns), preLoweringConfig)))
          return signalPassFailure();
        if (!listener.changed) {
          quiesced = true;
          break;
        }
      }
      // If the loop never settles a future pattern set may rely on a
      // rewrite the safety cap silently dropped — surface it so the next
      // maintainer can raise the cap or find the bouncing pattern.
      if (!quiesced)
        funcOp.emitWarning()
            << "convert-onnx-to-hip: pre-lowering round loop hit kMaxRounds="
            << kMaxRounds << " without quiescence";
    }
    // Run ConstantOfShape folding BEFORE `lowerOnnxConstants` so it can
    // still see the original `onnx.Constant` (or `onnx.Shape`) as the
    // shape input.  Once `lowerOnnxConstants` externalises the constant,
    // the IR becomes `memref.global` with a null `initial_value` (data
    // lives in `constants.bin`) and the fold can no longer reach it.
    // Roots on `onnx.ConstantOfShape`, disjoint from the pre-lowering
    // patterns above (which root on `onnx.Gather` and `onnx.Tanh`), so
    // ordering and pattern-set separation are both safe.
    {
      mlir::RewritePatternSet preFoldPatterns(ctx);
      populateConstantOfShapeConversionPatterns(preFoldPatterns, ctx);
      mlir::GreedyRewriteConfig cfg;
      cfg.setStrictness(mlir::GreedyRewriteStrictness::ExistingOps);
      if (mlir::failed(mlir::applyPatternsGreedily(
              funcOp, std::move(preFoldPatterns), cfg)))
        return signalPassFailure();
    }
    if (mlir::failed(
            lowerOnnxConstants(module, funcOp, minElems, extState.get())))
      return signalPassFailure();
    lowerOnnxReturns(funcOp);
    if (mlir::failed(convertComputeOps(funcOp, ctx)))
      return signalPassFailure();
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

  // Clean up onnx.NoValue and onnx.EntryPoint, plus any other unregistered
  // onnx.* op that ended up with no uses after conversion. The latter case
  // is the dead-shape-arithmetic pattern shipped by some HF ONNX exports:
  // a Shape/Gather/Unsqueeze/Concat chain whose computed shape feeds a
  // Reshape that lowered to tensor.expand_shape via static type info, so
  // the computed-shape operand is never read. Without this DCE,
  // one-shot-bufferize trips on the unregistered op because it has
  // tensor-typed operands but no bufferization interface, and the whole
  // pipeline aborts with "op was not bufferized" — which is silent (CPU
  // fallback) at the EP level.
  //
  // The FastGelu fusion erases its primitive chain inline in
  // reverse-topological order via the rewriter, and the Gather/Shape
  // fold leaves its now-unused `onnx.Shape` and `onnx.Constant` operands
  // alive on purpose (they may be shared across many Gather sites). This
  // walk catches all those single-layer `use_empty` survivors — Shape
  // ops shared across Gather instances that all folded, index constants,
  // and dead-shape-arithmetic survivors from upstream exports.
  llvm::SmallVector<mlir::Operation *> toErase;
  module.walk([&](mlir::Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "onnx.EntryPoint")
      toErase.push_back(op);
    else if (name.starts_with("onnx.") && op->use_empty())
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

  // Finalize externalization: emit the constants.bin sidecar or per-entry
  // source descriptors (or both, in hybrid mode), write the JSON manifest
  // when a full sidecar is produced, and stamp module attributes.
  //
  // Three emit modes:
  //   * skipDataWrite=false  — full sidecar (Workflow A: EPContext export
  //     + offline hip-compiler). All ConstantInfo.source remain NONE; the
  //     runtime bulk-loads model.constants.bin and hipMemcpy's it once.
  //   * skipDataWrite=true, no mem-addr entries — pure streaming. Per-entry
  //     descriptors only (Splat / FileRef); no sidecar written.
  //   * skipDataWrite=true, has mem-addr entries — hybrid. Mem-addr bytes
  //     are packed into a *partial* sidecar at compact 64B-aligned offsets;
  //     the descriptor for each mem-addr entry becomes SidecarSource with
  //     its sidecar offset. file-ref / splat entries keep their streaming
  //     descriptors. The runtime per-entry path uploads from a single
  //     reusable staging buffer regardless of source mix, bounding host
  //     peak to the largest single tensor instead of total constants size.
  if (extState && extState->constantIndex > 0) {
    // Module attributes shared across all emit modes.
    module->setAttr("hip.constants_file",
                    mlir::StringAttr::get(ctx, extState->binFileName));
    module->setAttr("hipdnn.constant_sizes",
                    mlir::DenseI64ArrayAttr::get(ctx, extState->constantSizes));
    module->setAttr(
        "hipdnn.constant_offsets",
        mlir::DenseI64ArrayAttr::get(ctx, extState->constantOffsets));

    if (!extState->skipDataWrite) {
      // Stream constants.bin via the shared helper (preserves the 1 MB
      // tile pattern for splats so peak host memory is bounded; file-ref
      // entries stream from disk into the writer with no in-memory copy).
      llvm::SmallVector<mlir::hip::ConstantEntry> entries;
      entries.reserve(extState->constantHostPtrs.size());
      for (size_t i = 0; i < extState->constantHostPtrs.size(); ++i) {
        const auto &h = extState->constantHostPtrs[i];
        mlir::hip::ConstantEntry e;
        e.name = extState->constantNames[i];
        e.offset = extState->constantOffsets[i];
        e.size = extState->constantSizes[i];
        e.data = h.ptr;
        e.splat_elem_size = h.splatElemSize;
        e.file_path = h.filePath;
        e.file_offset = h.fileOffset;
        entries.push_back(std::move(e));
      }
      if (!mlir::hip::writeConstantsBinToFileSystem(
              fs, extState->binFileName,
              std::vector<mlir::hip::ConstantEntry>(entries.begin(),
                                                    entries.end()),
              extState->currentOffset)) {
        module.emitError(
            "failed to write constants binary file via FileSystem: " +
            extState->binFileName);
        return signalPassFailure();
      }

      // JSON manifest (only meaningful when constants.bin is produced).
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
    } else if (!extState->constantHostPtrs.empty()) {
      // skipDataWrite=true: per-entry descriptors with optional partial
      // sidecar for mem-addr entries (hybrid).
      //
      // Six parallel arrays, all indexed by constantIndex:
      //   constant_source_kinds:        0=NONE, 1=Splat, 2=FileRef, 3=Sidecar
      //   constant_splat_elem_values:   elem bytes packed into i64 (0 unless
      //   splat) constant_splat_elem_sizes:    elem byte count 1/2/4/8 (0
      //   unless splat) constant_file_paths:          absolute OS path (empty
      //   unless file-ref) constant_file_offsets:        byte offset within
      //   file (0 unless file-ref) constant_sidecar_offsets:     byte offset
      //   within partial sidecar
      //                                 (0 unless mem-addr / Sidecar)
      int64_t count = static_cast<int64_t>(extState->constantHostPtrs.size());
      llvm::SmallVector<int32_t> kinds(count, 0);
      llvm::SmallVector<int64_t> splatValues(count, 0);
      llvm::SmallVector<int64_t> splatElemSizes(count, 0);
      llvm::SmallVector<mlir::Attribute> filePaths;
      filePaths.reserve(count);
      llvm::SmallVector<int64_t> fileOffsets(count, 0);
      llvm::SmallVector<int64_t> sidecarOffsets(count, 0);

      // Pass 1: collect mem-addr entries into a compact partial sidecar
      // layout. Each entry is 64B-aligned within the sidecar (matches the
      // GPU blob alignment used elsewhere, so writeConstantsBinToFileSystem
      // emits identical zero padding logic).
      constexpr int64_t kSidecarAlign = 64;
      llvm::SmallVector<mlir::hip::ConstantEntry> partialEntries;
      int64_t sidecarPos = 0;
      int64_t memAddrCount = 0, fileRefCount = 0, splatCount = 0;
      for (int64_t i = 0; i < count; ++i) {
        const auto &h = extState->constantHostPtrs[i];
        if (!h.filePath.empty()) {
          ++fileRefCount;
        } else if (h.splatElemSize > 0) {
          ++splatCount;
        } else {
          ++memAddrCount;
          int64_t off = llvm::alignTo(sidecarPos, kSidecarAlign);
          sidecarOffsets[i] = off;
          mlir::hip::ConstantEntry e;
          e.name = extState->constantNames[i];
          e.offset = off;
          e.size = extState->constantSizes[i];
          e.data = h.ptr;
          partialEntries.push_back(std::move(e));
          sidecarPos = off + extState->constantSizes[i];
        }
      }

      // Pass 2: write the partial sidecar (mem-addr bytes only). When
      // there are no mem-addr entries the sidecar is omitted entirely
      // (pure streaming model — runtime never opens constants_filename).
      if (!partialEntries.empty()) {
        if (!mlir::hip::writeConstantsBinToFileSystem(
                fs, extState->binFileName,
                std::vector<mlir::hip::ConstantEntry>(partialEntries.begin(),
                                                      partialEntries.end()),
                sidecarPos)) {
          module.emitError(
              "failed to write partial mem-addr sidecar via FileSystem: " +
              extState->binFileName);
          return signalPassFailure();
        }
        llvm::errs() << "[ConvertOnnxToHipPass] hybrid: " << memAddrCount
                     << " mem-addr -> partial sidecar ("
                     << llvm::format("%.1f", sidecarPos / (1024.0 * 1024.0))
                     << " MB), " << fileRefCount << " file-ref + " << splatCount
                     << " splat -> streaming\n";
      } else {
        llvm::errs() << "[ConvertOnnxToHipPass] streaming: " << fileRefCount
                     << " file-ref + " << splatCount
                     << " splat -> per-entry descriptors\n";
      }

      // Pass 3: stamp per-entry source descriptors. mem-addr entries get
      // their sidecarOffsets[i] from pass 1; the rest stay at 0 in that
      // slot which is fine because GenerateInterface only reads it for
      // kind==3 (Sidecar).
      for (int64_t i = 0; i < count; ++i) {
        const auto &entry = extState->constantHostPtrs[i];
        std::string path;
        if (!entry.filePath.empty()) {
          kinds[i] = 2; // FileRef
          path = entry.filePath;
          fileOffsets[i] = entry.fileOffset;
        } else if (entry.splatElemSize > 0) {
          kinds[i] = 1; // Splat
          splatElemSizes[i] = entry.splatElemSize;
          // Left-pack up to 8 bytes of element data into a uint64 carrier
          // so we can ship it through a DenseI64ArrayAttr.
          size_t n =
              static_cast<size_t>(std::min<int64_t>(splatElemSizes[i], 8));
          std::memcpy(&splatValues[i], entry.ptr, n);
        } else {
          kinds[i] = 3; // Sidecar (mem-addr packed into partial sidecar)
        }
        filePaths.push_back(mlir::StringAttr::get(ctx, path));
      }

      module->setAttr("hipdnn.constant_source_kinds",
                      mlir::DenseI32ArrayAttr::get(ctx, kinds));
      module->setAttr("hipdnn.constant_splat_elem_values",
                      mlir::DenseI64ArrayAttr::get(ctx, splatValues));
      module->setAttr("hipdnn.constant_splat_elem_sizes",
                      mlir::DenseI64ArrayAttr::get(ctx, splatElemSizes));
      module->setAttr("hipdnn.constant_file_paths",
                      mlir::ArrayAttr::get(ctx, filePaths));
      module->setAttr("hipdnn.constant_file_offsets",
                      mlir::DenseI64ArrayAttr::get(ctx, fileOffsets));
      module->setAttr("hipdnn.constant_sidecar_offsets",
                      mlir::DenseI64ArrayAttr::get(ctx, sidecarOffsets));
    }

    LLVM_DEBUG(llvm::dbgs()
               << "externalized " << extState->constantIndex << " constants ("
               << extState->currentOffset << " bytes) to "
               << extState->binFileName
               << (extState->skipDataWrite ? " (per-entry descriptors)" : "")
               << "\n");
  }
  logSubpass("finalize");

  if (timing) {
    llvm::errs() << "[ConvertOnnxToHipPass] total: "
                 << llvm::format("%.3f", elapsed_since(passStart)) << "s\n";
  }
}

} // namespace

std::unique_ptr<mlir::Pass>
createConvertOnnxToHipPass(morphizen::FileSystem *fs, int64_t minNumElements,
                           bool skipConstantData) {
  return std::make_unique<ConvertOnnxToHipPass>(fs, minNumElements,
                                                skipConstantData);
}

} // namespace hip
} // namespace mlir
