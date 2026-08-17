/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ExternalizeConstants.cpp - Commit hip.constant storage policy ------===//
//
// This pass intentionally has three phases:
//   1. validate every carrier/order and build a compatibility plan;
//   2. serialize the planned artifacts;
//   3. commit globals, replacements, and module metadata to the IR.
//
// A failed validation or artifact write therefore leaves the IR unchanged.
// FileSystem only exposes create_writer + fwrite, not rename/remove, so a
// failed later write can still leave a partial artifact. Source files are
// validated only in modes that read them. The shared ConstantsIO contract also
// does not report FileWriter short writes. This pass does not claim filesystem
// transactionality or sink guarantees that the current APIs cannot provide.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/Support/ConstantsIO.h"
#include "hip/Support/DiskFileSystem.h"
#include "morphizen-foundation/file_io.hpp"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#define DEBUG_TYPE "hip-externalize-constants"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_EXTERNALIZECONSTANTSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kConstantAlignment = 64;

/// For single-element signless-integer extern constants, stash the host-visible
/// value on the global so compile-time folds (e.g. axis-0 Gather ->
/// tensor.extract_slice) can emit static subview offsets instead of a runtime
/// host load from GPU-resident constants.bin.
static void attachCompileTimeScalarAttr(memref::GlobalOp global,
                                        RankedTensorType tensorType,
                                        const void *hostBytes) {
  if (!hostBytes || tensorType.getNumElements() != 1)
    return;
  auto intTy = dyn_cast<IntegerType>(tensorType.getElementType());
  if (!intTy || !intTy.isSignlessInteger() ||
      (intTy.getWidth() != 32 && intTy.getWidth() != 64))
    return;
  int64_t value = intTy.getWidth() == 64
                      ? *reinterpret_cast<const int64_t *>(hostBytes)
                      : static_cast<int64_t>(
                            *reinterpret_cast<const int32_t *>(hostBytes));
  global->setAttr(
      "hip.compile_time_scalar",
      IntegerAttr::get(IntegerType::get(global.getContext(), 64), value));
}

using SourceKind = hip::ConstantOp::SourceKind;

struct ConstantPlan {
  hip::ConstantOp op;
  RankedTensorType type;
  DenseElementsAttr value;
  SourceKind source = SourceKind::Inline;
  bool externalize = false;
  bool splat = false;
  int64_t numElements = 0;
  int64_t size = 0;
  /// Dense index in the generated constant metadata arrays.
  int64_t index = -1;
  /// Offset in the logical full constants blob and `hip.external_data`.
  int64_t blobOffset = 0;
  /// Process-local address of producer-owned constant storage.
  int64_t hostAddress = 0;
  /// Offset in the original external-data file named by `filePath`.
  int64_t sourceFileOffset = 0;
  /// Offset in the partial constants blob used by streaming memory sources.
  int64_t partialBlobOffset = 0;
  int64_t splatElementSize = 0;
  std::string filePath;
  std::string symbolName;
  std::string constantName;
  std::vector<char> ownedData;

  const void *data() const {
    if (!ownedData.empty())
      return ownedData.data();
    if (source == SourceKind::Memory)
      return reinterpret_cast<const void *>(
          static_cast<uintptr_t>(hostAddress));
    return value ? value.getRawData().data() : nullptr;
  }
};

std::string moduleBaseName(ModuleOp module) {
  if (auto sym =
          module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
    return sym.getValue().str();
  return "model";
}

std::string sanitizeForMlirIdentifier(StringRef raw) {
  std::string result;
  result.reserve(raw.size());
  bool lastWasUnderscore = true;
  for (char c : raw) {
    bool keep = std::isalnum(static_cast<unsigned char>(c)) || c == '_';
    char next = keep ? c : '_';
    if (next == '_') {
      if (!lastWasUnderscore)
        result.push_back(next);
      lastWasUnderscore = true;
    } else {
      result.push_back(next);
      lastWasUnderscore = false;
    }
  }
  while (!result.empty() && result.back() == '_')
    result.pop_back();
  return result;
}

std::string elementTypeToString(Type elementType) {
  return llvm::TypeSwitch<Type, std::string>(elementType)
      .Case([](Float16Type) { return std::string("f16"); })
      .Case([](BFloat16Type) { return std::string("bf16"); })
      .Case([](Float32Type) { return std::string("f32"); })
      .Case([](Float64Type) { return std::string("f64"); })
      .Case([](IntegerType type) {
        return "i" + std::to_string(type.getWidth());
      })
      .Default([](Type type) {
        std::string text;
        llvm::raw_string_ostream os(text);
        type.print(os);
        return text;
      });
}

FailureOr<int64_t> checkedNumElements(hip::ConstantOp op,
                                      RankedTensorType type) {
  int64_t count = 1;
  for (int64_t dim : type.getShape())
    if (llvm::MulOverflow(count, dim, count)) {
      op.emitError("result element count overflows int64");
      return failure();
    }
  return count;
}

FailureOr<int64_t> checkedByteSize(hip::ConstantOp op, RankedTensorType type,
                                   int64_t numElements) {
  int64_t elementBytes =
      static_cast<int64_t>((type.getElementTypeBitWidth() + 7) / 8);
  int64_t size = 0;
  if (llvm::MulOverflow(numElements, elementBytes, size)) {
    op.emitError("result byte size overflows int64");
    return failure();
  }
  return size;
}

FailureOr<int64_t> checkedAlign(hip::ConstantOp op, int64_t value) {
  if (value > std::numeric_limits<int64_t>::max() - (kConstantAlignment - 1)) {
    op.emitError("constant layout alignment overflows int64");
    return failure();
  }
  return llvm::alignTo(value, kConstantAlignment);
}

LogicalResult validateFileRange(hip::ConstantOp op, StringRef path,
                                int64_t offset, int64_t size) {
  FilePtr file(std::fopen(path.str().c_str(), "rb"), &std::fclose);
  if (!file)
    return op.emitError("failed to open external data file: ") << path;
  if (fileSeek(file.get(), 0, SEEK_END) != 0)
    return op.emitError("failed to seek external data file: ") << path;
  int64_t fileSize = fileTell(file.get());
  if (fileSize < 0)
    return op.emitError("failed to determine external data file size: ")
           << path;
  if (offset > fileSize || size > fileSize - offset)
    return op.emitError("external data range [")
           << offset << ", " << (offset + size) << ") exceeds file size "
           << fileSize << ": " << path;
  return success();
}

LogicalResult expandI1Data(ConstantPlan &plan) {
  if (!plan.value || !plan.type.getElementType().isInteger(1) || plan.splat)
    return success();
  plan.ownedData.reserve(static_cast<size_t>(plan.numElements));
  for (const llvm::APInt &value : plan.value.getValues<llvm::APInt>())
    plan.ownedData.push_back(value.getBoolValue() ? 1 : 0);
  if (static_cast<int64_t>(plan.ownedData.size()) != plan.size)
    return plan.op.emitError("failed to expand i1 dense storage to one byte "
                             "per tensor element");
  return success();
}

FailureOr<DenseElementsAttr> denseFromBytes(ConstantPlan &plan) {
  ArrayRef<char> bytes(plan.ownedData);
  if (plan.type.getElementType().isInteger(1)) {
    // External i1 sources follow the runtime's one-byte-per-element contract,
    // while DenseElementsAttr bit-packs i1 storage. Normalize only this special
    // representation; all byte-aligned element types use their raw buffer.
    SmallVector<llvm::APInt> values;
    values.reserve(static_cast<size_t>(plan.numElements));
    for (char byte : bytes)
      values.emplace_back(/*numBits=*/1, static_cast<unsigned char>(byte) != 0);
    return DenseElementsAttr::get(plan.type, values);
  }
  DenseElementsAttr value =
      DenseElementsAttr::getFromRawBuffer(plan.type, bytes);
  if (!value)
    return plan.op.emitError(
        "external source bytes are not valid raw storage for carrier type");
  return value;
}

struct OrderedCarrier {
  hip::ConstantOp op;
  int category = 1;
  int64_t order = 0;
  int64_t walkOrder = 0;
};

FailureOr<SmallVector<hip::ConstantOp>>
collectOrderedCarriers(ModuleOp module) {
  SmallVector<OrderedCarrier> ordered;
  llvm::SmallDenseSet<int64_t, 16> explicitOrders;
  int64_t walkOrder = 0;
  bool valid = true;

  module.walk([&](hip::ConstantOp op) {
    OrderedCarrier carrier;
    carrier.op = op;
    carrier.walkOrder = walkOrder++;

    IntegerAttr explicitOrder = op.getSerializationOrderAttr();
    if (!explicitOrder) {
      carrier.order = carrier.walkOrder;
      ordered.push_back(carrier);
      return;
    }
    carrier.order = explicitOrder.getInt();
    carrier.category = 0;
    if (!explicitOrders.insert(carrier.order).second) {
      op.emitError("duplicate constant serialization order ") << carrier.order;
      valid = false;
      return;
    }
    ordered.push_back(carrier);
  });
  if (!valid)
    return failure();

  llvm::stable_sort(ordered,
                    [](const OrderedCarrier &lhs, const OrderedCarrier &rhs) {
                      if (lhs.category != rhs.category)
                        return lhs.category < rhs.category;
                      return lhs.category == 1 ? lhs.walkOrder < rhs.walkOrder
                                               : lhs.order < rhs.order;
                    });

  SmallVector<hip::ConstantOp> result;
  result.reserve(ordered.size());
  for (const OrderedCarrier &carrier : ordered)
    result.push_back(carrier.op);
  return result;
}

class ConstantExternalizer {
public:
  ConstantExternalizer(ModuleOp module, morphizen::FileSystem &fs,
                       int64_t minElements, bool skipData,
                       bool allowMemorySources)
      : module(module), fs(fs), minElements(minElements), skipData(skipData),
        allowMemorySources(allowMemorySources),
        binFileName(moduleBaseName(module) + ".constants.bin") {}

  LogicalResult buildPlan() {
    if (minElements < 0)
      return module.emitError(
          "externalize-min-num-elements must be non-negative");
    if (failed(rejectStaleState()))
      return failure();

    FailureOr<SmallVector<hip::ConstantOp>> constants =
        collectOrderedCarriers(module);
    if (failed(constants))
      return failure();

    llvm::StringSet<> plannedSymbols;
    int64_t currentOffset = 0;
    for (hip::ConstantOp op : *constants)
      if (failed(planConstant(op, currentOffset, plannedSymbols)))
        return failure();

    totalBlobSize = currentOffset;
    if (skipData && failed(planStreamingLayout()))
      return failure();
    return success();
  }

  LogicalResult writeArtifacts() {
    if (externalizedCount == 0)
      return success();
    return skipData ? writeStreamingArtifact() : writeFullArtifacts();
  }

  void applyPlan() {
    if (externalizedCount != 0)
      setModuleMetadata();

    for (ConstantPlan &plan : plans) {
      if (plan.externalize)
        commitExternal(plan);
      else
        commitInline(plan);
    }
  }

private:
  LogicalResult planConstant(hip::ConstantOp op, int64_t &currentOffset,
                             llvm::StringSet<> &plannedSymbols) {
    ConstantPlan plan;
    plan.op = op;
    plan.type = op.getResult().getType();
    plan.source = op.getSourceKind();

    FailureOr<int64_t> count = checkedNumElements(op, plan.type);
    if (failed(count))
      return failure();
    plan.numElements = *count;
    FailureOr<int64_t> size = checkedByteSize(op, plan.type, plan.numElements);
    if (failed(size))
      return failure();
    plan.size = *size;

    if (plan.source == SourceKind::Inline) {
      if (failed(planInlineSource(plan)))
        return failure();
    } else if (failed(planExternalSource(plan))) {
      return failure();
    }

    if (plan.externalize &&
        failed(assignExternalStorage(plan, currentOffset, plannedSymbols)))
      return failure();
    plans.push_back(std::move(plan));
    return success();
  }

  LogicalResult planInlineSource(ConstantPlan &plan) {
    plan.value = cast<DenseElementsAttr>(plan.op.getValueAttr());
    plan.splat = plan.value.isSplat();
    plan.externalize = minElements > 0 && plan.numElements >= minElements;
    if (plan.splat) {
      plan.splatElementSize =
          static_cast<int64_t>(plan.value.getRawData().size());
      int64_t expectedElementSize =
          (plan.type.getElementTypeBitWidth() + 7) / 8;
      if (plan.splatElementSize != expectedElementSize)
        return plan.op.emitError("splat raw element size ")
               << plan.splatElementSize << " does not match expected size "
               << expectedElementSize;
      return success();
    }

    if (failed(expandI1Data(plan)))
      return failure();
    if (plan.ownedData.empty() &&
        static_cast<int64_t>(plan.value.getRawData().size()) != plan.size)
      return plan.op.emitError("dense raw byte size ")
             << plan.value.getRawData().size()
             << " does not match result byte size " << plan.size;
    return success();
  }

  LogicalResult planExternalSource(ConstantPlan &plan) {
    plan.externalize = minElements > 0;
    if (plan.source == SourceKind::Memory) {
      if (!allowMemorySources)
        return plan.op.emitError(
            "memory-address sources require production externalization with "
            "an injected FileSystem");
      plan.hostAddress = plan.op.getMemoryAddressAttr().getInt();
    } else {
      StringRef location = plan.op.getLocationAttr().getValue();
      plan.filePath = location.str();
      plan.sourceFileOffset = plan.op.getOffsetAttr().getInt();
      // Full serialization and inline materialization consume the source now.
      // Streaming preserves the file reference and deliberately does not
      // require the source file to be present at compile time.
      if ((!skipData || !plan.externalize) &&
          failed(validateFileRange(plan.op, location, plan.sourceFileOffset,
                                   plan.size)))
        return failure();
    }

    if (!plan.externalize)
      return materializeExternalSource(plan);
    return success();
  }

  LogicalResult materializeExternalSource(ConstantPlan &plan) {
    plan.ownedData.resize(static_cast<size_t>(plan.size));
    if (plan.source == SourceKind::Memory) {
      if (plan.size > 0)
        std::memcpy(plan.ownedData.data(),
                    reinterpret_cast<const void *>(
                        static_cast<uintptr_t>(plan.hostAddress)),
                    static_cast<size_t>(plan.size));
    } else if (plan.size > 0) {
      FilePtr file(std::fopen(plan.filePath.c_str(), "rb"), &std::fclose);
      if (!file || fileSeek(file.get(), plan.sourceFileOffset, SEEK_SET) != 0 ||
          std::fread(plan.ownedData.data(), 1, static_cast<size_t>(plan.size),
                     file.get()) != static_cast<size_t>(plan.size))
        return plan.op.emitError("failed to read external data file: ")
               << plan.filePath;
    }

    FailureOr<DenseElementsAttr> materialized = denseFromBytes(plan);
    if (failed(materialized))
      return failure();
    plan.value = *materialized;
    return success();
  }

  LogicalResult assignExternalStorage(ConstantPlan &plan,
                                      int64_t &currentOffset,
                                      llvm::StringSet<> &plannedSymbols) {
    FailureOr<int64_t> aligned = checkedAlign(plan.op, currentOffset);
    if (failed(aligned))
      return failure();
    plan.blobOffset = *aligned;
    if (plan.blobOffset > std::numeric_limits<int64_t>::max() - plan.size)
      return plan.op.emitError("constant layout range overflows int64");
    currentOffset = plan.blobOffset + plan.size;

    if (externalizedCount >
        static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      return plan.op.emitError("constant index overflows int64");
    plan.index = static_cast<int64_t>(externalizedCount++);
    setNames(plan);
    if (SymbolTable::lookupSymbolIn(module, plan.symbolName) ||
        !plannedSymbols.insert(plan.symbolName).second)
      return plan.op.emitError("externalized constant symbol collision: @")
             << plan.symbolName;
    return success();
  }

  LogicalResult planStreamingLayout() {
    int64_t currentOffset = 0;
    for (ConstantPlan &plan : plans) {
      if (!plan.externalize || plan.source == SourceKind::File || plan.splat)
        continue;
      FailureOr<int64_t> aligned = checkedAlign(plan.op, currentOffset);
      if (failed(aligned))
        return failure();
      plan.partialBlobOffset = *aligned;
      if (plan.partialBlobOffset >
          std::numeric_limits<int64_t>::max() - plan.size)
        return plan.op.emitError(
            "hybrid constant layout range overflows int64");
      currentOffset = plan.partialBlobOffset + plan.size;
    }
    partialBlobSize = currentOffset;
    return success();
  }

  void setModuleMetadata() {
    SmallVector<int64_t> sizes;
    SmallVector<int64_t> offsets;
    SmallVector<int32_t> sourceKinds;
    SmallVector<int64_t> splatValues;
    SmallVector<int64_t> splatElementSizes;
    SmallVector<Attribute> filePaths;
    SmallVector<int64_t> fileOffsets;
    SmallVector<int64_t> memOffsets;
    sizes.reserve(externalizedCount);
    offsets.reserve(externalizedCount);

    MLIRContext *ctx = module.getContext();
    StringAttr emptyPath = StringAttr::get(ctx, "");
    for (const ConstantPlan &plan : plans) {
      if (!plan.externalize)
        continue;
      sizes.push_back(plan.size);
      offsets.push_back(plan.blobOffset);
      if (!skipData)
        continue;

      ConstantMetadataSourceKind kind;
      int64_t splatValue = 0;
      Attribute filePath = emptyPath;
      if (plan.source == SourceKind::File) {
        kind = ConstantMetadataSourceKind::File;
        filePath = StringAttr::get(ctx, plan.filePath);
      } else if (plan.splat) {
        kind = ConstantMetadataSourceKind::Splat;
        std::memcpy(
            &splatValue, plan.data(),
            static_cast<size_t>(std::min<int64_t>(plan.splatElementSize, 8)));
      } else {
        kind = ConstantMetadataSourceKind::Memory;
      }
      sourceKinds.push_back(static_cast<int32_t>(kind));
      splatValues.push_back(splatValue);
      splatElementSizes.push_back(plan.splatElementSize);
      filePaths.push_back(filePath);
      fileOffsets.push_back(plan.sourceFileOffset);
      memOffsets.push_back(plan.partialBlobOffset);
    }

    module->setAttr("hip.constants_file", StringAttr::get(ctx, binFileName));
    module->setAttr("hipdnn.constant_sizes",
                    DenseI64ArrayAttr::get(ctx, sizes));
    module->setAttr("hipdnn.constant_offsets",
                    DenseI64ArrayAttr::get(ctx, offsets));
    if (!skipData)
      return;
    module->setAttr("hipdnn.constant_source_kinds",
                    DenseI32ArrayAttr::get(ctx, sourceKinds));
    module->setAttr("hipdnn.constant_splat_elem_values",
                    DenseI64ArrayAttr::get(ctx, splatValues));
    module->setAttr("hipdnn.constant_splat_elem_sizes",
                    DenseI64ArrayAttr::get(ctx, splatElementSizes));
    module->setAttr("hipdnn.constant_file_paths",
                    ArrayAttr::get(ctx, filePaths));
    module->setAttr("hipdnn.constant_file_offsets",
                    DenseI64ArrayAttr::get(ctx, fileOffsets));
    module->setAttr("hipdnn.constant_mem_offsets",
                    DenseI64ArrayAttr::get(ctx, memOffsets));
  }

  LogicalResult rejectStaleState() {
    static constexpr const char *attrs[] = {
        "hip.constants_file",
        "hipdnn.constant_sizes",
        "hipdnn.constant_offsets",
        "hipdnn.constant_source_kinds",
        "hipdnn.constant_splat_elem_values",
        "hipdnn.constant_splat_elem_sizes",
        "hipdnn.constant_file_paths",
        "hipdnn.constant_file_offsets",
        "hipdnn.constant_mem_offsets",
    };
    for (const char *attr : attrs)
      if (module->hasAttr(attr))
        return module.emitError("hip-externalize-constants found stale `")
               << attr << "` metadata (duplicate invocation is unsupported)";

    bool foundExternalGlobal = false;
    module.walk([&](memref::GlobalOp global) {
      foundExternalGlobal |= global->hasAttr("hip.external_data");
    });
    if (foundExternalGlobal)
      return module.emitError(
          "hip-externalize-constants found pre-existing hip.external_data "
          "global (duplicate or stale externalization)");
    return success();
  }

  void setNames(ConstantPlan &plan) {
    plan.symbolName = "hip_ext_constant_";
    std::string fragment;
    if (StringAttr hint = plan.op.getSymbolNameHintAttr())
      fragment = sanitizeForMlirIdentifier(hint.getValue());
    if (!fragment.empty())
      plan.symbolName += fragment + "_";
    plan.symbolName += std::to_string(plan.index);

    if (StringAttr sourceName = plan.op.getSourceNameAttr())
      plan.constantName = sourceName.getValue().str();
  }

  ConstantEntry makeEntry(const ConstantPlan &plan, int64_t offset) const {
    ConstantEntry entry;
    entry.name = plan.constantName;
    entry.offset = offset;
    entry.size = plan.size;
    entry.data = plan.data();
    entry.splat_elem_size = plan.splatElementSize;
    entry.file_path = plan.filePath;
    entry.file_offset = plan.sourceFileOffset;
    return entry;
  }

  LogicalResult writeFullArtifacts() {
    std::vector<ConstantEntry> entries;
    llvm::json::Array manifestEntries;
    for (const ConstantPlan &plan : plans) {
      if (!plan.externalize)
        continue;
      entries.push_back(makeEntry(plan, plan.blobOffset));
      llvm::json::Array shape;
      for (int64_t dim : plan.type.getShape())
        shape.push_back(dim);
      llvm::json::Object item;
      item["name"] = plan.symbolName;
      item["shape"] = std::move(shape);
      item["element_type"] = elementTypeToString(plan.type.getElementType());
      item["offset"] = plan.blobOffset;
      item["size"] = plan.size;
      item["alignment"] = kConstantAlignment;
      manifestEntries.push_back(std::move(item));
    }
    if (!writeConstantsBinToFileSystem(&fs, binFileName, entries,
                                       totalBlobSize))
      return module.emitError(
                 "failed to write constants binary file via FileSystem: ")
             << binFileName;

    llvm::json::Object manifest;
    manifest["version"] = 1;
    manifest["binary_file"] = binFileName;
    manifest["num_constants"] = static_cast<int64_t>(externalizedCount);
    manifest["total_bytes"] = totalBlobSize;
    manifest["constants"] = std::move(manifestEntries);
    std::string json;
    llvm::raw_string_ostream os(json);
    os << llvm::formatv("{0:2}", llvm::json::Value(std::move(manifest)));

    std::string jsonName = moduleBaseName(module) + ".constants.json";
    auto writer = fs.create_writer_template(jsonName.c_str());
    if (!writer)
      return module.emitError(
                 "failed to open constants manifest via FileSystem: ")
             << jsonName;
    if (writer->fwrite(json.data(), json.size()) != json.size())
      return module.emitError(
                 "short write to constants manifest via FileSystem: ")
             << jsonName;
    return success();
  }

  LogicalResult writeStreamingArtifact() {
    std::vector<ConstantEntry> partialEntries;
    for (const ConstantPlan &plan : plans) {
      if (!plan.externalize || plan.source == SourceKind::File || plan.splat)
        continue;
      partialEntries.push_back(makeEntry(plan, plan.partialBlobOffset));
      partialEntries.back().file_path.clear();
      partialEntries.back().splat_elem_size = 0;
    }

    if (!partialEntries.empty() &&
        !writeConstantsBinToFileSystem(&fs, binFileName, partialEntries,
                                       partialBlobSize))
      return module.emitError("failed to write partial mem-addr constants file "
                              "via FileSystem: ")
             << binFileName;
    return success();
  }

  void commitInline(ConstantPlan &plan) {
    OpBuilder builder(plan.op);
    auto constant =
        arith::ConstantOp::create(builder, plan.op.getLoc(), plan.value);
    plan.op.getResult().replaceAllUsesWith(constant.getResult());
    plan.op.erase();
  }

  void commitExternal(ConstantPlan &plan) {
    MemRefType memrefType =
        MemRefType::get(plan.type.getShape(), plan.type.getElementType());
    OpBuilder moduleBuilder(module.getBody(), module.getBody()->begin());
    DictionaryAttr externalData = moduleBuilder.getDictionaryAttr({
        moduleBuilder.getNamedAttr("index",
                                   moduleBuilder.getI64IntegerAttr(plan.index)),
        moduleBuilder.getNamedAttr(
            "offset", moduleBuilder.getI64IntegerAttr(plan.blobOffset)),
        moduleBuilder.getNamedAttr("size",
                                   moduleBuilder.getI64IntegerAttr(plan.size)),
    });
    auto global = memref::GlobalOp::create(
        moduleBuilder, plan.op.getLoc(), plan.symbolName,
        moduleBuilder.getStringAttr("private"), memrefType,
        /*initial_value=*/nullptr, /*constant=*/false,
        moduleBuilder.getI64IntegerAttr(kConstantAlignment));
    global->setAttr("hip.external_data", externalData);
    attachCompileTimeScalarAttr(global, plan.type, plan.data());

    OpBuilder builder(plan.op);
    auto getGlobal = memref::GetGlobalOp::create(builder, plan.op.getLoc(),
                                                 memrefType, plan.symbolName);
    auto tensor = bufferization::ToTensorOp::create(
        builder, plan.op.getLoc(), plan.type, getGlobal,
        /*restrict=*/builder.getUnitAttr(), /*writable=*/nullptr);
    plan.op.getResult().replaceAllUsesWith(tensor.getResult());
    plan.op.erase();
  }

  ModuleOp module;
  morphizen::FileSystem &fs;
  int64_t minElements;
  bool skipData;
  bool allowMemorySources;
  std::string binFileName;
  size_t externalizedCount = 0;
  int64_t totalBlobSize = 0;
  int64_t partialBlobSize = 0;
  std::vector<ConstantPlan> plans;
};

struct ExternalizeConstantsPass
    : public impl::ExternalizeConstantsPassBase<ExternalizeConstantsPass> {
  using ExternalizeConstantsPassBase::ExternalizeConstantsPassBase;

  ExternalizeConstantsPass(morphizen::FileSystem *fs, int64_t minElements,
                           bool skipConstantData)
      : fileSystem(fs), fsMinElements(minElements),
        fsSkipConstantData(skipConstantData) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::unique_ptr<DiskFileSystem> fallback;
    morphizen::FileSystem *fs = fileSystem;
    bool allowMemorySources = fs != nullptr;
    int64_t minElements =
        fileSystem ? fsMinElements : externalizeMinNumElements.getValue();
    bool skipData =
        fileSystem ? fsSkipConstantData : skipConstantData.getValue();
    if (!fs) {
      StringRef outputDir = externalizeOutputDir.getValue();
      fallback = std::make_unique<DiskFileSystem>(
          outputDir.empty() ? "." : outputDir.str().c_str());
      fs = fallback.get();
    }

    ConstantExternalizer externalizer(module, *fs, minElements, skipData,
                                      allowMemorySources);
    if (failed(externalizer.buildPlan()) ||
        failed(externalizer.writeArtifacts())) {
      signalPassFailure();
      return;
    }
    externalizer.applyPlan();
  }

  morphizen::FileSystem *fileSystem = nullptr;
  int64_t fsMinElements = 0;
  bool fsSkipConstantData = false;
};

} // namespace

std::unique_ptr<mlir::Pass>
createExternalizeConstantsPass(morphizen::FileSystem *fs,
                               int64_t minNumElements, bool skipConstantData) {
  return std::make_unique<ExternalizeConstantsPass>(fs, minNumElements,
                                                    skipConstantData);
}

} // namespace hip
} // namespace mlir
