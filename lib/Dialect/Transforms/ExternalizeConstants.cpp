/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ExternalizeConstants.cpp - Commit hip.constant storage policy ------===//
//
// This pass intentionally has three phases:
//   1. validate every carrier/order and build an immutable compatibility plan;
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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
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
constexpr llvm::StringLiteral kOrtMemAddrTag = "*/_ORT_MEM_ADDR_/*";

enum class SourceKind { Inline, Memory, File };

struct ConstantPlan {
  hip::ConstantOp op;
  RankedTensorType type;
  DenseElementsAttr value;
  SourceKind source = SourceKind::Inline;
  bool externalize = false;
  bool splat = false;
  int64_t numElements = 0;
  int64_t size = 0;
  int64_t index = -1;
  int64_t offset = 0;
  int64_t memoryAddress = 0;
  int64_t fileOffset = 0;
  int64_t memOffset = 0;
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
          static_cast<uintptr_t>(memoryAddress));
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
  llvm::SmallSet<int64_t, 16> compilerOrders;
  int64_t walkOrder = 0;
  bool valid = true;

  module.walk([&](hip::ConstantOp op) {
    OrderedCarrier carrier;
    carrier.op = op;
    carrier.walkOrder = walkOrder++;

    Attribute originRaw = op->getAttr("hip.constant_origin");
    Attribute orderRaw = op->getAttr("hip.constant_order");
    if (!originRaw && !orderRaw) {
      carrier.order = carrier.walkOrder;
      ordered.push_back(carrier);
      return;
    }
    if (!originRaw || !orderRaw) {
      op.emitError("compiler-owned `hip.constant_origin` and "
                   "`hip.constant_order` must be present together");
      valid = false;
      return;
    }

    auto origin = dyn_cast<StringAttr>(originRaw);
    auto order = dyn_cast<IntegerAttr>(orderRaw);
    if (!origin || !order) {
      op.emitError("compiler-owned constant origin/order attributes have "
                   "invalid types");
      valid = false;
      return;
    }
    carrier.order = order.getInt();
    if (carrier.order < 0) {
      op.emitError("compiler-owned constant order must be non-negative");
      valid = false;
      return;
    }

    if (origin.getValue() != "onnx-imported" &&
        origin.getValue() != "onnx-synthesized") {
      op.emitError("unknown compiler-owned constant origin `")
          << origin.getValue() << "`";
      valid = false;
      return;
    }
    carrier.category = 0;
    if (!compilerOrders.insert(carrier.order).second) {
      op.emitError("duplicate compiler-owned constant order ") << carrier.order;
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

class ExternalizationPlan {
public:
  ExternalizationPlan(ModuleOp module, morphizen::FileSystem &fs,
                      int64_t minElements, bool skipData)
      : module(module), fs(fs), minElements(minElements), skipData(skipData),
        binFileName(moduleBaseName(module) + ".constants.bin") {}

  LogicalResult validateAndPlan() {
    if (minElements < 0)
      return module.emitError(
          "externalize-min-num-elements must be non-negative");
    if (failed(rejectStaleState()))
      return failure();

    FailureOr<SmallVector<hip::ConstantOp>> constants =
        collectOrderedCarriers(module);
    if (failed(constants))
      return failure();

    llvm::SmallSet<std::string, 16> plannedSymbols;
    int64_t currentOffset = 0;
    for (hip::ConstantOp op : *constants) {
      if (failed(op.verify()))
        return failure();

      ConstantPlan plan;
      plan.op = op;
      plan.type = op.getResult().getType();
      FailureOr<int64_t> count = checkedNumElements(op, plan.type);
      if (failed(count))
        return failure();
      plan.numElements = *count;
      FailureOr<int64_t> size =
          checkedByteSize(op, plan.type, plan.numElements);
      if (failed(size))
        return failure();
      plan.size = *size;

      if (auto value =
              dyn_cast_if_present<DenseElementsAttr>(op.getValueAttr())) {
        plan.value = value;
        plan.source = SourceKind::Inline;
        plan.splat = value.isSplat();
        plan.externalize = minElements > 0 && plan.numElements >= minElements;
        if (plan.splat) {
          plan.splatElementSize =
              static_cast<int64_t>(value.getRawData().size());
          int64_t expectedElementSize =
              (plan.type.getElementTypeBitWidth() + 7) / 8;
          if (plan.splatElementSize != expectedElementSize)
            return op.emitError("splat raw element size ")
                   << plan.splatElementSize << " does not match expected size "
                   << expectedElementSize;
        } else if (failed(expandI1Data(plan))) {
          return failure();
        } else if (plan.ownedData.empty() &&
                   static_cast<int64_t>(value.getRawData().size()) !=
                       plan.size) {
          return op.emitError("dense raw byte size ")
                 << value.getRawData().size()
                 << " does not match result byte size " << plan.size;
        }
      } else {
        StringRef location = op.getLocationAttr().getValue();
        plan.externalize = minElements > 0;
        if (location == kOrtMemAddrTag) {
          plan.source = SourceKind::Memory;
          plan.memoryAddress = op.getOffsetAttr().getInt();
        } else {
          plan.source = SourceKind::File;
          plan.filePath = location.str();
          plan.fileOffset = op.getOffsetAttr().getInt();
        }

        if (plan.externalize && !skipData && plan.source == SourceKind::File &&
            failed(validateFileRange(op, location, plan.fileOffset, plan.size)))
          return failure();

        if (!plan.externalize) {
          if (plan.source == SourceKind::File &&
              failed(
                  validateFileRange(op, location, plan.fileOffset, plan.size)))
            return failure();
          plan.ownedData.resize(static_cast<size_t>(plan.size));
          if (plan.source == SourceKind::Memory) {
            if (plan.size > 0)
              std::memcpy(plan.ownedData.data(),
                          reinterpret_cast<const void *>(
                              static_cast<uintptr_t>(plan.memoryAddress)),
                          static_cast<size_t>(plan.size));
          } else if (plan.size > 0) {
            FilePtr file(std::fopen(plan.filePath.c_str(), "rb"), &std::fclose);
            if (!file || fileSeek(file.get(), plan.fileOffset, SEEK_SET) != 0 ||
                std::fread(plan.ownedData.data(), 1,
                           static_cast<size_t>(plan.size),
                           file.get()) != static_cast<size_t>(plan.size))
              return op.emitError("failed to read external data file: ")
                     << plan.filePath;
          }
          FailureOr<DenseElementsAttr> materialized = denseFromBytes(plan);
          if (failed(materialized))
            return op.emitError(
                "failed to materialize external source as DenseElementsAttr");
          if (materialized->getType() != plan.type)
            return op.emitError("materialized external source type ")
                   << materialized->getType()
                   << " does not match carrier result type " << plan.type;
          plan.value = *materialized;
        }
      }

      if (plan.externalize) {
        FailureOr<int64_t> aligned = checkedAlign(op, currentOffset);
        if (failed(aligned))
          return failure();
        plan.offset = *aligned;
        if (plan.offset > std::numeric_limits<int64_t>::max() - plan.size)
          return op.emitError("constant layout range overflows int64");
        currentOffset = plan.offset + plan.size;
        if (externalizedCount >
            static_cast<size_t>(std::numeric_limits<int64_t>::max()))
          return op.emitError("constant index overflows int64");
        plan.index = static_cast<int64_t>(externalizedCount++);
        setNames(plan);
        if (SymbolTable::lookupSymbolIn(module, plan.symbolName) ||
            !plannedSymbols.insert(plan.symbolName).second)
          return op.emitError("externalized constant symbol collision: @")
                 << plan.symbolName;
      }
      plans.push_back(std::move(plan));
    }
    totalBlobSize = currentOffset;
    return success();
  }

  LogicalResult serialize() {
    if (externalizedCount == 0)
      return success();
    return skipData ? serializeStreamingOrHybrid() : serializeFull();
  }

  LogicalResult commit() {
    for (ConstantPlan &plan : plans) {
      if (plan.externalize)
        commitExternal(plan);
      else if (failed(commitInline(plan)))
        return failure();
    }

    if (externalizedCount == 0)
      return success();
    MLIRContext *ctx = module.getContext();
    module->setAttr("hip.constants_file", StringAttr::get(ctx, binFileName));
    module->setAttr("hipdnn.constant_sizes",
                    DenseI64ArrayAttr::get(ctx, sizes));
    module->setAttr("hipdnn.constant_offsets",
                    DenseI64ArrayAttr::get(ctx, offsets));
    if (skipData) {
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
    return success();
  }

private:
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
    std::string nodeName;
    if (auto attr = plan.op->getAttrOfType<StringAttr>("onnx_node_name"))
      nodeName = attr.getValue().str();

    plan.symbolName = "hip_ext_constant_";
    std::string fragment = sanitizeForMlirIdentifier(nodeName);
    if (!fragment.empty())
      plan.symbolName += fragment + "_";
    plan.symbolName += std::to_string(plan.index);

    plan.constantName = nodeName;
    if (plan.constantName.empty())
      if (auto outputs = plan.op->getAttrOfType<ArrayAttr>("node.outputs"))
        if (!outputs.empty())
          if (auto name = dyn_cast<StringAttr>(outputs[0]))
            plan.constantName = name.getValue().str();

    sizes.push_back(plan.size);
    offsets.push_back(plan.offset);
  }

  ConstantEntry makeEntry(const ConstantPlan &plan, int64_t offset) const {
    ConstantEntry entry;
    entry.name = plan.constantName;
    entry.offset = offset;
    entry.size = plan.size;
    entry.data = plan.data();
    entry.splat_elem_size = plan.splatElementSize;
    entry.file_path = plan.filePath;
    entry.file_offset = plan.fileOffset;
    return entry;
  }

  LogicalResult serializeFull() {
    std::vector<ConstantEntry> entries;
    llvm::json::Array manifestEntries;
    for (const ConstantPlan &plan : plans) {
      if (!plan.externalize)
        continue;
      entries.push_back(makeEntry(plan, plan.offset));
      llvm::json::Array shape;
      for (int64_t dim : plan.type.getShape())
        shape.push_back(dim);
      llvm::json::Object item;
      item["name"] = plan.symbolName;
      item["shape"] = std::move(shape);
      item["element_type"] = elementTypeToString(plan.type.getElementType());
      item["offset"] = plan.offset;
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

  LogicalResult serializeStreamingOrHybrid() {
    sourceKinds.assign(externalizedCount, 0);
    splatValues.assign(externalizedCount, 0);
    splatElementSizes.assign(externalizedCount, 0);
    fileOffsets.assign(externalizedCount, 0);
    memOffsets.assign(externalizedCount, 0);
    filePaths.assign(externalizedCount,
                     StringAttr::get(module.getContext(), ""));

    std::vector<ConstantEntry> partialEntries;
    int64_t memPosition = 0;
    for (ConstantPlan &plan : plans) {
      if (!plan.externalize)
        continue;
      size_t i = static_cast<size_t>(plan.index);
      if (plan.source == SourceKind::File) {
        sourceKinds[i] = 2;
        fileOffsets[i] = plan.fileOffset;
        filePaths[i] = StringAttr::get(module.getContext(), plan.filePath);
      } else if (plan.splat) {
        sourceKinds[i] = 1;
        splatElementSizes[i] = plan.splatElementSize;
        std::memcpy(
            &splatValues[i], plan.data(),
            static_cast<size_t>(std::min<int64_t>(plan.splatElementSize, 8)));
      } else {
        sourceKinds[i] = 3;
        FailureOr<int64_t> aligned = checkedAlign(plan.op, memPosition);
        if (failed(aligned))
          return failure();
        plan.memOffset = *aligned;
        memOffsets[i] = plan.memOffset;
        if (plan.memOffset > std::numeric_limits<int64_t>::max() - plan.size)
          return plan.op.emitError(
              "hybrid constant layout range overflows int64");
        memPosition = plan.memOffset + plan.size;
        partialEntries.push_back(makeEntry(plan, plan.memOffset));
        partialEntries.back().file_path.clear();
        partialEntries.back().splat_elem_size = 0;
      }
    }

    if (!partialEntries.empty() &&
        !writeConstantsBinToFileSystem(&fs, binFileName, partialEntries,
                                       memPosition))
      return module.emitError("failed to write partial mem-addr constants file "
                              "via FileSystem: ")
             << binFileName;
    return success();
  }

  LogicalResult commitInline(ConstantPlan &plan) {
    OpBuilder builder(plan.op);
    auto constant =
        arith::ConstantOp::create(builder, plan.op.getLoc(), plan.value);
    plan.op.getResult().replaceAllUsesWith(constant.getResult());
    plan.op.erase();
    return success();
  }

  void commitExternal(ConstantPlan &plan) {
    MemRefType memrefType =
        MemRefType::get(plan.type.getShape(), plan.type.getElementType());
    OpBuilder moduleBuilder(module.getBody(), module.getBody()->begin());
    DictionaryAttr externalData = moduleBuilder.getDictionaryAttr({
        moduleBuilder.getNamedAttr("index",
                                   moduleBuilder.getI64IntegerAttr(plan.index)),
        moduleBuilder.getNamedAttr(
            "offset", moduleBuilder.getI64IntegerAttr(plan.offset)),
        moduleBuilder.getNamedAttr("size",
                                   moduleBuilder.getI64IntegerAttr(plan.size)),
    });
    auto global = memref::GlobalOp::create(
        moduleBuilder, plan.op.getLoc(), plan.symbolName,
        moduleBuilder.getStringAttr("private"), memrefType,
        /*initial_value=*/nullptr, /*constant=*/false,
        moduleBuilder.getI64IntegerAttr(kConstantAlignment));
    global->setAttr("hip.external_data", externalData);

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
  std::string binFileName;
  size_t externalizedCount = 0;
  int64_t totalBlobSize = 0;
  std::vector<ConstantPlan> plans;
  SmallVector<int64_t> sizes;
  SmallVector<int64_t> offsets;
  SmallVector<int32_t> sourceKinds;
  SmallVector<int64_t> splatValues;
  SmallVector<int64_t> splatElementSizes;
  SmallVector<Attribute> filePaths;
  SmallVector<int64_t> fileOffsets;
  SmallVector<int64_t> memOffsets;
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

    ExternalizationPlan plan(module, *fs, minElements, skipData);
    if (failed(plan.validateAndPlan()) || failed(plan.serialize()) ||
        failed(plan.commit()))
      signalPassFailure();
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
