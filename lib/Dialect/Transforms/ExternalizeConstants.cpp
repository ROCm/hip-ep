/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- ExternalizeConstants.cpp - Externalize hip.constant carriers -------===//
//
// The --hip-externalize-constants pass is the compile-time, dialect-agnostic
// half of constant handling (its runtime complement is the already-standalone
// hip-resolve-extern-constants pass). It walks every `hip.constant` carrier --
// whether lowered from onnx.Constant by convert-onnx-to-hip or emitted by a
// downstream plugin for its own weights -- and serializes each one uniformly:
//
//   * value tensors at/above `externalize-min-num-elements` become an extern
//     `memref.global` (+ a `hip.external_data` {index, offset, size} attr)
//     whose bytes are appended to the `<model>.constants.bin` sidecar, then
//     bridged back to tensor land via `memref.get_global` +
//     `bufferization.to_tensor`;
//   * smaller value tensors fold to `arith.constant` (kept inline in the DLL);
//   * ORT external-data references (`location`/`offset`/`size`) are always
//     externalized -- either a live host pointer (mem-addr) or an on-disk
//     (path, offset) file range streamed by the runtime.
//
// Splitting this out of convert-onnx-to-hip is what lets plugins participate:
// they only emit `hip.constant`, with no access to any private state here.
//
//   Before:  %w = hip.constant {value = dense<...> : tensor<8x8xf32>}
//                    : tensor<8x8xf32>
//   After (>= threshold):
//     memref.global "private" @hip_ext_constant_0 : memref<8x8xf32>
//         {alignment = 64, hip.external_data = {index = 0, offset = 0, size}}
//     %g = memref.get_global @hip_ext_constant_0 : memref<8x8xf32>
//     %w = bufferization.to_tensor %g restrict
//            : memref<8x8xf32> to tensor<8x8xf32>
//   After (< threshold):  %w = arith.constant dense<...> : tensor<8x8xf32>
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHip/ConstantsIO.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/ConstantMetadata.h"
#include "hip/Dialect/Transforms/Passes.h"
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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#define DEBUG_TYPE "hip-externalize-constants"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_EXTERNALIZECONSTANTSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// GPU-blob alignment (bytes), shared by every externalized constant and the
/// partial mem-addr sidecar; matches the runtime memory-pool alignment.
constexpr int64_t kConstantAlignment = 64;

/// Sentinel `location` meaning "`offset` is a live host pointer" (ORT zero-copy
/// external data), as opposed to an absolute on-disk file path.
constexpr llvm::StringLiteral kOrtMemAddrTag = "*/_ORT_MEM_ADDR_/*";

/// Short element-type tag for the informational .constants.json manifest (the
/// runtime keys off sizes/offsets, not this string). Signless integers map to
/// `iN`; anything else falls back to the generic type printer.
std::string elementTypeToString(Type elemType) {
  return llvm::TypeSwitch<Type, std::string>(elemType)
      .Case([](Float16Type) { return std::string("f16"); })
      .Case([](BFloat16Type) { return std::string("bf16"); })
      .Case([](Float32Type) { return std::string("f32"); })
      .Case([](Float64Type) { return std::string("f64"); })
      .Case([](IntegerType intTy) -> std::string {
        if (intTy.isSignless())
          return "i" + std::to_string(intTy.getWidth());
        std::string s;
        llvm::raw_string_ostream os(s);
        Type(intTy).print(os);
        return s;
      })
      .Default([](Type ty) {
        std::string s;
        llvm::raw_string_ostream os(s);
        ty.print(os);
        return s;
      });
}

/// Sanitize an ONNX node name into a valid MLIR bare-identifier fragment.
/// Mirrors OnnxToHipUtils::sanitizeForMlirIdentifier so externalized global
/// symbol names are identical to the pre-split (in-convert) output.
std::string sanitizeForMlirIdentifier(StringRef raw) {
  std::string sanitized;
  sanitized.reserve(raw.size());
  for (char c : raw)
    sanitized.push_back(
        std::isalnum(static_cast<unsigned char>(c)) || c == '_' ? c : '_');
  // Collapse runs of '_' and trim leading/trailing ones.
  std::string result;
  result.reserve(sanitized.size());
  bool lastWasUnderscore = true;
  for (char c : sanitized) {
    if (c == '_') {
      if (!lastWasUnderscore)
        result.push_back(c);
      lastWasUnderscore = true;
    } else {
      result.push_back(c);
      lastWasUnderscore = false;
    }
  }
  while (!result.empty() && result.back() == '_')
    result.pop_back();
  return result;
}

/// A validated ORT external-data reference read off a location-carrying
/// hip.constant.
struct ExternalRef {
  StringRef location;
  int64_t offset = 0;
  int64_t size = 0;
  bool isMemAddr = false; ///< location == kOrtMemAddrTag (offset is a host ptr)
};

/// Read + validate the `location`/`offset`/`size` attributes; emit a diagnostic
/// and return failure on malformed input.
FailureOr<ExternalRef> readExternalRef(hip::ConstantOp constOp) {
  auto locAttr = constOp.getLocationAttr();
  auto offsetAttr = constOp.getOffsetAttr();
  auto sizeAttr = constOp.getSizeAttr();
  if (!locAttr || !offsetAttr || !sizeAttr) {
    constOp.emitError(
        "hip.constant with location missing location/offset/size");
    return failure();
  }
  ExternalRef ref;
  ref.location = locAttr.getValue();
  ref.offset = offsetAttr.getInt();
  ref.size = sizeAttr.getInt();
  ref.isMemAddr = (ref.location == kOrtMemAddrTag);
  if (ref.size <= 0) {
    constOp.emitError("hip.constant has invalid size");
    return failure();
  }
  if (ref.isMemAddr && ref.offset == 0) {
    constOp.emitError("hip.constant mem-addr has null address");
    return failure();
  }
  return ref;
}

/// Replace a carrier with an inline `arith.constant` (below-threshold fold and
/// the offline / no-externalize path).
void replaceWithArithConstant(Operation *constOp, DenseElementsAttr valueAttr) {
  OpBuilder builder(constOp);
  auto arithConst =
      arith::ConstantOp::create(builder, constOp->getLoc(), valueAttr);
  constOp->getResult(0).replaceAllUsesWith(arithConst.getResult());
  constOp->erase();
}

/// No-externalize path (threshold disabled): materialize a location-carrying
/// carrier's bytes inline as an arith.constant. mem-addr reads the live host
/// pointer; file-ref does a one-shot fread.
LogicalResult materializeLocationInline(hip::ConstantOp constOp,
                                        RankedTensorType tensorType) {
  FailureOr<ExternalRef> ref = readExternalRef(constOp);
  if (failed(ref))
    return failure();

  if (ref->isMemAddr) {
    auto rawData = ArrayRef<char>(
        reinterpret_cast<const char *>(static_cast<uintptr_t>(ref->offset)),
        ref->size);
    replaceWithArithConstant(
        constOp, DenseElementsAttr::getFromRawBuffer(tensorType, rawData));
    return success();
  }

  std::vector<char> buf(static_cast<size_t>(ref->size));
  std::ifstream ifs(ref->location.str(), std::ios::binary);
  if (!ifs)
    return constOp.emitError("failed to open external data file: ")
           << ref->location;
  ifs.seekg(ref->offset);
  ifs.read(buf.data(), ref->size);
  if (!ifs)
    return constOp.emitError("short read from external data file: ")
           << ref->location;
  replaceWithArithConstant(
      constOp, DenseElementsAttr::getFromRawBuffer(
                   tensorType, ArrayRef<char>(buf.data(), buf.size())));
  return success();
}

/// Accumulates the constants-blob layout across the module walk and emits the
/// extern globals, the `<model>.constants.bin` sidecar (or per-entry
/// descriptors in skip-data mode), and the module-level metadata attributes.
/// One instance per pass run; only created when externalization is enabled
/// (threshold > 0).
class ConstantExternalizer {
public:
  ConstantExternalizer(ModuleOp module, morphizen::FileSystem &fs,
                       bool skipDataWrite)
      : module_(module), ctx_(module.getContext()), fs_(fs),
        skipDataWrite_(skipDataWrite),
        binFileName_(moduleBaseName(module) + ".constants.bin") {}

  /// Externalize an inline-`value` carrier already known to be at/above the
  /// threshold. Splat values store only the single element (the sidecar writer
  /// tiles it), so peak host memory stays bounded regardless of tensor size.
  void externalizeValue(hip::ConstantOp constOp, RankedTensorType tensorType,
                        DenseElementsAttr valueAttr) {
    int64_t elemBytes = (tensorType.getElementTypeBitWidth() + 7) / 8;
    int64_t byteSize = valueAttr.getNumElements() * elemBytes;
    ArrayRef<char> raw = valueAttr.getRawData();
    HostEntry entry;
    entry.ptr = raw.data();
    if (valueAttr.isSplat())
      entry.splatElemSize = static_cast<int64_t>(raw.size());
    recordEntry(constOp, tensorType, byteSize, std::move(entry));
  }

  /// Externalize a location-carrying carrier (ORT external data): mem-addr ->
  /// the live host pointer, file-ref -> the on-disk (path, offset).
  LogicalResult externalizeLocation(hip::ConstantOp constOp,
                                    RankedTensorType tensorType) {
    FailureOr<ExternalRef> ref = readExternalRef(constOp);
    if (failed(ref))
      return failure();
    HostEntry entry;
    if (ref->isMemAddr) {
      entry.ptr =
          reinterpret_cast<const void *>(static_cast<uintptr_t>(ref->offset));
    } else {
      entry.filePath = ref->location.str();
      entry.fileOffset = ref->offset;
    }
    recordEntry(constOp, tensorType, ref->size, std::move(entry));
    return success();
  }

  bool empty() const { return constantIndex_ == 0; }

  /// Stamp `hip.constants_file`, write the bytes (full sidecar or partial
  /// mem-addr sidecar), and emit the `hipdnn.constants` descriptor array. The
  /// writers do the I/O and return the per-constant descriptors; this owns the
  /// module attributes so both modes emit them the same way.
  LogicalResult finalize() {
    module_->setAttr(constant_meta::kConstantsFileAttr,
                     StringAttr::get(ctx_, binFileName_));
    FailureOr<SmallVector<Attribute>> descriptors =
        skipDataWrite_ ? writePerEntryDescriptors() : writeFullSidecar();
    if (failed(descriptors))
      return failure();
    module_->setAttr(constant_meta::kConstantsAttr,
                     ArrayAttr::get(ctx_, *descriptors));
    return success();
  }

private:
  /// One record per externalized constant, in emission (index) order. The
  /// active source is discriminated by the fields: splat if `splatElemSize >
  /// 0`, file-ref if `filePath` is non-empty, otherwise mem-addr/dense (`ptr`).
  struct HostEntry {
    const void *ptr = nullptr;
    int64_t splatElemSize = 0;
    std::string filePath;
    int64_t fileOffset = 0;
  };

  static std::string moduleBaseName(ModuleOp module) {
    if (auto sym =
            module->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      return sym.getValue().str();
    return "model";
  }

  /// Append one constant to the 64B-aligned running layout and rewrite it to
  /// the extern `memref.global` + `get_global`/`to_tensor` bridge.
  void recordEntry(Operation *constOp, RankedTensorType tensorType,
                   int64_t byteSize, HostEntry entry) {
    int64_t entryOffset = llvm::alignTo(currentOffset_, kConstantAlignment);
    currentOffset_ = entryOffset + byteSize;

    std::string name = makeGlobalName(constOp);
    recordManifest(name, tensorType, entryOffset, byteSize);
    sizes_.push_back(byteSize);
    offsets_.push_back(entryOffset);
    hostEntries_.push_back(std::move(entry));

    emitExternGlobalBridge(constOp, tensorType, name, entryOffset, byteSize);
    ++constantIndex_;
  }

  /// Reproduce the pre-split naming:
  /// hip_ext_constant_<sanitized-onnx-name>_<idx> (falling back to
  /// node.outputs[0], then a bare index). The raw ONNX name is stashed in
  /// `names_` for the ConstantEntry.
  std::string makeGlobalName(Operation *constOp) {
    std::string name = "hip_ext_constant_";
    std::string onnxName;
    if (auto nodeName = constOp->getAttrOfType<StringAttr>("onnx_node_name")) {
      onnxName = nodeName.getValue().str();
      if (std::string frag = sanitizeForMlirIdentifier(nodeName.getValue());
          !frag.empty())
        name += frag + "_";
    }
    // Initializers carry their tensor name in node.outputs[0], not the
    // (possibly empty) onnx_node_name.
    if (onnxName.empty())
      if (auto outputs = constOp->getAttrOfType<ArrayAttr>("node.outputs"))
        if (!outputs.empty())
          if (auto s = dyn_cast<StringAttr>(outputs.getValue().front()))
            onnxName = s.getValue().str();
    name += std::to_string(constantIndex_);
    names_.push_back(onnxName);
    return name;
  }

  void recordManifest(StringRef name, RankedTensorType tensorType,
                      int64_t offset, int64_t byteSize) {
    llvm::json::Array shape;
    for (int64_t dim : tensorType.getShape())
      shape.push_back(dim);
    manifest_.push_back(llvm::json::Object{
        {"name", name},
        {"shape", std::move(shape)},
        {"element_type", elementTypeToString(tensorType.getElementType())},
        {"offset", offset},
        {"size", byteSize},
        {"alignment", kConstantAlignment},
    });
  }

  void emitExternGlobalBridge(Operation *constOp, RankedTensorType tensorType,
                              StringRef name, int64_t offset,
                              int64_t byteSize) {
    auto memrefType =
        MemRefType::get(tensorType.getShape(), tensorType.getElementType());

    OpBuilder moduleBuilder(module_.getBody(), module_.getBody()->begin());
    auto externalData = moduleBuilder.getDictionaryAttr({
        moduleBuilder.getNamedAttr(
            "index", moduleBuilder.getI64IntegerAttr(constantIndex_)),
        moduleBuilder.getNamedAttr("offset",
                                   moduleBuilder.getI64IntegerAttr(offset)),
        moduleBuilder.getNamedAttr("size",
                                   moduleBuilder.getI64IntegerAttr(byteSize)),
    });
    auto globalOp = memref::GlobalOp::create(
        moduleBuilder, constOp->getLoc(), name,
        /*sym_visibility=*/moduleBuilder.getStringAttr("private"),
        /*type=*/memrefType, /*initial_value=*/nullptr, /*constant=*/false,
        /*alignment=*/moduleBuilder.getI64IntegerAttr(kConstantAlignment));
    globalOp->setAttr("hip.external_data", externalData);

    OpBuilder builder(constOp);
    auto getGlobal = memref::GetGlobalOp::create(builder, constOp->getLoc(),
                                                 memrefType, name);
    auto toTensor = bufferization::ToTensorOp::create(
        builder, constOp->getLoc(), tensorType, getGlobal.getResult(),
        /*restrict=*/builder.getUnitAttr(), /*writable=*/nullptr);
    constOp->getResult(0).replaceAllUsesWith(toTensor.getResult());
    constOp->erase();
  }

  /// Build one `hipdnn.constants` descriptor dictionary for constant `i`.
  /// offset/size/kind are always present; kind-specific keys are added only for
  /// the matching kind, so each entry is self-describing.
  DictionaryAttr
  makeConstantDict(int64_t i, ConstantSourceKind kind, int64_t splatValue = 0,
                   int64_t splatElemSize = 0, StringRef filePath = {},
                   int64_t fileOffset = 0, int64_t sidecarOffset = 0) {
    namespace cm = constant_meta;
    Builder b(ctx_);
    SmallVector<NamedAttribute> f;
    f.emplace_back(b.getStringAttr(cm::kOffset),
                   b.getI64IntegerAttr(offsets_[i]));
    f.emplace_back(b.getStringAttr(cm::kSize), b.getI64IntegerAttr(sizes_[i]));
    f.emplace_back(b.getStringAttr(cm::kKind),
                   b.getI64IntegerAttr(static_cast<int64_t>(kind)));
    switch (kind) {
    case ConstantSourceKind::Splat:
      f.emplace_back(b.getStringAttr(cm::kSplatValue),
                     b.getI64IntegerAttr(splatValue));
      f.emplace_back(b.getStringAttr(cm::kSplatElemSize),
                     b.getI64IntegerAttr(splatElemSize));
      break;
    case ConstantSourceKind::FileRef:
      f.emplace_back(b.getStringAttr(cm::kFilePath), b.getStringAttr(filePath));
      f.emplace_back(b.getStringAttr(cm::kFileOffset),
                     b.getI64IntegerAttr(fileOffset));
      break;
    case ConstantSourceKind::Sidecar:
      f.emplace_back(b.getStringAttr(cm::kSidecarOffset),
                     b.getI64IntegerAttr(sidecarOffset));
      break;
    case ConstantSourceKind::None:
      break;
    }
    return b.getDictionaryAttr(f);
  }

  /// Full-sidecar mode: stream every constant's bytes into constants.bin and
  /// write the JSON manifest; returns one `kind = None` descriptor per
  /// constant.
  FailureOr<SmallVector<Attribute>> writeFullSidecar() {
    std::vector<mlir::hip::ConstantEntry> entries;
    entries.reserve(hostEntries_.size());
    for (auto [i, h] : llvm::enumerate(hostEntries_)) {
      mlir::hip::ConstantEntry e;
      e.name = names_[i];
      e.offset = offsets_[i];
      e.size = sizes_[i];
      e.data = h.ptr;
      e.splat_elem_size = h.splatElemSize;
      e.file_path = h.filePath;
      e.file_offset = h.fileOffset;
      entries.push_back(std::move(e));
    }
    if (!mlir::hip::writeConstantsBinToFileSystem(&fs_, binFileName_, entries,
                                                  currentOffset_)) {
      module_.emitError("failed to write constants binary file: " +
                        binFileName_);
      return failure();
    }

    std::string jsonPath = moduleBaseName(module_) + ".constants.json";
    llvm::json::Object manifest;
    manifest["version"] = 1;
    manifest["binary_file"] = binFileName_;
    manifest["num_constants"] = constantIndex_;
    manifest["total_bytes"] = currentOffset_;
    manifest["constants"] = std::move(manifest_);
    auto jsonWriter = fs_.create_writer_template(jsonPath.c_str());
    if (!jsonWriter) {
      module_.emitError("failed to open constants manifest: " + jsonPath);
      return failure();
    }
    std::string jsonStr;
    llvm::raw_string_ostream jsonOs(jsonStr);
    jsonOs << llvm::formatv("{0:2}", llvm::json::Value(std::move(manifest)));
    jsonWriter->fwrite(jsonStr.data(), jsonStr.size());

    // Every constant's bytes live in the full sidecar at its offset, so all
    // descriptors are kind = None.
    SmallVector<Attribute> dicts;
    dicts.reserve(sizes_.size());
    for (size_t i : llvm::seq(sizes_.size()))
      dicts.push_back(makeConstantDict(i, ConstantSourceKind::None));
    return dicts;
  }

  /// Skip-data mode: pack mem-addr bytes into a partial sidecar and return one
  /// self-describing descriptor per constant. mem-addr entries become Sidecar
  /// (their bytes are only live during this compile); splat/file-ref entries
  /// stream from their descriptor and carry no sidecar bytes.
  FailureOr<SmallVector<Attribute>> writePerEntryDescriptors() {
    // Pass 1: pack mem-addr bytes into a compact 64B-aligned partial sidecar.
    SmallVector<int64_t> sidecarOffsets(hostEntries_.size(), 0);
    std::vector<mlir::hip::ConstantEntry> partialEntries;
    int64_t sidecarPos = 0;
    for (auto [i, h] : llvm::enumerate(hostEntries_)) {
      if (!h.filePath.empty() || h.splatElemSize > 0)
        continue; // file-ref / splat stream from their descriptor
      int64_t off = llvm::alignTo(sidecarPos, kConstantAlignment);
      sidecarOffsets[i] = off;
      mlir::hip::ConstantEntry e;
      e.name = names_[i];
      e.offset = off;
      e.size = sizes_[i];
      e.data = h.ptr;
      partialEntries.push_back(std::move(e));
      sidecarPos = off + sizes_[i];
    }
    if (!partialEntries.empty() &&
        !mlir::hip::writeConstantsBinToFileSystem(&fs_, binFileName_,
                                                  partialEntries, sidecarPos)) {
      module_.emitError("failed to write partial mem-addr sidecar: " +
                        binFileName_);
      return failure();
    }

    // Pass 2: one self-describing descriptor per constant.
    SmallVector<Attribute> dicts;
    dicts.reserve(hostEntries_.size());
    for (auto [i, h] : llvm::enumerate(hostEntries_)) {
      if (!h.filePath.empty()) {
        dicts.push_back(makeConstantDict(i, ConstantSourceKind::FileRef,
                                         /*splatValue=*/0, /*splatElemSize=*/0,
                                         h.filePath, h.fileOffset));
      } else if (h.splatElemSize > 0) {
        // Left-pack up to 8 element bytes into an i64 carrier.
        int64_t splatValue = 0;
        std::memcpy(&splatValue, h.ptr,
                    static_cast<size_t>(std::min<int64_t>(h.splatElemSize, 8)));
        dicts.push_back(makeConstantDict(i, ConstantSourceKind::Splat,
                                         splatValue, h.splatElemSize));
      } else {
        dicts.push_back(makeConstantDict(i, ConstantSourceKind::Sidecar,
                                         /*splatValue=*/0, /*splatElemSize=*/0,
                                         /*filePath=*/{}, /*fileOffset=*/0,
                                         sidecarOffsets[i]));
      }
    }
    return dicts;
  }

  ModuleOp module_;
  MLIRContext *ctx_;
  morphizen::FileSystem &fs_;
  bool skipDataWrite_;
  std::string binFileName_;

  int64_t currentOffset_ = 0;
  int64_t constantIndex_ = 0;
  llvm::SmallVector<int64_t> sizes_;
  llvm::SmallVector<int64_t> offsets_;
  llvm::SmallVector<std::string> names_;
  llvm::SmallVector<HostEntry> hostEntries_;
  llvm::json::Array manifest_;
};

struct ExternalizeConstantsPass
    : public impl::ExternalizeConstantsPassBase<ExternalizeConstantsPass> {
  using ExternalizeConstantsPassBase::ExternalizeConstantsPassBase;

  ExternalizeConstantsPass(morphizen::FileSystem *fs, int64_t minNumElements,
                           bool skipConstantData = false)
      : fileSystem_(fs), fsMinNumElements_(minNumElements),
        skipConstantData_(skipConstantData) {}

  void runOnOperation() override;

  morphizen::FileSystem *fileSystem_ = nullptr;
  int64_t fsMinNumElements_ = 0;
  bool skipConstantData_ = false;
};

void ExternalizeConstantsPass::runOnOperation() {
  ModuleOp module = getOperation();

  // FileSystem + threshold + skip-data come from the ctor (EP live-compile) or
  // the CLI pass options (standalone hip-mlir-opt). CLI falls back to a
  // DiskFileSystem rooted at externalize-output-dir.
  std::unique_ptr<DiskFileSystem> fallbackFs;
  morphizen::FileSystem *fs = fileSystem_;
  int64_t minElems =
      fileSystem_ ? fsMinNumElements_ : externalizeMinNumElements.getValue();
  bool skipData = fileSystem_ ? skipConstantData_ : skipConstantData.getValue();
  if (!fs) {
    StringRef dir = externalizeOutputDir.getValue();
    fallbackFs =
        std::make_unique<DiskFileSystem>(dir.empty() ? "." : dir.str().c_str());
    fs = fallbackFs.get();
  }

  const bool externalizing = minElems > 0;
  std::optional<ConstantExternalizer> externalizer;
  if (externalizing)
    externalizer.emplace(module, *fs, skipData);

  // Collect first: rewriting/erasing carriers during the walk would invalidate
  // it.
  SmallVector<hip::ConstantOp> constants;
  module.walk([&](hip::ConstantOp c) { constants.push_back(c); });

  for (hip::ConstantOp constOp : constants) {
    auto tensorType = dyn_cast<RankedTensorType>(constOp.getResult().getType());
    if (!tensorType) {
      constOp.emitError("hip.constant has non-ranked result type");
      return signalPassFailure();
    }

    if (auto valueAttr =
            dyn_cast_if_present<DenseElementsAttr>(constOp.getValueAttr())) {
      // Location constants are always externalized; value constants only at or
      // above the element-count threshold, else they fold inline.
      if (externalizing && valueAttr.getNumElements() >= minElems)
        externalizer->externalizeValue(constOp, tensorType, valueAttr);
      else
        replaceWithArithConstant(constOp, valueAttr);
      continue;
    }
    if (constOp.getLocationAttr()) {
      LogicalResult r =
          externalizing ? externalizer->externalizeLocation(constOp, tensorType)
                        : materializeLocationInline(constOp, tensorType);
      if (failed(r))
        return signalPassFailure();
      continue;
    }
    constOp.emitError("hip.constant missing value or location attribute");
    return signalPassFailure();
  }

  if (externalizer && !externalizer->empty())
    if (failed(externalizer->finalize()))
      return signalPassFailure();
}

} // namespace

std::unique_ptr<mlir::Pass>
createExternalizeConstantsPass(morphizen::FileSystem *fs,
                               int64_t minNumElements, bool skipConstantData) {
  return std::make_unique<ExternalizeConstantsPass>(fs, minNumElements,
                                                    skipConstantData);
}

} // namespace hip
} // namespace mlir
