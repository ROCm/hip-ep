/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ExternalizeConstants.cpp - hipsr.constant -> constants file -------===//
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Support/ConstantsIO.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectResourceBlobManager.h"

#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <vector>

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_HIPSREXTERNALIZECONSTANTSPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

// Constants-file alignment: every constant starts on a 64-byte boundary (GPU
// alignment; matches the hip.* constants-file layout).
constexpr int64_t kConstantAlignment = 64;

constexpr llvm::StringLiteral kFileKeyPrefix = "file|";

struct HipsrExternalizeConstantsPass
    : impl::HipsrExternalizeConstantsPassBase<HipsrExternalizeConstantsPass> {
  using impl::HipsrExternalizeConstantsPassBase<
      HipsrExternalizeConstantsPass>::HipsrExternalizeConstantsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    Builder builder(ctx);

    // Phase 1: collect entries and stamp offset/size across the whole module.
    // Module-scoped so a single cumulative offset feeds one shared constants
    // file (see Passes.td). Independent of the FileSystem so the IR transform
    // is deterministic with or without a sink.
    std::vector<hip::ConstantEntry> entries;
    int64_t filePos = 0;
    int64_t constantIndex = 0;
    module.walk([&](ConstantOp c) {
      hip::ConstantEntry entry;
      ElementsAttr value = c.getValue();
      llvm::ArrayRef<char> bytes;
      if (auto resource = dyn_cast<DenseResourceElementsAttr>(value)) {
        bytes = resource.getData();
        StringRef key = resource.getRawHandle().getKey();
        if (key.consume_front(kFileKeyPrefix)) {
          // File-ref: keep the on-disk reference (path/offset); this pass does
          // not read the (possibly multi-GB) weight file.
          auto [path, offsetText] = key.rsplit('|');
          entry.file_path = path.str();
          offsetText.getAsInteger(10, entry.file_offset);
        } else {
          entry.data = bytes.data();
        }
      } else {
        bytes = cast<DenseElementsAttr>(value).getRawData();
        entry.data = bytes.data();
      }

      int64_t size = static_cast<int64_t>(bytes.size());
      int64_t offset = llvm::alignTo(filePos, kConstantAlignment);
      entry.offset = offset;
      entry.size = size;
      entry.splat_elem_size = 0; // MorphiZen never emits splat constants.

      c.setOffsetAttr(builder.getI64IntegerAttr(offset));
      c.setSizeAttr(builder.getI64IntegerAttr(size));
      c.setIndexAttr(builder.getI64IntegerAttr(constantIndex++));
      filePos = offset + size;

      entries.push_back(std::move(entry));
    });

    // Phase 2: write the constants file, only when a FileSystem was injected
    // (e.g. by the compile driver). Standalone runs (hip-mlir-opt) inject none,
    // so the pass is a pure IR transform there.
    morphizen::FileSystem *fs = nullptr;
    if (auto *dialect = ctx->getLoadedDialect<HipsrDialect>()) {
      fs = dialect->getFileSystem();
    }
    if (fs && !entries.empty()) {
      if (!hip::writeConstantsBinToFileSystem(
              fs, constantsFile, entries,
              llvm::alignTo(filePos, kConstantAlignment))) {
        module.emitError("failed to write constants file: ") << constantsFile;
        signalPassFailure();
      }
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
