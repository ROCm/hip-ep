/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ExternalizeConstants.cpp - hipsr.constant -> sidecar --------------===//
//
// Phase 2 of the hipsr constant subsystem. Two stages inside one pass:
//
//   Phase 1 (always): walk each opted-in hipsr.constant, assign a 64-byte
//     aligned cumulative offset, and stamp offset/size on the op (append-only
//     -- value/source are kept). Builds one hip::ConstantEntry per constant.
//   Phase 2 (only when a FileSystem is injected on the dialect): write the
//     entries to the sidecar via writeConstantsBinToFileSystem.
//
// file_source entries carry their file_path/offset only (data = nullptr) so
// this pass does not read the weight file; inline / mem_source entries carry a
// byte view (getDataValues). See Passes.td.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Dialect/Hipsr/IR/HipsrConstantOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Support/ConstantsIO.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <vector>

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_HIPSREXTERNALIZECONSTANTSPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

// Sidecar alignment: every constant starts on a 64-byte boundary (GPU
// alignment; matches the hip.* sidecar layout).
constexpr int64_t kConstantAlignment = 64;

struct HipsrExternalizeConstantsPass
    : impl::HipsrExternalizeConstantsPassBase<HipsrExternalizeConstantsPass> {
  using impl::HipsrExternalizeConstantsPassBase<
      HipsrExternalizeConstantsPass>::HipsrExternalizeConstantsPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    Builder builder(ctx);

    // Phase 1: collect entries and stamp offset/size across the whole module.
    // Module-scoped so a single cumulative offset feeds one shared sidecar
    // (see Passes.td). Independent of the FileSystem so the IR transform is
    // deterministic with or without a sink.
    std::vector<hip::ConstantEntry> entries;
    int64_t filePos = 0;
    module.walk([&](ConstantOp c) {
      if (!c.shouldExternalize() || c.isExternalized()) {
        return;
      }

      hip::ConstantEntry entry;
      int64_t size = 0;
      if (auto fileSrc =
              llvm::dyn_cast_or_null<FileSourceAttr>(c.getSourceAttr())) {
        // File-ref: keep the on-disk reference (path/offset); this pass does
        // not read the (possibly multi-GB) weight file.
        entry.file_path = fileSrc.getPath().getValue().str();
        entry.file_offset = fileSrc.getOffset();
        size = fileSrc.getSize();
      } else {
        // inline value or mem_source: a byte view. getDataValues does not read
        // the bytes here (it only forms the pointer/size); the writer reads
        // them in Phase 2.
        llvm::ArrayRef<uint8_t> bytes = c.getDataValues<uint8_t>();
        entry.data = bytes.data();
        size = static_cast<int64_t>(bytes.size());
      }

      int64_t offset = llvm::alignTo(filePos, kConstantAlignment);
      entry.offset = offset;
      entry.size = size;
      entry.splat_elem_size = 0; // MorphiZen never emits splat constants.

      c.setOffsetAttr(builder.getI64IntegerAttr(offset));
      c.setSizeAttr(builder.getI64IntegerAttr(size));
      filePos = offset + size;

      entries.push_back(std::move(entry));
    });

    // Phase 2: write the sidecar, only when a FileSystem was injected (e.g. by
    // the compile driver). Standalone runs (hip-mlir-opt) inject none, so the
    // pass is a pure IR transform there.
    morphizen::FileSystem *fs = nullptr;
    if (auto *dialect = ctx->getLoadedDialect<HipsrDialect>()) {
      fs = dialect->getFileSystem();
    }
    if (fs && !entries.empty()) {
      if (!hip::writeConstantsBinToFileSystem(
              fs, constantsFile, entries,
              llvm::alignTo(filePos, kConstantAlignment))) {
        module.emitError("failed to write constants sidecar: ")
            << constantsFile;
        signalPassFailure();
      }
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
