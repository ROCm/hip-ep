/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/PluginRegistry.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>
#include <utility>
#include <vector>

namespace hip::compiler {

// ============================================================================
// Per-process plugin registry storage and vtable.
//
// The HipEpPluginRegistry class is a thin facade -- its inline thunks
// dispatch through a function-pointer table populated here. We keep
// the storage in this TU so the header has no std::vector dependency
// and the storage layout can grow in PRs 3 and 4 without touching
// the public ABI.
//
// Concurrency: plugin registration is one-shot. dispatchPluginRegis-
// trationsOnce() runs every plugin's RegisterCallbacks behind a
// std::call_once, which serializes all writes. Reads (passesForSlot,
// future PR-3/PR-4 accessors) happen after that completes. No
// additional locking is needed -- this storage is effectively
// "written once at startup, read many afterwards", which is the same
// model llvm/mlir's pass registries use.
// ============================================================================

namespace {

struct Storage {
  // PR 2: (slot, passName) pairs recorded by plugin
  // requestPipelineSlot calls.
  std::vector<std::pair<PipelineSlot, std::string>> slotRequests;

  // PR 3 will append: bitcode buffers contributed by plugins.
  // PR 4 will append: library paths and library names.
};

Storage &storage() {
  static Storage s;
  return s;
}

void requestPipelineSlotImpl(void * /*self*/, int slot, const char *name,
                             std::size_t nameLen) {
  // We own the std::string copy because the plugin's StringRef may
  // point into the plugin DLL's read-only data, and the registry must
  // outlive any individual RegisterCallbacks call.
  storage().slotRequests.emplace_back(static_cast<PipelineSlot>(slot),
                                      std::string(name, nameLen));
}

void addRuntimeBitcodeImpl(void * /*self*/, const void * /*data*/,
                           std::size_t /*sizeBytes*/) {
  // PR 3 fills this in.
}

void addLibraryPathImpl(void * /*self*/, const char * /*path*/,
                        std::size_t /*pathLen*/) {
  // PR 4 fills this in.
}

void addLibraryImpl(void * /*self*/, const char * /*name*/,
                    std::size_t /*nameLen*/) {
  // PR 4 fills this in.
}

const HipEpPluginRegistry::VTable g_vtable = {
    /* requestPipelineSlot = */ &requestPipelineSlotImpl,
    /* addRuntimeBitcode   = */ &addRuntimeBitcodeImpl,
    /* addLibraryPath      = */ &addLibraryPathImpl,
    /* addLibrary          = */ &addLibraryImpl,
};

} // namespace

HipEpPluginRegistry &getProcessPluginRegistry() {
  // The `self` pointer is unused by the current vtable functions
  // (storage is accessed through a function-local static), so we
  // can pass nullptr. Future per-instance state would use it.
  static HipEpPluginRegistry instance(&g_vtable, /*self=*/nullptr);
  return instance;
}

llvm::SmallVector<llvm::StringRef> pluginPassesForSlot(PipelineSlot slot) {
  llvm::SmallVector<llvm::StringRef> result;
  for (const auto &[s, name] : storage().slotRequests) {
    if (s == slot)
      result.emplace_back(name);
  }
  return result;
}

} // namespace hip::compiler
