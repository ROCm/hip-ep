/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/PluginRegistry.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>
#include <vector>

namespace hip::compiler {

// ============================================================================
// Per-process plugin registry storage and vtable.
//
// The HipEpPluginRegistry class is a thin facade -- its inline thunks
// dispatch through a function-pointer table populated here. The
// storage lives in this TU so the header has no std::vector
// dependency.
//
// Concurrency: plugin registration is one-shot.
// dispatchPluginRegistrationsOnce() (StaticPlugins.cpp) runs every
// statically-linked plugin's registration entry behind a std::call_once, which
// serializes all writes. Reads (passesForSlot and the bitcode / library
// accessors below) happen after that completes, and each accessor calls
// dispatchPluginRegistrationsOnce() itself as a defensive idempotent step, so a
// future caller that bypasses CompilerDriver::compile (a new tool, a unit test,
// etc.) still sees plugin-contributed state.
// ============================================================================

namespace {

// Number of PipelineSlot enumerators. Must be kept in sync with the
// PipelineSlot enum in PluginRegistry.h. We assert below that no
// enumerator drifts past this value; adding a new one is the only
// supported change (append-only across versions, per the public
// header's contract).
constexpr int kPipelineSlotCount = 7;
static_assert(static_cast<int>(PipelineSlot::AfterGenerateInterface) ==
                  kPipelineSlotCount - 1,
              "kPipelineSlotCount drifted from the PipelineSlot enum -- "
              "update PluginRegistry.cpp to match.");

struct Storage {
  // (slot, passName) pairs recorded by plugin requestPipelineSlot calls.
  std::vector<std::pair<PipelineSlot, std::string>> slotRequests;

  // Bitcode buffers contributed by plugin addRuntimeBitcode calls. Host-owned:
  // the bytes live in `bitcodeStorage`, and `bitcodeBuffers[i].data` points at
  // `bitcodeStorage[i].data()`.
  //
  // `std::vector<unsigned char>` (not `std::string`) is deliberate: growing the
  // outer vector move-constructs each element, and an SSO-sized `std::string`
  // would relocate its bytes, invalidating a `data()` pointer saved earlier. A
  // heap-backed `std::vector` move just steals the pointer, so the bytes -- and
  // the recorded `data()` -- never move.
  std::vector<std::vector<unsigned char>> bitcodeStorage;
  std::vector<PluginBitcodeBuffer> bitcodeBuffers;

  // Library search paths and library names contributed by plugin
  // addLibraryPath / addLibrary calls. The plugin's StringRef may point into
  // transient storage, so the registry owns the std::string copies.
  std::vector<std::string> libraryPaths;
  std::vector<std::string> libraries;

  // Dialect-registration callbacks contributed by plugin
  // addDialectRegistration calls. Plain function pointers (the public API
  // requires non-capturing callbacks), so no ownership concern -- they point
  // into the plugin's code (linked into the host binary), which is mapped for
  // the process lifetime.
  std::vector<void (*)(mlir::DialectRegistry &)> dialectRegistrations;
};

Storage &storage() {
  static Storage s;
  return s;
}

void requestPipelineSlotImpl(void * /*self*/, int slot, const char *name,
                             std::size_t nameLen) {
  // Defensively reject an out-of-range slot loudly rather than record a value
  // pluginPassesForSlot() would silently never match. With static linking the
  // plugin and host share this header, so this should not happen -- an
  // out-of-range value here means a plugin bug (a bad cast or uninitialized
  // slot), which is worth surfacing.
  if (slot < 0 || slot >= kPipelineSlotCount) {
    llvm::errs() << "[plugin] WARNING: requestPipelineSlot(slot=" << slot
                 << ", name='";
    llvm::errs().write(name, nameLen);
    llvm::errs() << "') -- slot value is out of range [0, "
                 << kPipelineSlotCount << "). Dropping this request.\n";
    return;
  }
  // We own the std::string copy because the plugin's StringRef may
  // point into the plugin's read-only data, and the registry must
  // outlive any individual plugin registration call.
  storage().slotRequests.emplace_back(static_cast<PipelineSlot>(slot),
                                      std::string(name, nameLen));
}

void addRuntimeBitcodeImpl(void * /*self*/, const void *data,
                           std::size_t sizeBytes) {
  // Empty buffer: log and skip. Without this guard we'd hand a
  // 0-byte MemoryBuffer to llvm::parseBitcodeFile, which fails with
  // "file too small to contain bitcode header" (see
  // llvm/lib/Bitcode/Reader/BitcodeReader.cpp::hasInvalidBitcodeHeader)
  // -- a confusing failure mode for a plugin author who simply has
  // no bitcode to contribute on this build.
  if (sizeBytes == 0 || data == nullptr) {
    llvm::errs() << "[plugin] WARNING: addRuntimeBitcode called with an "
                 << "empty buffer (size=" << sizeBytes
                 << "); skipping. Wrap the "
                 << "addRuntimeBitcode call in `if (size != 0)` if your plugin "
                 << "produces bitcode conditionally.\n";
    return;
  }

  // Copy the bytes: the plugin's pointer/lifetime is hard to enforce across the
  // registration boundary (stack buffer, aliased view, ...), so the host owns
  // the copy. A one-time startup cost of ~100 kB-1 MB.
  auto &store = storage();
  const auto *bytes = static_cast<const unsigned char *>(data);
  store.bitcodeStorage.emplace_back(bytes, bytes + sizeBytes);
  const auto &owned = store.bitcodeStorage.back();
  store.bitcodeBuffers.push_back(
      PluginBitcodeBuffer{owned.data(), owned.size()});
}

void addLibraryPathImpl(void * /*self*/, const char *path,
                        std::size_t pathLen) {
  storage().libraryPaths.emplace_back(path, pathLen);
}

void addLibraryImpl(void * /*self*/, const char *name, std::size_t nameLen) {
  storage().libraries.emplace_back(name, nameLen);
}

void addDialectRegistrationImpl(void * /*self*/,
                                void (*registerFn)(mlir::DialectRegistry &)) {
  if (registerFn == nullptr) {
    llvm::errs() << "[plugin] WARNING: addDialectRegistration called "
                 << "with a null callback; skipping.\n";
    return;
  }
  storage().dialectRegistrations.push_back(registerFn);
}

const HipEpPluginRegistry::VTable g_vtable = {
    /* requestPipelineSlot     = */ &requestPipelineSlotImpl,
    /* addRuntimeBitcode       = */ &addRuntimeBitcodeImpl,
    /* addLibraryPath          = */ &addLibraryPathImpl,
    /* addLibrary              = */ &addLibraryImpl,
    /* addDialectRegistration  = */ &addDialectRegistrationImpl,
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
  dispatchPluginRegistrationsOnce();
  llvm::SmallVector<llvm::StringRef> result;
  for (const auto &[s, name] : storage().slotRequests) {
    if (s == slot)
      result.emplace_back(name);
  }
  return result;
}

llvm::SmallVector<PluginBitcodeBuffer> pluginBitcodeBuffers() {
  dispatchPluginRegistrationsOnce();
  // Return a value copy so callers cannot alias internal storage; the buffers
  // are tiny and there are typically 0-2 of them.
  return llvm::to_vector(storage().bitcodeBuffers);
}

llvm::SmallVector<std::string> pluginLibraryPaths() {
  dispatchPluginRegistrationsOnce();
  return llvm::to_vector(storage().libraryPaths);
}

llvm::SmallVector<std::string> pluginLibraries() {
  dispatchPluginRegistrationsOnce();
  return llvm::to_vector(storage().libraries);
}

llvm::SmallVector<void (*)(mlir::DialectRegistry &)>
pluginDialectRegistrations() {
  dispatchPluginRegistrationsOnce();
  return llvm::to_vector(storage().dialectRegistrations);
}

} // namespace hip::compiler
