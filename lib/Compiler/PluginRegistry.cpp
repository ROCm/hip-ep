/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/PluginRegistry.h"

#include "hip/Compiler/PluginLoader.h"

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
// Concurrency: plugin registration is one-shot. dispatchPluginRegis-
// trationsOnce() runs every plugin's RegisterCallbacks behind a
// std::call_once, which serializes all writes. Reads (passesForSlot
// and the bitcode / library accessors below) happen after that completes,
// and each accessor calls dispatchPluginRegistrationsOnce() itself
// as a defensive idempotent step, so a future caller that bypasses
// CompilerDriver::compile (a new tool, a unit test, etc.) still
// sees plugin-contributed state.
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
  // the bytes live in `bitcodeStorage`, and `bitcodeBuffers[i].data` is
  // `bitcodeStorage[i].data()`.
  //
  // Type choice: `std::vector<unsigned char>` (rather than
  // `std::string`) avoids the small-string-optimization trap: when
  // the outer vector grows past its capacity, every inner element
  // is move-constructed into the new buffer. For an SSO-sized
  // `std::string` the moved-into copy lives in a different memory
  // address, invalidating any `data()` pointer we saved earlier.
  // `std::vector` of any size is always heap-backed, so move just
  // steals the heap pointer and the bytes never relocate -- the
  // `data()` we recorded in `bitcodeBuffers` remains valid.
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
  // into the plugin DLL's code, which stays mapped for the process lifetime.
  std::vector<void (*)(mlir::DialectRegistry &)> dialectRegistrations;
};

Storage &storage() {
  static Storage s;
  return s;
}

void requestPipelineSlotImpl(void * /*self*/, int slot, const char *name,
                             std::size_t nameLen) {
  // Reject out-of-range slots loudly. A plugin built against a
  // newer header that defines additional slots will pass an int
  // we don't know how to dispatch; recording it would be a silent
  // miss because pluginPassesForSlot() filters by slot equality.
  if (slot < 0 || slot >= kPipelineSlotCount) {
    llvm::errs() << "[plugin-loader] WARNING: requestPipelineSlot(slot=" << slot
                 << ", name='";
    llvm::errs().write(name, nameLen);
    llvm::errs() << "') -- slot value is out of range [0, "
                 << kPipelineSlotCount
                 << "). The plugin was likely built against a newer "
                 << "PluginRegistry.h. Dropping this request.\n";
    return;
  }
  // We own the std::string copy because the plugin's StringRef may
  // point into the plugin DLL's read-only data, and the registry must
  // outlive any individual RegisterCallbacks call.
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
    llvm::errs() << "[plugin-loader] WARNING: addRuntimeBitcode called with an "
                 << "empty buffer (size=" << sizeBytes
                 << "); skipping. Wrap the "
                 << "addRuntimeBitcode call in `if (size != 0)` if your plugin "
                 << "produces bitcode conditionally.\n";
    return;
  }

  // Copy the bytes. The plugin's pointer/lifetime guarantees are
  // hard to enforce across a DLL boundary (a stack buffer, a
  // dlclose'd plugin, an aliased view, etc. are all possible
  // misuses), so the host owns the copy. Vendor runtime bitcode
  // is typically 100 kB-1 MB and the copy happens once per process
  // at startup, well below noise -- earlier versions of this code
  // borrowed the pointer and the safety risk was not worth the
  // savings.
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
    llvm::errs() << "[plugin-loader] WARNING: addDialectRegistration called "
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
  // Defensive: ensure plugin registrations have run before reading
  // out their results. dispatchPluginRegistrationsOnce is
  // std::call_once-guarded so this is essentially free after the
  // first call.
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
  // Copy out a SmallVector view of the buffers. Buffers are tiny
  // (struct of 16 bytes on x64) and there are typically 0-2 of
  // them, so the copy is negligible and saves callers from worrying
  // about mutation between the call and use.
  llvm::SmallVector<PluginBitcodeBuffer> result;
  result.reserve(storage().bitcodeBuffers.size());
  for (const auto &buf : storage().bitcodeBuffers)
    result.push_back(buf);
  return result;
}

llvm::SmallVector<std::string> pluginLibraryPaths() {
  dispatchPluginRegistrationsOnce();
  llvm::SmallVector<std::string> result;
  result.reserve(storage().libraryPaths.size());
  for (const auto &p : storage().libraryPaths)
    result.push_back(p);
  return result;
}

llvm::SmallVector<std::string> pluginLibraries() {
  dispatchPluginRegistrationsOnce();
  llvm::SmallVector<std::string> result;
  result.reserve(storage().libraries.size());
  for (const auto &l : storage().libraries)
    result.push_back(l);
  return result;
}

llvm::SmallVector<void (*)(mlir::DialectRegistry &)>
pluginDialectRegistrations() {
  dispatchPluginRegistrationsOnce();
  llvm::SmallVector<void (*)(mlir::DialectRegistry &)> result;
  result.reserve(storage().dialectRegistrations.size());
  for (auto fn : storage().dialectRegistrations)
    result.push_back(fn);
  return result;
}

} // namespace hip::compiler
