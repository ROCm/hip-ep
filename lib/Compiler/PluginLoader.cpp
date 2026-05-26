/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/PluginLoader.h"
#include "hip/Compiler/PluginRegistry.h"
#include "hip/debug_log.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>
#include <vector>

namespace hip::compiler {

namespace {

constexpr const char *kEntrySymbol = "hipEpGetPluginInfo";
constexpr const char *kPluginsEnvVar = "HIP_EP_PLUGINS";

// Path-separator for HIP_EP_PLUGINS. We use ';' on every platform —
// Windows would clash with Unix ':' over drive letters (C:\foo:bar)
// so semicolon is the safer choice. Documented in PluginLoader.h.
constexpr char kPluginsEnvSeparator = ';';

llvm::Error makeError(const llvm::Twine &msg) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), msg);
}

void splitPluginPaths(llvm::StringRef envValue,
                      std::vector<std::string> &outPaths) {
  llvm::StringRef remainder = envValue;
  while (!remainder.empty()) {
    auto sep = remainder.find(kPluginsEnvSeparator);
    llvm::StringRef token =
        (sep == llvm::StringRef::npos) ? remainder : remainder.substr(0, sep);
    token = token.trim();
    if (!token.empty())
      outPaths.emplace_back(token.str());
    if (sep == llvm::StringRef::npos)
      break;
    remainder = remainder.substr(sep + 1);
  }
}

} // namespace

llvm::Expected<HipEpPluginLoader>
HipEpPluginLoader::Load(const std::string &filename) {
  std::string err;

  // `LoadLibraryPermanently` returns true on FAILURE (matches the
  // upstream `llvm::sys::DynamicLibrary` API surface). We don't need
  // the returned `DynamicLibrary` from this call — we just want the
  // file to be permanently loaded into the process so subsequent
  // `getPermanentLibrary` calls succeed.
  if (llvm::sys::DynamicLibrary::LoadLibraryPermanently(filename.c_str(),
                                                        &err)) {
    return makeError(llvm::Twine("Could not load plugin '") + filename +
                     "': " + err);
  }

  llvm::sys::DynamicLibrary library =
      llvm::sys::DynamicLibrary::getPermanentLibrary(filename.c_str(), &err);
  if (!library.isValid()) {
    return makeError(llvm::Twine("Plugin '") + filename +
                     "' loaded but DynamicLibrary handle is invalid: " + err);
  }

  void *entrySymbol = library.getAddressOfSymbol(kEntrySymbol);
  if (!entrySymbol) {
    return makeError(llvm::Twine("Plugin '") + filename +
                     "' does not export '" + kEntrySymbol +
                     "'. Is this actually a hip-compiler plugin?");
  }

  using EntryFnTy = HipEpPluginLibraryInfo();
  auto entryFn = reinterpret_cast<EntryFnTy *>(entrySymbol);
  HipEpPluginLibraryInfo info = entryFn();

  if (info.APIVersion != HIP_EP_PLUGIN_API_VERSION) {
    return makeError(
        llvm::Twine("Plugin '") + filename +
        "' has incompatible API version: plugin reports " +
        llvm::Twine(info.APIVersion) + ", hip-compiler expects " +
        llvm::Twine(HIP_EP_PLUGIN_API_VERSION) +
        ". Rebuild the plugin against the matching hip-compiler headers.");
  }

  if (!info.PluginName) {
    return makeError(llvm::Twine("Plugin '") + filename +
                     "' returned a null PluginName.");
  }
  if (!info.PluginVersion) {
    return makeError(llvm::Twine("Plugin '") + filename +
                     "' returned a null PluginVersion.");
  }

  return HipEpPluginLoader(filename, library, info);
}

const std::vector<HipEpPluginLoader> &loadPluginsOnce() {
  static const std::vector<HipEpPluginLoader> plugins = [] {
    std::vector<HipEpPluginLoader> loaded;

    std::string envValue = hip_get_env(kPluginsEnvVar);
    if (envValue.empty())
      return loaded;

    std::vector<std::string> paths;
    splitPluginPaths(envValue, paths);

    for (const std::string &path : paths) {
      auto plugin = HipEpPluginLoader::Load(path);
      if (!plugin) {
        // Bad plugin path is non-fatal: log under HIPDNN_EP_DEBUG and
        // continue. We do not want a typo in HIP_EP_PLUGINS to block
        // compilation when the user is iterating.
        std::string msg;
        llvm::raw_string_ostream(msg)
            << "[plugin-loader] failed to load '" << path
            << "': " << llvm::toString(plugin.takeError()) << "\n";
        COMPILER_DEBUG_LOG(msg);
        continue;
      }

      COMPILER_DEBUG_LOG("[plugin-loader] loaded "
                         << plugin->getPluginName() << " "
                         << plugin->getPluginVersion() << " from "
                         << plugin->getFilename() << "\n");
      loaded.push_back(std::move(*plugin));
    }

    return loaded;
  }();
  return plugins;
}

// ============================================================================
// HipEpPluginRegistry stub method bodies.
//
// In PR 1 these are intentionally empty. Subsequent PRs fill them in:
//   PR 2: requestPipelineSlot() records into a per-process registry
//         consulted by lib/Dialect/Transforms/Pipelines.cpp.
//   PR 3: addRuntimeBitcode() records buffers consumed by
//         lib/Target/LLVM/LLVMBackend.cpp::linkRuntimeModule.
//   PR 4: addLibraryPath() / addLibrary() record entries appended in
//         lib/Compiler/CompilerDriver.cpp::discoverLibraries.
//
// They live in this translation unit (and not their own
// PluginRegistry.cpp) until PR 2 grows their bodies — keeping the PR
// 1 file count minimal.
// ============================================================================

void HipEpPluginRegistry::requestPipelineSlot(PipelineSlot, llvm::StringRef) {
  // PR 2 fills this in.
}

void HipEpPluginRegistry::addRuntimeBitcode(const void *, std::size_t) {
  // PR 3 fills this in.
}

void HipEpPluginRegistry::addLibraryPath(llvm::StringRef) {
  // PR 4 fills this in.
}

void HipEpPluginRegistry::addLibrary(llvm::StringRef) {
  // PR 4 fills this in.
}

} // namespace hip::compiler
