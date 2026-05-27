/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Compiler/PluginLoader.h"
#include "hip/Compiler/PluginRegistry.h"
#include "hip/debug_log.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <exception>
#include <mutex>
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

// Split `HIP_EP_PLUGINS` on `;`, trim whitespace, and skip empties.
//
// Deduplicates by exact-string match so `foo.dll;foo.dll` produces
// one entry. We deliberately do not canonicalize paths (no
// `realpath` / `GetFullPathName`): a vendor pinning their plugin
// through both an absolute path and a symlink should ideally pin
// once; if they accidentally list both spellings the worst case is
// two plugin loads, which the underlying OS deduplicates at the
// HMODULE / dlopen-refcount level. The visible-to-the-user
// dedup catches the common `foo.dll;foo.dll` typo and keeps the
// implementation honest about what it guarantees.
void splitPluginPaths(llvm::StringRef envValue,
                      std::vector<std::string> &outPaths) {
  llvm::SmallSet<std::string, 4> seen;
  llvm::StringRef remainder = envValue;
  while (!remainder.empty()) {
    auto sep = remainder.find(kPluginsEnvSeparator);
    llvm::StringRef token =
        (sep == llvm::StringRef::npos) ? remainder : remainder.substr(0, sep);
    token = token.trim();
    if (!token.empty()) {
      std::string canonical = token.str();
      if (seen.insert(canonical).second)
        outPaths.emplace_back(std::move(canonical));
    }
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
        // Bad plugin path is non-fatal but loud: warn unconditionally
        // (not gated on HIPDNN_EP_DEBUG) so a typo or missing-DLL
        // mistake surfaces immediately rather than producing a silent
        // "is my plugin loaded?" mystery.
        llvm::errs() << "[plugin-loader] WARNING: failed to load '" << path
                     << "': " << llvm::toString(plugin.takeError()) << "\n";
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

void dispatchPluginRegistrationsOnce() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    // The registry returned here is a facade whose methods all
    // dispatch through a vtable that writes into per-process storage
    // in PluginRegistry.cpp. The plugin DLL therefore never needs to
    // resolve a hip-compiler symbol across the DLL boundary; it only
    // calls inline thunks that read function pointers off the
    // registry instance.
    HipEpPluginRegistry &registry = getProcessPluginRegistry();
    for (const auto &plugin : loadPluginsOnce()) {
      // A plugin that throws across the DLL boundary is technically
      // undefined behaviour (CRT / libstdc++ versions may not match
      // between host and plugin). We bound the blast radius: catch
      // anything escaping `RegisterCallbacks`, log it, and continue
      // with the next plugin. This is similar in spirit to
      // `mlir-opt`'s handling of plugin-load failures, which also
      // chooses degrade-and-continue over abort-the-host.
      try {
        plugin.registerCallbacks(registry);
      } catch (const std::exception &e) {
        llvm::errs() << "[plugin-loader] WARNING: '" << plugin.getPluginName()
                     << "' RegisterCallbacks threw std::exception: " << e.what()
                     << "; skipping further callbacks for this "
                     << "plugin.\n";
      } catch (...) {
        llvm::errs() << "[plugin-loader] WARNING: '" << plugin.getPluginName()
                     << "' RegisterCallbacks threw a non-std exception; "
                     << "skipping further callbacks for this plugin.\n";
      }
    }
  });
}

} // namespace hip::compiler
